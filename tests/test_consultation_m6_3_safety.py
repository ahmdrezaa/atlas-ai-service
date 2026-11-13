"""Offline safety checks plus explicit, strict xfails for unresolved semantics."""

import copy
import hashlib
import json
from concurrent import futures
from unittest.mock import Mock

import grpc
import pytest
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.models import HistoricalSolution
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.solution import (
    GeneratedSolutionTurn,
    solution_provider_schema,
    solution_to_proto,
)
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.server import BearerAuthInterceptor
from scripts.consultation_m6_3_evaluation import CORPUS, ROOT, case_input, evaluate

CORPUS_DATA = json.loads(CORPUS.read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in CORPUS_DATA["cases"]}
UNRESOLVED = {
    "a_independent_coordinate_fact": "Factual predicate inside shared coordination is unresolved",
    "a_two_speculations": "Modality and relevance inside shared coordination are unresolved",
    "a_bakery_causal": "Existing causal guard rejects a legitimate tentative bakery cause",
    "b_original": "Shared nominal coordination cannot yet establish semantic relevance",
}


def mapped(request, payload):
    return solution_to_proto(
        GeneratedSolutionTurn.model_validate(payload).solution_output(),
        prepare_context(request),
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case["id"],
            marks=[pytest.mark.xfail(strict=True, reason=UNRESOLVED[case["id"]])]
            if case["id"] in UNRESOLVED
            else [],
        )
        for case in CASES.values()
        if case["expected"] != "ambiguous"
    ],
)
def test_frozen_semantic_expectations(case):
    request, payload = case_input(case)
    before_request, before_payload = request.SerializeToString(), copy.deepcopy(payload)
    if case["expected"] == "valid":
        response = mapped(request, payload)
        assert all(
            getattr(response, field.name) for field in response.DESCRIPTOR.fields
        )
        assert ai_pb2.Solution.FromString(response.SerializeToString()) == response
    else:
        with pytest.raises(ConsultationFailure):
            mapped(request, payload)
    assert request.SerializeToString() == before_request
    assert payload == before_payload


def test_frozen_comparison_tracks_errors_without_treating_ambiguity_as_truth():
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == (
        "530a77b08f2c014ad72a87c872b9bb0b68010f698a5e6bb92762fad432e7ebe5"
    )
    ledger = json.loads(
        (ROOT / "evaluation/m6_3_grounding_safety.json").read_text(encoding="utf-8")
    )
    current = evaluate()
    before = ledger["before"]["new_corpus"]
    assert current["corpus_sha256"] == before["corpus_sha256"]
    assert current["counts"]["valid_rejected"] == before["counts"]["valid_rejected"]
    assert current["counts"]["invalid_accepted"] < before["counts"]["invalid_accepted"]
    previous = {c["id"]: c for c in before["cases"]}
    for case in current["cases"]:
        if case["expected"] == "invalid" and case["accepted"]:
            assert previous[case["id"]]["accepted"]  # No newly unsafe acceptance.
            assert case["id"] in UNRESOLVED
    assert sum(c["expected"] == "ambiguous" for c in current["cases"]) == 2


@pytest.mark.parametrize("connector", ["tetapi", "namun", "padahal", "sedangkan"])
@pytest.mark.parametrize("supported", [True, False])
def test_contrast_fact_needs_actual_attributed_owner_proof(connector, supported):
    request, payload = case_input(CASES["a_reported_denial"])
    fact = "Mesin kasir tidak rusak." if supported else "Mesin kasir rusak."
    payload["hypotheses"][0]["statement"] = (
        f"Hipotesis: antrean mungkin panjang {connector} Owner menyatakan: {fact}"
    )
    if supported:
        mapped(request, payload)
    else:
        with pytest.raises(ConsultationFailure, match="hypotheses"):
            mapped(request, payload)


def test_attribution_of_uncertain_text_still_requires_owner_proof():
    request, payload = case_input(CASES["a_reported_denial"])
    payload["hypotheses"][0]["statement"] = (
        "Hipotesis: antrean mungkin panjang tetapi Owner menyatakan: kasir mungkin rusak."
    )
    with pytest.raises(ConsultationFailure, match="hypotheses"):
        mapped(request, payload)


@pytest.mark.parametrize("connector", [" dan ", " serta ", ". ", "; "])
@pytest.mark.parametrize("reverse", [False, True])
def test_each_independent_action_needs_its_own_relevance(connector, reverse):
    request, payload = case_input(CASES["queue_base"])
    parts = ["Amati antrean kasir", "evaluasi harga Bitcoin"]
    if reverse:
        parts.reverse()
    payload["recommendations"][0]["description"] = connector.join(parts) + "."
    with pytest.raises(ConsultationFailure, match="recommendations"):
        mapped(request, payload)


@pytest.mark.parametrize("blank", ["", " ", "\n\t", "\u00a0"])
def test_nonblank_policy_is_local_not_a_historical_or_provider_schema_change(blank):
    request, payload = case_input(CASES["queue_base"])
    payload["recommendations"][0]["description"] = blank
    historical = HistoricalSolution.model_validate(
        {k: v for k, v in payload.items() if k != "memory_candidates"}
    )
    assert historical.recommendations[0].description == blank
    description = solution_provider_schema()["properties"]["recommendations"]["items"][
        "properties"
    ]["description"]
    assert description == {"type": "string"}
    with pytest.raises(ConsultationFailure, match="description is blank") as error:
        mapped(request, payload)
    assert error.value.code == grpc.StatusCode.INTERNAL


@pytest.mark.parametrize("event", [1, 2, 3, 4, 5])
def test_blank_output_on_authenticated_grpc_is_internal_not_request_error(event):
    request, payload = case_input(CASES["c_empty"])
    request.event = event
    assessment = {
        "decision": "GENERATE_SOLUTION",
        "reason": "Synthetic context sufficient",
        "problem_evidence_sufficient": True,
        "missing_operational_information": [],
        "relevant_business_constraints": [],
        "additional_question_material": False,
        "repeats_existing_information": False,
    }
    factory = Mock()
    factory.return_value.with_structured_output.side_effect = lambda schema, **kw: Mock(
        invoke=Mock(
            return_value=copy.deepcopy(
                payload if schema["title"] == "GeneratedSolutionTurn" else assessment
            )
        )
    )
    service = AtlasAIService()
    service.consultation_pipeline = ConsultationPipeline(
        ContextAssessor(
            GeminiSettings(model="mock-not-live", api_key=SecretStr("test-only")),
            factory,
        )
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[BearerAuthInterceptor("test-only")],
    )
    ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port
    server.start()
    try:
        with (
            grpc.insecure_channel(f"127.0.0.1:{port}") as channel,
            pytest.raises(grpc.RpcError) as error,
        ):
            ai_pb2_grpc.AIServiceStub(channel).ProcessConsultationTurn(
                request,
                timeout=5,
                metadata=(("authorization", "Bearer test-only"),),
            )
        assert error.value.code() == grpc.StatusCode.INTERNAL
        assert "description is blank" in error.value.details()
        assert factory.call_count == (1 if event == 4 else 2)
    finally:
        server.stop(0).wait(timeout=5)
