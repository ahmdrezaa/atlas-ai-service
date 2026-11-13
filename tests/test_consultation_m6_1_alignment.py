"""Exact synthetic M6 replays, adversarial title linkage and native mock gRPC."""

import copy
import json
from concurrent import futures
from pathlib import Path
from unittest.mock import Mock

import grpc
import pytest
from google.protobuf.json_format import ParseDict
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import (
    GroundingClaim,
    evaluate_claim,
    solution_claims,
)
from atlas_ai.consultation.memory import memory_candidates_to_proto
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.prompts import (
    SOLUTION_PROMPT_VERSION,
    SOLUTION_SYSTEM_PROMPT,
)
from atlas_ai.consultation.solution import (
    GeneratedSolutionTurn,
    solution_to_proto,
    validate_solution,
)
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.server import BearerAuthInterceptor
from scripts.consultation_m6_1_alignment import live_allowed, replay_recorded_outputs

ROOT = Path(__file__).resolve().parents[1]
M6 = json.loads(
    (ROOT / "evaluation/m6_consultation_e2e.json").read_text(encoding="utf-8")
)


def recorded(run_id):
    row = next(r for r in M6["runs"] if r["id"] == run_id)
    return ParseDict(
        row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
    ), copy.deepcopy(row["validation"]["synthetic_typed_output"])


def mapped(request, payload):
    return solution_to_proto(
        GeneratedSolutionTurn.model_validate(payload).solution_output(),
        prepare_context(request),
    )


def test_exact_m6_b_passes_without_changing_any_generated_text_or_reference():
    request, payload = recorded("M6-B-COMP")
    before = copy.deepcopy(payload)
    solution = mapped(request, payload)
    assert solution.recommendations[0].title == "Optimasi Alur Kerja Kasir"
    assert (
        solution.recommendations[0].description
        == payload["recommendations"][0]["description"]
    )
    assert payload == before
    candidate = memory_candidates_to_proto(
        GeneratedSolutionTurn.model_validate(payload).memory_candidates,
        prepare_context(request),
    )
    assert len(candidate) == 1
    assert candidate[0].fact_text == request.messages[-1].content
    assert candidate[0].source_owner_message_ids == [request.messages[-1].id]


@pytest.mark.parametrize(
    "title",
    [
        "Optimasi Alur Kerja Kasir",
        "Optimalisasi Pembagian Tugas Barista",
        "Penataan Kerja Kasir",
        "Perbaikan Kerja Kasir",
    ],
)
def test_action_labels_require_both_valid_description_and_cited_subject(title):
    request, payload = recorded("M6-B-COMP")
    payload["recommendations"][0]["title"] = title
    assert mapped(request, payload).recommendations[0].title == title


@pytest.mark.parametrize(
    "title",
    [
        "Optimasi Kasir Rusak",
        "Optimasi Kekurangan Barista",
        "Optimasi Barista Berjumlah Dua",
        "Optimasi Keuntungan Meningkat",
        "Optimasi Kasir Sudah Cepat",
        "Optimasi Kasir Saat Ini",
        "Optimasi Kasir Dijamin Untung",
        "Optimasi ROI Pasti",
        "Amati Kasir Dijamin Untung",
        "Optimasi Kasir 50%",
        "Optimasi Kasir Semua Pelanggan",
        "Optimasi Harga Bitcoin",
        "Optimasi Mesin Baru Kasir",
        "Ketersediaan Staf Kasir",
        "Kasir Rusak",
        "Optimasi Alur Kerja Kasir. Mesin rusak.",
        "Optimasi Alur Kerja Kasir; pelanggan puas",
        "Optimasi Alur Kerja Kasir karena keuntungan meningkat",
    ],
)
def test_resembling_a_label_never_proves_new_facts_or_guaranteed_outcomes(title):
    request, payload = recorded("M6-B-COMP")
    payload["recommendations"][0]["title"] = title
    with pytest.raises(ConsultationFailure):
        mapped(request, payload)


@pytest.mark.parametrize(
    "description",
    [
        "",
        "Mesin kasir rusak.",
        "Evaluasi alur kerja karena keuntungan meningkat.",
        "Evaluasi alur kerja. Mesin rusak.",
        "Evaluasi harga Bitcoin.",
        "Evaluasi dengan mesin yang sudah tersedia.",
    ],
)
def test_description_cannot_launder_an_unsupported_nominal_title(description):
    request, payload = recorded("M6-B-COMP")
    payload["recommendations"][0]["description"] = description
    with pytest.raises(ConsultationFailure):
        mapped(request, payload)


def test_a_valid_id_for_an_irrelevant_source_does_not_support_a_label():
    request, payload = recorded("M6-B-COMP")
    refs = payload["recommendations"][0]["references"]
    refs["owner_message_ids"] = []
    refs["memory_ids"] = [request.context.memories[1].id]  # Budget, not staff/cashier.
    with pytest.raises(ConsultationFailure):
        mapped(request, payload)


@pytest.mark.parametrize(
    "title,description",
    [
        ("Optimasi Kasir Lelah", "Evaluasi kasir lelah."),
        ("Optimasi Mesin Baru Kasir", "Evaluasi mesin baru di kasir."),
        ("Optimasi Kasir Otomatis", "Evaluasi kasir otomatis."),
        ("Optimasi Kasir Modern", "Evaluasi kasir modern."),
        ("Optimasi Barista Baru", "Evaluasi barista baru."),
    ],
)
def test_a_description_is_not_a_source_for_new_operational_modifiers(
    title, description
):
    request, payload = recorded("M6-B-COMP")
    payload["recommendations"][0].update(title=title, description=description)
    with pytest.raises(ConsultationFailure):
        mapped(request, payload)


@pytest.mark.parametrize(
    "metric", ["Durasi tunggu antrean kasir", "Catatan durasi pelayanan"]
)
def test_generic_metric_definitions_still_pass(metric):
    request, payload = recorded("M6-B-COMP")
    payload["success_indicators"][0]["metric"] = metric
    assert mapped(request, payload).success_indicators[0].metric == metric


@pytest.mark.parametrize(
    "metric",
    [
        "Waktu tunggu antrean kasir pada jam 17.00-19.00",
        "Waktu tunggu antrean kasir pada jam 10.00-12.00",
        "Waktu tunggu untuk 99 barista",
        "Rating rata-rata 4.9",
    ],
)
def test_metric_numbers_do_not_gain_implicit_memory_or_sql_provenance(metric):
    request, payload = recorded("M6-B-COMP")
    payload["success_indicators"][0]["metric"] = metric
    with pytest.raises(ConsultationFailure, match="Invalid solution output"):
        mapped(request, payload)


def test_prompt_measurement_schedule_is_a_proposal_not_current_memory_evidence():
    request, payload = recorded("M6-B-COMP")
    indicator = payload["success_indicators"][0]
    indicator["metric"] = "Durasi tunggu antrean kasir"
    indicator["timeframe"] = "Usulan: ukur selama satu minggu pada jam 17.00-19.00"
    assert (
        mapped(request, payload).success_indicators[0].timeframe
        == indicator["timeframe"]
    )
    assert SOLUTION_PROMPT_VERSION == "consultation-solution-v7"
    assert "metric has NO references field" in SOLUTION_SYSTEM_PROMPT
    assert "NO numbers" in SOLUTION_SYSTEM_PROMPT


def test_exact_c_stays_rejected_and_metric_only_change_does_not_repair_other_claims():
    request, payload = recorded("M6-C-COMP")
    with pytest.raises(ConsultationFailure, match="numeric factual"):
        mapped(request, payload)
    payload["success_indicators"][0]["metric"] = "Durasi tunggu antrean kasir"
    solution = GeneratedSolutionTurn.model_validate(payload).solution_output()
    validate_solution(
        solution, prepare_context(request)
    )  # Prospective prompt shape only.
    with pytest.raises(ConsultationFailure, match="grounding rejected"):
        mapped(request, payload)  # Original causal hypothesis remains rejected.


@pytest.mark.parametrize(
    "text,accepted",
    [
        ("Hipotesis: kecepatan pelayanan mungkin belum optimal saat jam ramai.", True),
        ("Hipotesis: antrean mungkin panjang; perlu diverifikasi.", True),
        ("Hipotesis: antrean terbukti disebabkan kekurangan staf.", False),
        ("Hipotesis: antrean mungkin panjang dan penyebabnya kekurangan staf.", False),
        ("Hipotesis: mayoritas pelanggan mungkin kecewa karena mesin rusak.", False),
        ("Hipotesis: harga Bitcoin mungkin berubah.", False),
    ],
)
def test_missing_d_wording_is_not_needed_to_test_scope_causality_and_relevance(
    text, accepted
):
    request, _ = recorded("M6-B-COMP")
    claim = GroundingClaim(
        claim_id="hypothesis",
        kind="hypothesis",
        text=text,
        references={
            "opinion_ids": [
                request.context.evidence[0].opinion_id,
                request.context.evidence[1].opinion_id,
            ]
        },
    )
    assert (
        evaluate_claim(claim, prepare_context(request)).label == "NOT_APPLICABLE"
    ) == accepted


def test_nominal_title_is_not_accepted_without_its_validated_description():
    request, payload = recorded("M6-B-COMP")
    claim = next(
        c
        for c in solution_claims(
            GeneratedSolutionTurn.model_validate(payload).solution_output()
        )
        if c.claim_id == "recommendations[0].title"
    )
    assert evaluate_claim(claim, prepare_context(request)).label == "NOT_APPLICABLE"
    assert (
        evaluate_claim(
            claim.model_copy(update={"recommendation_description": None}),
            prepare_context(request),
        ).label
        == "INSUFFICIENT_EVIDENCE"
    )


@pytest.mark.parametrize("field", ["quote", "opinion_id", "references", "number"])
def test_title_alignment_does_not_bypass_other_solution_guards(field):
    request, payload = recorded("M6-B-COMP")
    if field == "quote":
        payload["customer_evidence"][0]["quote"] = "Pelanggan tidak menulis ini."
    elif field == "opinion_id":
        payload["customer_evidence"][0]["opinion_id"] = request.messages[-1].id
    elif field == "references":
        payload["recommendations"][0]["references"]["owner_message_ids"] = [
            request.context.memories[0].id
        ]
    else:
        payload["findings"][0]["statement"] = "Sebanyak 99 opini mengeluh."
    with pytest.raises(ConsultationFailure):
        mapped(request, payload)


def test_replay_retains_exact_c_and_explicitly_missing_d():
    replay = replay_recorded_outputs(ROOT)
    assert replay["M6-B-COMP"]["structural_grounding_protobuf"] == "PASSED"
    assert replay["M6-C-COMP"]["structural"] == "REJECTED"
    assert replay["M6-D-COMP"]["exact_replay"] == "UNAVAILABLE"


@pytest.mark.parametrize("event", [1, 2, 3, 4, 5])
def test_exact_b_solution_and_owner_candidate_on_authenticated_grpc_all_events(event):
    request, payload = recorded("M6-B-COMP")
    request.event = event
    before = request.SerializeToString()
    assessment = {
        "decision": "GENERATE_SOLUTION",
        "reason": "Synthetic context sufficient",
        "problem_evidence_sufficient": True,
        "missing_operational_information": [],
        "relevant_business_constraints": [],
        "additional_question_material": False,
        "repeats_existing_information": False,
        "question": None,
    }
    factory = Mock()
    factory.return_value.with_structured_output.side_effect = lambda schema, **kwargs: (
        Mock(
            invoke=Mock(
                return_value=copy.deepcopy(
                    payload
                    if schema["title"] == "GeneratedSolutionTurn"
                    else assessment
                )
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
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            response = ai_pb2_grpc.AIServiceStub(channel).ProcessConsultationTurn(
                request, timeout=5, metadata=(("authorization", "Bearer test-only"),)
            )
        assert response.WhichOneof("result") == "solution"
        assert response.memory_candidates[0].source_owner_message_ids == [
            request.messages[-1].id
        ]
        assert response.model_id == "mock-not-live"
        assert response.prompt_version == SOLUTION_PROMPT_VERSION
        assert (
            ai_pb2.ProcessConsultationTurnResponse.FromString(
                response.SerializeToString()
            )
            == response
        )
        assert factory.call_count == (1 if event == 4 else 2)
        assert request.SerializeToString() == before
    finally:
        server.stop(0).wait(timeout=5)


@pytest.mark.parametrize("case", ["duplicate", "confirmed", "conflict", "wrong_owner"])
def test_recorded_candidate_keeps_existing_dedup_conflict_and_owner_policy(case):
    request, payload = recorded("M6-B-COMP")
    if case == "duplicate":
        payload["memory_candidates"] *= 2
    elif case == "confirmed":
        request.context.memories.add(
            id="30000000-0000-4000-8000-000000000099",
            fact_text=payload["memory_candidates"][0]["fact_text"],
        )
    elif case == "conflict":
        request.messages.add(
            id="20000000-0000-4000-8000-000000000099",
            role=ai_pb2.MESSAGE_ROLE_OWNER,
            content="Saat jam ramai, tiga barista menangani pesanan.",
        )
    else:
        payload["memory_candidates"][0]["source_owner_message_ids"] = [
            request.messages[0].id
        ]
    assert mapped(request, payload).findings
    result = memory_candidates_to_proto(
        GeneratedSolutionTurn.model_validate(payload).memory_candidates,
        prepare_context(request),
    )
    assert len(result) == (1 if case == "duplicate" else 0)


def test_p0_p3_five_section_solution_remains_valid():
    old = json.loads(
        (ROOT / "evaluation/p0_solution_live.json").read_text(encoding="utf-8")
    )
    row = next(r for r in old["runs"] if r["id"] == "P3")
    request = ParseDict(
        row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
    )
    payload = json.loads(row["provider_responses"][-1]["visible_text"])
    assert mapped(request, payload).recommendations


def test_live_gate_requires_offline_evidence_and_stops_after_success_or_two_calls():
    ledger = {"live_runs": []}
    assert not live_allowed(ledger, "L1", None)
    ledger.update(before={}, after={}, offline_verification={"passed": True})
    assert live_allowed(ledger, "L1", None)
    ledger["live_runs"] = [
        {"id": "L1", "status": "FAILED", "actual_gemini_invocations": 1}
    ]
    assert not live_allowed(ledger, "L1", "duplicate")
    assert not live_allowed(ledger, "L2", None)
    assert live_allowed(
        ledger, "L2", "Targeted request after investigating the recorded failure"
    )
    ledger["live_runs"][0]["status"] = "PASSED"
    assert not live_allowed(ledger, "L2", "Do not retry a successful turn")
    ledger["live_runs"][0].update(status="FAILED", actual_gemini_invocations=2)
    assert not live_allowed(ledger, "L2", "Budget exhausted")
