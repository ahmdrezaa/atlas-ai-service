"""Authorized language policy: diagnostics cannot replace objective evidence gates."""

import copy
import json
from pathlib import Path

import grpc
import pytest
from google.protobuf.json_format import ParseDict

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import ACCEPTED_LABELS, validate_grounding
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto

SCENARIOS = json.loads(
    (Path(__file__).resolve().parents[1] / "evaluation/m6_3_cases.json").read_text(
        encoding="utf-8"
    )
)["scenarios"]


def turn(scenario="queue"):
    data = copy.deepcopy(SCENARIOS[scenario])
    return ParseDict(data["request"], ai_pb2.ProcessConsultationTurnRequest()), data[
        "solution"
    ]


def check(request, payload):
    solution = GeneratedSolutionTurn.model_validate(payload).solution_output()
    context = prepare_context(request)
    return solution_to_proto(solution, context), validate_grounding(solution, context)


@pytest.mark.parametrize(
    "text",
    [
        "Evaluasi orbit satelit.",
        "Evaluasi orbit satelit untuk memperbaiki antrean kasir.",
        "Evaluasi harga Bitcoin dan amati antrean kasir.",
    ],
)
def test_relevance_warning_does_not_block_or_become_factual_support(text):
    request, payload = turn()
    payload["recommendations"][0]["description"] = text
    proto, diagnostics = check(request, payload)
    assert proto.recommendations[0].description == text
    assert any(
        d.claim_id == "recommendations[0].description"
        and d.label not in ACCEPTED_LABELS
        for d in diagnostics
    )


def test_tentative_plausibility_warning_is_not_a_proven_fact():
    request, payload = turn()
    payload["hypotheses"][0]["statement"] = "Hipotesis: orbit satelit mungkin berubah."
    proto, diagnostics = check(request, payload)
    assert proto.hypotheses[0].statement == payload["hypotheses"][0]["statement"]
    assert any(
        d.claim_id == "hypotheses[0].statement" and d.label not in ACCEPTED_LABELS
        for d in diagnostics
    )


@pytest.mark.parametrize(
    "scenario,text",
    [
        ("quality", "Penyempurnaan kualitas produk."),
        ("packaging", "Pemeriksaan kemasan."),
        ("bakery", "Pemeriksaan isian roti."),
    ],
)
def test_nominal_wording_with_complete_cited_object_is_not_workflow_limited(
    scenario, text
):
    request, payload = turn(scenario)
    payload["recommendations"][0]["description"] = text
    proto, diagnostics = check(request, payload)
    assert proto.recommendations[0].description == text
    assert any(
        d.claim_id == "recommendations[0].description"
        and d.label not in ACCEPTED_LABELS
        for d in diagnostics
    )


@pytest.mark.parametrize(
    "text",
    [
        "Evaluasi orbit satelit karena keuntungan meningkat.",
        "Evaluasi orbit satelit dengan mesin yang tersedia.",
        "Evaluasi orbit satelit. Mesin kasir rusak.",
        "Evaluasi orbit satelit yang sudah meningkatkan penjualan.",
        "Penataan mesin kasir rusak.",
    ],
)
def test_semantic_failure_cannot_mask_a_factual_premise_failure(text):
    request, payload = turn()
    payload["recommendations"][0]["description"] = text
    with pytest.raises(ConsultationFailure) as error:
        check(request, payload)
    assert error.value.code == grpc.StatusCode.INTERNAL


def test_source_words_cannot_be_recombined_into_a_new_operational_condition():
    request, payload = turn()
    request.messages[-1].content = "Mesin lama ada di kasir. Kasir baru bertugas."
    payload["recommendations"][0]["description"] = "Penataan mesin baru kasir."
    with pytest.raises(ConsultationFailure):
        check(request, payload)


def test_retained_causal_guard_is_an_explicit_conservative_limit():
    request, payload = turn("bakery")
    payload["hypotheses"][0]["statement"] = (
        "Hipotesis: isian roti mungkin sedikit karena takaran belum konsisten."
    )
    # Cause/premise scope cannot yet be separated safely; user safeguard 9 keeps
    # the guard rather than treating an arbitrary causal tail as proven evidence.
    with pytest.raises(ConsultationFailure):
        check(request, payload)


def test_source_bound_title_mismatch_is_diagnostic_but_an_invented_title_condition_blocks():
    request, payload = turn()
    payload["recommendations"][0].update(
        title="Optimasi Kasir", description="Evaluasi orbit satelit."
    )
    proto, diagnostics = check(request, payload)
    assert proto.recommendations[0].title == "Optimasi Kasir"
    assert any(
        d.claim_id == "recommendations[0].title" and d.label not in ACCEPTED_LABELS
        for d in diagnostics
    )
    payload["recommendations"][0]["title"] = "Optimasi Kasir Lelah"
    with pytest.raises(ConsultationFailure):
        check(request, payload)
