import json

import pytest

from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto
from scripts.consultation_m6_2_evaluation import CORPUS, ROOT, case_input, evaluate

CASES = json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_frozen_whole_solution_policy(case):
    request, payload = case_input(case)
    if case["valid"]:
        result = solution_to_proto(
            GeneratedSolutionTurn.model_validate(payload).solution_output(),
            prepare_context(request),
        )
        assert all(getattr(result, field.name) for field in result.DESCRIPTOR.fields)
    else:
        with pytest.raises(ConsultationFailure):
            solution_to_proto(
                GeneratedSolutionTurn.model_validate(payload).solution_output(),
                prepare_context(request),
            )


def test_frozen_before_after_comparison_has_no_new_false_acceptance():
    ledger = json.loads(
        (ROOT / "evaluation/m6_2_grounding_reliability.json").read_text(
            encoding="utf-8"
        )
    )
    current = evaluate()
    assert current["corpus_sha256"] == ledger["before"]["corpus_sha256"]
    assert (
        current["counts"]["valid_accepted"]
        > ledger["before"]["counts"]["valid_accepted"]
    )
    assert current["counts"]["invalid_accepted"] == 0
    assert not any(not c["valid"] and c["accepted"] for c in current["cases"])


@pytest.mark.parametrize(
    "field,text",
    [
        (
            "hypothesis",
            "Hipotesis: mesin kasir rusak berpotensi dipengaruhi koordinasi kasir dan barista.",
        ),
        (
            "hypothesis",
            "Hipotesis: antrean berpotensi dipengaruhi koordinasi kasir. Mesin rusak.",
        ),
        (
            "hypothesis",
            "Hipotesis: perlu ditelusuri apakah antrean terkait koordinasi kasir yang tersedia.",
        ),
        ("hypothesis", "Hipotesis: antrean mungkin terjadi karena dua barista rusak."),
        (
            "hypothesis",
            "Hipotesis: ada kemungkinan antrean berkaitan dengan investasi Bitcoin kasir.",
        ),
        ("description", "Penambahan barista kasir."),
        ("description", "Penataan mesin kasir rusak."),
        ("description", "Penataan alur kasir. Pendapatan meningkat."),
        ("description", "Penataan harga Bitcoin kasir."),
    ],
)
def test_additional_premise_and_resource_attacks(field, text):
    request, payload = case_input({"field": field, "text": text})
    with pytest.raises(ConsultationFailure):
        solution_to_proto(
            GeneratedSolutionTurn.model_validate(payload).solution_output(),
            prepare_context(request),
        )


def test_nominal_title_needs_a_consistent_independent_description():
    request, payload = case_input(
        {"field": "description", "text": "Penataan alur pelayanan kasir."}
    )
    payload["recommendations"][0]["title"] = "Penataan alur pelayanan kasir"
    solution_to_proto(
        GeneratedSolutionTurn.model_validate(payload).solution_output(),
        prepare_context(request),
    )
    payload["recommendations"][0]["description"] = "Evaluasi harga Bitcoin."
    with pytest.raises(ConsultationFailure):
        solution_to_proto(
            GeneratedSolutionTurn.model_validate(payload).solution_output(),
            prepare_context(request),
        )


def test_new_hypothesis_path_cannot_erase_customer_negation():
    request, payload = case_input(
        {
            "field": "hypothesis",
            "text": "Hipotesis: antrean lama berpotensi dipengaruhi koordinasi kasir dan barista.",
        }
    )
    evidence = request.context.evidence[0]
    evidence.evidence_text = evidence.full_feedback = "Antreannya tidak lama."
    payload["customer_evidence"][0]["quote"] = evidence.full_feedback
    payload["findings"][0]["statement"] = "Pelanggan menyebut: Antreannya tidak lama."
    with pytest.raises(ConsultationFailure, match="hypotheses"):
        solution_to_proto(
            GeneratedSolutionTurn.model_validate(payload).solution_output(),
            prepare_context(request),
        )
