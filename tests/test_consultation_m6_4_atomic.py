"""Frozen atomic-grounding checks; unresolved semantics remain visible failures."""

import hashlib
import json

import pytest

from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import ACCEPTED_LABELS, validate_grounding
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto
from scripts.consultation_m6_3_evaluation import case_input
from scripts.consultation_m6_4_evaluation import CORPUS, ROOT, evaluate

DATA = json.loads(CORPUS.read_text(encoding="utf-8"))
SEMANTIC_ONLY = {
    c["id"]
    for c in DATA["cases"]
    if c["id"].endswith(("_untuk_attack", "_agar_attack"))
    or c["id"] == "product_different_financial_topic"
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case["id"],
            marks=pytest.mark.xfail(
                strict=True,
                reason="Unresolved coordination/modal scope or tentative causal interpretation",
            )
            if case["id"].endswith(
                ("_coordinate_fact", "_coordinate_speculation", "_causal")
            )
            else [],
        )
        for case in DATA["cases"]
    ],
)
def test_frozen_extension_semantic_expectation(case):
    request, payload = case_input(case, DATA)

    def check():
        return solution_to_proto(
            GeneratedSolutionTurn.model_validate(payload).solution_output(),
            prepare_context(request),
        )

    if case["expected"] == "valid" or case["id"] in SEMANTIC_ONLY:
        result = check()
        assert all(getattr(result, field.name) for field in result.DESCRIPTOR.fields)
        if case["id"] in SEMANTIC_ONLY:
            diagnostics = validate_grounding(
                GeneratedSolutionTurn.model_validate(payload).solution_output(),
                prepare_context(request),
            )
            assert any(d.label not in ACCEPTED_LABELS for d in diagnostics)
    else:
        with pytest.raises(ConsultationFailure):
            check()


def test_frozen_measurement_has_only_authorized_policy_differences():
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == (
        "16d4cdbb90d9ddabfcd904faf081ea75be413ddbc8353bae59abba901348c3d5"
    )
    ledger = json.loads(
        (ROOT / "evaluation/m6_4_grounding.json").read_text(encoding="utf-8")
    )
    current = evaluate()
    assert current["corpus_sha256"] == ledger["after"]["m6_4"]["corpus_sha256"]
    previous = {c["id"]: c for c in ledger["after"]["m6_4"]["cases"]}
    for case in current["cases"]:
        before = previous[case["id"]]
        if case["expected"] == "valid" and before["accepted"]:
            assert case["accepted"]
        if case["expected"] == "invalid" and not before["accepted"]:
            assert case["accepted"] == (case["id"] in SEMANTIC_ONLY)
    assert {
        c["id"]
        for c in current["cases"]
        if c["accepted"] != previous[c["id"]]["accepted"]
    } == SEMANTIC_ONLY
