"""Frozen atomic-grounding checks; unresolved semantics remain visible failures."""

import hashlib
import json

import pytest

from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto
from scripts.consultation_m6_3_evaluation import case_input
from scripts.consultation_m6_4_evaluation import CORPUS, ROOT, evaluate

DATA = json.loads(CORPUS.read_text(encoding="utf-8"))


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

    if case["expected"] == "valid":
        result = check()
        assert all(getattr(result, field.name) for field in result.DESCRIPTOR.fields)
    else:
        with pytest.raises(ConsultationFailure):
            check()


def test_frozen_measurement_has_no_new_false_acceptance_or_rejection():
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == (
        "16d4cdbb90d9ddabfcd904faf081ea75be413ddbc8353bae59abba901348c3d5"
    )
    ledger = json.loads(
        (ROOT / "evaluation/m6_4_grounding.json").read_text(encoding="utf-8")
    )
    current = evaluate()
    assert current == ledger["after"]["m6_4"]
    previous = {c["id"]: c for c in ledger["before"]["m6_4"]["cases"]}
    for case in current["cases"]:
        before = previous[case["id"]]
        if case["expected"] == "valid" and before["accepted"]:
            assert case["accepted"]
        if case["expected"] == "invalid" and not before["accepted"]:
            assert not case["accepted"]
    assert (
        current["counts"]["invalid_accepted"]
        < ledger["before"]["m6_4"]["counts"]["invalid_accepted"]
    )
