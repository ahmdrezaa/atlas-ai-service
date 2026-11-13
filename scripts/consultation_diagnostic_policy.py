"""Measure the authorized diagnostic policy against immutable historical labels."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import (
    ACCEPTED_LABELS,
    evaluate_claim,
    solution_claims,
)
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto
from scripts.consultation_m6_2_evaluation import case_input as input_m6_2
from scripts.consultation_m6_3_evaluation import ROOT, case_input


def evaluate():
    baseline_path = ROOT / "evaluation/m6_4_grounding.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["after"]
    result = {
        "baseline_commit": "fd2e416a0e779dfcf9283aa1078cc80ef43c1be8",
        "baseline_source": str(baseline_path.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "policy": "evidence blocking; separable relevance/wording diagnostic; mixed factual guards retained",
        "external_calls": 0,
        "corpora": {},
    }
    for milestone in ("m6_2", "m6_3", "m6_4"):
        path = ROOT / f"evaluation/{milestone}_cases.json"
        corpus = json.loads(path.read_text(encoding="utf-8"))
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != baseline[milestone]["corpus_sha256"]
        ):
            raise ValueError("Historical corpus hash changed")
        prior = {c["id"]: c for c in baseline[milestone]["cases"]}
        rows = []
        for case in corpus["cases"]:
            request, payload = (
                input_m6_2(case) if milestone == "m6_2" else case_input(case, corpus)
            )
            context = prepare_context(request)
            diagnostics = []
            try:
                solution = GeneratedSolutionTurn.model_validate(
                    payload
                ).solution_output()
                diagnostics = [
                    evaluate_claim(c, context).model_dump(mode="json")
                    for c in solution_claims(solution)
                ]
                solution_to_proto(solution, context)
                accepted, error = True, None
            except (ConsultationFailure, ValidationError) as exc:
                accepted, error = (
                    False,
                    str(exc)
                    if isinstance(exc, ConsultationFailure)
                    else "invalid_schema",
                )
            expected = case.get("expected", "valid" if case.get("valid") else "invalid")
            rows.append(
                {
                    "id": case["id"],
                    "historical_quality_label": expected,
                    "baseline_response_accepted": prior[case["id"]]["accepted"],
                    "response_accepted": accepted,
                    "error": error,
                    "diagnostics": diagnostics,
                    "negative_diagnostic_ids": [
                        d["claim_id"]
                        for d in diagnostics
                        if d["label"] not in ACCEPTED_LABELS
                    ],
                }
            )
        result["corpora"][milestone] = {
            "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "historical_baseline_counts": baseline[milestone]["counts"],
            "current_counts_against_historical_quality_labels": {
                f"{label}_{outcome}": sum(
                    r["historical_quality_label"] == label
                    and r["response_accepted"] == (outcome == "accepted")
                    for r in rows
                )
                for label in ("valid", "invalid", "ambiguous")
                for outcome in ("accepted", "rejected")
            },
            "changed_response_ids": [
                r["id"]
                for r in rows
                if r["baseline_response_accepted"] != r["response_accepted"]
            ],
            "cases": rows,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing policy ledger")
    result = evaluate()
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v["current_counts_against_historical_quality_labels"]
                for k, v in result["corpora"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
