"""Offline atomic-grounding evaluation; frozen cases, no provider calls."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import evaluate_claim, solution_claims
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto
from scripts.consultation_m6_2_evaluation import evaluate as evaluate_m6_2
from scripts.consultation_m6_3_evaluation import ROOT, case_input
from scripts.consultation_m6_3_evaluation import evaluate as evaluate_m6_3

CORPUS = ROOT / "evaluation/m6_4_cases.json"


def evaluate():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    rows = []
    for case in corpus["cases"]:
        request, payload = case_input(case, corpus)
        context = prepare_context(request)
        decisions = []
        try:
            solution = GeneratedSolutionTurn.model_validate(payload).solution_output()
            decisions = [
                evaluate_claim(c, context).model_dump(mode="json")
                for c in solution_claims(solution)
            ]
            result = solution_to_proto(solution, context)
            assert all(getattr(result, f.name) for f in result.DESCRIPTOR.fields)
            accepted, reason = True, "structural_grounding_protobuf_passed"
        except (ConsultationFailure, ValidationError) as exc:
            accepted, reason = (
                False,
                str(exc) if isinstance(exc, ConsultationFailure) else "invalid_schema",
            )
        rows.append(
            {**case, "accepted": accepted, "reason": reason, "claims": decisions}
        )
    return {
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "counts": {
            f"{expected}_{outcome}": sum(
                r["expected"] == expected and r["accepted"] == (outcome == "accepted")
                for r in rows
            )
            for expected in ("valid", "invalid", "ambiguous")
            for outcome in ("accepted", "rejected")
        },
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {
            "milestone": "M6.4",
            "baseline": "55b30e37b79ef48a9eaa9889c09d8d32df4d06a6",
            "external_calls": 0,
        }
    )
    if args.phase in ledger:
        parser.error("Phase already recorded")
    result = {"m6_2": evaluate_m6_2(), "m6_3": evaluate_m6_3(), "m6_4": evaluate()}
    if "before" in ledger and any(
        ledger["before"][k]["corpus_sha256"] != v["corpus_sha256"]
        for k, v in result.items()
    ):
        parser.error("Frozen corpus changed")
    ledger[args.phase] = result
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v["counts"] for k, v in result.items()}))


if __name__ == "__main__":
    main()
