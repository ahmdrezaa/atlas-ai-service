"""Frozen, offline whole-Solution safety evaluation; no provider imports/calls here."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from google.protobuf.json_format import ParseDict
from pydantic import ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import evaluate_claim, solution_claims
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto
from scripts.consultation_m6_2_evaluation import evaluate as evaluate_m6_2

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evaluation/m6_3_cases.json"


def case_input(case, corpus=None):
    corpus = corpus or json.loads(CORPUS.read_text(encoding="utf-8"))
    fixture = copy.deepcopy(corpus["scenarios"][case["scenario"]])
    request = ParseDict(fixture["request"], ai_pb2.ProcessConsultationTurnRequest())
    payload = fixture["solution"]
    section, field = case["field"].split(".")
    payload[section][0][field] = case["value"]
    if "references" in case:
        payload[section][0]["references"] = case["references"]
    return request, payload


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
                evaluate_claim(claim, context).model_dump(mode="json")
                for claim in solution_claims(solution)
            ]
            proto = solution_to_proto(solution, context)
            assert all(getattr(proto, f.name) for f in proto.DESCRIPTOR.fields)
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
            "milestone": "M6.3",
            "baseline": "8519dfb11ccee8b9c329d1c4b62ddffde1448755",
            "external_calls": 0,
        }
    )
    if args.phase in ledger:
        parser.error("Phase already recorded")
    result = {"new_corpus": evaluate(), "m6_2": evaluate_m6_2()}
    if "before" in ledger and any(
        ledger["before"][key]["corpus_sha256"] != result[key]["corpus_sha256"]
        for key in result
    ):
        parser.error("Frozen corpus changed")
    ledger[args.phase] = result
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value["counts"] for key, value in result.items()}))


if __name__ == "__main__":
    main()
