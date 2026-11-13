"""Frozen synthetic whole-Solution evaluation. No provider or repair invocation."""

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
from atlas_ai.consultation.solution import GeneratedSolutionTurn, solution_to_proto

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evaluation/m6_2_cases.json"


def case_input(case):
    m6 = json.loads(
        (ROOT / "evaluation/m6_consultation_e2e.json").read_text(encoding="utf-8")
    )
    row = next(r for r in m6["runs"] if r["id"] == "M6-B-COMP")
    request = ParseDict(
        row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
    )
    payload = copy.deepcopy(row["validation"]["synthetic_typed_output"])
    refs = {
        "opinion_ids": [x.opinion_id for x in request.context.evidence],
        "owner_message_ids": [request.messages[-1].id],
        "memory_ids": [request.context.memories[-1].id],
    }
    field, text = case["field"], case["text"]
    if field in {"finding", "hypothesis"}:
        payload["findings" if field == "finding" else "hypotheses"][0].update(
            statement=text, references=refs
        )
    elif field in {"title", "description"}:
        payload["recommendations"][0][field] = text
        payload["recommendations"][0]["references"] = refs
        if field == "description":
            payload["recommendations"][0]["title"] = "Evaluasi pembagian tugas kasir"
    elif field in {"metric", "target", "timeframe"}:
        payload["success_indicators"][0][field] = text
    elif field == "quote":
        payload["customer_evidence"][0]["quote"] = text
    else:
        payload["findings"][0]["references"] = {
            "opinion_ids": [request.context.memories[0].id]
            if text == "namespace"
            else [request.messages[0].id]
            if text == "ai"
            else ["10000000-0000-4000-8000-000000000099"]
        }
    return request, payload


def evaluate():
    cases = json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        request, payload = case_input(case)
        try:
            solution_to_proto(
                GeneratedSolutionTurn.model_validate(payload).solution_output(),
                prepare_context(request),
            )
            accepted, reason = True, "all_structural_grounding_protobuf_checks_passed"
        except (ConsultationFailure, ValidationError) as error:
            accepted, reason = (
                False,
                str(error)
                if isinstance(error, ConsultationFailure)
                else "invalid_schema",
            )
        rows.append({**case, "accepted": accepted, "reason": reason})
    return {
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "counts": {
            "valid_accepted": sum(c["valid"] and c["accepted"] for c in rows),
            "valid_rejected": sum(c["valid"] and not c["accepted"] for c in rows),
            "invalid_accepted": sum(not c["valid"] and c["accepted"] for c in rows),
            "invalid_rejected": sum(not c["valid"] and not c["accepted"] for c in rows),
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
        else {"milestone": "M6.2", "external_calls": 0}
    )
    if args.phase in ledger:
        parser.error("Phase already recorded")
    result = evaluate()
    if (
        "before" in ledger
        and ledger["before"]["corpus_sha256"] != result["corpus_sha256"]
    ):
        parser.error("Frozen corpus changed")
    ledger[args.phase] = result
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
