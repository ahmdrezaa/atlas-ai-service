"""Offline exact M6 replay and at most two fresh synthetic recovery RPCs.

Replays never count as live success and never modify the original M6 ledger.
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from google.protobuf.json_format import MessageToDict, ParseDict

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import ConsultationFailure, GeminiSettings
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import (
    cited_sources,
    evaluate_claim,
    solution_claims,
)
from atlas_ai.consultation.memory import memory_candidates_to_proto
from atlas_ai.consultation.solution import (
    GeneratedSolutionTurn,
    solution_to_proto,
    validate_solution,
)
from scripts.consultation_live_readiness import run_case
from scripts.consultation_m6_e2e import finish_rpc_row, scenario_request
from scripts.gemini_live_diagnostics import load_environment


def replay_recorded_outputs(root=Path(".")):
    original = json.loads(
        (root / "evaluation/m6_consultation_e2e.json").read_text(encoding="utf-8")
    )
    result = {}
    for run_id in ("M6-B-COMP", "M6-C-COMP", "M6-D-COMP"):
        row = next(r for r in original["runs"] if r["id"] == run_id)
        payload = row.get("validation", {}).get("synthetic_typed_output")
        if payload is None:
            result[run_id] = {
                "exact_replay": "UNAVAILABLE",
                "original_output_missing": True,
                "recorded_verdicts": row["validation"].get("decisions", []),
                "is_false_positive": "UNKNOWN",
            }
            continue
        request = ParseDict(
            row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
        )
        context = prepare_context(request)
        turn = GeneratedSolutionTurn.model_validate(payload)
        solution = turn.solution_output()
        claims = solution_claims(solution)
        details = {
            "exact_replay": "UNMODIFIED_TYPED_OUTPUT_NOT_A_LIVE_INVOCATION",
            "original_typed_output_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "pydantic": "PASSED",
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "kind": c.kind,
                    "text": c.text,
                    "references": c.references.model_dump(mode="json"),
                    "sources": cited_sources(c, context),
                    "decision": evaluate_claim(c, context).model_dump(mode="json"),
                }
                for c in claims
            ],
            "memory_candidates_posthoc_not_rpc": [
                MessageToDict(m, preserving_proto_field_name=True)
                for m in memory_candidates_to_proto(turn.memory_candidates, context)
            ],
        }
        try:
            validate_solution(solution, context)
            details["structural"] = "PASSED"
            solution_to_proto(solution, context)
            details["structural_grounding_protobuf"] = "PASSED"
        except ConsultationFailure as error:
            details.setdefault("structural", "REJECTED")
            details["structural_grounding_protobuf"] = "REJECTED"
            details["sanitized_application_error"] = str(error)
        result[run_id] = details
    return result


def history_hashes(root=Path(".")):
    return {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted((root / "evaluation").glob("*.json"))
        if p.name != "m6_1_alignment.json"
    }


def live_allowed(ledger, run_id, reason):
    runs = ledger["live_runs"]
    return (
        "before" in ledger
        and "after" in ledger
        and ledger.get("offline_verification", {}).get("passed") is True
        and len(runs) < 2
        and sum(r["actual_gemini_invocations"] for r in runs) < 2
        and not any(r["id"] == run_id or r["status"] == "PASSED" for r in runs)
        and (not runs or bool(reason and reason.strip()))
    )


def run_live(args, ledger, parser):
    if not args.id or not live_allowed(ledger, args.id, args.reason):
        parser.error(
            "Offline checks/bounded spend gate/targeted second-attempt reason required; no request sent"
        )
    if not 40 <= args.turn_budget <= 45:
        parser.error("Use the audited 40-45s development budget within a 60s RPC")
    load_environment(args.env_file)
    production = GeminiSettings.from_env()
    settings = GeminiSettings.model_validate(
        {
            **production.model_dump(),
            "model": args.model or production.model,
            "timeout_seconds": args.turn_budget,
        }
    )
    if not settings.model or not settings.api_key.get_secret_value().strip():
        parser.error("Missing Gemini configuration; no request sent")
    request = scenario_request("D")  # GENERATE_NOW + durable OWNER fact, one call.
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    row = finish_rpc_row(
        run_case(
            "grpc",
            settings,
            request=request,
            allowed_results={"solution"},
            api_version=args.api_version,
            thinking=args.thinking,
        ),
        request,
    )
    row.update(
        id=args.id,
        category_of_test="REAL_PROVIDER_AUTHENTICATED_GRPC_NOT_REPLAY",
        model=settings.model,
        production_model_unchanged=production.model,
        diagnostic_model_override=settings.model != production.model,
        started_at=started_at,
        ended_at=datetime.now(UTC).isoformat(),
        total_duration_seconds=round(time.perf_counter() - started, 6),
        attempt_reason=args.reason
        or "Fresh synthetic GENERATE_NOW after offline alignment checks",
    )
    row["implementation_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path("src/atlas_ai/consultation/grounding.py"),
            Path("src/atlas_ai/consultation/prompts.py"),
        )
    }
    typed = row.get("validation", {}).get("synthetic_typed_output", {})
    if row.get("status") == "PASSED":
        row["memory_proposal_handling"] = {
            "valid_typed_proposals": len(typed.get("memory_candidates", [])),
            "returned_candidates": len(row["response"].get("memory_candidates", [])),
            "filtered_candidates_do_not_fail_solution": True,
        }
    ledger["live_runs"].append(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["before", "after", "live"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--id")
    parser.add_argument("--model")
    parser.add_argument("--api-version", choices=["v1", "v1beta"])
    parser.add_argument("--thinking", choices=["low", "medium", "high"])
    parser.add_argument("--turn-budget", type=float, default=45)
    parser.add_argument(
        "--reason",
        help="Mandatory justification for the one possible additional targeted request",
    )
    args = parser.parse_args()
    ledger = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {"milestone": "M6.1", "live_runs": []}
    )
    if ledger.get("milestone") != "M6.1" or args.phase in ledger:
        parser.error("Wrong ledger or phase already recorded")
    if args.phase == "live":
        row = run_live(args, ledger, parser)
        args.output.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "id",
                        "model",
                        "status",
                        "actual_grpc_status",
                        "category",
                        "rpc_duration_seconds",
                        "actual_gemini_invocations",
                        "all_response_validations",
                        "memory_validation",
                    )
                },
                ensure_ascii=False,
            )
        )
        return
    ledger[args.phase] = replay_recorded_outputs()
    current = history_hashes()
    if args.phase == "before":
        ledger["historical_ledger_sha256"] = current
    else:
        ledger["historical_ledgers_unchanged"] = (
            current == ledger["historical_ledger_sha256"]
        )
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                run: {
                    k: row.get(k)
                    for k in (
                        "exact_replay",
                        "structural",
                        "structural_grounding_protobuf",
                        "sanitized_application_error",
                    )
                }
                for run, row in ledger[args.phase].items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
