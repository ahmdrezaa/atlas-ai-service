"""Bounded, synthetic M6 diagnostics using the existing authenticated gRPC runner.

Each CLI execution permits one preflight or one RPC, never a retry/fallback loop.
Non-D scenarios require an HTTP-200 provider response for the same configuration.
Provider responsiveness never counts as application validation success.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from google.protobuf.json_format import ParseDict

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import GeminiSettings
from atlas_ai.consultation.context import prepare_context
from scripts.consultation_live_readiness import case_request, run_case
from scripts.gemini_live_diagnostics import load_environment, run_stage

MAX_ATTEMPTS = 7  # One preflight, production D, explicit comparator D, A/B/C/E.
MAX_INVOCATIONS = 11  # Worst case includes assessment + solution on A/B/C/E.
OWNER_FACT_ID = "20000000-0000-4000-8000-000000000002"
OWNER_FACT = "Pesanan dicatat manual di kasir."
SECTIONS = (
    "findings",
    "customer_evidence",
    "hypotheses",
    "recommendations",
    "success_indicators",
)


def scenario_request(scenario, previous_solution=None):
    request = case_request("start" if scenario in {"A", "B", "C"} else "grpc")
    request.context.shop.name = "Kedai sintetis M6"
    request.request_id = f"50000000-0000-4000-8000-{ord(scenario):012d}"
    request.event = {
        "A": ai_pb2.CONSULTATION_EVENT_START,
        "B": ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
        "C": ai_pb2.CONSULTATION_EVENT_SKIP_QUESTION,
        "D": ai_pb2.CONSULTATION_EVENT_GENERATE_NOW,
        "E": ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
    }[scenario]
    if scenario in {"B", "C"}:
        request.messages.add(
            id="90000000-0000-4000-8000-000000000001",
            role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
            content="Berapa barista yang bekerja saat jam ramai?",
            topic_key="staffing",
        )
        request.asked_topics.append("staffing")
        request.question_count = 1
        if scenario == "B":
            request.messages.add(
                id="20000000-0000-4000-8000-000000000003",
                role=ai_pb2.MESSAGE_ROLE_OWNER,
                content="Saat jam ramai, dua barista menangani pesanan.",
            )
        else:
            request.skipped_topics.append("staffing")
    if scenario == "D":  # F piggybacks here; no separate Gemini request.
        request.messages.add(
            id=OWNER_FACT_ID, role=ai_pb2.MESSAGE_ROLE_OWNER, content=OWNER_FACT
        )
    if scenario == "E":
        if previous_solution is None:
            raise ValueError(
                "E requires a previously validated Solution; no invented live result"
            )
        request.previous_solution.CopyFrom(previous_solution)
        request.messages.add(
            id="90000000-0000-4000-8000-000000000002",
            role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
            solution=previous_solution,
        )
        request.messages.add(
            id="20000000-0000-4000-8000-000000000004",
            role=ai_pb2.MESSAGE_ROLE_OWNER,
            content="Revisi langkah observasi agar berfokus pada alur pencatatan pesanan, tanpa pembelian alat.",
        )
    prepare_context(request)  # Fail before spending any API request.
    return request


def context_observations(request):
    context = prepare_context(request)
    latest = context.latest_owner_message
    return {
        "history_order_preserved": [m.id for m in context.conversation_history]
        == [m.id for m in request.messages],
        "latest_owner_history_occurrences": sum(
            m.id == latest.id for m in context.conversation_history
        )
        if latest
        else 0,
        "latest_owner_source_occurrences": int(
            latest.id in context.sources.owner_messages
        )
        if latest
        else 0,
        "latest_owner_is_separate_context_view": latest is not None,
        "ai_ids_excluded_from_owner_sources": all(
            m.id not in context.sources.owner_messages
            for m in request.messages
            if m.role == ai_pb2.MESSAGE_ROLE_ATLAS_AI
        ),
        "previous_solution_source_kind": context.previous_solution.source_kind
        if context.previous_solution
        else None,
        "confirmed_memory_count": len(context.confirmed_memories),
        "asked_topics": list(context.asked_topics),
        "skipped_topics": list(context.skipped_topics),
        "question_count": context.question_count,
    }


def sanitize_row(row):
    """Keep usage/output digests, never raw SDK error bodies or provider prompts."""
    row.pop("provider_errors", None)
    row.pop("application_error", None)
    for response in row.get("provider_responses", []):
        raw = response.pop("visible_text", "")
        response["output_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        response["output_utf8_bytes"] = len(raw.encode())
    return row


def finish_rpc_row(row, request):
    row = sanitize_row(row)
    invocations = row.get("sdk_invocations", [])
    row["actual_gemini_invocations"] = len(invocations)
    row["dedicated_memory_invocations"] = 0
    row["event"] = ai_pb2.ConsultationEvent.Name(request.event)
    row["context_observations"] = context_observations(request)
    timings = row.get("stage_durations_seconds", {})
    row["validation_reached"] = {
        "pydantic": any(k.endswith(".pydantic") for k in timings),
        "solution_structural": "solution.structural" in timings,
        "solution_grounding": "solution.grounding" in timings,
        "memory": "memory.validation_and_mapping" in timings,
    }
    statuses = [e["http_status"] for e in row.get("http_events", [])]
    if row.get("status") == "FAILED":
        if any(code in statuses for code in (503, 504, 408, 429)):
            code = next(code for code in (503, 504, 408, 429) if code in statuses)
            row["category"] = (
                "PROVIDER_RATE_LIMIT" if code == 429 else f"PROVIDER_HTTP_{code}"
            )
        elif any(
            i.get("exception_type")
            in {"ReadTimeout", "ConnectTimeout", "TimeoutError", "PoolTimeout"}
            for i in invocations
        ):
            row["category"] = "CLIENT_TIMEOUT"
        elif row.get("actual_grpc_status") == "DEADLINE_EXCEEDED":
            row["category"] = (
                "GRPC_CALLER_DEADLINE"
                if row.get("rpc_duration_seconds", 0) >= row["rpc_timeout_seconds"]
                else "APPLICATION_DEADLINE"
            )
        if row.get("category") == "PARSER_SCHEMA_FAILURE":
            row["category"] = (
                "LOCAL_PYDANTIC_REJECTION"
                if row["validation_reached"]["pydantic"]
                else "STRUCTURED_OUTPUT_PARSING_FAILURE"
            )
        row["all_response_validations"] = "NOT_PASSED"
        return row
    response = row["response"]
    kind = row["result_kind"]
    expected = (
        1
        if request.event == ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
        or kind == "next_question"
        or request.question_count >= 4
        else 2
    )
    row["expected_gemini_invocations"] = expected
    # Rechecked against current OWNER sources; production already filtered these.
    owners = prepare_context(request).sources.owner_messages
    candidates = response.get("memory_candidates", [])
    row["memory_validation"] = {
        "returned_count": len(candidates),
        "live_candidate_demonstrated": bool(candidates),
        "empty_is_valid": True,
        "owner_sources_valid": all(
            c.get("source_owner_message_ids")
            and all(s in owners for s in c["source_owner_message_ids"])
            for c in candidates
        ),
        "provenance": [
            {
                **c,
                "owner_source_texts": {
                    s: owners[s].content
                    for s in c["source_owner_message_ids"]
                    if s in owners
                },
            }
            for c in candidates
        ],
    }
    row["response_checks"] = {
        "oneof": kind in {"solution", "next_question"}
        and ("solution" in response) != ("next_question" in response),
        "metadata": bool(response.get("model_id") and response.get("prompt_version")),
        "protobuf_round_trip": row["protobuf_round_trip"],
        "invocation_count": len(invocations) == expected,
        "request_unchanged": row.get("server_request_unchanged", False),
        "memory_owner_provenance": row["memory_validation"]["owner_sources_valid"],
    }
    if kind == "solution":
        row["response_checks"]["five_nonempty_sections"] = all(
            response["solution"].get(k) for k in SECTIONS
        )
        row["response_checks"]["production_validation_stages"] = all(
            k in timings
            for k in (
                "solution.pydantic",
                "solution.structural",
                "solution.grounding",
                "solution.protobuf_mapping",
                "memory.validation_and_mapping",
            )
        )
    else:
        question = response["next_question"]
        row["response_checks"]["no_repeated_topic"] = question["topic_key"] not in {
            *request.asked_topics,
            *request.skipped_topics,
        }
        row["response_checks"]["question_guard"] = (
            row.get("validation", {}).get("question_guard") == "PASSED"
        )
    passed = all(row["response_checks"].values())
    row["all_response_validations"] = "PASSED" if passed else "FAILED"
    if not passed:
        row["status"] = "DIAGNOSTIC_CHECK_FAILED"
    return row


def attempt_allowed(runs, scenario, model, api_version, thinking):
    reserve = 1 if scenario in {"preflight", "D"} else 2
    if (
        len(runs) >= MAX_ATTEMPTS
        or sum(r["actual_gemini_invocations"] for r in runs) + reserve > MAX_INVOCATIONS
    ):
        return False
    if any(r["scenario"] == scenario and r["model"] == model for r in runs):
        return False
    if scenario == "preflight":
        return not runs
    if scenario == "D":
        return sum(r["scenario"] == "D" for r in runs) < 2
    matching = [
        r
        for r in runs
        if r["category_of_test"] == "C_REAL_PROVIDER_AUTHENTICATED_GRPC"
        and r["model"] == model
        and r.get("requested_api_version") == api_version
        and r.get("requested_thinking") == thinking
    ]
    return bool(matching) and any(
        event["http_status"] == 200 for event in matching[-1].get("http_events", [])
    )


def historical_solution(runs, model, run_id=None):
    """Historical input only, never replayed as the output of the new RPC."""
    row = next(
        (
            r
            for r in runs
            if r.get("actual_grpc_status") == "OK"
            and r["status"] == "PASSED"
            and r.get("response", {}).get("solution")
            and (r.get("model") or r.get("model_override")) == model
            and (run_id is None or r["id"] == run_id)
        ),
        None,
    )
    if row is None:
        raise ValueError(
            "A successful, authenticated live Solution is required as historical input"
        )
    return ParseDict(row["response"]["solution"], ai_pb2.Solution()), row["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument(
        "--scenario", choices=["preflight", "A", "B", "C", "D", "E"], required=True
    )
    parser.add_argument("--model", help="Explicit diagnostic override only")
    parser.add_argument("--turn-budget", type=float)
    parser.add_argument("--api-version", choices=["v1", "v1beta"])
    parser.add_argument("--thinking", choices=["low", "medium", "high"])
    parser.add_argument(
        "--previous-ledger",
        type=Path,
        help="Optional read-only historical Solution input for E, never evidence of this run passing",
    )
    parser.add_argument("--previous-run-id")
    args = parser.parse_args()
    ledger = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {"milestone": "M6", "synthetic_only": True, "runs": []}
    )
    if ledger.get("milestone") != "M6" or any(
        r["id"] == args.id for r in ledger["runs"]
    ):
        parser.error("Wrong ledger or duplicate ID; no request sent")
    load_environment(args.env_file)
    production = GeminiSettings.from_env()
    data = production.model_dump()
    if args.model:
        data["model"] = args.model
    if args.turn_budget is not None:
        data["timeout_seconds"] = args.turn_budget
    if args.scenario == "preflight":
        data.update(
            timeout_seconds=min(data["timeout_seconds"], 10), max_output_tokens=256
        )
    settings = GeminiSettings.model_validate(data)
    if not settings.model or not settings.api_key.get_secret_value().strip():
        parser.error("Missing configuration; no request sent")
    if not attempt_allowed(
        ledger["runs"], args.scenario, settings.model, args.api_version, args.thinking
    ):
        parser.error(
            "Bounded limit/duplicate scenario/provider responsiveness gate; no request sent"
        )
    previous = None
    if args.scenario == "E":
        previous_ledger = (
            json.loads(args.previous_ledger.read_text(encoding="utf-8"))
            if args.previous_ledger
            else ledger
        )
        try:
            previous, previous_run_id = historical_solution(
                previous_ledger["runs"], settings.model, args.previous_run_id
            )
        except ValueError:
            parser.error(
                "E requires a successful authenticated live Solution as historical input"
            )
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    if args.scenario == "preflight":
        row = sanitize_row(
            run_stage(
                "lc_simple",
                settings,
                api_version=args.api_version,
                thinking=args.thinking,
            )
        )
        row["actual_gemini_invocations"] = len(row.get("sdk_invocations", []))
        row["category_of_test"] = "C_PREFLIGHT_NOT_GRPC_NOT_E2E"
    else:
        request = scenario_request(args.scenario, previous)
        row = run_case(
            "grpc",
            settings,
            request=request,
            allowed_results={"solution"}
            if args.scenario == "D"
            else {"solution", "next_question"},
            api_version=args.api_version,
            thinking=args.thinking,
        )
        row = finish_rpc_row(row, request)
        row["category_of_test"] = "C_REAL_PROVIDER_AUTHENTICATED_GRPC"
        if args.scenario == "E":
            row["previous_solution_run_id"] = previous_run_id
            row["previous_solution_input_ledger"] = (
                args.previous_ledger.name
                if args.previous_ledger
                else "current_m6_ledger"
            )
            row["previous_solution_role"] = (
                "HISTORICAL_AI_INPUT_ONLY_NOT_A_NEW_LIVE_RESULT"
            )
    row.update(
        id=args.id,
        scenario=args.scenario,
        model=settings.model,
        diagnostic_model_override=settings.model != production.model,
        requested_api_version=args.api_version,
        requested_thinking=args.thinking,
        started_at=started_at,
        ended_at=datetime.now(UTC).isoformat(),
        total_duration_seconds=round(time.perf_counter() - started, 6),
    )
    ledger.update(
        production_model_unchanged=production.model,
        production_turn_budget_seconds=production.timeout_seconds,
        production_output_token_budget=production.max_output_tokens,
        limits={
            "attempts": MAX_ATTEMPTS,
            "gemini_invocations": MAX_INVOCATIONS,
            "automatic_retries": 0,
        },
        packages={
            p: importlib.metadata.version(p)
            for p in (
                "google-genai",
                "langchain-google-genai",
                "langchain-core",
                "pydantic",
                "grpcio",
                "protobuf",
            )
        },
    )
    ledger["runs"].append(row)
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: row.get(k)
                for k in (
                    "id",
                    "scenario",
                    "model",
                    "status",
                    "category",
                    "actual_grpc_status",
                    "http_events",
                    "rpc_duration_seconds",
                    "actual_gemini_invocations",
                    "all_response_validations",
                    "memory_validation",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
