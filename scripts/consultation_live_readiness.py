"""Bounded 4C live tests through the real pipeline; no production model changes."""

import argparse
import hashlib
import json
import secrets
import time
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict
from pydantic import ValidationError

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import evaluate_claim, solution_claims
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.solution import (
    GeneratedSolution,
    GeneratedSolutionTurn,
    solution_to_proto,
    validate_solution,
)
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.server import BearerAuthInterceptor
from scripts.gemini_live_diagnostics import (
    Capture,
    classify_error,
    load_environment,
    synthetic_request,
)

MODEL = "gemini-3.6-flash"  # Test override only; never persisted to .env.


def case_request(case):
    request = synthetic_request()
    request.context.shop.name = "Kedai sintetis 4C"
    request.context.shop.category = "Kafe"
    request.context.shop.scale = "Kecil"
    request.context.evidence.add(
        opinion_id="10000000-0000-4000-8000-000000000002",
        opinion_text="Pelayanan lambat",
        evidence_text="Pelayanannya lambat.",
        full_feedback="Pelayanannya lambat.",
        rating=3,
        sentiment=ai_pb2.SENTIMENT_NEGATIVE,
        severity=ai_pb2.SEVERITY_NOT_SEVERE,
    )
    request.context.stats.unique_feedback_count = 2
    request.context.stats.opinion_count = 2
    request.context.stats.negative_count = 2
    request.context.stats.not_severe_count = 2
    facts = [
        "Jam ramai pukul 17.00 sampai 19.00 setiap hari.",
        "Anggaran tambahan belum tersedia; langkah awal harus tanpa pembelian alat.",
        "Waktu observasi tersedia selama satu minggu.",
        "Owner ingin mengurangi antrean dengan memperbaiki alur kasir.",
    ]
    if case != "start":
        request.event = ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
        facts.append("Jumlah barista dua orang.")
        request.messages.add(
            id="20000000-0000-4000-8000-000000000001",
            role=ai_pb2.MESSAGE_ROLE_OWNER,
            content="Buat langkah observasi tanpa biaya. Penyebab antrean belum diketahui.",
        )
    for index, fact in enumerate(facts, start=1):
        request.context.memories.add(
            id=f"30000000-0000-4000-8000-{index:012d}", fact_text=fact
        )
    return request


def inspect_solution_output(capture, request):
    """Inspect real synthetic model output after the timed call; no repair/inference."""
    if not capture.responses:
        return {"structural": "NOT_REACHED", "grounding": "NOT_REACHED"}
    try:
        turn = GeneratedSolutionTurn.model_validate_json(
            capture.responses[-1]["visible_text"]
        )
        result = turn.solution_output()
    except ValidationError as error:
        return {
            "pydantic_schema": "REJECTED",
            "structural": "REJECTED",
            "grounding": "NOT_REACHED",
            "exception_type": type(error).__name__,
        }
    context = prepare_context(request)
    details = {
        "pydantic_schema": "PASSED",
        # This runner only sends synthetic fixtures; retain parsed claims for
        # diagnosis without recording prompts or raw provider error bodies.
        "synthetic_typed_output": turn.model_dump(mode="json"),
        "section_lengths": {
            key: len(getattr(result, key)) for key in GeneratedSolution.model_fields
        },
    }
    try:
        validate_solution(result, context)
    except ConsultationFailure as error:
        return {
            **details,
            "structural": "REJECTED",
            "grounding": "NOT_REACHED",
            "application_error": str(error),
        }
    decisions = [
        evaluate_claim(claim, context).model_dump(mode="json")
        for claim in solution_claims(result)
    ]
    return {
        **details,
        "structural": "PASSED",
        "grounding": "PASSED"
        if all(d["label"] in {"SUPPORTED", "NOT_APPLICABLE"} for d in decisions)
        else "REJECTED",
        "decisions": decisions,
    }


def revalidate_saved_live_solution(runs):
    """A spend gate only, never a substitute for the fresh gRPC invocation.

    Keep the original failed row untouched. Revalidate its exact HTTP-200 output
    after a local false-rejection fix; do not repair it or return it as the RPC result.
    """
    for row in runs:
        if (
            row["case"] != "solution"
            or not row.get("provider_responses")
            or not any(
                event["http_status"] == 200 for event in row.get("http_events", [])
            )
        ):
            continue
        raw = row["provider_responses"][-1]["visible_text"]
        try:
            request = ParseDict(
                row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
            )
            result = GeneratedSolutionTurn.model_validate_json(raw).solution_output()
            solution_to_proto(result, prepare_context(request))
        except (ValueError, ConsultationFailure):
            continue
        return {
            "type": "UNMODIFIED_SAVED_LIVE_OUTPUT_REVALIDATED_NOT_NEW_LIVE_SUCCESS",
            "original_run_id": row["id"],
            "original_status": row["status"],
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "pydantic_structural_grounding_protobuf": "PASSED",
            "used_as_rpc_response": False,
        }
    return None


def run_case(
    case,
    settings,
    *,
    rpc_timeout=60,
    memory_smoke=False,
    request=None,
    allowed_results=None,
    api_version="v1beta",
    thinking="low",
):
    request = case_request(case) if request is None else request
    if memory_smoke:
        request.messages.add(
            id="20000000-0000-4000-8000-000000000002",
            role=ai_pb2.MESSAGE_ROLE_OWNER,
            content="Pesanan dicatat manual di kasir.",
        )
    capture = Capture()

    def factory(**kwargs):
        return capture.model_factory(
            api_version=api_version, thinking=thinking, **kwargs
        )

    pipeline = ConsultationPipeline(ContextAssessor(settings, factory))
    row = {
        "case": case,
        "model_override": settings.model,
        "turn_budget_seconds": settings.timeout_seconds,
        "output_token_budget": settings.max_output_tokens,
        "thinking": thinking.upper() if thinking else "provider_default",
        "api_version": api_version or "sdk_default",
        "started_at": datetime.now(UTC).isoformat(),
        "synthetic_request": MessageToDict(request, preserving_proto_field_name=True),
        "automatic_retries": 0,
        "bounded_corrections": 0,
        "stage_durations_seconds": {},
        "rpc_timeout_seconds": rpc_timeout,
        "configured_rpc_headroom_seconds": rpc_timeout - settings.timeout_seconds,
    }
    original_run = pipeline.run

    def measured_run(*args, **kwargs):
        row["server_entry_time_remaining_seconds"] = kwargs.get("time_remaining")
        if settings.timeout_seconds >= 40 and (
            kwargs.get("time_remaining") is None
            or kwargs["time_remaining"] - settings.timeout_seconds < 10
        ):
            raise ConsultationFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Development budget requires 10s caller headroom at entry",
            )
        kwargs["timings"] = row["stage_durations_seconds"]
        call_started = time.perf_counter()
        before = args[0].SerializeToString()
        try:
            return original_run(*args, **kwargs)
        finally:
            row["server_request_unchanged"] = before == args[0].SerializeToString()
            row["pipeline_duration_seconds"] = round(
                time.perf_counter() - call_started, 6
            )

    pipeline.run = measured_run
    started = time.perf_counter()
    try:
        if case == "grpc":
            token = secrets.token_urlsafe(32)
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=2),
                interceptors=[BearerAuthInterceptor(token)],
            )
            service = AtlasAIService()
            service.consultation_pipeline = pipeline
            ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
            port = server.add_insecure_port("127.0.0.1:0")
            if not port:
                raise RuntimeError("Test service could not bind")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    rpc_started = time.perf_counter()
                    row["rpc_started_at"] = datetime.now(UTC).isoformat()
                    try:
                        response = ai_pb2_grpc.AIServiceStub(
                            channel
                        ).ProcessConsultationTurn(
                            request,
                            timeout=rpc_timeout,
                            metadata=(
                                ("authorization", f"Bearer {token}"),
                                ("x-request-id", request.request_id),
                            ),
                        )
                    finally:
                        row["rpc_ended_at"] = datetime.now(UTC).isoformat()
                        row["rpc_duration_seconds"] = round(
                            time.perf_counter() - rpc_started, 6
                        )
                        if "pipeline_duration_seconds" in row:
                            row["grpc_overhead_observed_seconds"] = round(
                                row["rpc_duration_seconds"]
                                - row["pipeline_duration_seconds"],
                                6,
                            )
            finally:
                server.stop(0).wait(timeout=5)
            row["actual_grpc_status"] = "OK"
            row["rpc_timeout_seconds"] = rpc_timeout
        else:
            response = pipeline.run(request, time_remaining=rpc_timeout)
        row["result_kind"] = response.WhichOneof("result")
        row["response"] = MessageToDict(response, preserving_proto_field_name=True)
        row["protobuf_round_trip"] = (
            ai_pb2.ProcessConsultationTurnResponse.FromString(
                response.SerializeToString()
            )
            == response
        )
        if row["result_kind"] == "solution":
            row["go_required_references_nonempty"] = all(
                item.references.opinion_ids
                or item.references.owner_message_ids
                or item.references.memory_ids
                for section in ("findings", "hypotheses", "recommendations")
                for item in getattr(response.solution, section)
            )
            if not row["go_required_references_nonempty"]:
                raise ConsultationFailure(
                    grpc.StatusCode.INTERNAL,
                    "Go requires a nonempty current reference for every finding/hypothesis/recommendation",
                )
        expected = "next_question" if case == "start" else "solution"
        row["status"] = (
            "PASSED"
            if row["result_kind"] in (allowed_results or {expected})
            else "UNEXPECTED_RESULT"
        )
        if memory_smoke:
            owners = prepare_context(request).sources.owner_messages
            row["memory_validation"] = {
                "returned_count": len(response.memory_candidates),
                "owner_sources_valid": all(
                    memory.source_owner_message_ids
                    and all(
                        source_id in owners
                        for source_id in memory.source_owner_message_ids
                    )
                    for memory in response.memory_candidates
                ),
                "provenance": [
                    {
                        "fact_text": memory.fact_text,
                        "source_owner_message_ids": list(
                            memory.source_owner_message_ids
                        ),
                        "owner_messages": [
                            owners[source_id].model_dump(mode="json")
                            for source_id in memory.source_owner_message_ids
                        ],
                    }
                    for memory in response.memory_candidates
                ],
            }
            if not response.memory_candidates:
                row["status"] = "EMPTY_MEMORY_CANDIDATES"
    except Exception as error:  # noqa: BLE001 -- sanitized application/provider boundary.
        row["status"] = "FAILED"
        row.update(classify_error(error, capture.http_events))
        if isinstance(error, grpc.RpcError):
            row["actual_grpc_status"] = error.code().name
            row["grpc_status_equivalent"] = error.code().name
            row["grpc_status_source"] = "actual_grpc"
            # Only classify known application messages. Never persist wire details.
            for prefix, category in (
                ("Solution grounding rejected", "GROUNDING_REJECTION"),
                ("Invalid structured", "PARSER_SCHEMA_FAILURE"),
                ("Invalid solution", "STRUCTURAL_REJECTION"),
                ("Assessment returned an invalid", "QUESTION_GUARD_REJECTION"),
            ):
                if (error.details() or "").startswith(prefix):
                    row["category"] = category
        elif hasattr(error, "code") and isinstance(error.code, grpc.StatusCode):
            row["application_error"] = str(error)
            if error.code == grpc.StatusCode.INTERNAL:
                message = str(error)
                if message.startswith("Solution grounding rejected"):
                    row["category"] = "GROUNDING_REJECTION"
                elif message.startswith("Invalid structured"):
                    row["category"] = "PARSER_SCHEMA_FAILURE"
                elif message.startswith("Invalid solution"):
                    row["category"] = "STRUCTURAL_REJECTION"
    row.update(
        duration_seconds=round(time.perf_counter() - started, 3),
        http_events=capture.http_events,
        provider_errors=capture.provider_errors,
        effective_requests=capture.requests,
        provider_responses=capture.responses,
        sdk_invocations=capture.invocations,
        ended_at=datetime.now(UTC).isoformat(),
    )
    if row.get("result_kind") == "solution" or (
        case != "start" and row.get("result_kind") != "next_question"
    ):
        row["validation"] = inspect_solution_output(capture, request)
    elif row.get("result_kind") == "next_question":
        row["validation"] = {"question_guard": "PASSED"}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["start", "solution", "grpc"], required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--turn-budget", type=float, default=30)
    parser.add_argument("--milestone", choices=["4C", "4C.1", "P0", "M5"], default="4C")
    parser.add_argument("--rpc-timeout", type=float, default=60)
    args = parser.parse_args()
    ledger = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {"milestone": args.milestone, "runs": []}
    )
    if ledger.get("milestone", args.milestone) != args.milestone:
        parser.error("Milestone does not match ledger; no request sent")
    if args.milestone in {"P0", "M5"} and not (
        40 <= args.turn_budget <= 45 and args.turn_budget + 10 <= args.rpc_timeout <= 60
    ):
        parser.error(
            "P0 requires a 40-45s budget with 10s+ headroom within the audited 60s caller"
        )
    limit = (
        3 if args.milestone == "P0" else 2 if args.milestone in {"4C.1", "M5"} else 6
    )
    if args.milestone == "M5" and args.case != "grpc":
        parser.error(
            "M5 uses one fresh gRPC smoke after offline schema/tests, not warmup calls"
        )
    if len(ledger["runs"]) >= limit or any(
        row["id"] == args.id for row in ledger["runs"]
    ):
        parser.error("Bounded run limit/duplicate ID; no request sent")
    if args.milestone in {"4C.1", "P0"} and (
        args.case == "start"
        or sum(row["case"] == args.case for row in ledger["runs"])
        >= (2 if args.milestone == "P0" and args.case == "solution" else 1)
    ):
        parser.error("4C.1 permits one solution and one conditional gRPC call")
    prerequisites = (
        ("solution",) if args.milestone in {"4C.1", "P0"} else ("start", "solution")
    )
    prerequisites_passed = all(
        any(row["case"] == case and row["status"] == "PASSED" for row in ledger["runs"])
        for case in prerequisites
    )
    if args.milestone == "M5":
        # Schema + mock gRPC are verified offline first. One live invocation can
        # test Solution and candidate metadata together, without an extra warmup.
        prerequisites_passed = True
    replay = None
    if args.case == "grpc" and args.milestone == "P0" and not prerequisites_passed:
        replay = revalidate_saved_live_solution(ledger["runs"])
        prerequisites_passed = replay is not None
    if args.case == "grpc" and not prerequisites_passed:
        parser.error("Both direct live paths must pass before live gRPC testing")
    load_environment(args.env_file)
    original = GeminiSettings.from_env()
    settings = GeminiSettings.model_validate(
        {**original.model_dump(), "model": MODEL, "timeout_seconds": args.turn_budget}
    )
    if not settings.api_key.get_secret_value():
        parser.error("API key missing; no request sent")
    result = run_case(
        args.case,
        settings,
        rpc_timeout=args.rpc_timeout,
        memory_smoke=args.milestone == "M5",
    )
    if replay is not None:
        result["spend_prerequisite"] = replay
    result["id"] = args.id
    ledger["production_model_unchanged"] = original.model
    ledger["runs"].append(result)
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"synthetic_request", "provider_responses", "response"}
    }
    summary["usage"] = [response["usage"] for response in result["provider_responses"]]
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
