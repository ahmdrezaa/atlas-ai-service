"""Bounded 4C live tests through the real pipeline; no production model changes."""

import argparse
import json
import secrets
import time
from concurrent import futures
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict
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
from atlas_ai.consultation.solution import GeneratedSolution, validate_solution
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
        result = GeneratedSolution.model_validate_json(
            capture.responses[-1]["visible_text"]
        )
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


def run_case(case, settings, *, rpc_timeout=60):
    request = case_request(case)
    capture = Capture()

    def factory(**kwargs):
        return capture.model_factory(api_version="v1beta", thinking="low", **kwargs)

    pipeline = ConsultationPipeline(ContextAssessor(settings, factory))
    row = {
        "case": case,
        "model_override": settings.model,
        "turn_budget_seconds": settings.timeout_seconds,
        "output_token_budget": settings.max_output_tokens,
        "thinking": "LOW",
        "api_version": "v1beta",
        "synthetic_request": MessageToDict(request, preserving_proto_field_name=True),
        "automatic_retries": 0,
        "bounded_corrections": 0,
        "rpc_timeout_seconds": rpc_timeout,
        "configured_rpc_headroom_seconds": rpc_timeout - settings.timeout_seconds,
    }
    original_run = pipeline.run

    def measured_run(*args, **kwargs):
        row["server_entry_time_remaining_seconds"] = kwargs.get("time_remaining")
        call_started = time.perf_counter()
        try:
            return original_run(*args, **kwargs)
        finally:
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
                    row["rpc_duration_seconds"] = round(
                        time.perf_counter() - rpc_started, 6
                    )
                    row["grpc_overhead_observed_seconds"] = round(
                        row["rpc_duration_seconds"] - row["pipeline_duration_seconds"],
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
            "PASSED" if row["result_kind"] == expected else "UNEXPECTED_RESULT"
        )
    except Exception as error:  # noqa: BLE001 -- sanitized application/provider boundary.
        row["status"] = "FAILED"
        row.update(classify_error(error, capture.http_events))
        if isinstance(error, grpc.RpcError):
            row["actual_grpc_status"] = error.code().name
        elif hasattr(error, "code") and isinstance(error.code, grpc.StatusCode):
            row["application_error"] = str(error)
    row.update(
        duration_seconds=round(time.perf_counter() - started, 3),
        http_events=capture.http_events,
        provider_errors=capture.provider_errors,
        effective_requests=capture.requests,
        provider_responses=capture.responses,
    )
    if case != "start":
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
    parser.add_argument("--milestone", choices=["4C", "4C.1"], default="4C")
    args = parser.parse_args()
    ledger = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {"milestone": args.milestone, "runs": []}
    )
    if ledger.get("milestone", args.milestone) != args.milestone:
        parser.error("Milestone does not match ledger; no request sent")
    limit = 2 if args.milestone == "4C.1" else 6
    if len(ledger["runs"]) >= limit or any(
        row["id"] == args.id for row in ledger["runs"]
    ):
        parser.error("Bounded run limit/duplicate ID; no request sent")
    if args.milestone == "4C.1" and (
        args.case == "start" or any(row["case"] == args.case for row in ledger["runs"])
    ):
        parser.error("4C.1 permits one solution and one conditional gRPC call")
    prerequisites = ("solution",) if args.milestone == "4C.1" else ("start", "solution")
    if args.case == "grpc" and not all(
        any(row["case"] == case and row["status"] == "PASSED" for row in ledger["runs"])
        for case in prerequisites
    ):
        parser.error("Both direct live paths must pass before live gRPC testing")
    load_environment(args.env_file)
    original = GeminiSettings.from_env()
    settings = GeminiSettings.model_validate(
        {**original.model_dump(), "model": MODEL, "timeout_seconds": args.turn_budget}
    )
    if not settings.api_key.get_secret_value():
        parser.error("API key missing; no request sent")
    result = run_case(args.case, settings)
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
