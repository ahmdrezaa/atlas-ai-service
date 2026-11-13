"""One explicit live request per invocation; synthetic data and allowlisted telemetry.

No automatic retries, model switching, prompt/grounding changes, or private inputs.
Run with PYTHONPATH=src after installing the existing project dependencies.
"""

import argparse
import importlib.metadata
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import grpc
import httpx
from google import genai
from google.genai import types
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
    validate_question,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.solution import SolutionGenerator


class DiagnosticReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ATLAS_OK"]


def load_environment(path: Path) -> None:
    allowed = {
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_TIMEOUT_SECONDS",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "CONSULTATION_MAX_CONTEXT_BYTES",
    }
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in allowed:
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def synthetic_request():
    request = ai_pb2.ProcessConsultationTurnRequest(
        session_id="40000000-0000-4000-8000-000000000001",
        request_id="50000000-0000-4000-8000-000000000001",
        event=ai_pb2.CONSULTATION_EVENT_START,
    )
    request.context.issue.id = "60000000-0000-4000-8000-000000000001"
    request.context.issue.title = "Antrean"
    request.context.shop.id = "70000000-0000-4000-8000-000000000001"
    request.context.shop.name = "Kedai sintetis diagnostik"
    request.context.stats.CopyFrom(
        ai_pb2.IssueStats(
            unique_feedback_count=1,
            opinion_count=1,
            average_rating=3,
            negative_count=1,
            not_severe_count=1,
        )
    )
    request.context.evidence.add(
        opinion_id="10000000-0000-4000-8000-000000000001",
        opinion_text="Antrean lama",
        evidence_text="Antreannya lama sekali.",
        full_feedback="Antreannya lama sekali.",
        rating=3,
        sentiment=ai_pb2.SENTIMENT_NEGATIVE,
        severity=ai_pb2.SEVERITY_NOT_SEVERE,
    )
    return request


def classify_error(error, http_events):
    """Never serialize exception messages, response bodies, headers or credentials."""
    chain, seen = [], set()
    current = error
    client_timeout = parsing = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        chain.append(
            {
                "type": type(current).__name__,
                "http_code": code
                if isinstance(code, int) and 100 <= code <= 599
                else None,
            }
        )
        client_timeout |= isinstance(current, (TimeoutError, httpx.TimeoutException))
        parsing |= isinstance(current, (ValidationError, OutputParserException))
        current = current.__cause__ or current.__context__
    codes = [event["http_status"] for event in http_events] + [
        item["http_code"] for item in chain if item["http_code"]
    ]
    if 504 in codes:
        category, status = "PROVIDER_HTTP_504", "DEADLINE_EXCEEDED"
    elif 429 in codes:
        category, status = "RATE_LIMIT", "RESOURCE_EXHAUSTED"
    elif client_timeout:
        category, status = "CLIENT_TIMEOUT", "DEADLINE_EXCEEDED"
    elif parsing:
        category, status = "STRUCTURED_OUTPUT_PARSE_ERROR", "INTERNAL"
    elif any(code >= 400 for code in codes):
        category, status = "PROVIDER_HTTP_ERROR", "UNAVAILABLE"
    elif isinstance(error, ConsultationFailure):
        status = error.code.name
        category = (
            "APPLICATION_VALIDATION"
            if error.code == grpc.StatusCode.INTERNAL
            else "APPLICATION_DEADLINE"
            if error.code == grpc.StatusCode.DEADLINE_EXCEEDED
            else "APPLICATION_FAILURE"
        )
    else:
        category, status = "SDK_OR_TRANSPORT_ERROR", "UNAVAILABLE"
    if isinstance(error, ConsultationFailure):
        status = error.code.name  # Preserve the actual existing application mapping.
    return {
        "category": category,
        "grpc_status_equivalent": status,
        "grpc_status_source": "application_exception"
        if isinstance(error, ConsultationFailure)
        else "diagnostic_mapping_only",
        "exception_chain": chain,
    }


class Capture:
    def __init__(self):
        self.http_events = []
        self.responses = []
        self.requests = []
        self.provider_errors = []
        self.invocations = []

    def response_hook(self, response):
        self.http_events.append(
            {
                "http_status": response.status_code,
                "host": response.request.url.host,
                "path": response.request.url.path,
            }
        )

        if response.status_code == 400:
            try:
                response.read()
                error = response.json().get("error", {})
            except (ValueError, httpx.HTTPError):
                return
            message = str(error.get("message", ""))
            for value in (
                response.request.headers.get("x-goog-api-key"),
                response.request.headers.get("authorization"),
                response.request.url.params.get("key"),
            ):
                if value:
                    message = message.replace(value, "[REDACTED]")
            self.provider_errors.append(
                {"http_status": 400, "message_redacted": message[:3000]}
            )

    def record(self, response):
        usage = response.usage_metadata
        self.responses.append(
            {
                "model_version": response.model_version,
                "finish_reasons": [
                    str(
                        getattr(
                            candidate.finish_reason, "value", candidate.finish_reason
                        )
                    )
                    for candidate in response.candidates or []
                ],
                "usage": {
                    key: getattr(usage, key, None)
                    for key in (
                        "prompt_token_count",
                        "candidates_token_count",
                        "thoughts_token_count",
                        "total_token_count",
                        "cached_content_token_count",
                    )
                }
                if usage
                else None,
                "visible_text": "".join(
                    part.text or ""
                    for candidate in response.candidates or []
                    for part in candidate.content.parts or []
                    if not part.thought
                )
                if response.candidates and all(c.content for c in response.candidates)
                else "",
            }
        )

    def model_factory(self, *, api_version=None, thinking=None, **kwargs):
        options = {"client_args": {"event_hooks": {"response": [self.response_hook]}}}
        if api_version is not None:
            options["api_version"] = api_version
        if thinking is not None:
            options["thinking_config"] = {"thinking_level": thinking}
        model = ChatGoogleGenerativeAI(**kwargs, **options)
        self.requests.append(
            {
                "api_version": model.client._api_client._http_options.api_version,
                "timeout_seconds": kwargs.get("timeout"),
                "max_output_tokens": kwargs.get("max_output_tokens"),
                "thinking_level_requested": thinking or "provider_default",
            }
        )
        original = model.client.models.generate_content

        def generate(**request):
            config = request.get("config")
            content = list(request.get("contents") or [])
            if config.system_instruction:
                content.append(config.system_instruction)
            texts = [
                part.text
                for item in content
                for part in getattr(item, "parts", None) or []
                if part.text
            ]
            self.requests[-1].update(
                {
                    "actual_prompt_text_utf8_bytes": sum(
                        len(text.encode("utf-8")) for text in texts
                    ),
                    "response_json_schema_utf8_bytes": len(
                        json.dumps(config.response_json_schema).encode("utf-8")
                    )
                    if config.response_json_schema
                    else 0,
                    "effective_temperature": config.temperature,
                    "effective_thinking_config": config.thinking_config.model_dump(
                        mode="json", exclude_none=True
                    )
                    if config.thinking_config
                    else None,
                }
            )
            started = time.perf_counter()
            invocation = {
                "started_at": datetime.now(UTC).isoformat(),
                "schema_title": (config.response_json_schema or {}).get("title"),
            }
            self.invocations.append(invocation)
            http_start = len(self.http_events)
            try:
                response = original(**request)
            except Exception as error:  # Record types/codes, never bodies; re-raise.
                invocation["exception_type"] = type(error).__name__
                code = getattr(error, "code", None)
                invocation["provider_code"] = code if isinstance(code, int) else None
                raise
            finally:
                elapsed = round(time.perf_counter() - started, 3)
                self.requests[-1]["provider_elapsed_seconds"] = elapsed
                invocation.update(
                    ended_at=datetime.now(UTC).isoformat(),
                    duration_seconds=elapsed,
                    http_statuses=[
                        e["http_status"] for e in self.http_events[http_start:]
                    ],
                )
            self.record(response)
            return response

        model.client.models.generate_content = generate
        return model


def run_stage(stage, settings, *, api_version=None, thinking=None):
    capture = Capture()
    started = time.perf_counter()
    row = {
        "stage": stage,
        "model": settings.model,
        "synthetic_only": True,
        "configured_timeout_seconds": settings.timeout_seconds,
        "output_token_budget": settings.max_output_tokens,
        "requested_api_version": api_version,
        "thinking_level_requested": thinking or "provider_default",
        "automatic_retries": 0,
    }
    try:
        with tracing_context(enabled=False):
            if stage == "sdk_plain":
                options = types.HttpOptions(
                    timeout=int(settings.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=0),
                    client_args={"event_hooks": {"response": [capture.response_hook]}},
                )
                if api_version is not None:
                    options.api_version = api_version
                with genai.Client(
                    api_key=settings.api_key.get_secret_value(),
                    vertexai=False,
                    http_options=options,
                ) as client:
                    capture.requests.append(
                        {"api_version": client._api_client._http_options.api_version}
                    )
                    response = client.models.generate_content(
                        model=settings.model,
                        contents="Reply with exactly ATLAS_OK and nothing else.",
                        config=types.GenerateContentConfig(
                            max_output_tokens=settings.max_output_tokens,
                            thinking_config=types.ThinkingConfig(
                                thinking_level=thinking
                            )
                            if thinking
                            else None,
                        ),
                    )
                    capture.record(response)
                    row["result"] = {
                        "expected_text_matched": response.text is not None
                        and response.text.strip() == "ATLAS_OK"
                    }
            else:

                def factory(**kwargs):
                    return capture.model_factory(
                        api_version=api_version, thinking=thinking, **kwargs
                    )

                if stage == "lc_simple":
                    model = factory(
                        model=settings.model,
                        api_key=settings.api_key,
                        timeout=settings.timeout_seconds,
                        max_retries=0,
                        temperature=0,
                        max_output_tokens=settings.max_output_tokens,
                        cache=False,
                        vertexai=False,
                    )
                    result = model.with_structured_output(
                        DiagnosticReply, method="json_schema"
                    ).invoke(
                        "Return status ATLAS_OK using the supplied schema.",
                        config={"callbacks": []},
                    )
                    row["result"] = result.model_dump()
                else:
                    request = synthetic_request()
                    if stage == "solution":
                        request.event = ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
                    context = prepare_context(request)
                    assessor = ContextAssessor(settings, factory)
                    if stage == "assessment":
                        result = assessor.assess(
                            context, time_remaining=settings.timeout_seconds
                        )
                        row["typed_output_valid"] = True
                        if result.decision == "ASK_QUESTION":
                            validate_question(result.question, context)
                        row["result"] = {
                            "decision": result.decision,
                            "question": result.question.model_dump()
                            if result.question
                            else None,
                        }
                    else:
                        result, _ = SolutionGenerator(assessor).generate(
                            context, time_remaining=settings.timeout_seconds
                        )
                        row["result"] = {
                            "protobuf_type": result.DESCRIPTOR.full_name,
                            "section_lengths": {
                                field: len(getattr(result, field))
                                for field in [
                                    "findings",
                                    "customer_evidence",
                                    "hypotheses",
                                    "recommendations",
                                    "success_indicators",
                                ]
                            },
                        }
            row["status"] = "OK"
            row["grpc_status_equivalent"] = "OK"
    except Exception as error:  # noqa: BLE001 -- diagnostic boundary must sanitize SDK errors.
        row["status"] = "FAILED"
        row.update(classify_error(error, capture.http_events))
        if (
            isinstance(error, ConsultationFailure)
            and row["category"] == "APPLICATION_VALIDATION"
        ):
            row["application_error"] = str(
                error
            )  # Application messages contain only safe paths/labels.
    row.update(
        duration_seconds=round(time.perf_counter() - started, 3),
        http_events=capture.http_events,
        effective_requests=capture.requests,
        provider_responses=capture.responses,
        sdk_invocations=capture.invocations,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument(
        "--stage",
        choices=["sdk_plain", "lc_simple", "assessment", "solution"],
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--api-version", choices=["v1", "v1beta"])
    parser.add_argument("--thinking", choices=["low", "medium", "high"])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--output-tokens", type=int)
    args = parser.parse_args()
    ledger = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {"diagnostic_version": 1, "invocations": []}
    )
    if any(row["id"] == args.id for row in ledger["invocations"]):
        parser.error("Invocation ID already recorded; no request sent")
    if len(ledger["invocations"]) >= 8:
        parser.error("Bounded diagnostic request limit reached; no request sent")
    load_environment(args.env_file)
    data = GeminiSettings.from_env().model_dump()
    if args.model:
        data["model"] = args.model
    if args.timeout is not None:
        data["timeout_seconds"] = args.timeout
    if args.output_tokens is not None:
        data["max_output_tokens"] = args.output_tokens
    settings = GeminiSettings.model_validate(data)
    if not settings.model or not settings.api_key.get_secret_value():
        parser.error("GEMINI_MODEL and GEMINI_API_KEY required; no request sent")
    ledger["packages"] = {
        name: importlib.metadata.version(name)
        for name in [
            "google-genai",
            "langchain-google-genai",
            "langchain-core",
            "pydantic",
            "httpx",
        ]
    }
    row = run_stage(
        args.stage, settings, api_version=args.api_version, thinking=args.thinking
    )
    row["id"] = args.id
    ledger["invocations"].append(row)
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Generated content is synthetic but keep full responses in the artifact only.
    summary = {key: value for key, value in row.items() if key != "provider_responses"}
    summary["usage"] = [r["usage"] for r in row["provider_responses"]]
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
