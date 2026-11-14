"""Explicit synthetic evaluation only. Production tracing and code are untouched.

Local write-ahead reservations prevent duplicate Gemini calls after interruption.
Remote operations are separate, recoverable SDK operations on captured results.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from google.protobuf.json_format import MessageToDict, ParseDict
from langsmith import Client, tracing_context
from langsmith.utils import (
    LangSmithAuthError,
    LangSmithConnectionError,
    LangSmithNotFoundError,
    LangSmithRequestTimeout,
)
from pydantic import BaseModel, ValidationError
from urllib3.util import Retry

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import (
    ACCEPTED_LABELS,
    GROUNDING_RULE_VERSION,
    evaluate_claim,
    solution_claims,
)
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.prompts import SOLUTION_PROMPT, SOLUTION_PROMPT_VERSION
from atlas_ai.consultation.solution import (
    GeneratedSolutionTurn,
    solution_provider_schema,
    solution_to_proto,
    validate_solution,
)
from scripts.gemini_live_diagnostics import Capture, classify_error

ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "atlas-consultation-m7-1-synthetic-v1"
SOURCE_HASHES = {
    "m7_1_consultation_quality.json": "e72e98dd39de03d6a7a87c0bf066673d152febb00305d176d2232afa0119f098",
    "m7_1_quality_review_bundle.json": "39df1574d7547de8a8b541066e345bb47f3fab44a23763298c6a2023a2982bc3",
}
SCENARIO_IDS = (
    "bakery_filling",
    "bakery_quality",
    "cafe_queue",
    "cafe_packaging",
    "msme_pricing",
    "msme_orders",
    "owner_constraints",
)
ALIASES = {
    "cafe_queues": "cafe_queue",
    "cafe_takeaway": "cafe_packaging",
    "pricing": "msme_pricing",
    "order_mistakes": "msme_orders",
}
ENV_NAMES = {
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TIMEOUT_SECONDS",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "CONSULTATION_MAX_CONTEXT_BYTES",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
}
CAP = 7


class EvaluationError(Exception):
    """Only fixed categories and configuration NAMES, never raw exception strings."""

    def __init__(self, category, names=()):
        super().__init__(category)
        self.category, self.names = category, list(names)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def load_environment(path):
    if path:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in ENV_NAMES:
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@contextmanager
def quiet_sdk_logs():
    loggers = [
        logging.getLogger(name)
        for name in (
            "langsmith",
            "google",
            "langchain_google_genai",
            "httpx",
            "httpcore",
            "urllib3",
        )
    ]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def scrub(value):
    """Defense in depth for model echoes; never change the payload being validated."""
    secrets = [os.getenv(name, "") for name in ("GEMINI_API_KEY", "LANGSMITH_API_KEY")]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(scrub(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@contextmanager
def locked(path):
    lock = Path(str(path) + ".lock")
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise EvaluationError("LEDGER_LOCKED") from None
    try:
        yield
    finally:
        lock.unlink()


def load_manifest():
    documents = {}
    for name, expected in SOURCE_HASHES.items():
        data = (ROOT / "evaluation" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise EvaluationError("FIXTURE_INTEGRITY_MISMATCH")
        documents[name] = json.loads(data)
    report, bundle = documents.values()
    if (
        SOLUTION_PROMPT_VERSION != "consultation-solution-v8"
        or bundle["prompt_version"] != SOLUTION_PROMPT_VERSION
    ):
        raise EvaluationError("PROMPT_VERSION_MISMATCH")
    if bundle["provider_schema"] != solution_provider_schema():
        raise EvaluationError("PROVIDER_SCHEMA_MISMATCH")
    if (
        tuple(c["id"] for c in report["cases"]) != SCENARIO_IDS
        or tuple(c["id"] for c in bundle["cases"]) != SCENARIO_IDS
    ):
        raise EvaluationError("SCENARIO_IDENTITY_MISMATCH")
    examples = []
    for row, review in zip(report["cases"], bundle["cases"], strict=True):
        request = ParseDict(
            row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
        )
        if request.event != ai_pb2.CONSULTATION_EVENT_GENERATE_NOW:
            raise EvaluationError("ONE_INVOCATION_EVENT_REQUIRED")
        if request != ParseDict(
            review["request"], ai_pb2.ProcessConsultationTurnRequest()
        ):
            raise EvaluationError("REQUEST_IDENTITY_MISMATCH")
        context = prepare_context(request)
        data = context.model_dump(mode="json", exclude={"sources"})
        data["event_name"] = ai_pb2.ConsultationEvent.Name(request.event)
        messages = [
            {"role": m.type, "content": m.content}
            for m in SOLUTION_PROMPT.format_prompt(
                context_json=json.dumps(data, ensure_ascii=False)
            ).to_messages()
        ]
        if messages != review["messages"]:
            raise EvaluationError("PROMPT_CONTEXT_MISMATCH")
        # No predefined Solution is included in dataset inputs or reference outputs.
        inputs = {"scenario_id": row["id"], "request": row["synthetic_request"]}
        references = {
            "factual_boundaries": row["factual_boundaries"],
            "expected_invalid_factual_claims": [
                row["factual_boundaries"][-1].removeprefix("Never use as a finding: ")
            ],
            "reviewer_guidance": "Evaluate actual source support and multiple valid recommendations, not textual equality with fixture outputs.",
            "rubric": report["rubric"],
        }
        examples.append(
            {
                "scenario_id": row["id"],
                "inputs": inputs,
                "reference_outputs": references,
                "fixture_hash": digest(row["synthetic_request"]),
                "prompt_hash": digest(messages),
                "requested_example_id": str(
                    uuid5(NAMESPACE_URL, DATASET_NAME + "/" + row["id"])
                ),
            }
        )
    return {
        "dataset_name": DATASET_NAME,
        "source_hashes": SOURCE_HASHES,
        "aliases": ALIASES,
        "prompt_version": SOLUTION_PROMPT_VERSION,
        "grounding_rule_version": GROUNDING_RULE_VERSION,
        "rubric": report["rubric"],
        "examples": examples,
    }


def prepare(path):
    if Path(path).exists():
        raise EvaluationError("LEDGER_ALREADY_EXISTS")
    manifest = load_manifest()
    experiment = str(uuid4())
    ledger = {
        "workflow": "atlas-m7.2",
        "created_at": now(),
        "experiment_id": experiment,
        "experiment_name": "atlas-m7-2-v8-" + experiment[:12],
        "manifest_hash": digest(manifest),
        "manifest": manifest,
        "sdk_versions": {
            p: importlib.metadata.version(p)
            for p in ("langsmith", "langchain-google-genai", "google-genai", "pydantic")
        },
        "request_cap": CAP,
        "scope": "Solution GENERATE_NOW only; Assessment quality not evaluated",
        "configuration_presence": {
            name: bool(os.getenv(name, "").strip()) for name in sorted(ENV_NAMES)
        },
        "dataset": None,
        "project": None,
        "settings": None,
        "stop_reason": None,
        "workflow_errors": [],
        "cases": [
            {
                "scenario_id": name,
                "attempt_status": "NOT_ATTEMPTED",
                "reservation": False,
                "provider_attempts": 0,
                "run_id": str(uuid5(UUID(experiment), name)),
                "upload_status": "NOT_APPLICABLE",
                "review": None,
            }
            for name in SCENARIO_IDS
        ],
    }
    write_json(path, ledger)
    return ledger


def read_ledger(path):
    ledger = json.loads(Path(path).read_text(encoding="utf-8"))
    trusted_hash = digest(load_manifest())
    if (
        ledger.get("workflow") != "atlas-m7.2"
        or ledger.get("request_cap") != CAP
        or ledger.get("manifest_hash") != trusted_hash
        or digest(ledger.get("manifest")) != trusted_hash
    ):
        raise EvaluationError("LEDGER_IDENTITY_MISMATCH")
    if [c["scenario_id"] for c in ledger["cases"]] != list(SCENARIO_IDS):
        raise EvaluationError("LEDGER_CASE_MISMATCH")
    for row in ledger["cases"]:
        if (
            type(row["provider_attempts"]) is not int
            or row["provider_attempts"] not in (0, 1)
            or (row["provider_attempts"] and not row["reservation"])
            or row["run_id"]
            != str(uuid5(UUID(ledger["experiment_id"]), row["scenario_id"]))
        ):
            raise EvaluationError("LEDGER_ATTEMPT_MISMATCH")
    return ledger


def require_configuration(gemini=False):
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        raise EvaluationError("MISSING_LANGSMITH_CONFIGURATION", ["LANGSMITH_API_KEY"])
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    url = urlsplit(endpoint)
    if (
        url.scheme != "https"
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise EvaluationError("INVALID_LANGSMITH_ENDPOINT", ["LANGSMITH_ENDPOINT"])
    workspace = os.getenv("LANGSMITH_WORKSPACE_ID", "").strip()
    if workspace:
        try:
            UUID(workspace)
        except ValueError:
            raise EvaluationError(
                "INVALID_LANGSMITH_WORKSPACE", ["LANGSMITH_WORKSPACE_ID"]
            ) from None
    if gemini:
        missing = [
            n
            for n in ("GEMINI_API_KEY", "GEMINI_MODEL")
            if not os.getenv(n, "").strip()
        ]
        if missing:
            raise EvaluationError("MISSING_GEMINI_CONFIGURATION", missing)
        try:
            return GeminiSettings.from_env()
        except ConsultationFailure:
            raise EvaluationError(
                "INVALID_GEMINI_CONFIGURATION",
                [
                    "GEMINI_TIMEOUT_SECONDS",
                    "GEMINI_MAX_OUTPUT_TOKENS",
                    "CONSULTATION_MAX_CONTEXT_BYTES",
                ],
            ) from None
    return None


def make_client():
    require_configuration()
    return Client(
        api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        api_key=os.environ["LANGSMITH_API_KEY"],
        workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
        auto_batch_tracing=False,
        tracing_sampling_rate=1,
        omit_traced_runtime_info=True,
        retry_config=Retry(total=0, connect=0, read=0, redirect=0, status=0),
        timeout_ms=10000,
        hide_inputs=False,
        hide_outputs=False,
        prefer_profile_auth=False,
    )


def remote_error(exc):
    if isinstance(exc, LangSmithAuthError):
        return "LANGSMITH_AUTHENTICATION_FAILURE"
    if isinstance(exc, (LangSmithConnectionError, LangSmithRequestTimeout)):
        return "LANGSMITH_CONNECTIVITY_FAILURE"
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return (
        "LANGSMITH_AUTHENTICATION_FAILURE"
        if code in (401, 403)
        else "LANGSMITH_SDK_OR_API_FAILURE"
    )


def safe_url(value):
    if not isinstance(value, str):
        return None
    url = urlsplit(value)
    secret_query = any(
        any(
            word in key.casefold()
            for word in ("key", "token", "authorization", "password", "secret")
        )
        for key, _ in parse_qsl(url.query)
    )
    known_secret = any(
        os.getenv(name) and os.environ[name] in value
        for name in ("GEMINI_API_KEY", "LANGSMITH_API_KEY")
    )
    return (
        value
        if url.scheme == "https"
        and url.netloc
        and not (
            url.username or url.password or url.fragment or secret_query or known_secret
        )
        else None
    )


def dataset_metadata(ledger):
    return {
        "synthetic_only": True,
        "manifest_hash": ledger["manifest_hash"],
        "prompt_version": SOLUTION_PROMPT_VERSION,
    }


def ensure_dataset(ledger, client, *, create=False, progress=lambda: None):
    expected = dataset_metadata(ledger)
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except LangSmithNotFoundError:
        if not create:
            raise EvaluationError("DATASET_NOT_REGISTERED") from None
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Approved ATLAS M7.1 synthetic contexts; no golden Solution text.",
            metadata=expected,
        )
    if any((dataset.metadata or {}).get(k) != v for k, v in expected.items()):
        raise EvaluationError("REMOTE_DATASET_IDENTITY_MISMATCH")
    if ledger["dataset"] and ledger["dataset"]["id"] != str(dataset.id):
        raise EvaluationError("REMOTE_DATASET_IDENTITY_MISMATCH")
    ledger["dataset"] = {
        "id": str(dataset.id),
        "url": safe_url(dataset.url),
        "example_ids": {},
        "status": "VERIFYING",
    }
    progress()
    rows = list(
        client.list_examples(dataset_id=dataset.id, limit=8, inline_s3_urls=False)
    )
    desired = {e["requested_example_id"]: e for e in ledger["manifest"]["examples"]}
    if any(str(row.id) not in desired for row in rows):
        raise EvaluationError("REMOTE_DATASET_CONTENT_MISMATCH")
    existing = {str(row.id): row for row in rows}
    # Verify every existing row before creating missing rows. Never overwrite.
    for row in rows:
        reference = desired[str(row.id)]
        if (
            row.inputs != reference["inputs"]
            or row.outputs != reference["reference_outputs"]
            or (row.metadata or {}).get("fixture_hash") != reference["fixture_hash"]
        ):
            raise EvaluationError("REMOTE_EXAMPLE_CONTENT_MISMATCH")
    example_ids = {}
    for eid, example in desired.items():
        if eid not in existing:
            if not create:
                raise EvaluationError("REMOTE_DATASET_INCOMPLETE")
            row = client.create_example(
                inputs=example["inputs"],
                reference_outputs=example["reference_outputs"],
                dataset_id=dataset.id,
                example_id=eid,
                metadata={
                    "scenario_id": example["scenario_id"],
                    "fixture_hash": example["fixture_hash"],
                    "prompt_version": SOLUTION_PROMPT_VERSION,
                },
            )
            if str(row.id) != eid:
                raise EvaluationError("REMOTE_EXAMPLE_ID_MISMATCH")
        example_ids[example["scenario_id"]] = eid
        ledger["dataset"]["example_ids"] = dict(example_ids)
        progress()
    ledger["dataset"]["status"] = "VERIFIED"
    progress()


def ensure_project(ledger, client):
    metadata = {
        **dataset_metadata(ledger),
        "experiment_id": ledger["experiment_id"],
        "configured_model": ledger["settings"]["model"],
        "scope": ledger["scope"],
    }
    try:
        project = client.read_project(project_name=ledger["experiment_name"])
    except LangSmithNotFoundError:
        project = client.create_project(
            ledger["experiment_name"],
            reference_dataset_id=ledger["dataset"]["id"],
            metadata=metadata,
            upsert=False,
            num_examples=CAP,
            num_repetitions=1,
            evaluator_keys=[
                "atlas." + k
                for k in (
                    "provider_response",
                    "pydantic",
                    "structural",
                    "grounding",
                    "protobuf",
                    "fully_accepted",
                )
            ],
        )
    if str(project.reference_dataset_id) != ledger["dataset"]["id"] or any(
        (project.extra or {}).get("metadata", {}).get(k) != v
        for k, v in metadata.items()
    ):
        raise EvaluationError("REMOTE_EXPERIMENT_IDENTITY_MISMATCH")
    if ledger["project"] and ledger["project"]["id"] != str(project.id):
        raise EvaluationError("REMOTE_EXPERIMENT_IDENTITY_MISMATCH")
    ledger["project"] = {"id": str(project.id), "url": safe_url(project.url)}


class EvaluationCapture(Capture):
    """Reuse SDK telemetry but never capture provider error bodies or headers."""

    def __init__(self, on_attempt):
        super().__init__()
        self.on_attempt, self.payload = on_attempt, None
        self.prompt_hash = None
        self.expected_prompt_hash = None

    def response_hook(self, response):
        self.http_events.append({"http_status": response.status_code})

    def model_factory(self, **kwargs):
        model = super().model_factory(**kwargs)
        original = model.client.models.generate_content

        def generate(**request):
            self.on_attempt()
            return original(**request)

        model.client.models.generate_content = generate

        def structured(schema, **options):
            runnable = model.with_structured_output(schema, **options)

            def invoke(*args, **config):
                self.prompt_hash = digest(
                    [
                        {"role": m.type, "content": m.content}
                        for m in args[0].to_messages()
                    ]
                )
                if (
                    self.expected_prompt_hash
                    and self.prompt_hash != self.expected_prompt_hash
                ):
                    raise EvaluationError("ACTUAL_PROMPT_MISMATCH")
                result = runnable.invoke(*args, **config)
                self.payload = (
                    result.model_dump() if isinstance(result, BaseModel) else result
                )
                return result

            return SimpleNamespace(invoke=invoke)

        return SimpleNamespace(with_structured_output=structured)


def inspect_payload(request, payload):
    """Offline diagnostic revalidation; never replaces actual runtime status."""
    result = {k: None for k in ("pydantic", "structural", "grounding", "protobuf")}
    result.update(
        diagnostics=[],
        valid_source_reference_counts=None,
        first_blocking_rejection=None,
    )
    if payload is None:
        return result
    try:
        typed = GeneratedSolutionTurn.model_validate(payload).solution_output()
    except ValidationError:
        result["pydantic"] = False
        result["first_blocking_rejection"] = "Invalid structured solution output"
        return result
    result["pydantic"] = True
    context = prepare_context(request)
    result["five_sections_present"] = all(
        getattr(typed, n)
        for n in (
            "findings",
            "customer_evidence",
            "hypotheses",
            "recommendations",
            "success_indicators",
        )
    )
    result["valid_source_reference_counts"] = {
        key: sum(
            sid in registry
            for item in (*typed.findings, *typed.hypotheses, *typed.recommendations)
            for sid in getattr(item.references, key)
        )
        for key, registry in (
            ("opinion_ids", context.sources.opinions),
            ("owner_message_ids", context.sources.owner_messages),
            ("memory_ids", context.sources.memories),
        )
    }
    try:
        validate_solution(typed, context)
    except ConsultationFailure as exc:
        result["structural"] = False
        result["first_blocking_rejection"] = str(exc)
        return result
    result["structural"] = True
    evidence = [
        evaluate_claim(c, context, semantic_checks=False)
        for c in solution_claims(typed)
    ]
    result["diagnostics"] = [
        evaluate_claim(c, context).model_dump(mode="json")
        for c in solution_claims(typed)
    ]
    failed = next((c for c in evidence if c.label not in ACCEPTED_LABELS), None)
    if failed:
        result["grounding"] = False
        result["first_blocking_rejection"] = {
            "claim_id": failed.claim_id,
            "label": failed.label.value,
            "reason": failed.reason,
        }
        return result
    result["grounding"] = True
    try:
        mapped = solution_to_proto(typed, context)
        result["protobuf"] = (
            ai_pb2.Solution.FromString(mapped.SerializeToString()) == mapped
        )
    except Exception:  # noqa: BLE001 - sanitize all unexpected protobuf/library failures.
        result["protobuf"] = False
        result["first_blocking_rejection"] = "Protobuf mapping failure"
    return result


def invoke_case(
    request, settings, row, on_attempt, *, capture_factory=EvaluationCapture
):
    capture, timings = capture_factory(on_attempt), {}
    capture.expected_prompt_hash = row["prompt_hash"]
    started = time.perf_counter()
    response, failure = None, None
    try:
        with tracing_context(enabled=False):
            response = ConsultationPipeline(
                ContextAssessor(settings, capture.model_factory)
            ).run(request, timings=timings)
    except Exception as exc:  # noqa: BLE001 - provider exception bodies must never escape.
        failure = classify_error(exc, capture.http_events)
    row.update(
        ended_at=now(),
        latency_seconds=round(time.perf_counter() - started, 6),
        timings=timings,
    )
    inspected = inspect_payload(request, capture.payload)
    runtime = {}
    for stage in ("pydantic", "structural", "grounding", "protobuf"):
        key = "solution." + ("protobuf_mapping" if stage == "protobuf" else stage)
        runtime[stage] = inspected[stage] if key in timings else None
    if failure and failure["grpc_status_equivalent"] == "DEADLINE_EXCEEDED":
        # An offline recheck cannot turn a production timeout into success.
        for stage in ("grounding", "protobuf"):
            if runtime[stage] is True and response is None:
                runtime[stage] = None
    statuses = [event["http_status"] for event in capture.http_events]
    category = (
        "ACCEPTED"
        if response is not None
        else (failure or {}).get("category", "APPLICATION_FAILURE")
    )
    if any(code in (401, 403) for code in statuses) or any(
        c.get("http_code") in (401, 403)
        for c in (failure or {}).get("exception_chain", [])
    ):
        category = "GEMINI_AUTHENTICATION_FAILURE"
    elif 400 in statuses or 404 in statuses:
        category = "GEMINI_REQUEST_REJECTED_AUTH_MODEL_OR_SCHEMA"
    elif failure and failure["grpc_status_equivalent"] == "DEADLINE_EXCEEDED":
        pass  # Preserve the actual timeout, irrespective of offline payload checks.
    elif capture.responses and capture.payload is None:
        category = "STRUCTURED_OUTPUT_PARSE_FAILURE"
    elif inspected["pydantic"] is False:
        category = "PYDANTIC_VALIDATION_FAILURE"
    elif inspected["structural"] is False or inspected["grounding"] is False:
        category = "FACTUAL_EVIDENCE_REJECTION"
    elif inspected["protobuf"] is False:
        category = "PROTOBUF_MAPPING_FAILURE"
    actual_response = bool(capture.responses)
    row.update(
        provider_outcome=category,
        provider_http_statuses=statuses,
        model_response_received=actual_response,
        structured_output_parse_status="ACCEPTED"
        if capture.payload is not None
        else "FAILED"
        if actual_response
        else "NO_RESPONSE",
        attempt_status="FULLY_ACCEPTED"
        if response is not None
        else "STRUCTURED_OUTPUT_EVIDENCE_FAILED"
        if capture.payload is not None
        else "MODEL_RESPONSE_PARSE_FAILED"
        if actual_response
        else "PROVIDER_ATTEMPTED_NO_RESPONSE"
        if row["provider_attempts"]
        else "PRE_PROVIDER_FAILURE",
        captured_payload=capture.payload,
        provider_responses=capture.responses,
        returned_model_identifier=capture.responses[-1].get("model_version")
        if capture.responses
        else None,
        token_usage=capture.responses[-1].get("usage") if capture.responses else None,
        finish_reasons=capture.responses[-1].get("finish_reasons")
        if capture.responses
        else None,
        provider_requests=capture.requests,
        failure=failure,
        offline_revalidation=inspected,
        provider_invocations=capture.invocations,
        actual_prompt_hash=getattr(capture, "prompt_hash", None),
        validation=runtime,
        fully_accepted=response is not None,
        response=MessageToDict(response, preserving_proto_field_name=True)
        if response
        else None,
    )
    return row


def feedback(client, row, key, score, *, comment, source, project_id):
    fid = str(uuid5(UUID(row["run_id"]), key))
    comment, source = scrub(comment), scrub(source)
    existing = next(
        (
            f
            for f in client.list_feedback(
                run_ids=[row["run_id"]], feedback_key=[key], limit=100
            )
            if str(f.id) == fid
        ),
        None,
    )
    if existing:
        if existing.score != score or existing.comment != comment:
            raise EvaluationError("REMOTE_FEEDBACK_CONTENT_MISMATCH")
        return
    client.create_feedback(
        run_id=row["run_id"],
        key=key,
        score=score,
        comment=comment,
        feedback_id=fid,
        source_info=source,
        feedback_source_type="api",
        stop_after_attempt=1,
        session_id=project_id,
        feedback_config={
            "type": "continuous",
            "min": 0,
            "max": 2 if key.startswith("human.") else 1,
        },
    )


def upload_case(ledger, row, client):
    example = next(
        e
        for e in ledger["manifest"]["examples"]
        if e["scenario_id"] == row["scenario_id"]
    )
    inputs = example["inputs"]
    outputs = scrub(
        {
            k: row.get(k)
            for k in (
                "provider_outcome",
                "captured_payload",
                "provider_responses",
                "validation",
                "offline_revalidation",
                "fully_accepted",
                "response",
            )
        }
    )
    metadata = {
        k: row.get(k)
        for k in (
            "scenario_id",
            "configured_model",
            "prompt_version",
            "grounding_rule_version",
            "prompt_hash",
            "fixture_hash",
            "provider_attempts",
        )
    }
    metadata.update(experiment_id=ledger["experiment_id"], synthetic_only=True)
    try:
        remote = client.read_run(
            row["run_id"],
            project_id=ledger["project"]["id"],
            start_time=datetime.fromisoformat(row["started_at"]),
        )
    except LangSmithNotFoundError:
        client.create_run(
            name="ATLAS Solution " + row["scenario_id"],
            inputs=inputs,
            run_type="chain",
            project_name=ledger["experiment_name"],
            id=row["run_id"],
            reference_example_id=ledger["dataset"]["example_ids"][row["scenario_id"]],
            start_time=datetime.fromisoformat(row["started_at"]),
            end_time=datetime.fromisoformat(row["ended_at"]),
            outputs=outputs,
            error=None if row["fully_accepted"] else row["provider_outcome"],
            extra={"metadata": metadata},
        )
        remote = client.read_run(
            row["run_id"],
            project_id=ledger["project"]["id"],
            start_time=datetime.fromisoformat(row["started_at"]),
        )
    mismatches = [
        key
        for key, matches in {
            "inputs": remote.inputs == inputs,
            "outputs": remote.outputs == outputs,
            "reference_example_id": str(remote.reference_example_id)
            == ledger["dataset"]["example_ids"][row["scenario_id"]],
            "project_id": str(remote.session_id) == ledger["project"]["id"],
            "metadata": all(
                (remote.extra or {}).get("metadata", {}).get(k) == v
                for k, v in metadata.items()
            ),
        }.items()
        if not matches
    ]
    if mismatches:
        row["remote_verification_mismatch_fields"] = mismatches
        raise EvaluationError("REMOTE_RUN_CONTENT_MISMATCH")
    row["run_url"] = safe_url(remote.url)
    if not row["run_url"]:
        try:
            row["run_url"] = safe_url(
                client.get_run_url(run=remote, project_id=ledger["project"]["id"])
            )
        except Exception as exc:  # noqa: BLE001 - optional URL resolution must not lose the verified run.
            row["run_url_error"] = remote_error(exc)
    scores = {
        "provider_response": row["model_response_received"],
        **row["validation"],
        "fully_accepted": row["fully_accepted"],
    }
    for key, value in scores.items():
        if value is not None:
            feedback(
                client,
                row,
                "atlas." + key,
                int(value),
                comment="Code-based runtime stage result; not a semantic quality judgment.",
                source={"kind": "deterministic_code"},
                project_id=ledger["project"]["id"],
            )
    row["upload_status"] = "UPLOADED"
    row["upload_error"] = None


def run_live(
    path, *, authorized, phase, client_factory=make_client, invoke=invoke_case
):
    if not authorized:
        raise EvaluationError("EXPLICIT_LIVE_FLAG_REQUIRED")
    if phase not in {"smoke", "remaining"}:
        raise EvaluationError("INVALID_LIVE_PHASE")
    settings = require_configuration(gemini=True)
    with locked(path):
        ledger = read_ledger(path)
        snapshot = settings.model_dump(mode="json", exclude={"api_key"})
        if ledger["settings"] not in (None, snapshot):
            raise EvaluationError("EXPERIMENT_SETTINGS_CHANGED")
        if not ledger["dataset"]:
            raise EvaluationError("DATASET_NOT_REGISTERED")
        if ledger["stop_reason"]:
            raise EvaluationError("EXPERIMENT_STOPPED")
        if phase == "remaining" and not (
            ledger["cases"][0].get("fully_accepted")
            and ledger["cases"][0]["provider_attempts"] == 1
            and ledger["cases"][0]["upload_status"] == "UPLOADED"
        ):
            raise EvaluationError("SMOKE_GATE_NOT_PASSED")
        ledger["settings"] = snapshot
        client = client_factory()
        ensure_dataset(
            ledger, client, create=False, progress=lambda: write_json(path, ledger)
        )
        ensure_project(ledger, client)
        write_json(path, ledger)
        selection = ledger["cases"][:1] if phase == "smoke" else ledger["cases"][1:]
        for row in selection:
            if row["reservation"]:
                if row["upload_status"] != "UPLOADED":
                    raise EvaluationError(
                        "CAPTURE_NEEDS_UPLOAD_OR_MANUAL_INTERRUPTION_REVIEW"
                    )
                continue
            if sum(c["reservation"] for c in ledger["cases"]) >= CAP:
                raise EvaluationError("PROVIDER_CALL_CAP_REACHED")
            if any(
                c["reservation"] and c["upload_status"] != "UPLOADED"
                for c in ledger["cases"]
            ):
                raise EvaluationError("PRIOR_CAPTURE_UPLOAD_PENDING")
            example = next(
                e
                for e in ledger["manifest"]["examples"]
                if e["scenario_id"] == row["scenario_id"]
            )
            row.update(
                reservation=True,
                attempt_status="RESERVED_UNKNOWN_OUTCOME",
                upload_status="PENDING",
                started_at=now(),
                configured_model=settings.model,
                prompt_version=SOLUTION_PROMPT_VERSION,
                grounding_rule_version=GROUNDING_RULE_VERSION,
                prompt_hash=example["prompt_hash"],
                fixture_hash=example["fixture_hash"],
            )
            write_json(path, ledger)

            def on_attempt(current=row):
                if (
                    current["provider_attempts"] >= 1
                    or sum(c["provider_attempts"] for c in ledger["cases"]) >= CAP
                ):
                    raise EvaluationError("PROVIDER_CALL_CAP_REACHED")
                current["provider_attempts"] += 1
                current["attempt_status"] = "PROVIDER_IN_FLIGHT"
                write_json(path, ledger)

            request = ParseDict(
                example["inputs"]["request"], ai_pb2.ProcessConsultationTurnRequest()
            )
            try:
                invoke(request, settings, row, on_attempt)
                if row.get("fully_accepted") and row["provider_attempts"] != 1:
                    raise EvaluationError("INVOCATION_COUNT_MISMATCH")
            except Exception:  # noqa: BLE001 - persist the reservation even for unexpected local failures.
                # Reservation survives unexpected local failures; never regenerate.
                row.update(attempt_status="INTERRUPTED_UNKNOWN_OUTCOME", ended_at=now())
                write_json(path, ledger)
                raise EvaluationError("LOCAL_CAPTURE_INTERRUPTED") from None
            infrastructure = row["provider_outcome"] not in {
                "ACCEPTED",
                "FACTUAL_EVIDENCE_REJECTION",
                "PYDANTIC_VALIDATION_FAILURE",
                "STRUCTURED_OUTPUT_PARSE_FAILURE",
            }
            if infrastructure or (phase == "smoke" and not row["fully_accepted"]):
                ledger["stop_reason"] = row["provider_outcome"]
            write_json(path, ledger)  # capture + stop gate BEFORE any upload
            try:
                upload_case(ledger, row, client)
            except Exception as exc:  # noqa: BLE001 - upload failure cannot lose the local capture.
                row["upload_error"] = (
                    exc.category
                    if isinstance(exc, EvaluationError)
                    else remote_error(exc)
                )
                write_json(path, ledger)
                raise EvaluationError("EXPERIMENT_UPLOAD_FAILED") from None
            # Only accepted smoke opens the remaining phase. Other infrastructure
            # failures also stop the initial experiment, never trigger a retry.
            write_json(path, ledger)
            if ledger["stop_reason"]:
                break
        return ledger


def review_scorecard(row, review, rubric):
    scores, reasons = review.get("scores", {}), review.get("reasons", {})
    keys = {r["id"] for r in rubric}
    if set(scores) != keys or set(reasons) != keys:
        raise EvaluationError("REVIEW_RUBRIC_MISMATCH")
    supplied = {k: v for k, v in scores.items() if v is not None}
    if supplied and (
        not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
    ):
        raise EvaluationError("REVIEWER_REQUIRED")
    for key, value in supplied.items():
        if (
            type(value) is not int
            or value not in (0, 1, 2)
            or not isinstance(reasons[key], str)
            or not reasons[key].strip()
        ):
            raise EvaluationError("INVALID_HUMAN_REVIEW")
        if (
            not row.get("model_response_received")
            and key != "generation_validation_success"
        ):
            raise EvaluationError("NO_MODEL_OUTPUT_TO_REVIEW")
        if (
            key == "generation_validation_success"
            and value > 0
            and not row.get("fully_accepted")
        ):
            raise EvaluationError("FAILED_GENERATION_CANNOT_PASS_REVIEW")
        if not row.get("provider_attempts"):
            raise EvaluationError("UNATTEMPTED_CASE_CANNOT_BE_SCORED")
    complete = len(supplied) == len(keys)
    critical = [r["id"] for r in rubric if r.get("critical")]
    passed = (
        complete
        and row.get("fully_accepted", False)
        and row.get("model_response_received", False)
        and sum(supplied.values()) >= 14
        and all(v > 0 for v in supplied.values())
        and all(scores[k] == 2 for k in critical)
    )
    return {
        "status": "QUALITY_PASS"
        if passed
        else "HUMAN_QUALITY_REJECTION"
        if complete
        else "HUMAN_REVIEW_PENDING",
        "total": sum(supplied.values()) if complete else None,
        "critical_failure_flags": [
            k for k in critical if scores[k] is not None and scores[k] < 2
        ],
    }


def export_review(ledger, output):
    if Path(output).exists():
        existing = json.loads(Path(output).read_text(encoding="utf-8"))
        if (
            existing.get("experiment_id") != ledger["experiment_id"]
            or existing.get("rubric") != ledger["manifest"]["rubric"]
            or [c.get("scenario_id") for c in existing.get("cases", [])]
            != list(SCENARIO_IDS)
        ):
            raise EvaluationError("REVIEW_OUTPUT_IDENTITY_MISMATCH")
        if any(
            c.get("reviewer")
            or any(v is not None for v in c.get("scores", {}).values())
            for c in existing.get("cases", [])
        ):
            raise EvaluationError("ANNOTATED_REVIEW_MUST_NOT_BE_OVERWRITTEN")
    cases = []
    for row, example in zip(
        ledger["cases"], ledger["manifest"]["examples"], strict=True
    ):
        request = ParseDict(
            example["inputs"]["request"], ai_pb2.ProcessConsultationTurnRequest()
        )
        stored = row.get("review") or {}
        scorecard = stored.get("scorecard") or {}
        cases.append(
            {
                "scenario_id": row["scenario_id"],
                "run_id": row["run_id"],
                "run_url": row.get("run_url"),
                "inputs": example["inputs"],
                "reviewer_guidance": example["reference_outputs"],
                "attempt_status": row["attempt_status"],
                "captured_payload": row.get("captured_payload"),
                "provider_responses": row.get("provider_responses"),
                "runtime_validation": row.get("validation"),
                "offline_revalidation": inspect_payload(
                    request, row.get("captured_payload")
                ),
                "reviewer": scorecard.get("reviewer"),
                "scores": {
                    r["id"]: scorecard.get("scores", {}).get(r["id"])
                    for r in ledger["manifest"]["rubric"]
                },
                "reasons": {
                    r["id"]: scorecard.get("reasons", {}).get(r["id"])
                    for r in ledger["manifest"]["rubric"]
                },
                "review_status": stored.get("decision", {}).get(
                    "status", "HUMAN_REVIEW_PENDING"
                ),
            }
        )
    write_json(
        output,
        {
            "experiment_id": ledger["experiment_id"],
            "rubric": ledger["manifest"]["rubric"],
            "quality_gate": "Actual output + validation/protobuf success; >=14/16; no zeros; all critical dimensions=2.",
            "cases": cases,
        },
    )


def summary(ledger):
    rows = ledger["cases"]
    attempted = sum(bool(r["provider_attempts"]) for r in rows)
    accepted = sum(bool(r.get("fully_accepted")) for r in rows)
    uploaded = sum(r["upload_status"] == "UPLOADED" for r in rows)
    latencies = [
        r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None
    ]
    if not attempted:
        status = (
            "LIVE PARTIAL / BLOCKED"
            if ledger.get("workflow_errors") or ledger.get("stop_reason")
            else "OFFLINE READY — LIVE NOT RUN"
        )
    elif ledger.get("stop_reason") or attempted < CAP or uploaded != attempted:
        status = "LIVE PARTIAL / BLOCKED"
    elif all(
        r.get("review", {}).get("decision", {}).get("status")
        in {"QUALITY_PASS", "HUMAN_QUALITY_REJECTION"}
        and r.get("review", {}).get("feedback_status") == "UPLOADED"
        for r in rows
        if r.get("review")
    ) and all(r.get("review") for r in rows):
        status = "LIVE QUALITY REVIEWED"
    else:
        status = "LIVE CAPTURED — REVIEW PENDING"
    return {
        "status": status,
        "provider_attempts": attempted,
        "reserved_slots": sum(r["reservation"] for r in rows),
        "model_responses": sum(bool(r.get("model_response_received")) for r in rows),
        "fully_accepted": accepted,
        "uploaded": uploaded,
        "acceptance_per_attempt": accepted / attempted if attempted else None,
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_min_seconds": min(latencies) if latencies else None,
        "latency_max_seconds": max(latencies) if latencies else None,
        "scope": "Descriptive only, at most seven synthetic cases; no significance or model comparison claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "dataset", "live", "upload", "review", "submit-review"],
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--remote-setup", action="store_true")
    parser.add_argument("--remote-upload", action="store_true")
    parser.add_argument("--remote-feedback", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--phase", choices=["smoke", "remaining"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args()
    load_environment(args.env_file)
    # This is an isolated evaluation process. SDK warning/traceback bodies are
    # not telemetry; only the sanitized error categories below are printed.
    # No trace decorators and no global tracing toggle. Explicit API writes only.
    try:
        with tracing_context(enabled=False), quiet_sdk_logs():
            if args.command == "prepare":
                with locked(args.ledger):
                    ledger = prepare(args.ledger)
            elif args.command == "review":
                ledger = read_ledger(args.ledger)
                if not args.output or args.output.resolve() == args.ledger.resolve():
                    raise EvaluationError("NEW_REVIEW_OUTPUT_PATH_REQUIRED")
                export_review(ledger, args.output)
            elif args.command == "live":
                ledger = run_live(args.ledger, authorized=args.live, phase=args.phase)
            else:
                flag = {
                    "dataset": args.remote_setup,
                    "upload": args.remote_upload,
                    "submit-review": args.remote_feedback,
                }[args.command]
                if not flag:
                    raise EvaluationError("EXPLICIT_REMOTE_FLAG_REQUIRED")
                require_configuration()
                with locked(args.ledger):
                    ledger, client = read_ledger(args.ledger), make_client()
                    ensure_dataset(
                        ledger,
                        client,
                        create=args.command == "dataset",
                        progress=lambda: write_json(args.ledger, ledger),
                    )
                    if args.command in {"upload", "submit-review"}:
                        if not ledger["settings"]:
                            raise EvaluationError("NO_CAPTURED_EXPERIMENT")
                        ensure_project(ledger, client)
                    if args.command == "upload":
                        for row in ledger["cases"]:
                            if (
                                row["reservation"]
                                and row.get("ended_at")
                                and "fully_accepted" in row
                            ):
                                try:
                                    upload_case(ledger, row, client)
                                except Exception as exc:  # Preserve recovery diagnostics, then re-raise.
                                    row["upload_error"] = (
                                        exc.category
                                        if isinstance(exc, EvaluationError)
                                        else remote_error(exc)
                                    )
                                    row["upload_failure_type"] = type(exc).__name__
                                    write_json(args.ledger, ledger)
                                    raise
                                write_json(args.ledger, ledger)
                    elif args.command == "submit-review":
                        if not args.review_file:
                            raise EvaluationError("REVIEW_FILE_REQUIRED")
                        review = json.loads(
                            args.review_file.read_text(encoding="utf-8")
                        )
                        if review["experiment_id"] != ledger["experiment_id"]:
                            raise EvaluationError("REVIEW_EXPERIMENT_MISMATCH")
                        for scorecard in review["cases"]:
                            row = next(
                                r
                                for r in ledger["cases"]
                                if r["scenario_id"] == scorecard["scenario_id"]
                            )
                            if scorecard["run_id"] != row["run_id"]:
                                raise EvaluationError("REVIEW_RUN_MISMATCH")
                            decision = review_scorecard(
                                row, scorecard, ledger["manifest"]["rubric"]
                            )
                            row["review"] = {
                                "scorecard": scorecard,
                                "decision": decision,
                                "feedback_status": "PENDING",
                            }
                            write_json(args.ledger, ledger)
                            for key, score in scorecard["scores"].items():
                                if score is not None:
                                    if row["upload_status"] != "UPLOADED":
                                        raise EvaluationError(
                                            "RUN_UPLOAD_REQUIRED_FOR_REVIEW"
                                        )
                                    feedback(
                                        client,
                                        row,
                                        "human." + key,
                                        score,
                                        comment=scorecard["reasons"][key],
                                        source={
                                            "kind": "human_review",
                                            "reviewer": scorecard["reviewer"],
                                        },
                                        project_id=ledger["project"]["id"],
                                    )
                            row["review"]["feedback_status"] = "UPLOADED"
                            write_json(args.ledger, ledger)
                    write_json(args.ledger, ledger)
        print(
            json.dumps(
                scrub(
                    {
                        **summary(ledger),
                        "dataset": ledger["dataset"],
                        "experiment": ledger["project"],
                    }
                )
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary prints fixed categories, never traceback bodies.
        category = (
            exc.category if isinstance(exc, EvaluationError) else remote_error(exc)
        )
        error_recorded = False
        if args.ledger.exists():
            try:
                with locked(args.ledger):
                    ledger = read_ledger(args.ledger)
                    ledger["workflow_errors"].append(
                        {
                            "at": now(),
                            "category": category,
                            "exception_type": type(exc).__name__,
                            "configuration_names": exc.names
                            if isinstance(exc, EvaluationError)
                            else [],
                        }
                    )
                    write_json(args.ledger, ledger)
                    error_recorded = True
            except (OSError, EvaluationError, ValueError, KeyError):
                error_recorded = False  # Do not interfere with another live writer or an invalid ledger.
        print(
            json.dumps(
                {
                    "error": category,
                    "ledger_error_recorded": error_recorded,
                    "configuration_names": exc.names
                    if isinstance(exc, EvaluationError)
                    else [],
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
