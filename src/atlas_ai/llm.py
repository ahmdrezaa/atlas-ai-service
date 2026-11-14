"""Shared single-call OpenAI boundary for Path A (feedback) and Path B (consultation).

Native strict JSON schema output, no tools/agents, no SDK retries, no tracing of raw
content. Provider and SDK failures are mapped to sanitized gRPC status codes.
"""

import json
import os
import time
from collections.abc import Callable
from typing import Literal, TypeVar

import grpc
import openai
from httpx import TimeoutException
from langchain_core.exceptions import ContextOverflowError, OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from atlas_ai import observability

OutputModel = TypeVar("OutputModel", bound=BaseModel)

# Never start a provider request that cannot finish in a useful time; the caller's
# gRPC deadline always bounds the actual call.
MIN_CALL_SECONDS = 1.0
# One bounded retry for transient provider failures (5xx, dropped connection).
# 400/401/403/404/429/timeouts/invalid output are never retried here.
MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_HTTP_CODES = {500, 502, 503}


class AIFailure(Exception):
    """Sanitized application failure; never include provider response or secrets."""

    def __init__(self, code: grpc.StatusCode, message: str):
        super().__init__(message)
        self.code = code


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = ""
    api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    timeout_seconds: float = Field(default=30, gt=0, le=55, allow_inf_nan=False)
    max_context_bytes: int = Field(default=120000, ge=1024, le=1000000)
    max_output_tokens: int = Field(default=2048, ge=256, le=32768)
    # None = the model's default (reasoning models may reject temperature=0).
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    # Official per-1M-token prices (USD) for cost estimates; None = not configured.
    price_input_per_1m: float | None = Field(default=None, ge=0)
    price_output_per_1m: float | None = Field(default=None, ge=0)

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "CONSULTATION",
        timeout_env: str | None = None,
        timeout_default: str = "45",
        max_context_bytes_default: str = "120000",
        max_output_tokens_default: str = "2048",
        stage: str = "assessment",
    ) -> "LLMSettings":
        """Reads ``{prefix}_MODEL`` etc.; the API key and temperature are shared."""

        def env(name: str, default: str = "") -> str:
            return os.getenv(name, default).strip()

        try:
            return cls(
                model=env(f"{prefix}_MODEL"),
                api_key=SecretStr(env("OPENAI_API_KEY")),
                timeout_seconds=env(
                    timeout_env or f"{prefix}_TIMEOUT_SECONDS", timeout_default
                ),
                max_context_bytes=env(
                    f"{prefix}_MAX_CONTEXT_BYTES", max_context_bytes_default
                ),
                max_output_tokens=env(
                    f"{prefix}_MAX_OUTPUT_TOKENS", max_output_tokens_default
                ),
                temperature=env("LLM_TEMPERATURE") or None,
                reasoning_effort=env(f"{prefix}_REASONING_EFFORT") or None,
                price_input_per_1m=env(f"{prefix}_PRICE_INPUT_PER_1M_USD") or None,
                price_output_per_1m=env(f"{prefix}_PRICE_OUTPUT_PER_1M_USD") or None,
            )
        except ValidationError:
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Invalid {stage} configuration",
            ) from None

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if self.price_input_per_1m is None or self.price_output_per_1m is None:
            return None
        return round(
            (input_tokens or 0) / 1e6 * self.price_input_per_1m
            + (output_tokens or 0) / 1e6 * self.price_output_per_1m,
            8,
        )


def compact_provider_schema(schema: type[BaseModel]) -> dict:
    """Provider grammar only; the complete Pydantic model validates locally.

    The compact form keeps which fields are optional; ``strict_provider_schema``
    turns it into the OpenAI strict form at call time.
    """
    data = schema.model_json_schema()
    definitions = data.get("$defs", {})
    allowed = {
        "type",
        "properties",
        "items",
        "required",
        "enum",
        "anyOf",
        "additionalProperties",
    }

    def compact(value):
        if isinstance(value, list):
            return [compact(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return compact(definitions[value["$ref"].split("/")[-1]])
        return {
            key: (
                {name: compact(item) for name, item in part.items()}
                if key == "properties"
                else compact(part)
            )
            for key, part in value.items()
            if key in allowed
        }

    return {"title": schema.__name__, **compact(data)}


def strict_provider_schema(schema: dict) -> dict:
    """OpenAI strict structured outputs: every property is required and objects are
    closed. A field that was optional becomes nullable; ``drop_optional_nulls``
    removes those nulls again so local validation sees an omitted field."""

    def convert(node):
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {
            key: (
                {name: convert(item) for name, item in part.items()}
                if key == "properties"
                else convert(part)
            )
            for key, part in node.items()
        }
        if "properties" in node:
            required = set(node.get("required", []))
            out["properties"] = {
                name: item if name in required else _nullable(item)
                for name, item in out["properties"].items()
            }
            out["required"] = list(node["properties"])
            out["additionalProperties"] = False
        elif node.get("type") == "object":
            out["additionalProperties"] = False
        return out

    return convert(schema)


def _nullable(node: dict) -> dict:
    options = node.get("anyOf")
    if options is not None:
        if any(option.get("type") == "null" for option in options):
            return node
        return {**node, "anyOf": [*options, {"type": "null"}]}
    if node.get("type") == "null":
        return node
    return {"anyOf": [node, {"type": "null"}]}


def drop_optional_nulls(value, schema: dict):
    """Remove ``null`` for properties the compact schema did not require."""
    if isinstance(value, list):
        items = schema.get("items", {})
        return [drop_optional_nulls(item, items) for item in value]
    if not isinstance(value, dict):
        return value
    properties = schema.get("properties")
    if properties is None:
        # Pick the object branch of an anyOf (e.g. a nullable nested object).
        for option in schema.get("anyOf", []):
            if "properties" in option:
                return drop_optional_nulls(value, option)
        return value
    required = set(schema.get("required", []))
    return {
        key: drop_optional_nulls(item, properties.get(key, {}))
        for key, item in value.items()
        if not (item is None and key not in required)
    }


def invoke_llm_structured(
    settings: LLMSettings,
    model_factory: Callable,
    prompt_template: ChatPromptTemplate,
    schema: type[OutputModel],
    *,
    stage: str,
    input_data: dict,
    time_remaining: float | None = None,
    is_active: Callable[[], bool] | None = None,
    provider_schema: dict | None = None,
    timings: dict[str, float] | None = None,
    label: str = "Consultation",
    config_names: str = "CONSULTATION_MODEL and OPENAI_API_KEY",
) -> OutputModel:
    """One OpenAI call with structured output, bounded by settings and gRPC deadline.

    ``label`` only names the RPC in deadline/cancellation messages; Path B keeps
    its historical wording ("Consultation ...").
    """
    started = time.monotonic()
    timeout = settings.timeout_seconds
    if time_remaining is not None:
        timeout = min(timeout, time_remaining - 0.25)
    if timeout <= 0:
        raise AIFailure(grpc.StatusCode.DEADLINE_EXCEEDED, f"{label} deadline exceeded")
    if timeout < MIN_CALL_SECONDS:
        raise AIFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            f"{stage} budget too small for a provider call",
        )
    deadline = started + timeout
    if is_active is not None and not is_active():
        raise AIFailure(grpc.StatusCode.CANCELLED, f"{label} cancelled")

    # format_prompt is local formatting, not a traced Runnable invocation.
    prompt = prompt_template.format_prompt(
        context_json=json.dumps(input_data, ensure_ascii=False)
    )
    size = sum(len(message.content.encode("utf-8")) for message in prompt.to_messages())
    if size > settings.max_context_bytes:
        raise AIFailure(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            f"{stage} context exceeds configured byte limit; no sources were truncated",
        )
    if not settings.model or not settings.api_key.get_secret_value().strip():
        raise AIFailure(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"{config_names} must be configured",
        )
    compact = (
        provider_schema
        if provider_schema is not None
        else compact_provider_schema(schema)
    )
    output_schema = strict_provider_schema(compact)
    info = {
        "stage": stage.lower(),
        "model": settings.model,
        "attempts": 0,
        "retries": 0,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "status": "OK",
    }
    call_started = time.monotonic()
    try:
        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise AIFailure(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"{label} deadline exceeded",
                )
            info["attempts"] = attempt
            try:
                raw = _call_provider(
                    settings,
                    model_factory,
                    prompt,
                    output_schema,
                    timeout,
                    stage,
                    timings,
                )
                break
            except _Retryable as exc:
                # One bounded retry, only inside the same deadline and only if a
                # useful call still fits after the backoff.
                remaining = deadline - time.monotonic() - RETRY_BACKOFF_SECONDS
                if (
                    attempt >= MAX_ATTEMPTS
                    or remaining < MIN_CALL_SECONDS
                    or (is_active is not None and not is_active())
                ):
                    raise exc.failure from None
                info["retries"] += 1
                time.sleep(RETRY_BACKOFF_SECONDS)
        parsed, usage = _unpack(raw)
        if usage:
            info["input_tokens"] = usage.get("input_tokens")
            info["output_tokens"] = usage.get("output_tokens")
            info["cost_usd"] = settings.cost(
                info["input_tokens"], info["output_tokens"]
            )
        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump()
        stage_started = time.monotonic() if timings is not None else 0
        try:
            result = schema.model_validate(drop_optional_nulls(parsed, compact))
        except ValidationError:
            raise AIFailure(
                grpc.StatusCode.INTERNAL, f"Invalid structured {stage.lower()} output"
            ) from None
        finally:
            if timings is not None:
                timings[f"{stage.lower()}.pydantic"] = time.monotonic() - stage_started
        if is_active is not None and not is_active():
            raise AIFailure(grpc.StatusCode.CANCELLED, f"{label} cancelled")
        if time.monotonic() >= deadline:
            raise AIFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} deadline exceeded"
            )
        return result
    except AIFailure as exc:
        info["status"] = exc.code.name
        raise
    finally:
        info["latency_ms"] = round((time.monotonic() - call_started) * 1000, 1)
        observability.record_llm_call(info)


class _Retryable(Exception):
    def __init__(self, failure: AIFailure):
        super().__init__(str(failure))
        self.failure = failure


def _model_kwargs(settings: LLMSettings, timeout: float) -> dict:
    kwargs = {
        "model": settings.model,
        "api_key": settings.api_key,
        "timeout": timeout,
        "max_retries": 0,
        "max_tokens": settings.max_output_tokens,
        "cache": False,
    }
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if settings.reasoning_effort is not None:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return kwargs


def _call_provider(
    settings, model_factory, prompt, output_schema, timeout, stage, timings
):
    """One provider request. Maps SDK errors to sanitized AIFailure codes."""
    try:
        # Explicit strict JSON schema: no tools/agents, no SDK retries, no traces.
        with tracing_context(enabled=False), observability.span(f"llm.{stage.lower()}"):
            model = model_factory(**_model_kwargs(settings, timeout))
            # include_raw exposes usage_metadata (token counts) next to the parsed
            # output; the parsed value is validated locally afterwards.
            structured = model.with_structured_output(
                output_schema, method="json_schema", strict=True, include_raw=True
            )
            stage_started = time.monotonic() if timings is not None else 0
            try:
                return structured.invoke(prompt, config={"callbacks": []})
            finally:
                if timings is not None:
                    timings[f"{stage.lower()}.langchain_invoke_and_parse"] = (
                        time.monotonic() - stage_started
                    )
    except (
        OutputParserException,
        openai.LengthFinishReasonError,
        openai.ContentFilterFinishReasonError,
    ):
        raise AIFailure(
            grpc.StatusCode.INTERNAL, f"Invalid structured {stage.lower()} output"
        ) from None
    except (TimeoutError, TimeoutException, openai.APITimeoutError):
        raise AIFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} provider timed out"
        ) from None
    except ContextOverflowError:
        raise AIFailure(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            f"{stage} exceeds model context limit",
        ) from None
    except openai.APIConnectionError:
        # Connection dropped before a response: transient, retried once.
        raise _Retryable(
            AIFailure(grpc.StatusCode.UNAVAILABLE, f"{stage} provider unavailable")
        ) from None
    # SDK/transport exceptions must never expose raw request text or secrets.
    except Exception as exc:  # noqa: BLE001
        raise _status_failure(_status_code(exc), stage) from None


def _status_code(exc: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def _status_failure(code: int | None, stage: str) -> Exception:
    if code in {408, 504}:
        return AIFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} provider timed out"
        )
    if code in {401, 403, 404}:
        # Wrong key, no access to the model, or unknown model: configuration.
        return AIFailure(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"{stage} provider rejected the configured credentials or model",
        )
    if code in {400, 422}:
        # Our request was rejected (configuration/schema): not a transient outage.
        return AIFailure(
            grpc.StatusCode.INTERNAL, f"{stage} provider rejected the request"
        )
    failure = AIFailure(grpc.StatusCode.UNAVAILABLE, f"{stage} provider unavailable")
    if code in RETRYABLE_HTTP_CODES:
        return _Retryable(failure)
    return failure


def _unpack(result) -> tuple[object, dict | None]:
    """Accept include_raw output ({raw, parsed, parsing_error}) or a plain value."""
    if isinstance(result, dict) and {"raw", "parsed"} <= set(result):
        if result.get("parsing_error") is not None or result.get("parsed") is None:
            return {"__invalid__": True}, _usage(result.get("raw"))
        return result["parsed"], _usage(result.get("raw"))
    return result, None


def _usage(raw) -> dict | None:
    usage = getattr(raw, "usage_metadata", None)
    return dict(usage) if usage else None
