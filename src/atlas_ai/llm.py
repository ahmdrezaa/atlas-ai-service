"""Shared single-call Gemini boundary for Path A (feedback) and Path B (consultation).

Native JSON schema output, no tools/agents, no automatic retries, no tracing of raw
content. Provider and SDK failures are mapped to sanitized gRPC status codes.
"""

import json
import os
import time
from collections.abc import Callable
from typing import TypeVar

import grpc
from httpx import TimeoutException
from langchain_core.exceptions import ContextOverflowError, OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

OutputModel = TypeVar("OutputModel", bound=BaseModel)

# Gemini API rejects shorter request deadlines with HTTP 400 ("Minimum allowed
# deadline is 10s"), observed 2026-11-14. Never send one; fail fast instead.
PROVIDER_MIN_TIMEOUT_SECONDS = 10.0


class AIFailure(Exception):
    """Sanitized application failure; never include provider response or secrets."""

    def __init__(self, code: grpc.StatusCode, message: str):
        super().__init__(message)
        self.code = code


class GeminiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = ""
    api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    timeout_seconds: float = Field(default=30, gt=0, le=55, allow_inf_nan=False)
    max_context_bytes: int = Field(default=120000, ge=1024, le=1000000)
    max_output_tokens: int = Field(default=2048, ge=256, le=8192)

    @classmethod
    def from_env(
        cls,
        *,
        model_env: str = "GEMINI_MODEL",
        api_key_envs: tuple[str, ...] = ("GEMINI_API_KEY",),
        timeout_env: str = "GEMINI_TIMEOUT_SECONDS",
        timeout_default: str = "45",
        max_context_bytes_env: str = "CONSULTATION_MAX_CONTEXT_BYTES",
        max_context_bytes_default: str = "120000",
        max_output_tokens_env: str = "GEMINI_MAX_OUTPUT_TOKENS",
        max_output_tokens_default: str = "2048",
        stage: str = "assessment",
    ) -> "GeminiSettings":
        """Defaults are the Path B (consultation) variable names and values."""
        api_key = next(
            (
                os.getenv(name, "")
                for name in api_key_envs
                if os.getenv(name, "").strip()
            ),
            "",
        )
        try:
            return cls(
                model=os.getenv(model_env, "").strip(),
                api_key=SecretStr(api_key),
                timeout_seconds=os.getenv(timeout_env, timeout_default),
                max_context_bytes=os.getenv(
                    max_context_bytes_env, max_context_bytes_default
                ),
                max_output_tokens=os.getenv(
                    max_output_tokens_env, max_output_tokens_default
                ),
            )
        except ValidationError:
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Invalid {stage} configuration",
            ) from None


def compact_provider_schema(schema: type[BaseModel]) -> dict:
    """Native provider grammar only; the complete Pydantic model validates locally."""
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


def invoke_gemini_structured(
    settings: GeminiSettings,
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
    config_names: str = "GEMINI_MODEL and GEMINI_API_KEY",
) -> OutputModel:
    """One Gemini call with structured output, bounded by settings and gRPC deadline.

    ``label`` only names the RPC in deadline/cancellation messages; Path B keeps
    its historical wording ("Consultation ...").
    """
    started = time.monotonic()
    timeout = settings.timeout_seconds
    if time_remaining is not None:
        timeout = min(timeout, time_remaining - 0.25)
    if timeout <= 0:
        raise AIFailure(grpc.StatusCode.DEADLINE_EXCEEDED, f"{label} deadline exceeded")
    if timeout < PROVIDER_MIN_TIMEOUT_SECONDS:
        # The provider would reject it with HTTP 400; do not spend a request.
        raise AIFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            f"{stage} budget below the provider minimum deadline",
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
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise AIFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            f"{label} deadline exceeded",
        )
    # Local formatting takes milliseconds; never send less than the provider minimum.
    timeout = max(timeout, PROVIDER_MIN_TIMEOUT_SECONDS)
    try:
        # Explicit native JSON schema: no tools/agents, no retries, no raw traces.
        with tracing_context(enabled=False):
            model = model_factory(
                model=settings.model,
                api_key=settings.api_key,
                temperature=0,
                timeout=timeout,
                max_retries=0,
                max_output_tokens=settings.max_output_tokens,
                vertexai=False,
                cache=False,
            )
            structured = model.with_structured_output(
                provider_schema if provider_schema is not None else schema,
                method="json_schema",
            )
            stage_started = time.monotonic() if timings is not None else 0
            try:
                result = structured.invoke(prompt, config={"callbacks": []})
            finally:
                if timings is not None:
                    timings[f"{stage.lower()}.langchain_invoke_and_parse"] = (
                        time.monotonic() - stage_started
                    )
            if isinstance(result, BaseModel):
                result = result.model_dump()
            stage_started = time.monotonic() if timings is not None else 0
            try:
                result = schema.model_validate(result)
            finally:
                if timings is not None:
                    timings[f"{stage.lower()}.pydantic"] = (
                        time.monotonic() - stage_started
                    )
    except (ValidationError, OutputParserException):
        raise AIFailure(
            grpc.StatusCode.INTERNAL, f"Invalid structured {stage.lower()} output"
        ) from None
    except (TimeoutError, TimeoutException):
        raise AIFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} provider timed out"
        ) from None
    except ContextOverflowError:
        raise AIFailure(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            f"{stage} exceeds model context limit",
        ) from None
    # SDK/transport exceptions must never expose raw request text or secrets.
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None)
        if code in {408, 504}:
            raise AIFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"{stage} provider timed out",
            ) from None
        if code == 400:
            # Our request was rejected (configuration/schema): not a transient outage.
            raise AIFailure(
                grpc.StatusCode.INTERNAL, f"{stage} provider rejected the request"
            ) from None
        raise AIFailure(
            grpc.StatusCode.UNAVAILABLE, f"{stage} provider unavailable"
        ) from None
    if is_active is not None and not is_active():
        raise AIFailure(grpc.StatusCode.CANCELLED, f"{label} cancelled")
    if time.monotonic() >= deadline:
        raise AIFailure(grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} deadline exceeded")
    return result
