"""One Gemini assessment invocation plus deterministic question policy."""

import json
import os
import re
import time
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Literal, TypeVar

import grpc
from httpx import TimeoutException
from langchain_core.exceptions import ContextOverflowError, OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import tracing_context
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.memory import MemoryProposals
from atlas_ai.consultation.models import ConsultationContext
from atlas_ai.consultation.prompts import ASSESSMENT_PROMPT

QUESTION_LIMIT = 4  # Conservative cumulative cap until Go supplies cycle scope.
OutputModel = TypeVar("OutputModel", bound=BaseModel)


class ConsultationFailure(Exception):
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
    def from_env(cls) -> "GeminiSettings":
        try:
            return cls(
                model=os.getenv("GEMINI_MODEL", "").strip(),
                api_key=SecretStr(os.getenv("GEMINI_API_KEY", "")),
                timeout_seconds=os.getenv("GEMINI_TIMEOUT_SECONDS", "30"),
                max_context_bytes=os.getenv("CONSULTATION_MAX_CONTEXT_BYTES", "120000"),
                max_output_tokens=os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"),
            )
        except ValidationError:
            raise ConsultationFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Invalid assessment configuration",
            ) from None


class AssessmentQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    topic_key: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=500)


class SufficiencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    decision: Literal["ASK_QUESTION", "GENERATE_SOLUTION"]
    reason: str = Field(min_length=1, max_length=2000)
    problem_evidence_sufficient: bool
    missing_operational_information: list[str]
    relevant_business_constraints: list[str]
    additional_question_material: bool
    repeats_existing_information: bool
    question: AssessmentQuestion | None = None
    memory_candidates: MemoryProposals = Field(default_factory=list)

    @model_validator(mode="after")
    def check_decision(self):
        if not self.reason.strip():
            raise ValueError("Missing internal reason")
        if self.decision == "ASK_QUESTION":
            if (
                self.question is None
                or not self.additional_question_material
                or self.repeats_existing_information
            ):
                raise ValueError("An ask requires one material, non-repeated question")
        elif self.question is not None:
            raise ValueError("Generate decision must not include a question")
        return self


def must_generate(context: ConsultationContext) -> bool:
    return (
        context.event == ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
        or context.question_count >= QUESTION_LIMIT
    )


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def validate_question(
    question: AssessmentQuestion, context: ConsultationContext
) -> ai_pb2.NextQuestion:
    topic, text = question.topic_key.strip(), question.text.strip()
    invalid = not topic or len(topic.encode("utf-8")) > 100
    invalid |= not _normalized(topic) or any(
        unicodedata.category(char) == "Cc" for char in topic
    )
    invalid |= (
        not text or len(text) > 500 or text.count("?") != 1 or not text.endswith("?")
    )
    invalid |= any(char in text for char in "\n\r;" + "\x00")
    invalid |= bool(
        re.search(
            r"\b(dan|serta|and)\s+(berapa|bagaimana|apa|kapan|siapa|mengapa|how|what)\b",
            text,
            re.IGNORECASE,
        )
    )
    banned_topics = {*context.asked_topics, *context.skipped_topics}
    banned_topics.update(
        m.topic_key
        for m in context.conversation_history
        if m.role == ai_pb2.MESSAGE_ROLE_ATLAS_AI and m.topic_key
    )
    invalid |= _normalized(topic) in {_normalized(value) for value in banned_topics}
    for message in context.conversation_history:
        if message.role != ai_pb2.MESSAGE_ROLE_ATLAS_AI or not (
            message.topic_key or "?" in message.content
        ):
            continue
        invalid |= (
            SequenceMatcher(
                None, _normalized(text), _normalized(message.content)
            ).ratio()
            >= 0.88
        )
    if invalid or must_generate(context):
        raise ConsultationFailure(
            grpc.StatusCode.INTERNAL,
            "Assessment returned an invalid or repeated question",
        )
    return ai_pb2.NextQuestion(topic_key=topic, text=text)


class ContextAssessor:
    def __init__(
        self, settings: GeminiSettings, model_factory: Callable = ChatGoogleGenerativeAI
    ):
        self.settings = settings
        self.model_factory = model_factory

    def assess(
        self,
        context: ConsultationContext,
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
        timings: dict[str, float] | None = None,
    ) -> SufficiencyAssessment:
        if must_generate(context):
            return SufficiencyAssessment(
                decision="GENERATE_SOLUTION",
                reason="Generation forced by event or cumulative question cap; evidence not assessed.",
                problem_evidence_sufficient=False,
                missing_operational_information=[],
                relevant_business_constraints=[],
                additional_question_material=False,
                repeats_existing_information=False,
            )
        return invoke_structured(
            context,
            self.settings,
            self.model_factory,
            ASSESSMENT_PROMPT,
            SufficiencyAssessment,
            stage="Assessment",
            time_remaining=time_remaining,
            is_active=is_active,
            timings=timings,
            provider_schema=compact_provider_schema(SufficiencyAssessment),
        )


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


def invoke_structured(
    context: ConsultationContext,
    settings: GeminiSettings,
    model_factory: Callable,
    prompt_template: ChatPromptTemplate,
    schema: type[OutputModel],
    *,
    stage: str,
    time_remaining: float | None = None,
    is_active: Callable[[], bool] | None = None,
    input_data: dict | None = None,
    provider_schema: dict | None = None,
    timings: dict[str, float] | None = None,
) -> OutputModel:
    """Shared single-call Gemini boundary for assessment and solution generation."""
    started = time.monotonic()
    timeout = settings.timeout_seconds
    if time_remaining is not None:
        timeout = min(timeout, time_remaining - 0.25)
    if timeout <= 0:
        raise ConsultationFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, "Consultation deadline exceeded"
        )
    deadline = started + timeout
    if is_active is not None and not is_active():
        raise ConsultationFailure(grpc.StatusCode.CANCELLED, "Consultation cancelled")

    data = (
        dict(input_data)
        if input_data is not None
        else context.model_dump(mode="json", exclude={"sources"})
    )
    if input_data is None:
        data["event_name"] = ai_pb2.ConsultationEvent.Name(context.event)
    # format_prompt is local formatting, not a traced Runnable invocation.
    prompt = prompt_template.format_prompt(
        context_json=json.dumps(data, ensure_ascii=False)
    )
    size = sum(len(message.content.encode("utf-8")) for message in prompt.to_messages())
    if size > settings.max_context_bytes:
        raise ConsultationFailure(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            f"{stage} context exceeds configured byte limit; no sources were truncated",
        )
    if not settings.model or not settings.api_key.get_secret_value().strip():
        raise ConsultationFailure(
            grpc.StatusCode.FAILED_PRECONDITION,
            "GEMINI_MODEL and GEMINI_API_KEY must be configured",
        )
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise ConsultationFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            "Consultation deadline exceeded",
        )
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
        raise ConsultationFailure(
            grpc.StatusCode.INTERNAL, f"Invalid structured {stage.lower()} output"
        ) from None
    except (TimeoutError, TimeoutException):
        raise ConsultationFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} provider timed out"
        ) from None
    except ContextOverflowError:
        raise ConsultationFailure(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            f"{stage} exceeds model context limit",
        ) from None
    # SDK/transport exceptions must never expose raw request text or secrets.
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "code", None) in {408, 504}:
            raise ConsultationFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"{stage} provider timed out",
            ) from None
        raise ConsultationFailure(
            grpc.StatusCode.UNAVAILABLE, f"{stage} provider unavailable"
        ) from None
    if is_active is not None and not is_active():
        raise ConsultationFailure(grpc.StatusCode.CANCELLED, "Consultation cancelled")
    if time.monotonic() >= deadline:
        raise ConsultationFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, f"{stage} deadline exceeded"
        )
    return result
