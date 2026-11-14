"""One Gemini assessment invocation plus deterministic question policy."""

import re
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Literal, TypeVar

import grpc
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.memory import MemoryProposals
from atlas_ai.consultation.models import ConsultationContext
from atlas_ai.consultation.prompts import ASSESSMENT_PROMPT
from atlas_ai.llm import (
    AIFailure,
    GeminiSettings,
    compact_provider_schema,
    invoke_gemini_structured,
)

__all__ = [
    "QUESTION_LIMIT",
    "AssessmentQuestion",
    "ConsultationFailure",
    "ContextAssessor",
    "GeminiSettings",
    "SufficiencyAssessment",
    "compact_provider_schema",
    "invoke_structured",
    "must_generate",
    "validate_question",
]

QUESTION_LIMIT = 4  # Conservative cumulative cap until Go supplies cycle scope.
OutputModel = TypeVar("OutputModel", bound=BaseModel)


# Shared with Path A; kept under its historical name for existing imports.
ConsultationFailure = AIFailure


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
    """Consultation adapter over the shared Gemini boundary in ``atlas_ai.llm``."""
    if input_data is not None:
        data = dict(input_data)
    else:
        data = context.model_dump(mode="json", exclude={"sources"})
        data["event_name"] = ai_pb2.ConsultationEvent.Name(context.event)
    return invoke_gemini_structured(
        settings,
        model_factory,
        prompt_template,
        schema,
        stage=stage,
        input_data=data,
        time_remaining=time_remaining,
        is_active=is_active,
        provider_schema=provider_schema,
        timings=timings,
    )
