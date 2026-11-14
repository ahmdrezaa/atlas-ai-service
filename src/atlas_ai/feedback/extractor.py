"""Gemini opinion extraction for ProcessFeedback (Path A, P-A1)."""

import logging
import os
from collections.abc import Callable

import grpc
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.feedback.models import (
    ExtractionOutcome,
    ExtractionOutput,
    ProviderExtraction,
)
from atlas_ai.feedback.prompts import EXTRACTION_PROMPT, FEEDBACK_PROMPT_VERSION
from atlas_ai.feedback.validation import validate_feedback_request, validate_opinions
from atlas_ai.llm import (
    AIFailure,
    GeminiSettings,
    compact_provider_schema,
    invoke_gemini_structured,
)

logger = logging.getLogger(__name__)


class FeedbackLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # Provisional operational limits; not tuned values.
    max_opinions: int = Field(default=10, ge=1, le=50)
    max_text_chars: int = Field(default=5000, ge=100, le=50000)

    @classmethod
    def from_env(cls) -> "FeedbackLimits":
        try:
            return cls(
                max_opinions=os.getenv("FEEDBACK_MAX_OPINIONS", "10"),
                max_text_chars=os.getenv("FEEDBACK_MAX_TEXT_CHARS", "5000"),
            )
        except ValidationError:
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Invalid feedback configuration",
            ) from None


def feedback_settings_from_env() -> GeminiSettings:
    """Path A configuration, separate from Path B. The model never falls back."""
    return GeminiSettings.from_env(
        model_env="FEEDBACK_GEMINI_MODEL",
        api_key_envs=("FEEDBACK_GEMINI_API_KEY", "GEMINI_API_KEY"),
        timeout_env="FEEDBACK_EXTRACTION_TIMEOUT_SECONDS",
        timeout_default="20",
        max_context_bytes_env="FEEDBACK_MAX_CONTEXT_BYTES",
        max_context_bytes_default="32000",
        max_output_tokens_env="FEEDBACK_MAX_OUTPUT_TOKENS",
        max_output_tokens_default="1024",
        stage="feedback",
    )


def extraction_input(request: ai_pb2.ProcessFeedbackRequest) -> dict:
    """Only what extraction needs: no customer name, no candidate issues/vectors."""
    return {
        "feedback_text": request.feedback.text,
        "rating": request.feedback.rating,
        "shop": {
            "name": request.shop.name,
            "description": request.shop.description,
            "category": request.shop.category,
            "scale": request.shop.scale,
        },
    }


class FeedbackExtractor:
    def __init__(
        self,
        settings: GeminiSettings,
        model_factory: Callable = ChatGoogleGenerativeAI,
        limits: FeedbackLimits | None = None,
    ):
        self.settings = settings
        self.model_factory = model_factory
        self.limits = limits or FeedbackLimits()

    @classmethod
    def from_env(cls) -> "FeedbackExtractor":
        return cls(feedback_settings_from_env(), limits=FeedbackLimits.from_env())

    def extract(
        self,
        request: ai_pb2.ProcessFeedbackRequest,
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
        timings: dict[str, float] | None = None,
    ) -> ExtractionOutcome:
        validate_feedback_request(request, max_text_chars=self.limits.max_text_chars)
        output = invoke_gemini_structured(
            self.settings,
            self.model_factory,
            EXTRACTION_PROMPT,
            ExtractionOutput,
            stage="Extraction",
            label="Feedback",
            config_names="FEEDBACK_GEMINI_MODEL and GEMINI_API_KEY",
            input_data=extraction_input(request),
            time_remaining=time_remaining,
            is_active=is_active,
            provider_schema=compact_provider_schema(ProviderExtraction),
            timings=timings,
        )
        outcome = validate_opinions(
            output.opinions,
            request.feedback.text,
            max_opinions=self.limits.max_opinions,
        )
        # Counts and reason codes only; never feedback text or customer data.
        logger.info(
            "feedback_extraction feedback_id=%s prompt_version=%s raw=%d kept=%d dropped=%s",
            request.feedback.id,
            FEEDBACK_PROMPT_VERSION,
            outcome.raw_count,
            len(outcome.opinions),
            ",".join(reason.value for reason in outcome.dropped) or "-",
        )
        return outcome
