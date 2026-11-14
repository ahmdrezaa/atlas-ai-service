"""Internal Path A contracts. These are NOT the protobuf wire format."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FeedbackModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OpinionDraft(FeedbackModel):
    """One opinion exactly as the LLM proposed it; validated item by item."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    opinion_text: Annotated[NonBlank, StringConstraints(max_length=500)]
    evidence_text: Annotated[NonBlank, StringConstraints(max_length=1000)]
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    severity: Literal["SEVERE", "NOT_SEVERE"]
    variant: Annotated[str, StringConstraints(max_length=200)] | None = None
    issue_title_hint: Annotated[NonBlank, StringConstraints(max_length=120)]


class ProviderOpinion(FeedbackModel):
    """Provider grammar only (all strings). Local validation uses OpinionDraft."""

    opinion_text: str
    evidence_text: str
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    severity: Literal["SEVERE", "NOT_SEVERE"]
    variant: str | None = None
    issue_title_hint: str


class ProviderExtraction(FeedbackModel):
    opinions: list[ProviderOpinion]


class ExtractionOutput(FeedbackModel):
    """Top-level shape only: one malformed item must not discard valid siblings."""

    opinions: list[dict]


class DropReason(str, Enum):
    INVALID_FIELDS = "INVALID_FIELDS"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    NEGATION_CROPPED = "NEGATION_CROPPED"
    DUPLICATE = "DUPLICATE"


class ValidatedOpinion(FeedbackModel):
    opinion_text: str
    # Always an exact substring of the original feedback text.
    evidence_text: str
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    severity: Literal["SEVERE", "NOT_SEVERE"]
    variant: str | None
    issue_title_hint: str


class ExtractionOutcome(FeedbackModel):
    opinions: tuple[ValidatedOpinion, ...]
    dropped: tuple[DropReason, ...] = ()
    raw_count: int = Field(ge=0)
