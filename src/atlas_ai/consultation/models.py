"""Internal snapshots, not a replacement for the protobuf input contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IssueSummary(ContextModel):
    id: str
    title: str


class Period(ContextModel):
    # RFC3339 strings preserve protobuf nanoseconds; None means not supplied.
    start_time: str | None = None
    end_time: str | None = None


class AggregateStatistics(ContextModel):
    source_kind: Literal["aggregate_statistics"] = "aggregate_statistics"
    unique_feedback_count: int
    opinion_count: int
    average_rating: float
    negative_count: int
    positive_count: int
    neutral_count: int
    severe_count: int
    not_severe_count: int


class CustomerEvidence(ContextModel):
    source_kind: Literal["customer_evidence_sample"] = "customer_evidence_sample"
    opinion_id: str
    opinion_text: str
    evidence_text: str
    sentiment: int
    severity: int
    full_feedback: str
    rating: float
    create_time: str | None = None


class ShopProfile(ContextModel):
    id: str
    name: str
    description: str
    category: str
    scale: str


class ConfirmedMemory(ContextModel):
    source_kind: Literal["confirmed_business_memory"] = "confirmed_business_memory"
    id: str
    fact_text: str


class References(ContextModel):
    opinion_ids: tuple[str, ...] = ()
    owner_message_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()


class Finding(ContextModel):
    statement: str
    references: References | None = None


class SolutionEvidence(ContextModel):
    opinion_id: str
    quote: str
    note: str


class Hypothesis(ContextModel):
    statement: str
    confidence: str
    references: References | None = None


class Recommendation(ContextModel):
    title: str
    description: str
    priority: str
    references: References | None = None


class SuccessIndicator(ContextModel):
    metric: str
    target: str
    timeframe: str


class HistoricalSolution(ContextModel):
    """Past AI output, including its claims/references, never new factual evidence."""

    source_kind: Literal["historical_ai_output"] = "historical_ai_output"
    findings: tuple[Finding, ...]
    customer_evidence: tuple[SolutionEvidence, ...]
    hypotheses: tuple[Hypothesis, ...]
    recommendations: tuple[Recommendation, ...]
    success_indicators: tuple[SuccessIndicator, ...]


class ConversationMessage(ContextModel):
    id: str
    role: int
    source_kind: Literal["owner_statement", "historical_ai_output"]
    content: str
    solution: HistoricalSolution | None = None
    topic_key: str
    create_time: str | None = None


class ArchivedSolution(ContextModel):
    source_kind: Literal["historical_ai_output"] = "historical_ai_output"
    session_id: str
    archive_time: str | None = None
    solution: HistoricalSolution | None = None


class SourceRegistry(ContextModel):
    """Separate ID namespaces. AI messages and archived citations are excluded."""

    opinions: dict[str, CustomerEvidence] = Field(default_factory=dict)
    owner_messages: dict[str, ConversationMessage] = Field(default_factory=dict)
    memories: dict[str, ConfirmedMemory] = Field(default_factory=dict)


class ConsultationContext(ContextModel):
    session_id: str
    request_id: str
    issue: IssueSummary
    period: Period | None
    aggregate_statistics: AggregateStatistics | None
    representative_evidence: tuple[CustomerEvidence, ...]
    shop: ShopProfile
    confirmed_memories: tuple[ConfirmedMemory, ...]
    conversation_history: tuple[ConversationMessage, ...]
    latest_owner_message: ConversationMessage | None
    previous_solution: HistoricalSolution | None
    prior_archived_solution: ArchivedSolution | None
    event: int
    asked_topics: tuple[str, ...]
    skipped_topics: tuple[str, ...]
    question_count: int
    sources: SourceRegistry
