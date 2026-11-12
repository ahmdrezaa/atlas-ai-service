"""Lossless context preparation; no prompts, inference, or session persistence."""

from typing import TypeVar

from google.protobuf.json_format import MessageToDict

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.models import (
    AggregateStatistics,
    ArchivedSolution,
    ConfirmedMemory,
    ConsultationContext,
    ConversationMessage,
    CustomerEvidence,
    HistoricalSolution,
    IssueSummary,
    Period,
    ShopProfile,
    SourceRegistry,
)
from atlas_ai.consultation.validation import validate_request

T = TypeVar("T")


def _index(pairs: list[tuple[str, T]], namespace: str) -> dict[str, T]:
    result: dict[str, T] = {}
    for source_id, value in pairs:
        if source_id in result and result[source_id] != value:
            # Never silently overwrite conflicting sources for a citable ID.
            raise ValueError(f"Conflicting {namespace} ID in consultation context")
        result[source_id] = value
    return result


def prepare_context(
    request: ai_pb2.ProcessConsultationTurnRequest,
) -> ConsultationContext:
    """Validate and copy the official request into explicitly attributed sections.

    Preserve every message in Go's supplied order, even for long histories.
    No truncation/summarization is done before a model/token budget exists.
    Text is untrusted data: retain quotes verbatim rather than stripping possible
    instructions. These models do not constitute prompt-injection protection.
    """
    validate_request(request)
    # Preserve defaults and enum numbers. Pydantic restores JSON int64 strings
    # to integers without float conversion; absent message fields stay absent.
    data = MessageToDict(
        request,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
        always_print_fields_with_no_presence=True,
    )
    snapshot = data["context"]
    evidence = tuple(
        CustomerEvidence.model_validate(item) for item in snapshot["evidence"]
    )
    memories = tuple(
        ConfirmedMemory.model_validate(item) for item in snapshot["memories"]
    )
    history = tuple(
        ConversationMessage.model_validate(
            {
                **item,
                "source_kind": "owner_statement"
                if item["role"] == ai_pb2.MESSAGE_ROLE_OWNER
                else "historical_ai_output",
            }
        )
        for item in data["messages"]
    )
    # Also reject conflicts between OWNER and AI messages with the same ID.
    _index([(item.id, item) for item in history], "message")
    owners = tuple(item for item in history if item.role == ai_pb2.MESSAGE_ROLE_OWNER)
    return ConsultationContext(
        session_id=request.session_id,
        request_id=request.request_id,
        issue=IssueSummary.model_validate(snapshot["issue"]),
        period=Period.model_validate(snapshot["period"])
        if "period" in snapshot
        else None,
        aggregate_statistics=AggregateStatistics.model_validate(snapshot["stats"])
        if "stats" in snapshot
        else None,
        representative_evidence=evidence,
        shop=ShopProfile.model_validate(snapshot["shop"]),
        confirmed_memories=memories,
        conversation_history=history,
        latest_owner_message=owners[-1] if owners else None,
        previous_solution=HistoricalSolution.model_validate(data["previous_solution"])
        if "previous_solution" in data
        else None,
        prior_archived_solution=ArchivedSolution.model_validate(
            snapshot["prior_issue_solution"]
        )
        if "prior_issue_solution" in snapshot
        else None,
        event=request.event,
        asked_topics=tuple(request.asked_topics),
        skipped_topics=tuple(request.skipped_topics),
        question_count=request.question_count,
        sources=SourceRegistry(
            opinions=_index([(item.opinion_id, item) for item in evidence], "opinion"),
            owner_messages=_index(
                [(item.id, item) for item in owners], "owner message"
            ),
            memories=_index([(item.id, item) for item in memories], "memory"),
        ),
    )
