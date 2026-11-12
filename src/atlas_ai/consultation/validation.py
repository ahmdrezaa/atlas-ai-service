import math
from uuid import UUID

from google.protobuf.timestamp_pb2 import Timestamp

from atlas.ai.v1 import ai_pb2

VALID_EVENTS = {
    ai_pb2.CONSULTATION_EVENT_START,
    ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
    ai_pb2.CONSULTATION_EVENT_SKIP_QUESTION,
    ai_pb2.CONSULTATION_EVENT_GENERATE_NOW,
    ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
}


def _validate_uuid(value: str, field: str) -> None:
    try:
        UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: expected UUID") from exc


def _validate_timestamp(value: Timestamp, field: str) -> None:
    try:
        value.ToJsonString()
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {field}: expected protobuf Timestamp") from exc


def validate_request(request: ai_pb2.ProcessConsultationTurnRequest) -> None:
    """Validate structure, leaving sufficiency and turn policy to the AI layer.

    Optional snapshots/history may be absent; empty evidence is legal input.
    Historical solutions are not re-grounded against the current snapshot.
    Go owns authorization, counters, session transitions, and persistent state.
    """
    for field in ("session_id", "request_id"):
        _validate_uuid(getattr(request, field), field)

    if request.event not in VALID_EVENTS:
        raise ValueError("Invalid consultation event")
    if request.question_count < 0:
        raise ValueError("Invalid question count")

    if not request.HasField("context"):
        raise ValueError("Missing context snapshot")
    context = request.context
    for field in ("issue", "shop"):
        if not context.HasField(field):
            raise ValueError(f"Missing {field} context")
        _validate_uuid(getattr(context, field).id, f"context.{field}.id")

    if context.HasField("period"):
        period = context.period
        for field in ("start_time", "end_time"):
            if period.HasField(field):
                _validate_timestamp(getattr(period, field), f"context.period.{field}")
        if period.HasField("start_time") and period.HasField("end_time"):
            start = (period.start_time.seconds, period.start_time.nanos)
            end = (period.end_time.seconds, period.end_time.nanos)
            if start > end:
                raise ValueError("Invalid context.period: start_time after end_time")

    if context.HasField("stats"):
        stats = context.stats
        # Do not compare counts with evidence: evidence is only a sample.
        for field in stats.DESCRIPTOR.fields:
            value = getattr(stats, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid context.stats.{field.name}")

    for index, opinion in enumerate(context.evidence):
        field = f"context.evidence[{index}]"
        _validate_uuid(opinion.opinion_id, f"{field}.opinion_id")
        if opinion.sentiment not in {
            ai_pb2.SENTIMENT_NEGATIVE,
            ai_pb2.SENTIMENT_POSITIVE,
            ai_pb2.SENTIMENT_NEUTRAL,
        }:
            raise ValueError(f"Invalid {field}.sentiment")
        if opinion.severity not in ai_pb2.Severity.values():
            raise ValueError(f"Invalid {field}.severity")
        if not math.isfinite(opinion.rating):
            raise ValueError(f"Invalid {field}.rating")
        if opinion.HasField("create_time"):
            _validate_timestamp(opinion.create_time, f"{field}.create_time")

    for index, memory in enumerate(context.memories):
        _validate_uuid(memory.id, f"context.memories[{index}].id")

    if context.HasField("prior_issue_solution"):
        prior = context.prior_issue_solution
        _validate_uuid(prior.session_id, "context.prior_issue_solution.session_id")
        if prior.HasField("archive_time"):
            _validate_timestamp(
                prior.archive_time, "context.prior_issue_solution.archive_time"
            )

    for index, message in enumerate(request.messages):
        field = f"messages[{index}]"
        _validate_uuid(message.id, f"{field}.id")
        if message.role not in {
            ai_pb2.MESSAGE_ROLE_OWNER,
            ai_pb2.MESSAGE_ROLE_ATLAS_AI,
        }:
            raise ValueError(f"Invalid {field}.role")
        if message.role == ai_pb2.MESSAGE_ROLE_OWNER and not message.content.strip():
            raise ValueError(f"Invalid {field}.content: owner message cannot be empty")
        if message.HasField("create_time"):
            _validate_timestamp(message.create_time, f"{field}.create_time")

    if request.event in {
        ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
        ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
    }:
        if not request.messages:
            raise ValueError("Missing owner message")
        if request.messages[-1].role != ai_pb2.MESSAGE_ROLE_OWNER:
            raise ValueError("Latest message must be from owner")
