import re
from uuid import uuid4

import pytest

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.validation import validate_request


def test_valid_events_are_accepted_without_mutating_request(valid_request):
    before = valid_request.SerializeToString()
    validate_request(valid_request)
    assert valid_request.SerializeToString() == before


def test_valid_request_without_evidence_cannot_generate(
    valid_request, generate_assessor
):
    with pytest.raises(ConsultationFailure, match="verifiable customer evidence"):
        ConsultationPipeline(generate_assessor).run(valid_request)


def test_pipeline_rejects_invalid_input_before_ai(invalid_request):
    request, error = invalid_request
    with pytest.raises(ValueError, match=re.escape(error)):
        ConsultationPipeline().run(request)


@pytest.mark.parametrize("path", ["context", "context.issue", "context.shop"])
def test_required_snapshot_messages(request_factory, path):
    request = request_factory()
    target = request.context if "." in path else request
    target.ClearField(path.split(".")[-1])
    with pytest.raises(ValueError, match="Missing"):
        validate_request(request)


@pytest.mark.parametrize(
    "event",
    [
        ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
        ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
    ],
)
@pytest.mark.parametrize("invalid", ["missing", "ai-last", "empty"])
def test_reply_events_require_latest_nonempty_owner(request_factory, event, invalid):
    request = request_factory(event)
    if invalid == "missing":
        request.ClearField("messages")
    elif invalid == "ai-last":
        request.messages.add(
            id=str(uuid4()),
            role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
            content="Pertanyaan",
        )
    else:
        request.messages[-1].content = ""
    with pytest.raises(ValueError, match="owner"):
        validate_request(request)


def test_period_order_including_nanoseconds(request_factory):
    request = request_factory()
    request.context.period.start_time.seconds = 10
    request.context.period.start_time.nanos = 2
    request.context.period.end_time.seconds = 10
    request.context.period.end_time.nanos = 1
    with pytest.raises(ValueError, match="start_time after end_time"):
        validate_request(request)
    request.context.period.end_time.nanos = 2
    validate_request(request)


def test_no_extra_business_requirements_on_optional_input(valid_request):
    # No minimum evidence, upper counter limit, or required prior solution.
    valid_request.question_count = 12
    valid_request.context.stats.SetInParent()  # Proto3 zero values are legal.
    valid_request.context.period.start_time.seconds = 0  # Partial period is legal.
    validate_request(valid_request)


def test_historical_solution_sources_need_not_exist_in_current_snapshot(
    request_factory,
):
    request = request_factory()
    previous = request.previous_solution
    previous.findings.add(statement="Kondisi terdahulu").references.opinion_ids.append(
        str(uuid4())
    )
    request.context.prior_issue_solution.session_id = str(uuid4())
    request.context.prior_issue_solution.solution.CopyFrom(previous)
    request.messages.add(
        id=str(uuid4()),
        role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
        solution=previous,
    )  # AI solution messages may have no text content.
    validate_request(request)


def test_present_snapshot_and_history_fields(request_factory):
    request = request_factory(ai_pb2.CONSULTATION_EVENT_OWNER_REPLY)
    request.context.period.start_time.FromJsonString("2026-01-01T00:00:00Z")
    request.context.period.end_time.FromJsonString("2026-11-12T00:00:00Z")
    request.context.stats.opinion_count = 20  # Evidence is a sample, not all 20.
    request.context.stats.average_rating = 4.5
    opinion = request.context.evidence.add(
        opinion_id=str(uuid4()),
        sentiment=ai_pb2.SENTIMENT_NEUTRAL,
        severity=ai_pb2.SEVERITY_UNSPECIFIED,
        rating=0,
    )
    opinion.create_time.CopyFrom(request.context.period.start_time)
    request.messages[-1].create_time.CopyFrom(request.context.period.end_time)
    request.context.memories.add(id=str(uuid4()), fact_text="Dua barista.")
    prior = request.context.prior_issue_solution
    prior.session_id = str(uuid4())
    prior.archive_time.CopyFrom(request.context.period.start_time)
    validate_request(request)
