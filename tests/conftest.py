import copy
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ContextAssessor,
    GeminiSettings,
    SufficiencyAssessment,
)
from atlas_ai.consultation.solution import GeneratedSolution


@pytest.fixture(scope="module")
def generate_assessor():
    assessor = Mock(spec=ContextAssessor)
    assessor.settings = GeminiSettings(model="mock-gemini")
    assessor.assess = Mock(
        return_value=SufficiencyAssessment(
            decision="GENERATE_SOLUTION",
            reason="Konteks cukup atau pertanyaan tidak material.",
            problem_evidence_sufficient=True,
            missing_operational_information=[],
            relevant_business_constraints=[],
            additional_question_material=False,
            repeats_existing_information=False,
        )
    )
    return assessor


@pytest.fixture
def solution_request(valid_request):
    request = valid_request
    request.context.stats.CopyFrom(
        ai_pb2.IssueStats(
            opinion_count=10,
            unique_feedback_count=8,
            average_rating=3.5,
            negative_count=6,
            positive_count=3,
            neutral_count=1,
            severe_count=1,
            not_severe_count=9,
        )
    )
    request.context.evidence.add(
        opinion_id=str(uuid4()),
        opinion_text="Antrean lambat",
        evidence_text="Antreannya lama sekali.",
        full_feedback="Rotinya enak. Antreannya lama sekali.",
        sentiment=ai_pb2.SENTIMENT_NEGATIVE,
        severity=ai_pb2.SEVERITY_NOT_SEVERE,
    )
    request.context.memories.add(id=str(uuid4()), fact_text="Jumlah barista 2 orang.")
    return request


@pytest.fixture
def solution_payload(solution_request):
    oid = solution_request.context.evidence[0].opinion_id
    refs = {"opinion_ids": [oid], "owner_message_ids": [], "memory_ids": []}
    return {
        "findings": [
            {"statement": "Pelanggan menyebut antrean lama.", "references": refs},
            {
                "statement": "Jumlah barista 2 orang.",
                "references": {"memory_ids": [solution_request.context.memories[0].id]},
            },
        ],
        "customer_evidence": [
            {
                "opinion_id": oid,
                "quote": "Antreannya lama sekali.",
                "note": "Keluhan antrean.",
            }
        ],
        "hypotheses": [
            {
                "statement": "Hipotesis: alur pesanan mungkin menghambat antrean; perlu diverifikasi.",
                "confidence": "",
                "references": refs,
            }
        ],
        "recommendations": [
            {
                "title": "Amati alur pesanan",
                "description": "Usulan: amati 2 shift sebelum mengubah alur.",
                "priority": "medium",
                "references": refs,
            }
        ],
        "success_indicators": [
            {
                "metric": "Durasi antrean yang dicatat owner",
                "target": "Usulan: catat menit tunggu per pesanan",
                "timeframe": "Usulan: evaluasi setelah 1 minggu",
            }
        ],
    }


@pytest.fixture
def solution_assessor(solution_payload):
    factory = Mock()

    def structured(schema, **kwargs):
        result = Mock()
        if schema is GeneratedSolution or (
            isinstance(schema, dict) and schema.get("title") == "GeneratedSolutionTurn"
        ):
            result.invoke.return_value = copy.deepcopy(solution_payload)
        else:
            result.invoke.return_value = {
                "decision": "GENERATE_SOLUTION",
                "reason": "Cukup untuk langkah observasi.",
                "problem_evidence_sufficient": True,
                "missing_operational_information": [],
                "relevant_business_constraints": [],
                "additional_question_material": False,
                "repeats_existing_information": False,
                "question": None,
            }
        return result

    factory.return_value.with_structured_output.side_effect = structured
    return ContextAssessor(
        GeminiSettings(model="mock-gemini", api_key=SecretStr("test-only")), factory
    )


EVENTS = [
    ai_pb2.CONSULTATION_EVENT_START,
    ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
    ai_pb2.CONSULTATION_EVENT_SKIP_QUESTION,
    ai_pb2.CONSULTATION_EVENT_GENERATE_NOW,
    ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
]


@pytest.fixture
def request_factory():
    def make(event=ai_pb2.CONSULTATION_EVENT_START):
        request = ai_pb2.ProcessConsultationTurnRequest(
            session_id=str(uuid4()),
            request_id=str(uuid4()),
            event=event,
            context=ai_pb2.ContextSnapshot(
                issue=ai_pb2.IssueSummary(id=str(uuid4())),
                shop=ai_pb2.ShopProfile(id=str(uuid4())),
            ),
        )
        if event in {
            ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
            ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
        }:
            request.messages.add(
                id=str(uuid4()),
                role=ai_pb2.MESSAGE_ROLE_OWNER,
                content="Dua barista bekerja pada sore hari.",
            )
        if event == ai_pb2.CONSULTATION_EVENT_SKIP_QUESTION:
            request.asked_topics.append("staffing")
            request.skipped_topics.append("staffing")
            request.question_count = 1
            request.messages.add(
                id=str(uuid4()),
                role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
                content="Berapa barista yang bekerja?",
                topic_key="staffing",
            )
        return request

    return make


@pytest.fixture(params=EVENTS, ids=lambda event: ai_pb2.ConsultationEvent.Name(event))
def valid_request(request, request_factory):
    return request_factory(request.param)


INVALID_CASES = [
    ("session_id", "", "session_id"),
    ("session_id", "not-a-uuid", "session_id"),
    ("request_id", "", "request_id"),
    ("request_id", "invalid", "request_id"),
    ("event", 0, "consultation event"),
    ("event", 999, "consultation event"),
    ("question_count", -1, "question count"),
    ("context.issue.id", "", "context.issue.id"),
    ("context.issue.id", "invalid", "context.issue.id"),
    ("context.shop.id", "", "context.shop.id"),
    ("context.shop.id", "invalid", "context.shop.id"),
    ("messages.0.id", "invalid", "messages[0].id"),
    ("messages.0.role", 0, "messages[0].role"),
    ("messages.0.role", 999, "messages[0].role"),
    ("messages.0.role", ai_pb2.MESSAGE_ROLE_ATLAS_AI, "Latest message"),
    ("messages.0.content", " \n\t", "owner message cannot be empty"),
    ("context.stats.opinion_count", -1, "context.stats.opinion_count"),
    ("context.stats.average_rating", float("nan"), "context.stats.average_rating"),
    ("context.stats.average_rating", float("inf"), "context.stats.average_rating"),
    ("context.stats.average_rating", -0.1, "context.stats.average_rating"),
    ("context.evidence.0.opinion_id", "invalid", "evidence[0].opinion_id"),
    ("context.evidence.0.sentiment", 0, "evidence[0].sentiment"),
    ("context.evidence.0.sentiment", 999, "evidence[0].sentiment"),
    ("context.evidence.0.severity", 999, "evidence[0].severity"),
    ("context.evidence.0.rating", float("nan"), "evidence[0].rating"),
    ("context.evidence.0.rating", float("inf"), "evidence[0].rating"),
    ("context.memories.0.id", "invalid", "memories[0].id"),
    (
        "context.prior_issue_solution.session_id",
        "invalid",
        "prior_issue_solution.session_id",
    ),
]
for field in (
    "unique_feedback_count",
    "negative_count",
    "positive_count",
    "neutral_count",
    "severe_count",
    "not_severe_count",
):
    INVALID_CASES.append((f"context.stats.{field}", -1, f"context.stats.{field}"))
for field in (
    "context.period.start_time",
    "context.period.end_time",
    "context.evidence.0.create_time",
    "messages.0.create_time",
    "context.prior_issue_solution.archive_time",
):
    INVALID_CASES.extend(
        [
            (f"{field}.seconds", 253402300800, field.replace(".0", "[0]")),
            (f"{field}.nanos", -1, field.replace(".0", "[0]")),
            (f"{field}.nanos", 1000000000, field.replace(".0", "[0]")),
        ]
    )


@pytest.fixture(params=INVALID_CASES, ids=lambda case: f"{case[0]}={case[1]}")
def invalid_request(request, request_factory):
    """Shared malformed snapshots for validator and real gRPC boundary tests."""
    value = request_factory(ai_pb2.CONSULTATION_EVENT_OWNER_REPLY)
    value.context.evidence.add(
        opinion_id=str(uuid4()),
        sentiment=ai_pb2.SENTIMENT_NEGATIVE,
        severity=ai_pb2.SEVERITY_UNSPECIFIED,
    )
    value.context.memories.add(id=str(uuid4()), fact_text="Dua barista.")
    value.context.prior_issue_solution.session_id = str(uuid4())
    path, replacement, error = request.param
    parts = path.split(".")
    target = value
    for part in parts[:-1]:
        target = target[int(part)] if part.isdecimal() else getattr(target, part)
    setattr(target, parts[-1], replacement)
    return value, error


@pytest.fixture
def feedback_request_factory():
    """Path A request with valid IDs; tests override fields as needed."""

    def make(text="Rotinya empuk, tapi isian coklatnya sedikit.", rating=4.0):
        shop_id = str(uuid4())
        return ai_pb2.ProcessFeedbackRequest(
            shop_id=shop_id,
            feedback=ai_pb2.Feedback(
                id=str(uuid4()),
                customer_name="Rahasia Pelanggan",
                rating=rating,
                text=text,
            ),
            shop=ai_pb2.ShopProfile(
                id=shop_id, name="Toko Roti Uji", category="Bakery", scale="Mikro"
            ),
        )

    return make
