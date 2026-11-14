from concurrent import futures
from unittest.mock import Mock

import grpc
import pytest

from atlas.ai.v1 import ai_pb2_grpc
from atlas_ai.consultation.assessment import (
    AssessmentQuestion,
    ConsultationFailure,
    SufficiencyAssessment,
)
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.server import BearerAuthInterceptor


@pytest.fixture(scope="module")
def consultation_stub(generate_assessor):
    # Exercise generated serializers, the existing interceptor, and real adapter.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[BearerAuthInterceptor("test-only-token")],
    )
    service = AtlasAIService()
    service.consultation_pipeline = ConsultationPipeline(generate_assessor)
    ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port != 0
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=5)
            yield ai_pb2_grpc.AIServiceStub(channel)
    finally:
        server.stop(grace=0).wait(timeout=5)


def call(stub, request, authorization="Bearer test-only-token"):
    return stub.ProcessConsultationTurn(
        request,
        timeout=30,
        metadata=(
            ("authorization", authorization),
            ("x-request-id", request.request_id),
        ),
    )


def test_solution_without_evidence_returns_failed_precondition(
    consultation_stub, valid_request
):
    with pytest.raises(grpc.RpcError) as error:
        call(consultation_stub, valid_request)
    assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "verifiable customer evidence" in error.value.details()


def test_invalid_requests_return_invalid_argument(consultation_stub, invalid_request):
    request, expected_error = invalid_request
    with pytest.raises(grpc.RpcError) as error:
        call(consultation_stub, request)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert expected_error in error.value.details()


def test_missing_context_returns_invalid_argument(consultation_stub, valid_request):
    valid_request.ClearField("context")
    with pytest.raises(grpc.RpcError) as error:
        call(consultation_stub, valid_request)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_reply_without_owner_returns_invalid_argument(
    consultation_stub, request_factory
):
    from atlas.ai.v1 import ai_pb2

    for event in (
        ai_pb2.CONSULTATION_EVENT_OWNER_REPLY,
        ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION,
    ):
        request = request_factory(event)
        request.ClearField("messages")
        with pytest.raises(grpc.RpcError) as error:
            call(consultation_stub, request)
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_existing_auth_still_rejects_invalid_token(consultation_stub, request_factory):
    with pytest.raises(grpc.RpcError) as error:
        call(consultation_stub, request_factory(), authorization="Bearer invalid")
    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_assessment_question_and_metadata_on_wire(
    consultation_stub, request_factory, generate_assessor, monkeypatch
):
    result = SufficiencyAssessment(
        decision="ASK_QUESTION",
        reason="Operational gap",
        problem_evidence_sufficient=True,
        missing_operational_information=["Budget"],
        relevant_business_constraints=[],
        additional_question_material=True,
        repeats_existing_information=False,
        question=AssessmentQuestion(
            topic_key="budget", text="Berapa anggaran yang tersedia?"
        ),
    )
    monkeypatch.setattr(generate_assessor, "assess", Mock(return_value=result))
    response = call(consultation_stub, request_factory())
    assert response.WhichOneof("result") == "next_question"
    assert response.next_question.topic_key == "budget"
    assert response.model_id == "mock-gemini"
    assert response.prompt_version == "consultation-assessment-v6"
    assert not response.memory_candidates


@pytest.mark.parametrize(
    "code",
    [
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.FAILED_PRECONDITION,
        grpc.StatusCode.CANCELLED,
    ],
)
def test_assessment_failures_keep_status_on_wire(
    consultation_stub, request_factory, generate_assessor, monkeypatch, code
):
    monkeypatch.setattr(
        generate_assessor,
        "assess",
        Mock(side_effect=ConsultationFailure(code, "Safe error")),
    )
    with pytest.raises(grpc.RpcError) as error:
        call(consultation_stub, request_factory())
    assert error.value.code() == code
    assert error.value.details() == "Safe error"
