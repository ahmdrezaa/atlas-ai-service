import socket
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.feedback.embeddings import (
    DEFAULT_MODEL_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    E5Embedder,
)
from atlas_ai.feedback.extractor import FeedbackExtractor
from atlas_ai.feedback.pipeline import FeedbackPipeline
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.llm import AIFailure, GeminiSettings
from atlas_ai.server import SERVICE_NAME, create_server, warm_up

SETTINGS = GeminiSettings(model="mock-gemini", api_key=SecretStr("test-only"))
TOKEN = "test-only-token"

TEXTURE = {
    "opinion_text": "Roti empuk",
    "evidence_text": "Rotinya empuk",
    "sentiment": "POSITIVE",
    "severity": "NOT_SEVERE",
    "variant": None,
    "issue_title_hint": "Tekstur Roti Empuk",
}
FILLING = {
    "opinion_text": "Isian coklat sedikit",
    "evidence_text": "isian coklatnya sedikit",
    "sentiment": "NEGATIVE",
    "severity": "NOT_SEVERE",
    "variant": "coklat",
    "issue_title_hint": "Porsi Isian Roti Terlalu Sedikit",
}


def gemini(payload=None, exc=None):
    factory = Mock()
    invoke = factory.return_value.with_structured_output.return_value.invoke
    if exc is not None:
        invoke.side_effect = exc
    else:
        invoke.return_value = payload
    return factory


class FakeEmbedder:
    def __init__(self):
        self.texts = None

    def embed(self, texts, **kwargs):
        self.texts = list(texts)
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def pipeline(payload=None, exc=None, embedder=None):
    embedder = embedder or FakeEmbedder()
    extractor = FeedbackExtractor(SETTINGS, gemini(payload, exc))
    return FeedbackPipeline(extractor=extractor, embedder=embedder), embedder


# --- pipeline -----------------------------------------------------------------------


def test_multi_opinion_response_mapping(feedback_request_factory):
    run, embedder = pipeline({"opinions": [TEXTURE, FILLING]})
    response = run.run(feedback_request_factory())
    first, second = response.opinions
    assert first.sentiment == ai_pb2.SENTIMENT_POSITIVE
    assert second.sentiment == ai_pb2.SENTIMENT_NEGATIVE
    assert second.severity == ai_pb2.SEVERITY_NOT_SEVERE
    assert second.evidence_text == "isian coklatnya sedikit"
    assert second.embedding_model == EMBEDDING_MODEL_ID
    assert len(second.embedding) == EMBEDDING_DIM
    # Embeddings are computed from opinion_text, never from evidence.
    assert embedder.texts == ["Roti empuk", "Isian coklat sedikit"]
    for item in response.opinions:
        assert item.issue.new_issue_title  # always set
        assert item.issue.existing_issue_id == ""


def test_severe_maps_to_proto(feedback_request_factory):
    severe = {**FILLING, "severity": "SEVERE"}
    response = pipeline({"opinions": [severe]})[0].run(feedback_request_factory())
    assert response.opinions[0].severity == ai_pb2.SEVERITY_SEVERE


def test_exact_title_reuses_existing_issue(feedback_request_factory):
    request = feedback_request_factory()
    issue_id = str(uuid4())
    request.candidate_issues.add(issue_id=str(uuid4()), title="Antrean Kasir Lama")
    request.candidate_issues.add(
        issue_id=issue_id, title="  porsi isian roti   TERLALU sedikit "
    )
    response = pipeline({"opinions": [TEXTURE, FILLING]})[0].run(request)
    texture, filling = response.opinions
    assert filling.issue.existing_issue_id == issue_id
    assert filling.issue.new_issue_title == "Porsi Isian Roti Terlalu Sedikit"
    assert texture.issue.existing_issue_id == ""


def test_similar_but_not_equal_title_is_a_new_issue(feedback_request_factory):
    request = feedback_request_factory()
    request.candidate_issues.add(
        issue_id=str(uuid4()), title="Porsi Isian Roti Sedikit"
    )
    response = pipeline({"opinions": [FILLING]})[0].run(request)
    assert response.opinions[0].issue.existing_issue_id == ""


def test_no_opinion_skips_embedding(feedback_request_factory):
    run, embedder = pipeline({"opinions": []})
    response = run.run(feedback_request_factory("Saya datang pukul 15.00."))
    assert list(response.opinions) == []
    assert embedder.texts is None


def test_embedding_failure_fails_the_request(feedback_request_factory):
    embedder = Mock()
    embedder.embed.side_effect = AIFailure(
        grpc.StatusCode.INTERNAL, "Invalid embedding values"
    )
    with pytest.raises(AIFailure) as error:
        pipeline({"opinions": [FILLING]}, embedder=embedder)[0].run(
            feedback_request_factory()
        )
    assert error.value.code == grpc.StatusCode.INTERNAL


def test_malformed_request_is_rejected_before_loading(feedback_request_factory):
    request = feedback_request_factory()
    request.shop_id = "bad"
    run = FeedbackPipeline()  # nothing injected: would need env + model files
    with pytest.raises(ValueError):
        run.run(request)
    assert run._extractor is None and run._embedder is None


def test_deadline_margin_is_applied(feedback_request_factory):
    factory = gemini({"opinions": [FILLING]})
    run = FeedbackPipeline(
        extractor=FeedbackExtractor(SETTINGS, factory), embedder=FakeEmbedder()
    )
    run.run(feedback_request_factory(), time_remaining=20)
    timeout = factory.call_args.kwargs["timeout"]
    assert timeout <= 20 - 0.5 - 0.25 + 0.01


def test_missing_model_files_is_failed_precondition(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_EMBEDDING_MODEL_DIR", str(tmp_path))
    service = AtlasAIService()
    assert warm_up(service) is False


# --- gRPC -------------------------------------------------------------------------


@pytest.fixture
def bound():
    """Start a server on a free port and return (stub, health_stub, health_servicer)."""

    def make(feedback_pipeline):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        service = AtlasAIService()
        service.feedback_pipeline = feedback_pipeline
        server, health_servicer, _ = create_server(
            TOKEN, f"127.0.0.1:{port}", service=service, max_workers=2
        )
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        make.cleanup.append((server, channel))
        return (
            ai_pb2_grpc.AIServiceStub(channel),
            health_pb2_grpc.HealthStub(channel),
            health_servicer,
        )

    make.cleanup = []
    yield make
    for server, channel in make.cleanup:
        channel.close()
        server.stop(grace=0).wait(timeout=5)


def call(stub, request, token=TOKEN):
    return stub.ProcessFeedback(
        request,
        timeout=30,
        metadata=(("authorization", f"Bearer {token}"), ("x-request-id", str(uuid4()))),
    )


def test_grpc_full_response(bound, feedback_request_factory):
    stub, _, _ = bound(pipeline({"opinions": [TEXTURE, FILLING]})[0])
    response = call(stub, feedback_request_factory())
    assert len(response.opinions) == 2
    assert all(len(item.embedding) == EMBEDDING_DIM for item in response.opinions)


@pytest.mark.parametrize(
    "exc, code",
    [
        (RuntimeError("provider down"), grpc.StatusCode.UNAVAILABLE),
        (TimeoutError(), grpc.StatusCode.DEADLINE_EXCEEDED),
    ],
)
def test_grpc_provider_failures(bound, feedback_request_factory, exc, code):
    stub, _, _ = bound(pipeline(exc=exc)[0])
    with pytest.raises(grpc.RpcError) as error:
        call(stub, feedback_request_factory())
    assert error.value.code() == code
    assert "provider down" not in error.value.details()


def test_grpc_invalid_output_is_internal(bound, feedback_request_factory):
    stub, _, _ = bound(
        pipeline({"opinions": [{**FILLING, "evidence_text": "made up"}]})[0]
    )
    with pytest.raises(grpc.RpcError) as error:
        call(stub, feedback_request_factory())
    assert error.value.code() == grpc.StatusCode.INTERNAL


def test_grpc_invalid_request(bound, feedback_request_factory):
    stub, _, _ = bound(pipeline({"opinions": []})[0])
    request = feedback_request_factory()
    request.shop.id = str(uuid4())
    with pytest.raises(grpc.RpcError) as error:
        call(stub, request)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_auth_still_required(bound, feedback_request_factory):
    stub, _, _ = bound(pipeline({"opinions": []})[0])
    # Non-ASCII values are rejected by the gRPC client before sending.
    for token in ("wrong", ""):
        with pytest.raises(grpc.RpcError) as error:
            call(stub, feedback_request_factory(), token=token)
        assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_health_without_token(bound):
    _, health_stub, health_servicer = bound(pipeline({"opinions": []})[0])
    for name in ("", SERVICE_NAME):
        status = health_stub.Check(
            health_pb2.HealthCheckRequest(service=name), timeout=5
        )
        assert status.status == health_pb2.HealthCheckResponse.NOT_SERVING
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    status = health_stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
    assert status.status == health_pb2.HealthCheckResponse.SERVING


# --- real E5 + mocked Gemini (inside the Docker image only) ---------------------------

real_model = pytest.mark.skipif(
    not (Path(DEFAULT_MODEL_DIR) / "model.onnx").is_file(),
    reason="E5 model not bundled in this environment",
)


@real_model
def test_grpc_with_real_embeddings(bound, feedback_request_factory):
    run = FeedbackPipeline(
        extractor=FeedbackExtractor(SETTINGS, gemini({"opinions": [TEXTURE, FILLING]})),
        embedder=E5Embedder(),
    )
    stub, _, _ = bound(run)
    response = call(stub, feedback_request_factory())
    first, second = response.opinions
    assert len(first.embedding) == EMBEDDING_DIM
    assert abs(sum(x * x for x in first.embedding) - 1.0) < 1e-4
    assert list(first.embedding) != list(second.embedding)
