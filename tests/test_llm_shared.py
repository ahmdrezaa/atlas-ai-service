from concurrent import futures
from uuid import uuid4

import grpc
import pytest

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.consultation import assessment
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.llm import AIFailure, GeminiSettings, compact_provider_schema
from atlas_ai.server import BearerAuthInterceptor


def test_consultation_names_still_point_to_shared_module():
    assert assessment.ConsultationFailure is AIFailure
    assert assessment.GeminiSettings is GeminiSettings
    assert assessment.compact_provider_schema is compact_provider_schema


def test_default_from_env_keeps_consultation_variable_names(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", " m ")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("CONSULTATION_MAX_CONTEXT_BYTES", "4096")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "512")
    settings = GeminiSettings.from_env()
    assert settings.model == "m"
    assert settings.api_key.get_secret_value() == "k"
    assert settings.timeout_seconds == 12
    assert settings.max_context_bytes == 4096
    assert settings.max_output_tokens == 512


def test_default_from_env_error_message_unchanged(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "bad")
    with pytest.raises(AIFailure, match="^Invalid assessment configuration$"):
        GeminiSettings.from_env()


def test_malformed_process_feedback_is_invalid_argument_over_grpc():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        interceptors=[BearerAuthInterceptor("test-only-token")],
    )
    ai_pb2_grpc.add_AIServiceServicer_to_server(AtlasAIService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = ai_pb2_grpc.AIServiceStub(channel)
            with pytest.raises(grpc.RpcError) as error:
                stub.ProcessFeedback(
                    ai_pb2.ProcessFeedbackRequest(shop_id=str(uuid4())),
                    timeout=5,
                    metadata=(("authorization", "Bearer test-only-token"),),
                )
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        server.stop(grace=0).wait(timeout=5)


def _invoke(factory, time_remaining):
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, SecretStr

    from atlas_ai.llm import invoke_gemini_structured

    class Out(BaseModel):
        ok: bool

    return invoke_gemini_structured(
        GeminiSettings(model="m", api_key=SecretStr("k")),
        factory,
        ChatPromptTemplate.from_messages([("human", "{context_json}")]),
        Out,
        stage="Test",
        input_data={},
        time_remaining=time_remaining,
    )


def test_budget_below_provider_minimum_fails_fast_without_a_call():
    from unittest.mock import Mock

    factory = Mock()
    with pytest.raises(AIFailure) as error:
        _invoke(factory, time_remaining=9.9)
    assert error.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "provider minimum deadline" in str(error.value)
    factory.assert_not_called()


def test_provider_400_is_internal_not_unavailable():
    from unittest.mock import Mock

    class Rejected(Exception):
        code = 400

    factory = Mock()
    factory.return_value.with_structured_output.return_value.invoke.side_effect = (
        Rejected("Minimum allowed deadline is 10s.")
    )
    with pytest.raises(AIFailure) as error:
        _invoke(factory, time_remaining=20)
    assert error.value.code == grpc.StatusCode.INTERNAL
    assert "Minimum allowed" not in str(error.value)
