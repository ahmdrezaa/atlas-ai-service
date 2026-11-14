from concurrent import futures
from uuid import uuid4

import grpc
import pytest
from openai_offline import OfflineOpenAI
from pydantic import BaseModel, ConfigDict

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.consultation import assessment
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.llm import (
    AIFailure,
    LLMSettings,
    compact_provider_schema,
    drop_optional_nulls,
    strict_provider_schema,
)
from atlas_ai.server import BearerAuthInterceptor


def test_consultation_names_still_point_to_shared_module():
    assert assessment.ConsultationFailure is AIFailure
    assert assessment.LLMSettings is LLMSettings
    assert assessment.compact_provider_schema is compact_provider_schema


def test_default_from_env_reads_consultation_variable_names(monkeypatch):
    monkeypatch.setenv("CONSULTATION_MODEL", " m ")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("CONSULTATION_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("CONSULTATION_MAX_CONTEXT_BYTES", "4096")
    monkeypatch.setenv("CONSULTATION_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("CONSULTATION_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_TEMPERATURE", "")
    settings = LLMSettings.from_env()
    assert settings.model == "m"
    assert settings.api_key.get_secret_value() == "k"
    assert settings.timeout_seconds == 12
    assert settings.max_context_bytes == 4096
    assert settings.max_output_tokens == 512
    assert settings.reasoning_effort == "low"
    assert settings.temperature is None


@pytest.mark.parametrize(
    "name,value",
    [
        ("CONSULTATION_TIMEOUT_SECONDS", "bad"),
        ("CONSULTATION_REASONING_EFFORT", "extreme"),
        ("LLM_TEMPERATURE", "hot"),
        ("CONSULTATION_PRICE_INPUT_PER_1M_USD", "-1"),
    ],
)
def test_default_from_env_error_message_unchanged(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(AIFailure, match="^Invalid assessment configuration$"):
        LLMSettings.from_env()


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

    from atlas_ai.llm import invoke_llm_structured

    class Out(BaseModel):
        ok: bool

    return invoke_llm_structured(
        LLMSettings(model="m", api_key=SecretStr("k")),
        factory,
        ChatPromptTemplate.from_messages([("human", "{context_json}")]),
        Out,
        stage="Test",
        input_data={},
        time_remaining=time_remaining,
    )


def test_budget_too_small_for_a_call_fails_fast_without_a_call():
    from unittest.mock import Mock

    factory = Mock()
    with pytest.raises(AIFailure) as error:
        _invoke(factory, time_remaining=1.0)
    assert error.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "budget too small" in str(error.value)
    factory.assert_not_called()


def test_short_budget_is_sent_as_is():
    # OpenAI has no minimum deadline: a 5 s budget is sent as ~4.75 s.
    provider = OfflineOpenAI({"ok": True})
    assert _invoke(provider, time_remaining=5).ok is True
    assert provider.model_kwargs[0]["timeout"] == pytest.approx(4.75, abs=0.05)


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


# --- OpenAI strict structured outputs ------------------------------------------------


class _Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    note: str = ""


class _Outer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    inner: _Inner | None = None
    items: list[_Inner] = []


def _closed_and_complete(node):
    if isinstance(node, list):
        return all(_closed_and_complete(item) for item in node)
    if not isinstance(node, dict):
        return True
    if "properties" in node:
        if node.get("additionalProperties") is not False:
            return False
        if set(node["required"]) != set(node["properties"]):
            return False
    return all(_closed_and_complete(value) for value in node.values())


def _invoke_outer(provider, settings=None):
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import SecretStr

    from atlas_ai.llm import invoke_llm_structured

    return invoke_llm_structured(
        settings or LLMSettings(model="m", api_key=SecretStr("k")),
        provider,
        ChatPromptTemplate.from_messages([("human", "{context_json}")]),
        _Outer,
        stage="Test",
        input_data={},
        time_remaining=10,
    )


def test_strict_schema_requires_every_field_and_closes_objects():
    compact = compact_provider_schema(_Outer)
    strict = strict_provider_schema(compact)
    assert _closed_and_complete(strict)
    # Optional fields became nullable; required ones did not.
    assert {"type": "null"} in strict["properties"]["items"]["anyOf"]
    assert {"type": "null"} in strict["properties"]["inner"]["anyOf"]
    assert strict["properties"]["name"] == {"type": "string"}
    # The compact form (used for prompt-version hashes) is unchanged.
    assert compact["required"] == ["name"]


def test_optional_nulls_are_dropped_before_local_validation():
    compact = compact_provider_schema(_Outer)
    data = {
        "name": "x",
        "inner": {"label": "a", "note": None},
        "items": [{"label": "b", "note": None}],
    }
    cleaned = drop_optional_nulls(data, compact)
    assert cleaned == {"name": "x", "inner": {"label": "a"}, "items": [{"label": "b"}]}
    assert _Outer.model_validate(cleaned).items[0].note == ""
    # A null for a REQUIRED field is kept, so validation still rejects it.
    assert drop_optional_nulls({"name": None}, compact) == {"name": None}
    assert drop_optional_nulls({"name": "x", "inner": None}, compact) == {"name": "x"}


def test_provider_nulls_for_optional_fields_validate_end_to_end():
    provider = OfflineOpenAI({"name": "x", "inner": None, "items": None})
    assert _invoke_outer(provider) == _Outer(name="x")
    sent = provider.requests[0]["response_format"]["json_schema"]
    assert sent["strict"] is True and _closed_and_complete(sent["schema"])


@pytest.mark.parametrize(
    "status,code,calls",
    [
        (503, grpc.StatusCode.UNAVAILABLE, 2),  # retried once, then fails
        (500, grpc.StatusCode.UNAVAILABLE, 2),
        (429, grpc.StatusCode.UNAVAILABLE, 1),  # rate limit: never retried
        (400, grpc.StatusCode.INTERNAL, 1),
        (401, grpc.StatusCode.FAILED_PRECONDITION, 1),
        (403, grpc.StatusCode.FAILED_PRECONDITION, 1),
        (404, grpc.StatusCode.FAILED_PRECONDITION, 1),  # unknown model
    ],
)
def test_real_openai_sdk_errors_are_mapped_and_sanitized(
    monkeypatch, status, code, calls
):
    monkeypatch.setattr("atlas_ai.llm.RETRY_BACKOFF_SECONDS", 0)
    provider = OfflineOpenAI(status, status)
    with pytest.raises(AIFailure) as error:
        _invoke(provider, time_remaining=10)
    assert error.value.code == code
    assert len(provider.requests) == calls
    assert "private provider body" not in str(error.value)


def test_real_openai_sdk_503_then_success(monkeypatch):
    monkeypatch.setattr("atlas_ai.llm.RETRY_BACKOFF_SECONDS", 0)
    provider = OfflineOpenAI(503, {"ok": True})
    assert _invoke(provider, time_remaining=10).ok is True
    assert len(provider.requests) == 2


def test_temperature_and_reasoning_effort_are_sent_only_when_configured():
    from pydantic import SecretStr

    keys = ("temperature", "reasoning_effort")
    provider = OfflineOpenAI({"name": "x"}, {"name": "x"})
    _invoke_outer(provider)
    _invoke_outer(
        provider,
        LLMSettings(
            model="m", api_key=SecretStr("k"), temperature=0, reasoning_effort="low"
        ),
    )
    default, configured = provider.requests
    assert not any(key in default for key in keys)
    assert {key: configured[key] for key in keys} == {
        "temperature": 0,
        "reasoning_effort": "low",
    }
