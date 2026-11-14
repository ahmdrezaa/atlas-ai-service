"""Fase 1: bounded provider retry, token usage, and per-RPC structured records."""

import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import grpc
import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, SecretStr

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai import observability
from atlas_ai.feedback.extractor import FeedbackExtractor
from atlas_ai.feedback.pipeline import FeedbackPipeline
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.llm import AIFailure, LLMSettings, invoke_llm_structured
from atlas_ai.server import create_server

SETTINGS = LLMSettings(model="mock-gemini", api_key=SecretStr("test-only"))
PROMPT = ChatPromptTemplate.from_messages([("human", "{context_json}")])


class Out(BaseModel):
    ok: bool


class ProviderError(Exception):
    def __init__(self, code):
        super().__init__(f"provider {code} with private details")
        self.code = code


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr("atlas_ai.llm.RETRY_BACKOFF_SECONDS", 0)


def factory_with(*results):
    factory = Mock()
    factory.return_value.with_structured_output.return_value.invoke.side_effect = list(
        results
    )
    return factory


def call(factory, time_remaining=30, settings=SETTINGS):
    with observability.request("Test", "req-1") as record:
        try:
            result = invoke_llm_structured(
                settings,
                factory,
                PROMPT,
                Out,
                stage="Unit",
                input_data={},
                time_remaining=time_remaining,
            )
        except AIFailure as exc:
            record.status = exc.code.name
            return exc, record
        return result, record


def raw(payload, input_tokens=120, output_tokens=30):
    return {
        "raw": SimpleNamespace(
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        ),
        "parsed": payload,
        "parsing_error": None,
    }


# --- retry -------------------------------------------------------------------------


def test_one_retry_on_provider_503_then_success():
    factory = factory_with(ProviderError(503), raw({"ok": True}))
    result, record = call(factory)
    assert result.ok is True
    (info,) = record.llm_calls
    assert info["attempts"] == 2 and info["retries"] == 1 and info["status"] == "OK"
    assert factory.call_count == 2
    # The retry stays inside the same deadline (shorter timeout on attempt 2).
    first, second = (c.kwargs["timeout"] for c in factory.call_args_list)
    assert second <= first


def test_two_503s_fail_after_one_retry():
    exc, record = call(factory_with(ProviderError(503), ProviderError(503)))
    assert exc.code == grpc.StatusCode.UNAVAILABLE
    assert "private details" not in str(exc)
    assert record.llm_calls[0]["attempts"] == 2


@pytest.mark.parametrize(
    "code,status",
    [
        (400, grpc.StatusCode.INTERNAL),
        (429, grpc.StatusCode.UNAVAILABLE),
        (504, grpc.StatusCode.DEADLINE_EXCEEDED),
    ],
)
def test_no_retry_for_non_5xx_or_timeouts(code, status):
    factory = factory_with(ProviderError(code), raw({"ok": True}))
    exc, record = call(factory)
    assert exc.code == status and factory.call_count == 1
    assert record.llm_calls[0]["retries"] == 0


def test_no_retry_when_budget_cannot_fit_another_call(monkeypatch):
    # Real 1 s backoff (sleep faked): 2.1 s budget - 1 s backoff < 1 s minimum call.
    monkeypatch.setattr("atlas_ai.llm.RETRY_BACKOFF_SECONDS", 1.0)
    monkeypatch.setattr("atlas_ai.llm.time.sleep", lambda seconds: None)
    factory = factory_with(ProviderError(503), raw({"ok": True}))
    exc, _ = call(factory, time_remaining=2.1)
    assert exc.code == grpc.StatusCode.UNAVAILABLE and factory.call_count == 1


def test_invalid_output_is_not_retried():
    factory = factory_with(raw({"wrong": 1}), raw({"ok": True}))
    exc, _ = call(factory)
    assert exc.code == grpc.StatusCode.INTERNAL and factory.call_count == 1


def test_parsing_error_from_include_raw_is_internal():
    factory = factory_with(
        {
            "raw": SimpleNamespace(usage_metadata=None),
            "parsed": None,
            "parsing_error": "x",
        }
    )
    exc, _ = call(factory)
    assert exc.code == grpc.StatusCode.INTERNAL


# --- tokens & cost -------------------------------------------------------------------


def test_token_usage_is_recorded_and_costed():
    priced = SETTINGS.model_copy(
        update={"price_input_per_1m": 0.10, "price_output_per_1m": 0.50}
    )
    _, record = call(factory_with(raw({"ok": True}, 1000, 500)), settings=priced)
    summary = record.summary()
    assert summary["tokens"] == {"input": 1000, "output": 500}
    expected = 1000 / 1e6 * 0.10 + 500 / 1e6 * 0.50
    assert summary["cost_usd"] == pytest.approx(expected)
    assert record.llm_calls[0]["cost_usd"] == pytest.approx(expected)
    assert record.llm_calls[0]["model"] == "mock-gemini"
    assert any(s["name"] == "llm.unit" for s in record.spans)


def test_each_call_is_priced_with_its_own_model():
    # Path A and Path B models have different official prices.
    cheap = SETTINGS.model_copy(
        update={"price_input_per_1m": 0.10, "price_output_per_1m": 0.50}
    )
    strong = SETTINGS.model_copy(
        update={"price_input_per_1m": 1.00, "price_output_per_1m": 5.00}
    )
    with observability.request("Test", "req-2") as record:
        for settings in (cheap, strong):
            invoke_llm_structured(
                settings,
                factory_with(raw({"ok": True}, 1_000_000, 100_000)),
                PROMPT,
                Out,
                stage="Unit",
                input_data={},
            )
    assert record.summary()["cost_usd"] == pytest.approx(0.15 + 1.50)


def test_cost_is_none_without_configured_prices():
    _, record = call(factory_with(raw({"ok": True})))
    assert record.summary()["cost_usd"] is None


def test_prices_come_from_per_path_environment(monkeypatch):
    from atlas_ai.feedback.extractor import feedback_settings_from_env

    monkeypatch.setenv("CONSULTATION_PRICE_INPUT_PER_1M_USD", "1.00")
    monkeypatch.setenv("CONSULTATION_PRICE_OUTPUT_PER_1M_USD", "5.00")
    monkeypatch.setenv("FEEDBACK_PRICE_INPUT_PER_1M_USD", "0.10")
    monkeypatch.setenv("FEEDBACK_PRICE_OUTPUT_PER_1M_USD", "0.50")
    consultation = LLMSettings.from_env()
    feedback = feedback_settings_from_env()
    assert consultation.cost(1_000_000, 1_000_000) == pytest.approx(6.00)
    assert feedback.cost(1_000_000, 1_000_000) == pytest.approx(0.60)


def test_plain_mock_output_still_works_without_usage():
    result, record = call(factory_with({"ok": True}))
    assert result.ok and record.llm_calls[0]["input_tokens"] is None


# --- per-RPC record over real gRPC -------------------------------------------------------


def test_rpc_record_has_request_id_status_spans_and_no_text(
    tmp_path, monkeypatch, feedback_request_factory
):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATLAS_TRACE_FILE", str(trace))
    secret_text = "Rotinya keras rahasia pelanggan"
    payload = {
        "opinions": [
            {
                "opinion_text": "Roti keras",
                "evidence_text": "Rotinya keras",
                "sentiment": "NEGATIVE",
                "severity": "NOT_SEVERE",
                "variant": None,
                "issue_title_hint": "Roti Terlalu Keras",
            }
        ]
    }
    embedder = Mock()
    embedder.embed.side_effect = lambda texts, **k: [[1.0] + [0.0] * 383 for _ in texts]
    pipeline = FeedbackPipeline(
        extractor=FeedbackExtractor(SETTINGS, factory_with(raw(payload, 800, 90))),
        embedder=embedder,
    )
    service = AtlasAIService()
    service.feedback_pipeline = pipeline
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server, _, _ = create_server(
        "tok", f"127.0.0.1:{port}", service=service, max_workers=1
    )
    server.start()
    request = feedback_request_factory(secret_text)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = ai_pb2_grpc.AIServiceStub(channel)
            stub.ProcessFeedback(
                request,
                timeout=30,
                metadata=(("authorization", "Bearer tok"), ("x-request-id", "job-42")),
            )
            bad = ai_pb2.ProcessFeedbackRequest(shop_id=str(uuid4()))
            with pytest.raises(grpc.RpcError):
                stub.ProcessFeedback(
                    bad,
                    timeout=30,
                    metadata=(
                        ("authorization", "Bearer tok"),
                        ("x-request-id", "job-43"),
                    ),
                )
    finally:
        server.stop(grace=0).wait(timeout=5)
    lines = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    ok, failed = lines
    assert ok["rpc"] == "ProcessFeedback" and ok["request_id"] == "job-42"
    assert ok["status"] == "OK" and ok["subject_id"] == request.feedback.id
    assert ok["tokens"] == {"input": 800, "output": 90}
    names = [s["name"] for s in ok["spans"]]
    assert {"extraction", "llm.extraction", "embedding", "matching"} <= set(names)
    assert failed["request_id"] == "job-43" and failed["status"] == "INVALID_ARGUMENT"
    text = trace.read_text(encoding="utf-8")
    assert secret_text not in text and "Rahasia Pelanggan" not in text
