import json

import grpc
import httpx
import pytest
from pydantic import ValidationError

from atlas_ai.consultation.assessment import ConsultationFailure
from scripts.gemini_live_diagnostics import (
    Capture,
    DiagnosticReply,
    classify_error,
    synthetic_request,
)


@pytest.mark.parametrize(
    "status,category,grpc_status",
    [
        (504, "PROVIDER_HTTP_504", "DEADLINE_EXCEEDED"),
        (503, "PROVIDER_HTTP_ERROR", "UNAVAILABLE"),
        (429, "RATE_LIMIT", "RESOURCE_EXHAUSTED"),
        (400, "PROVIDER_HTTP_ERROR", "UNAVAILABLE"),
    ],
)
def test_http_error_is_not_client_timeout(status, category, grpc_status):
    result = classify_error(
        RuntimeError("private-provider-message"), [{"http_status": status}]
    )
    assert result["category"] == category
    assert result["grpc_status_equivalent"] == grpc_status
    assert "private" not in json.dumps(result)


def test_client_read_timeout_without_http_response_is_separate():
    result = classify_error(httpx.ReadTimeout("private-key-in-url"), [])
    assert result["category"] == "CLIENT_TIMEOUT"
    assert not result["exception_chain"][0]["http_code"]
    assert "private" not in json.dumps(result)


def test_nested_sdk_error_code_is_preserved_without_message():
    class SDKServerError(Exception):
        code = 504

    original = SDKServerError("private-provider-body")
    error = ConsultationFailure(
        grpc.StatusCode.DEADLINE_EXCEEDED, "Safe application message"
    )
    error.__cause__ = original
    result = classify_error(error, [])
    assert result["category"] == "PROVIDER_HTTP_504"
    assert result["exception_chain"][-1]["http_code"] == 504
    assert "private" not in json.dumps(result)


def test_rate_limit_preserves_current_application_status_mapping():
    result = classify_error(
        ConsultationFailure(grpc.StatusCode.UNAVAILABLE, "safe"), [{"http_status": 429}]
    )
    assert result["category"] == "RATE_LIMIT"
    assert result["grpc_status_equivalent"] == "UNAVAILABLE"
    assert result["grpc_status_source"] == "application_exception"


def test_parsing_error_is_not_application_grounding_or_provider_failure():
    try:
        DiagnosticReply.model_validate({"status": "invalid-output"})
    except ValidationError as error:
        result = classify_error(error, [{"http_status": 200}])
    assert result["category"] == "STRUCTURED_OUTPUT_PARSE_ERROR"
    assert "invalid-output" not in json.dumps(result)


@pytest.mark.parametrize(
    "code,category",
    [
        (grpc.StatusCode.INTERNAL, "APPLICATION_VALIDATION"),
        (grpc.StatusCode.DEADLINE_EXCEEDED, "APPLICATION_DEADLINE"),
        (grpc.StatusCode.CANCELLED, "APPLICATION_FAILURE"),
    ],
)
def test_application_error_is_distinguished_from_transport(code, category):
    result = classify_error(ConsultationFailure(code, "safe"), [{"http_status": 200}])
    assert result["category"] == category


def test_response_hook_does_not_record_query_string_headers_or_key():
    response = httpx.Response(
        503,
        request=httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1/models/example:generateContent?key=private-key",
            headers={"x-goog-api-key": "private-key"},
        ),
    )
    capture = Capture()
    capture.response_hook(response)
    assert capture.http_events == [
        {
            "http_status": 503,
            "host": "generativelanguage.googleapis.com",
            "path": "/v1/models/example:generateContent",
        }
    ]
    assert "private-key" not in json.dumps(capture.http_events)


def test_diagnostic_request_is_synthetic_and_valid():
    from atlas_ai.consultation.context import prepare_context

    context = prepare_context(synthetic_request())
    assert context.shop.name == "Kedai sintetis diagnostik"
    assert len(context.sources.opinions) == 1
    assert not context.conversation_history
