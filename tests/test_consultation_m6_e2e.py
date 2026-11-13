"""Offline fixtures and real authenticated gRPC; all provider outputs are mocks."""

import copy
import json
from datetime import datetime
from unittest.mock import Mock

import pytest
from httpx import ReadTimeout
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import GeminiSettings
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.solution import GeneratedSolution, solution_to_proto
from scripts.consultation_live_readiness import run_case
from scripts.consultation_m6_e2e import (
    MAX_ATTEMPTS,
    OWNER_FACT,
    OWNER_FACT_ID,
    attempt_allowed,
    context_observations,
    finish_rpc_row,
    historical_solution,
    sanitize_row,
    scenario_request,
)
from scripts.gemini_live_diagnostics import Capture


@pytest.fixture
def m6_payload():
    refs = {
        "opinion_ids": ["10000000-0000-4000-8000-000000000001"],
        "owner_message_ids": [],
        "memory_ids": [],
    }
    return {
        "findings": [
            {"statement": "Pelanggan menyebut antrean lama.", "references": refs}
        ],
        "customer_evidence": [
            {
                "opinion_id": refs["opinion_ids"][0],
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
        "memory_candidates": [],
    }


@pytest.fixture
def m6_previous(m6_payload):
    return solution_to_proto(
        GeneratedSolution.model_validate(
            {k: v for k, v in m6_payload.items() if k != "memory_candidates"}
        ),
        prepare_context(scenario_request("D")),
    )


def mock_boundary(
    monkeypatch, payload, *, question=False, failure=None, http_status=None
):
    """Mock just the model factory, retaining native server/adapter/pipeline guards."""
    calls = []

    def factory(capture, **kwargs):
        def structured(schema, **unused):
            def invoke(*args, **unused_config):
                calls.append(schema["title"])
                capture.requests.append({"mock": True})
                capture.invocations.append(
                    {"schema_title": schema["title"], "mock": True}
                )
                if http_status:
                    capture.http_events.append({"http_status": http_status})
                if failure:
                    capture.invocations[-1]["exception_type"] = type(failure).__name__
                    raise failure
                if schema["title"] == "GeneratedSolutionTurn":
                    result = copy.deepcopy(payload)
                else:
                    result = {
                        "decision": "ASK_QUESTION" if question else "GENERATE_SOLUTION",
                        "reason": "Operational gap"
                        if question
                        else "Sufficient context",
                        "problem_evidence_sufficient": True,
                        "missing_operational_information": ["order_recording"]
                        if question
                        else [],
                        "relevant_business_constraints": [],
                        "additional_question_material": question,
                        "repeats_existing_information": False,
                        "question": {
                            "topic_key": "order_recording",
                            "text": "Bagaimana pesanan dicatat di kasir?",
                        }
                        if question
                        else None,
                    }
                capture.responses.append(
                    {"visible_text": json.dumps(result), "usage": None}
                )
                return result

            return Mock(invoke=invoke)

        return Mock(with_structured_output=structured)

    monkeypatch.setattr(Capture, "model_factory", factory)
    return calls


def settings():
    return GeminiSettings(model="mock-not-live", api_key=SecretStr("test-only"))


@pytest.mark.parametrize("scenario", list("ABCDE"))
def test_synthetic_scenarios_preserve_contract_and_owner_order(scenario, m6_previous):
    request = scenario_request(scenario, m6_previous if scenario == "E" else None)
    before = request.SerializeToString()
    observation = context_observations(request)
    assert observation["history_order_preserved"]
    assert observation["ai_ids_excluded_from_owner_sources"]
    assert observation["latest_owner_history_occurrences"] == (
        1 if scenario in "BDE" else 0
    )
    if scenario == "E":
        assert observation["previous_solution_source_kind"] == "historical_ai_output"
    if scenario == "C":
        assert observation["skipped_topics"] == ["staffing"]
    assert request.SerializeToString() == before


def test_continue_requires_actual_supplied_previous_solution():
    with pytest.raises(ValueError, match="previously validated"):
        scenario_request("E")


@pytest.mark.parametrize("scenario", list("ABCDE"))
def test_m6_authenticated_grpc_solution_all_events_empty_memory_is_valid(
    monkeypatch, m6_payload, m6_previous, scenario
):
    calls = mock_boundary(monkeypatch, m6_payload)
    request = scenario_request(scenario, m6_previous if scenario == "E" else None)
    row = finish_rpc_row(
        run_case(
            "grpc",
            settings(),
            request=request,
            allowed_results={"solution", "next_question"},
        ),
        request,
    )
    assert row["actual_grpc_status"] == "OK"
    assert row["status"] == "PASSED"
    assert row["all_response_validations"] == "PASSED"
    assert row["memory_validation"]["returned_count"] == 0
    assert row["memory_validation"]["empty_is_valid"]
    assert len(calls) == (1 if scenario == "D" else 2)
    assert all(row["response_checks"].values())
    assert all(row["validation_reached"].values())


@pytest.mark.parametrize("scenario", list("ABCE"))
def test_m6_grpc_next_question_allowed_without_repeated_topics(
    monkeypatch, m6_payload, m6_previous, scenario
):
    calls = mock_boundary(monkeypatch, m6_payload, question=True)
    request = scenario_request(scenario, m6_previous if scenario == "E" else None)
    row = finish_rpc_row(
        run_case(
            "grpc",
            settings(),
            request=request,
            allowed_results={"solution", "next_question"},
        ),
        request,
    )
    assert row["status"] == "PASSED"
    assert row["result_kind"] == "next_question"
    assert calls == ["SufficiencyAssessment"]
    assert row["all_response_validations"] == "PASSED"


@pytest.mark.parametrize("invalid", [False, True])
def test_m6_grpc_memory_optional_valid_owner_and_invalid_ai_sources(
    monkeypatch, m6_payload, invalid
):
    source = "90000000-0000-4000-8000-000000000001" if invalid else OWNER_FACT_ID
    m6_payload["memory_candidates"] = [
        {"fact_text": OWNER_FACT, "source_owner_message_ids": [source]}
    ]
    calls = mock_boundary(monkeypatch, m6_payload)
    request = scenario_request("D")
    request.messages.insert(
        0,
        ai_pb2.ConversationMessage(
            id="90000000-0000-4000-8000-000000000001",
            role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
            content=OWNER_FACT,
        ),
    )
    row = finish_rpc_row(run_case("grpc", settings(), request=request), request)
    assert row["status"] == "PASSED" and len(calls) == 1
    assert row["memory_validation"]["returned_count"] == (0 if invalid else 1)
    assert row["memory_validation"]["owner_sources_valid"]
    assert len(request.context.memories) == 5


@pytest.mark.parametrize(
    "http_status,code,category",
    [
        (503, "UNAVAILABLE", "PROVIDER_HTTP_503"),
        (504, "DEADLINE_EXCEEDED", "PROVIDER_HTTP_504"),
        (429, "UNAVAILABLE", "PROVIDER_RATE_LIMIT"),
        (None, "DEADLINE_EXCEEDED", "CLIENT_TIMEOUT"),
    ],
)
def test_failed_rpc_keeps_real_status_duration_and_sanitized_classification(
    monkeypatch, m6_payload, http_status, code, category
):
    error = (
        ReadTimeout("private provider body")
        if http_status is None
        else type("ProviderError", (Exception,), {"code": http_status})(
            "private provider body"
        )
    )
    mock_boundary(monkeypatch, m6_payload, failure=error, http_status=http_status)
    request = scenario_request("D")
    row = finish_rpc_row(run_case("grpc", settings(), request=request), request)
    assert row["actual_grpc_status"] == code
    assert row["category"] == category
    assert row["rpc_duration_seconds"] > 0
    assert datetime.fromisoformat(row["rpc_ended_at"]) >= datetime.fromisoformat(
        row["rpc_started_at"]
    )
    assert not row["validation_reached"]["solution_structural"]
    assert "private provider body" not in json.dumps(row)


def test_local_pydantic_rejection_distinct_from_provider_failure(
    monkeypatch, m6_payload
):
    m6_payload["findings"] = []
    mock_boundary(monkeypatch, m6_payload)
    request = scenario_request("D")
    row = finish_rpc_row(run_case("grpc", settings(), request=request), request)
    assert row["actual_grpc_status"] == "INTERNAL"
    assert row["category"] == "LOCAL_PYDANTIC_REJECTION"
    assert row["all_response_validations"] == "NOT_PASSED"


def test_artifact_removes_raw_provider_body_and_retains_usage_and_digest():
    result = sanitize_row(
        {
            "provider_errors": [{"message_redacted": "raw-body"}],
            "application_error": "body",
            "provider_responses": [
                {"visible_text": "synthetic", "usage": {"total_token_count": 10}}
            ],
        }
    )
    assert "raw-body" not in json.dumps(result)
    assert "visible_text" not in json.dumps(result)
    assert result["provider_responses"][0]["usage"]["total_token_count"] == 10
    assert len(result["provider_responses"][0]["output_sha256"]) == 64


def test_budget_and_responsiveness_gates_do_not_automatically_switch_or_retry():
    assert attempt_allowed([], "D", "prod", None, None)
    assert not attempt_allowed([], "A", "prod", None, None)
    failed = {
        "scenario": "D",
        "model": "prod",
        "actual_gemini_invocations": 1,
        "status": "FAILED",
        "category_of_test": "C_REAL_PROVIDER_AUTHENTICATED_GRPC",
    }
    assert not attempt_allowed([failed], "D", "prod", None, None)
    assert not attempt_allowed([failed], "A", "prod", None, None)
    assert attempt_allowed([failed], "D", "explicit-comparator", None, None)
    assert not attempt_allowed([failed] * MAX_ATTEMPTS, "D", "another", None, None)


def test_http_200_grounding_rejection_allows_other_scenarios_but_is_not_a_pass():
    row = {
        "scenario": "D",
        "model": "comparator",
        "actual_gemini_invocations": 1,
        "status": "FAILED",
        "category": "GROUNDING_REJECTION",
        "category_of_test": "C_REAL_PROVIDER_AUTHENTICATED_GRPC",
        "http_events": [{"http_status": 200}],
    }
    assert attempt_allowed([row], "A", "comparator", None, None)
    assert row["status"] == "FAILED"
    unavailable = {**row, "scenario": "A", "http_events": [{"http_status": 503}]}
    assert not attempt_allowed([row, unavailable], "B", "comparator", None, None)


def test_previous_live_solution_is_input_only_and_requires_actual_grpc_success():
    from pathlib import Path

    ledger = json.loads(
        (
            Path(__file__).resolve().parents[1] / "evaluation/p0_solution_live.json"
        ).read_text(encoding="utf-8")
    )
    previous, run_id = historical_solution(ledger["runs"], "gemini-3.6-flash", "P3")
    request = scenario_request("E", previous)
    assert run_id == "P3" and request.previous_solution == previous
    assert (
        prepare_context(request).previous_solution.source_kind == "historical_ai_output"
    )
    with pytest.raises(ValueError, match="historical input"):
        historical_solution(ledger["runs"], "gemini-3.6-flash", "P1")
