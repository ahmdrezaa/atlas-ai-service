"""B2: one LLM call drops off-topic advice; failures fail open."""

import copy
from unittest.mock import Mock

import grpc
import pytest
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import ContextAssessor, LLMSettings
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.relevance import (
    RELEVANCE_PROMPT,
    RelevanceSettings,
    check_relevance,
)
from atlas_ai.consultation.solution import GeneratedSolution
from atlas_ai.llm import AIFailure

GEMINI = LLMSettings(model="mock-gemini", api_key=SecretStr("test-only"))
ON = RelevanceSettings(enabled=True, timeout_seconds=10)
BITCOIN = {
    "title": "Evaluasi Harga Bitcoin",
    "description": "Usulan: pantau harga Bitcoin setiap minggu.",
    "priority": "low",
}


def judge(verdicts=None, exc=None):
    factory = Mock()
    invoke = factory.return_value.with_structured_output.return_value.invoke
    if exc is not None:
        invoke.side_effect = exc
    else:
        invoke.return_value = {"verdicts": verdicts or []}
    return factory


def with_bitcoin(payload):
    payload = copy.deepcopy(payload)
    payload["recommendations"].append(
        {**BITCOIN, "references": payload["recommendations"][0]["references"]}
    )
    return GeneratedSolution.model_validate(payload)


def test_off_topic_recommendation_is_dropped(solution_request, solution_payload):
    factory = judge([{"item_id": "recommendations[1]", "relevant": False}])
    solution, status, dropped = check_relevance(
        with_bitcoin(solution_payload),
        prepare_context(solution_request),
        GEMINI,
        factory,
        ON,
        time_remaining=30,
    )
    assert status == "ok" and dropped == ["recommendations[1]"]
    assert [r.title for r in solution.recommendations] == [
        solution_payload["recommendations"][0]["title"]
    ]
    # Its own short timeout, never more than the configured relevance budget.
    assert 9 < factory.call_args.kwargs["timeout"] <= 10


def test_all_recommendations_off_topic_rejects_the_turn(
    solution_request, solution_payload
):
    factory = judge(
        [
            {"item_id": "recommendations[0]", "relevant": False},
            {"item_id": "recommendations[1]", "relevant": False},
        ]
    )
    with pytest.raises(AIFailure) as error:
        check_relevance(
            with_bitcoin(solution_payload),
            prepare_context(solution_request),
            GEMINI,
            factory,
            ON,
            time_remaining=30,
        )
    assert error.value.code == grpc.StatusCode.INTERNAL
    assert "off-topic" in str(error.value)


def test_unknown_item_ids_are_ignored(solution_request, solution_payload):
    factory = judge([{"item_id": "recommendations[9]", "relevant": False}])
    solution, _, dropped = check_relevance(
        with_bitcoin(solution_payload),
        prepare_context(solution_request),
        GEMINI,
        factory,
        ON,
        time_remaining=30,
    )
    assert dropped == [] and len(solution.recommendations) == 2


@pytest.mark.parametrize(
    "exc", [RuntimeError("provider down"), TimeoutError(), ValueError("bad")]
)
def test_provider_failure_fails_open(solution_request, solution_payload, exc):
    solution, status, _ = check_relevance(
        with_bitcoin(solution_payload),
        prepare_context(solution_request),
        GEMINI,
        judge(exc=exc),
        ON,
        time_remaining=30,
    )
    assert status == "failed" and len(solution.recommendations) == 2


def test_invalid_judge_output_fails_open(solution_request, solution_payload):
    factory = Mock()
    factory.return_value.with_structured_output.return_value.invoke.return_value = {
        "wrong": 1
    }
    _, status, _ = check_relevance(
        with_bitcoin(solution_payload),
        prepare_context(solution_request),
        GEMINI,
        factory,
        ON,
        time_remaining=30,
    )
    assert status == "failed"


def test_short_budget_and_disabled_make_no_call(solution_request, solution_payload):
    factory = judge()
    solution = with_bitcoin(solution_payload)
    context = prepare_context(solution_request)
    assert (
        check_relevance(solution, context, GEMINI, factory, ON, time_remaining=10)[1]
        == "skipped_budget"
    )
    off = RelevanceSettings(enabled=False)
    assert check_relevance(solution, context, GEMINI, factory, off)[1] == "disabled"
    factory.assert_not_called()


def test_cancellation_is_not_swallowed(solution_request, solution_payload):
    with pytest.raises(AIFailure) as error:
        check_relevance(
            with_bitcoin(solution_payload),
            prepare_context(solution_request),
            GEMINI,
            judge(),
            ON,
            time_remaining=30,
            is_active=lambda: False,
        )
    assert error.value.code == grpc.StatusCode.CANCELLED


def test_judge_sees_issue_evidence_and_items_as_data(
    solution_request, solution_payload
):
    factory = judge()
    check_relevance(
        with_bitcoin(solution_payload),
        prepare_context(solution_request),
        GEMINI,
        factory,
        ON,
        time_remaining=30,
    )
    prompt = factory.return_value.with_structured_output.return_value.invoke.call_args[
        0
    ][0]
    system, human = prompt.to_messages()
    assert "UNTRUSTED_RELEVANCE_JSON" in human.content
    assert "Bitcoin" in human.content and "recommendations[1]" in human.content
    assert "Antreannya lama sekali." in human.content  # customer evidence
    assert "bakery" in system.content and "Bitcoin" in system.content  # example


def test_prompt_format():
    messages = RELEVANCE_PROMPT.format_prompt(context_json="{}").to_messages()
    assert [m.type for m in messages] == ["system", "human"]


def test_settings_from_env(monkeypatch):
    monkeypatch.delenv("CONSULTATION_RELEVANCE_CHECK", raising=False)
    monkeypatch.delenv("CONSULTATION_RELEVANCE_TIMEOUT_SECONDS", raising=False)
    settings = RelevanceSettings.from_env()
    assert settings.enabled and settings.timeout_seconds == 10
    monkeypatch.setenv("CONSULTATION_RELEVANCE_TIMEOUT_SECONDS", "5")
    assert RelevanceSettings.from_env().timeout_seconds == 5  # no 10 s minimum
    monkeypatch.setenv("CONSULTATION_RELEVANCE_TIMEOUT_SECONDS", "0.5")
    with pytest.raises(AIFailure) as error:
        RelevanceSettings.from_env()
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION


def test_pipeline_drops_off_topic_advice_end_to_end(
    solution_request, solution_payload, monkeypatch
):
    monkeypatch.setenv("CONSULTATION_RELEVANCE_CHECK", "true")
    solution_request.event = ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
    payload = copy.deepcopy(solution_payload)
    payload["recommendations"].append(
        {**BITCOIN, "references": payload["recommendations"][0]["references"]}
    )
    factory = Mock()

    def structured(schema, **kwargs):
        result = Mock()
        if schema.get("title") == "RelevanceOutput":
            result.invoke.return_value = {
                "verdicts": [{"item_id": "recommendations[1]", "relevant": False}]
            }
        else:
            result.invoke.return_value = copy.deepcopy(payload)
        return result

    factory.return_value.with_structured_output.side_effect = structured
    response = ConsultationPipeline(ContextAssessor(GEMINI, factory)).run(
        solution_request, time_remaining=50
    )
    titles = [r.title for r in response.solution.recommendations]
    assert "Evaluasi Harga Bitcoin" not in titles and titles
    assert factory.call_count == 2  # solution + relevance (GENERATE_NOW skips assess)
