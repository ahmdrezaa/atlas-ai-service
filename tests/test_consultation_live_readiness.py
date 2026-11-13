import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import httpx
import pytest
from google.protobuf.json_format import ParseDict
from pydantic import ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
    SufficiencyAssessment,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.grounding import evaluate_claim, solution_claims
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.prompts import ASSESSMENT_SYSTEM_PROMPT, PROMPT_VERSION
from atlas_ai.consultation.solution import (
    GeneratedSolution,
    solution_provider_schema,
    validate_solution,
)
from scripts.consultation_live_readiness import case_request
from scripts.gemini_live_diagnostics import Capture


def test_fixture_has_one_unknown_operational_gap_and_real_source_ids():
    start = prepare_context(case_request("start"))
    solution = prepare_context(case_request("solution"))
    assert len(start.representative_evidence) == 2
    assert all(
        source.opinion_id in start.sources.opinions
        for source in start.representative_evidence
    )
    assert not any(
        "Jumlah barista" in memory.fact_text for memory in start.confirmed_memories
    )
    assert any(
        "Jumlah barista dua orang" in memory.fact_text
        for memory in solution.confirmed_memories
    )
    assert len(solution.sources.owner_messages) == 1
    assert solution.question_count == 0


def test_assessment_prompt_clarifies_single_variable_without_relaxing_guard():
    assert PROMPT_VERSION == "consultation-assessment-v4"
    assert "ONLY ONE" in ASSESSMENT_SYSTEM_PROMPT
    assert "peak_hours_and_staffing" in ASSESSMENT_SYSTEM_PROMPT
    assert "One question mark is not enough" in ASSESSMENT_SYSTEM_PROMPT


def test_provider_schema_omits_grammar_bounds_but_local_schema_keeps_them():
    provider = solution_provider_schema()
    assert set(provider["required"]) == set(GeneratedSolution.model_fields)
    assert provider["properties"]["findings"]["items"]["type"] == "object"
    serialized = json.dumps(provider)
    for keyword in (
        '"$ref"',
        '"$defs"',
        '"minItems"',
        '"maxItems"',
        '"minLength"',
        '"maxLength"',
        '"default"',
    ):
        assert keyword not in serialized
    local = GeneratedSolution.model_json_schema()
    assert local["properties"]["findings"]["minItems"] == 1
    assert local["properties"]["recommendations"]["maxItems"] == 3
    with pytest.raises(ValidationError):
        GeneratedSolution.model_validate(
            {key: [] for key in GeneratedSolution.model_fields}
        )


def test_schema_contains_no_empty_string_enum():
    def inspect(value):
        if isinstance(value, dict):
            assert "" not in value.get("enum", [])
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(solution_provider_schema())


def test_null_is_only_an_unspecified_label_adapter():
    from atlas_ai.consultation.solution import GeneratedHypothesis

    result = GeneratedHypothesis.model_validate(
        {
            "statement": "Hipotesis: penyebab belum diketahui.",
            "confidence": None,
            "references": {},
        }
    )
    assert result.confidence == ""
    with pytest.raises(ValidationError):
        GeneratedHypothesis.model_validate(
            {"statement": result.statement, "confidence": "90%", "references": {}}
        )


def test_provider_error_instrumentation_redacts_keys():
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST",
            "https://example.com/generate?key=query-secret",
            headers={"x-goog-api-key": "header-secret"},
        ),
        json={"error": {"message": "Rejected query-secret and header-secret"}},
    )
    capture = Capture()
    capture.response_hook(response)
    assert "query-secret" not in json.dumps(capture.provider_errors)
    assert "header-secret" not in json.dumps(capture.provider_errors)
    assert "[REDACTED]" in capture.provider_errors[0]["message_redacted"]


def test_bounds_and_numeric_labels_remain_enforced_after_provider_adaptation():
    # Reuse the existing fixture-independent historical shape, then exercise local
    # output checks on dicts returned by the native JSON schema parser.
    from atlas_ai.consultation.solution import (
        GeneratedHypothesis,
        GeneratedRecommendation,
    )

    base = {"statement": "x" * 3001, "confidence": None, "references": {}}
    with pytest.raises(ValidationError):
        GeneratedHypothesis.model_validate(base)
    for bad in (0.9, "0.9", "90%"):
        with pytest.raises(ValidationError):
            GeneratedRecommendation.model_validate(
                {
                    "title": "Amati antrean",
                    "description": "",
                    "priority": bad,
                    "references": {},
                }
            )


def test_question_validation_time_cannot_escape_shared_turn_budget(monkeypatch):
    import atlas_ai.consultation.pipeline as module

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    assessor = Mock(spec=ContextAssessor)
    assessor.settings = GeminiSettings(model="mock")
    assessor.assess.return_value = SufficiencyAssessment(
        decision="ASK_QUESTION",
        reason="Staff count unknown",
        problem_evidence_sufficient=True,
        missing_operational_information=["staff count"],
        relevant_business_constraints=[],
        additional_question_material=True,
        repeats_existing_information=False,
        question={"topic_key": "staffing", "text": "Berapa staf yang melayani kasir?"},
    )
    original = module.validate_question

    def validate_then_expire(*args):
        result = original(*args)
        clock[0] = 31
        return result

    monkeypatch.setattr(module, "validate_question", validate_then_expire)
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(assessor).run(case_request("start"), time_remaining=60)
    assert error.value.code == grpc.StatusCode.DEADLINE_EXCEEDED


def test_live_grpc_cannot_run_before_both_direct_paths_pass(tmp_path, monkeypatch):
    import scripts.consultation_live_readiness as module

    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "runs": [
                    {"id": "C1", "case": "start", "status": "PASSED"},
                    {"id": "C5", "case": "solution", "status": "FAILED"},
                ]
            }
        ),
        encoding="utf-8",
    )
    env = Mock(
        side_effect=AssertionError("No environment/model access before prerequisites")
    )
    monkeypatch.setattr(module, "load_environment", env)
    monkeypatch.setattr(
        "sys.argv",
        ["readiness", "--output", str(ledger), "--case", "grpc", "--id", "C6"],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    env.assert_not_called()


def test_c5_numeric_false_rejection_fixed_without_whitelisting_all_its_language():
    ledger = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "evaluation/functional_live_readiness.json"
        ).read_text(encoding="utf-8")
    )
    row = next(row for row in ledger["runs"] if row["id"] == "C5")
    original = row["provider_responses"][0]["visible_text"]
    result = GeneratedSolution.model_validate_json(original)
    request = ParseDict(
        row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest()
    )
    context = prepare_context(request)
    # The original output is neither rewritten nor counted as a live success.
    with pytest.raises(ConsultationFailure, match="no current sources"):
        validate_solution(result, context)
    rejected = {
        claim.claim_id
        for claim in solution_claims(result)
        if evaluate_claim(claim, context).label not in {"SUPPORTED", "NOT_APPLICABLE"}
    }
    assert rejected == {
        "customer_evidence[0].note",
        "customer_evidence[1].note",
    }
    assert "17.00 sampai 19.00" in result.recommendations[0].description
    assert not result.recommendations[0].description.startswith("Usulan:")
    assert original == row["provider_responses"][0]["visible_text"]


def test_local_model_still_rejects_provider_omitted_constraints():
    ledger = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "evaluation/functional_live_readiness.json"
        ).read_text(encoding="utf-8")
    )
    row = next(row for row in ledger["runs"] if row["id"] == "C5")
    original = json.loads(row["provider_responses"][0]["visible_text"])
    from copy import deepcopy

    for mutate in (
        lambda data: data.update(unexpected="field"),
        lambda data: data["findings"][0].update(unexpected="field"),
        lambda data: data.update(recommendations=data["recommendations"] * 4),
        lambda data: data["customer_evidence"][0].update(quote="x" * 3001),
        lambda data: data["recommendations"][0].update(priority=0.99),
        lambda data: data["findings"][0].update(references={"opinion_ids": [123]}),
    ):
        data = deepcopy(original)
        mutate(data)
        with pytest.raises(ValidationError):
            GeneratedSolution.model_validate(data)


def test_4c1_live_grpc_requires_direct_solution_and_has_no_retries(
    tmp_path, monkeypatch
):
    import scripts.consultation_live_readiness as module

    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"milestone": "4C.1", "runs": []}), encoding="utf-8")
    env = Mock(side_effect=AssertionError("No API access before prerequisites"))
    monkeypatch.setattr(module, "load_environment", env)
    monkeypatch.setattr(
        "sys.argv",
        [
            "readiness",
            "--output",
            str(ledger),
            "--milestone",
            "4C.1",
            "--case",
            "grpc",
            "--id",
            "A2",
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    env.assert_not_called()
