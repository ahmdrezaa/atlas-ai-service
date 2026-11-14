"""One bounded Solution regeneration when grounding empties a section."""

import copy
import json
from unittest.mock import Mock

import grpc
import pytest
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2
from atlas_ai import observability
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    LLMSettings,
)
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.solution import revision_notes

SECRET = "rotinya beracun dan berbahaya"


def unsupported(payload):
    """Every finding unsupported by its cited source: the section empties."""
    bad = copy.deepcopy(payload)
    refs = bad["findings"][0]["references"]
    bad["findings"] = [
        {"statement": f"Pelanggan menyebut {SECRET}.", "references": refs}
    ]
    return bad


FAKE_QUOTE = "Kutipan yang tidak pernah ditulis."
FAKE_ID = "00000000-0000-4000-8000-000000000000"


def fabricated(payload):
    bad = copy.deepcopy(payload)
    bad["customer_evidence"][0]["quote"] = FAKE_QUOTE
    return bad


def invented_id(payload):
    bad = copy.deepcopy(payload)
    # references are shared between items in the fixture: replace, do not mutate.
    bad["findings"][0]["references"] = {
        **bad["findings"][0]["references"],
        "opinion_ids": [FAKE_ID],
    }
    return bad


def assessor_with(*drafts):
    """Solution calls return the drafts in order; records each prompt's JSON."""
    factory = Mock()
    seen: list[dict] = []
    queue = list(drafts)

    def structured(schema, **kwargs):
        def invoke(prompt, **unused):
            seen.append(json.loads(prompt.to_messages()[-1].content.split("\n", 1)[1]))
            return copy.deepcopy(queue.pop(0))

        return Mock(invoke=invoke)

    factory.return_value.with_structured_output.side_effect = structured
    assessor = ContextAssessor(
        LLMSettings(model="mock", api_key=SecretStr("test-only")), factory
    )
    return assessor, seen


@pytest.fixture
def generate_now(solution_request):
    solution_request.event = ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
    return solution_request


def test_rejected_draft_is_regenerated_once_with_codes_only(
    generate_now, solution_payload
):
    assessor, seen = assessor_with(unsupported(solution_payload), solution_payload)
    with observability.request("Test", "repair") as record:
        response = ConsultationPipeline(assessor).run(generate_now, time_remaining=45)
    assert response.HasField("solution")
    first, second = seen
    assert "revision_notes" not in first
    notes = second["revision_notes"]
    assert notes == [
        {
            "item": "findings[0].statement",
            "reason": "INSUFFICIENT_EVIDENCE:no_bounded_source_proof_for_clause",
        }
    ]
    # Only item keys and reason codes: never generated or source text.
    assert SECRET not in json.dumps(notes)
    assert [call["stage"] for call in record.llm_calls] == [
        "solution",
        "solution_repair",
    ]


def test_second_rejection_fails_the_turn(generate_now, solution_payload):
    assessor, seen = assessor_with(
        unsupported(solution_payload), unsupported(solution_payload)
    )
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(assessor).run(generate_now, time_remaining=45)
    assert error.value.code == grpc.StatusCode.INTERNAL
    assert len(seen) == 2


@pytest.mark.parametrize(
    "make_bad,item,reason",
    [
        (fabricated, "customer_evidence[0]", "quote_not_verbatim"),
        (invented_id, "findings[0]", "reference_id_unavailable"),
    ],
)
def test_fabricated_draft_is_discarded_and_rewritten_once(
    generate_now, solution_payload, make_bad, item, reason
):
    assessor, seen = assessor_with(make_bad(solution_payload), solution_payload)
    response = ConsultationPipeline(assessor).run(generate_now, time_remaining=45)
    assert response.HasField("solution")
    assert seen[1]["revision_notes"] == [{"item": item, "reason": reason}]
    # Nothing from the fabricated draft reaches the response or the notes.
    sent = str(response) + json.dumps(seen[1]["revision_notes"])
    assert FAKE_QUOTE not in sent and FAKE_ID not in sent


def test_fabrication_twice_fails_the_turn(generate_now, solution_payload):
    assessor, seen = assessor_with(
        fabricated(solution_payload), invented_id(solution_payload)
    )
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(assessor).run(generate_now, time_remaining=45)
    assert error.value.code == grpc.StatusCode.INTERNAL
    assert "namespace" in str(error.value)
    assert len(seen) == 2


def test_fabrication_without_budget_is_not_rewritten(generate_now, solution_payload):
    assessor, seen = assessor_with(fabricated(solution_payload), solution_payload)
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(assessor).run(generate_now, time_remaining=11)
    assert "not verbatim" in str(error.value)
    assert len(seen) == 1


def test_no_regeneration_without_budget(generate_now, solution_payload):
    assessor, seen = assessor_with(unsupported(solution_payload), solution_payload)
    with pytest.raises(ConsultationFailure):
        ConsultationPipeline(assessor).run(generate_now, time_remaining=11)
    assert len(seen) == 1


def test_regeneration_can_be_disabled(generate_now, solution_payload, monkeypatch):
    monkeypatch.setenv("CONSULTATION_SOLUTION_REPAIR", "false")
    assessor, seen = assessor_with(unsupported(solution_payload), solution_payload)
    with pytest.raises(ConsultationFailure):
        ConsultationPipeline(assessor).run(generate_now, time_remaining=45)
    assert len(seen) == 1


def test_revision_notes_are_sanitized_and_bounded():
    notes = revision_notes(
        [("findings[0].statement", 'X:y "injected <text>" {json}')] * 30
    )
    assert len(notes) == 20
    assert all(set(n["reason"]) <= set("XY:y injected text json") for n in notes)
