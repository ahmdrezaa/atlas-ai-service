import json
from concurrent import futures
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import grpc
import pytest
from httpx import ReadTimeout
from openai_offline import OfflineOpenAI
from pydantic import SecretStr, ValidationError
from solution_policy import rejected_or_absent

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    LLMSettings,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.prompts import SOLUTION_PROMPT_VERSION
from atlas_ai.consultation.solution import GeneratedSolution, solution_to_proto


def test_valid_solution_all_events(
    solution_request, solution_payload, solution_assessor
):
    before = solution_request.SerializeToString()
    response = ConsultationPipeline(solution_assessor).run(solution_request)
    assert response.WhichOneof("result") == "solution"
    assert all(len(getattr(response.solution, field)) for field in solution_payload)
    assert response.solution.findings[1].statement == "Jumlah barista 2 orang."
    assert (
        response.solution.customer_evidence[0].quote
        == solution_payload["customer_evidence"][0]["quote"]
    )
    assert response.model_id == "mock-gemini"
    assert response.prompt_version == SOLUTION_PROMPT_VERSION
    assert not response.memory_candidates
    assert solution_request.SerializeToString() == before
    expected = (
        1 if solution_request.event == ai_pb2.CONSULTATION_EVENT_GENERATE_NOW else 2
    )
    assert solution_assessor.model_factory.call_count == expected
    assert (
        ai_pb2.ProcessConsultationTurnResponse.FromString(response.SerializeToString())
        == response
    )


@pytest.mark.parametrize(
    "section,field,bad",
    [
        ("customer_evidence", "quote", "Pelanggan tidak pernah menulis ini."),
        ("customer_evidence", "quote", " "),
        ("customer_evidence", "opinion_id", "missing"),
        ("findings", "statement", "Snapshot mencatat [[stat:opinion_count=11]]."),
        ("findings", "statement", "Snapshot mencatat [[stat:negative_count=10]]."),
        ("findings", "statement", "Sebanyak 10 opini mengeluh."),
        ("findings", "statement", "Jumlah [[stat:unknown=10]]."),
        ("findings", "statement", "Jumlah [[stat:opinion_count=NaN]]."),
        ("findings", "statement", "Antrean disebabkan kekurangan staf."),
        ("hypotheses", "statement", "Kekurangan staf terbukti menjadi penyebab."),
        (
            "hypotheses",
            "statement",
            "Hipotesis: mungkin kekurangan staf terbukti menjadi penyebab.",
        ),
        ("hypotheses", "confidence", "0.99"),
        ("recommendations", "priority", "90%"),
        ("recommendations", "description", "Hasil sudah meningkat 20%."),
        ("success_indicators", "target", "Telah meningkat 20%"),
        ("success_indicators", "timeframe", "Sudah tercapai minggu lalu"),
        ("success_indicators", "metric", " "),
        ("findings", "statement", " "),
    ],
)
def test_invalid_output_rejected(
    solution_request, solution_payload, section, field, bad
):
    solution_payload[section][0][field] = bad
    before = len(solution_payload[section])
    outcome = rejected_or_absent(
        lambda: solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        ),
        bad,
    )
    if isinstance(outcome, ai_pb2.Solution):
        # P-B1: the invalid item was dropped (never rewritten); the rest remains.
        assert len(getattr(outcome, section)) == before - 1


@pytest.mark.parametrize(
    "namespace", ["opinion_ids", "owner_message_ids", "memory_ids"]
)
def test_reference_namespaces_not_interchangeable(
    solution_request, solution_payload, namespace
):
    # A valid UUID from a different namespace must not pass.
    wrong = (
        solution_request.context.memories[0].id
        if namespace == "opinion_ids"
        else solution_request.context.evidence[0].opinion_id
    )
    solution_payload["findings"][0]["references"] = {namespace: [wrong]}
    with pytest.raises(ConsultationFailure, match="namespace"):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )


def test_owner_and_memory_references_and_numeric_excerpt(
    solution_request, solution_payload
):
    if (
        not solution_request.messages
        or solution_request.messages[-1].role != ai_pb2.MESSAGE_ROLE_OWNER
    ):
        solution_request.messages.add(
            id=str(uuid4()),
            role=ai_pb2.MESSAGE_ROLE_OWNER,
            content="Shift sore 2 orang.",
        )
    owner = solution_request.messages[-1]
    owner.content = "Shift sore 2 orang."
    solution_payload["findings"][0] = {
        "statement": "Owner menyatakan: Shift sore 2 orang.",
        "references": {"owner_message_ids": [owner.id]},
    }
    solution_payload["recommendations"][0]["references"] = {
        "memory_ids": [solution_request.context.memories[0].id]
    }
    result = solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    assert result.findings[0].references.owner_message_ids == [owner.id]


@pytest.mark.parametrize(
    "field",
    [
        "findings",
        "customer_evidence",
        "hypotheses",
        "recommendations",
        "success_indicators",
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_all_sections_required(solution_payload, field, missing):
    if missing:
        del solution_payload[field]
    else:
        solution_payload[field] = []
    with pytest.raises(ValidationError):
        GeneratedSolution.model_validate(solution_payload)


def test_generation_no_evidence_does_not_invoke_provider(request_factory):
    assessor = ContextAssessor(LLMSettings(), Mock())
    request = request_factory(ai_pb2.CONSULTATION_EVENT_GENERATE_NOW)
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(assessor).run(request)
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assessor.model_factory.assert_not_called()


def test_stage_timings_are_request_local_and_include_validation_and_mapping(
    solution_request, solution_assessor
):
    timings = {}
    response = ConsultationPipeline(solution_assessor).run(
        solution_request, timings=timings
    )
    assert response.WhichOneof("result") == "solution"
    for stage in (
        "solution.langchain_invoke_and_parse",
        "solution.pydantic",
        "solution.structural",
        "solution.grounding",
        "solution.protobuf_mapping",
    ):
        assert timings[stage] >= 0
    assert all(isinstance(value, float) for value in timings.values())


@pytest.mark.parametrize("with_reference", [False, True])
def test_mvp_forbids_standalone_sql_even_with_a_real_but_unrelated_reference(
    solution_request, solution_payload, with_reference
):
    solution_payload["findings"][0]["statement"] = (
        "Snapshot mencatat [[stat:opinion_count=10]]."
    )
    if not with_reference:
        solution_payload["findings"][0]["references"] = {}
    rejected_or_absent(
        lambda: solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        ),
        "[[stat:",
        "Snapshot mencatat",
    )


def test_optional_evidence_note_can_be_omitted_without_inventing_an_explanation(
    solution_request, solution_payload
):
    del solution_payload["customer_evidence"][0]["note"]
    response = solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    assert response.customer_evidence[0].note == ""


@pytest.mark.parametrize(
    "result,code",
    [
        (None, grpc.StatusCode.INTERNAL),
        ({}, grpc.StatusCode.INTERNAL),
        ("bad", grpc.StatusCode.INTERNAL),
        (ReadTimeout("private"), grpc.StatusCode.DEADLINE_EXCEEDED),
        (RuntimeError("private"), grpc.StatusCode.UNAVAILABLE),
    ],
)
def test_generation_errors_not_invalid_argument(solution_request, result, code):
    solution_request.event = ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
    factory = Mock()
    invoke = factory.return_value.with_structured_output.return_value.invoke
    if isinstance(result, Exception):
        invoke.side_effect = result
    else:
        invoke.return_value = result
    assessor = ContextAssessor(
        LLMSettings(model="mock", api_key=SecretStr("test")), factory
    )
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(assessor).run(solution_request)
    assert error.value.code == code
    assert "private" not in str(error.value)


def test_revision_uses_latest_owner_but_historical_ids_are_not_sources(
    solution_request, solution_payload
):
    solution_request.event = ai_pb2.CONSULTATION_EVENT_CONTINUE_DISCUSSION
    solution_request.question_count = 8  # session cap (G2) forces generation
    solution_request.messages.add(
        id=str(uuid4()),
        role=ai_pb2.MESSAGE_ROLE_OWNER,
        content="Revisi agar biaya rendah.",
    )
    old_id = str(uuid4())
    solution_request.previous_solution.findings.add(
        statement="Claim lama"
    ).references.opinion_ids.append(old_id)
    factory = Mock()
    factory.return_value.with_structured_output.return_value.invoke.return_value = (
        solution_payload
    )
    assessor = ContextAssessor(
        LLMSettings(model="mock", api_key=SecretStr("test")), factory
    )
    ConsultationPipeline(assessor).run(solution_request)
    prompt = (
        factory.return_value.with_structured_output.return_value.invoke.call_args.args[
            0
        ]
    )
    messages = prompt.to_messages()
    data = json.loads(messages[-1].content.split("\n", 1)[1])
    assert data["latest_owner_message"]["content"] == "Revisi agar biaya rendah."
    assert data["previous_solution"]["source_kind"] == "historical_ai_output"
    assert factory.call_count == 1
    solution_payload["findings"][0]["references"] = {"opinion_ids": [old_id]}
    with pytest.raises(ConsultationFailure):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )


def test_solution_on_real_grpc_wire(solution_request, solution_assessor):
    from atlas.ai.v1 import ai_pb2_grpc
    from atlas_ai.grpc_service import AtlasAIService

    service = AtlasAIService()
    service.consultation_pipeline = ConsultationPipeline(solution_assessor)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            response = ai_pb2_grpc.AIServiceStub(channel).ProcessConsultationTurn(
                solution_request, timeout=30
            )
            assert response.WhichOneof("result") == "solution"
            assert response.prompt_version == SOLUTION_PROMPT_VERSION
            assert len(response.solution.customer_evidence) == 1
    finally:
        server.stop(0).wait(timeout=5)


def test_real_sdk_structured_solution_offline(solution_request, solution_payload):
    solution_request.event = ai_pb2.CONSULTATION_EVENT_GENERATE_NOW
    provider = OfflineOpenAI(solution_payload)
    assessor = ContextAssessor(
        LLMSettings(model="offline-test", api_key=SecretStr("test-only")), provider
    )
    response = ConsultationPipeline(assessor).run(solution_request)
    assert response.HasField("solution")
    (request,) = provider.requests
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in request


def test_total_deadline_shared_between_both_calls(
    solution_request, solution_assessor, monkeypatch
):
    solution_request.event = ai_pb2.CONSULTATION_EVENT_START
    clock = [0.0]
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr("atlas_ai.consultation.pipeline.time", timer)
    monkeypatch.setattr("atlas_ai.llm.time", timer)

    def build(**kwargs):
        clock[0] += 10
        return solution_assessor.model_factory.return_value

    solution_assessor.model_factory.side_effect = build
    ConsultationPipeline(solution_assessor).run(solution_request, time_remaining=30)
    calls = solution_assessor.model_factory.call_args_list
    assert calls[0].kwargs["timeout"] == pytest.approx(29.75)
    assert calls[1].kwargs["timeout"] == pytest.approx(19.75)


def test_statistic_missing_and_unavailable_references(
    solution_request, solution_payload
):
    solution_request.context.ClearField("stats")
    # Statistics are optional context, never an output-only dependency.
    assert solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    ).findings


def test_zero_and_decimal_statistics_are_context_only(
    solution_request, solution_payload
):
    solution_request.context.stats.neutral_count = 0
    solution_payload["findings"][1]["statement"] = (
        "Snapshot: [[stat:neutral_count=0]], [[stat:average_rating=3.5]]."
    )
    rejected_or_absent(
        lambda: solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        ),
        "[[stat:",
        "Snapshot:",
    )


def test_full_feedback_quote_allowed_but_normalized_opinion_is_not_source_quote(
    solution_request, solution_payload
):
    solution_payload["customer_evidence"][0]["quote"] = "Rotinya enak."
    solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    solution_payload["customer_evidence"][0]["quote"] = (
        solution_request.context.evidence[0].opinion_text
    )
    with pytest.raises(ConsultationFailure, match="verbatim"):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )


@pytest.mark.parametrize("kind", ["memory", "owner"])
def test_sourced_operational_numbers_are_not_ai_targets(
    solution_request, solution_payload, kind
):
    fact = "Jam ramai pukul 17.00 sampai 19.00 setiap hari."
    if kind == "memory":
        source = solution_request.context.memories.add(id=str(uuid4()), fact_text=fact)
        refs = {"memory_ids": [source.id]}
    else:
        source = solution_request.messages.add(
            id=str(uuid4()), role=ai_pb2.MESSAGE_ROLE_OWNER, content=fact
        )
        refs = {"owner_message_ids": [source.id]}
    action = solution_payload["recommendations"][0]
    action["references"] = refs
    action["description"] = "Amati antrean pada jam ramai 17.00 sampai 19.00."
    response = solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    assert response.recommendations[0].description == action["description"]
    assert not action["description"].startswith("Usulan:")


@pytest.mark.parametrize(
    "description,accepted",
    [
        (
            "Catat durasi transaksi kasir secara manual pada jam 17.00 hingga 19.00 tanpa membeli alat baru.",
            True,
        ),
        ("Catat durasi transaksi kasir pada jam 18.00 hingga 19.00.", False),
        ("Catat durasi transaksi kasir pada jam buka 17.00 hingga 19.00.", False),
        ("Catat durasi transaksi kasir pada jam 19.00 hingga 17.00.", False),
        ("Catat durasi transaksi kasir pukul 17.00–19.00.", True),
        ("Catat durasi transaksi kasir jam 17:00-19:00.", True),
        ("Catat durasi transaksi kasir pukul 18.00–19.00.", False),
        ("Usulan: catat durasi transaksi kasir pada jam 18.00 hingga 19.00.", True),
    ],
)
def test_action_schedule_reuses_a_cited_window_without_changing_its_meaning(
    solution_request, solution_payload, description, accepted
):
    memory = solution_request.context.memories.add(
        id=str(uuid4()), fact_text="Jam ramai pukul 17.00 sampai 19.00 setiap hari."
    )
    action = solution_payload["recommendations"][0]
    action["description"] = description
    action["references"] = {"memory_ids": [memory.id]}
    if accepted:
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )
    else:
        with pytest.raises(ConsultationFailure):
            solution_to_proto(
                GeneratedSolution.model_validate(solution_payload),
                prepare_context(solution_request),
            )


@pytest.mark.parametrize(
    "description,wrong_source",
    [
        ("Amati antrean pada jam ramai 18.00 sampai 19.00.", False),
        ("Amati antrean pada jam buka 17.00 sampai 19.00.", False),
        ("Amati antrean dengan jumlah barista 17 orang.", False),
        ("Amati antrean pada jam ramai 17.00 sampai 19.00.", True),
        ("Usulan: amati antrean pada jam ramai 18.00 sampai 19.00.", False),
        ("Amati antrean pada jam ramai 17.00 sampai 19.00 selama 99 hari.", False),
    ],
)
def test_operational_numbers_require_same_proposition_and_correct_citation(
    solution_request, solution_payload, description, wrong_source
):
    memory = solution_request.context.memories.add(
        id=str(uuid4()), fact_text="Jam ramai pukul 17.00 sampai 19.00 setiap hari."
    )
    action = solution_payload["recommendations"][0]
    action["description"] = description
    action["references"] = {
        "memory_ids": [
            solution_request.context.memories[0].id if wrong_source else memory.id
        ]
    }
    with pytest.raises(ConsultationFailure):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )


def test_unavailable_operational_reference_is_not_inferred_from_other_memories(
    solution_request, solution_payload
):
    action = solution_payload["recommendations"][0]
    action["description"] = "Amati antrean dengan jumlah barista 2 orang."
    action["references"] = {"memory_ids": [str(uuid4())]}
    with pytest.raises(ConsultationFailure, match="namespace"):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )


def test_cited_staff_count_and_proposed_target_remain_distinct(
    solution_request, solution_payload
):
    action = solution_payload["recommendations"][0]
    action["references"] = {"memory_ids": [solution_request.context.memories[0].id]}
    action["description"] = "Amati antrean dengan jumlah barista 2 orang."
    indicator = solution_payload["success_indicators"][0]
    indicator["target"] = "Usulan: durasi antrean di bawah 5 menit"
    indicator["timeframe"] = "Usulan: evaluasi setelah 7 hari"
    result = solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    assert result.success_indicators[0].target == indicator["target"]


def test_target_staffing_is_a_proposal_not_a_claim_about_existing_staff(
    solution_request, solution_payload
):
    solution_payload["success_indicators"][0]["target"] = (
        "Usulan: jumlah barista 3 orang"
    )
    result = solution_to_proto(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    assert result.success_indicators[0].target.endswith("3 orang")


def test_verified_sql_cannot_bypass_go_nonempty_reference_requirement(
    solution_request, solution_payload
):
    solution_payload["findings"][1]["references"] = {}
    # P-B1: an uncited finding is dropped, so Go never receives empty references.
    result = rejected_or_absent(
        lambda: solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        ),
        solution_payload["findings"][1]["statement"],
    )
    if isinstance(result, ai_pb2.Solution):
        assert all(
            f.references.opinion_ids
            or f.references.owner_message_ids
            or f.references.memory_ids
            for f in result.findings
        )


@pytest.mark.parametrize(
    "fact",
    [
        "Jam ramai pukul 17.00 sampai 19.00 belum diketahui.",
        "Bukan jam ramai pukul 17.00 sampai 19.00.",
        "Jam ramai bukan pukul 17.00 sampai 19.00.",
        "Usulan: jam ramai pukul 17.00 sampai 19.00.",
    ],
)
@pytest.mark.parametrize("with_positive", [False, True])
def test_uncertain_or_conflicting_operational_sources_do_not_prove_a_fact(
    solution_request, solution_payload, fact, with_positive
):
    uncertain = solution_request.context.memories.add(id=str(uuid4()), fact_text=fact)
    refs = [uncertain.id]
    if with_positive:
        positive = solution_request.context.memories.add(
            id=str(uuid4()), fact_text="Jam ramai pukul 17.00 sampai 19.00."
        )
        refs.append(positive.id)
    action = solution_payload["recommendations"][0]
    action["description"] = "Amati antrean pada jam ramai 17.00 sampai 19.00."
    action["references"] = {"memory_ids": refs}
    with pytest.raises(ConsultationFailure, match="operational number"):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )


@pytest.mark.parametrize(
    "statement",
    [
        "Jumlah opini 99.",
        "Jumlah feedback unik 99.",
        "Jumlah opini sembilan puluh sembilan.",
        "Jumlah feedback unik sembilan puluh sembilan.",
    ],
)
def test_owner_or_memory_cannot_override_issue_statistics(
    solution_request, solution_payload, statement
):
    memory = solution_request.context.memories.add(id=str(uuid4()), fact_text=statement)
    solution_payload["findings"][0] = {
        "statement": statement,
        "references": {"memory_ids": [memory.id]},
    }
    rejected_or_absent(
        lambda: solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        ),
        statement,
    )


@pytest.mark.parametrize(
    "quote,accepted",
    [
        ("Antreannya lama", True),
        ("lama sekali", True),
        ("Antrean lama", False),
        ("treannya lama", False),
        ("Antreannya cepat", False),
    ],
)
def test_quote_may_be_shorter_verbatim_excerpt(
    solution_request, solution_payload, quote, accepted
):
    solution_payload["customer_evidence"][0]["quote"] = quote
    output = GeneratedSolution.model_validate(solution_payload)
    if accepted:
        assert (
            solution_to_proto(output, prepare_context(solution_request))
            .customer_evidence[0]
            .quote
            == quote
        )
    else:
        with pytest.raises(ConsultationFailure, match="verbatim"):
            solution_to_proto(output, prepare_context(solution_request))


def test_quote_cannot_drop_customer_negation(solution_request, solution_payload):
    opinion = solution_request.context.evidence[0]
    opinion.full_feedback = "Antreannya tidak lama."
    opinion.evidence_text = (
        "lama."  # Even a cropped evidence_text cannot erase wider context.
    )
    solution_payload["customer_evidence"][0]["quote"] = "lama."
    with pytest.raises(ConsultationFailure, match="verbatim"):
        solution_to_proto(
            GeneratedSolution.model_validate(solution_payload),
            prepare_context(solution_request),
        )
