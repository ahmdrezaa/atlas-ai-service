"""Evidence integrity on the authenticated wire, independently of language policy."""

import copy
import json
from concurrent import futures
from pathlib import Path
from unittest.mock import Mock

import grpc
import pytest
from google.protobuf.json_format import ParseDict
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai.consultation.assessment import ContextAssessor, LLMSettings
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.server import BearerAuthInterceptor

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = json.loads(
    (ROOT / "evaluation/m6_3_cases.json").read_text(encoding="utf-8")
)["scenarios"]


@pytest.fixture
def synthetic_turn():
    scenario = copy.deepcopy(SCENARIOS["queue"])
    return (
        ParseDict(scenario["request"], ai_pb2.ProcessConsultationTurnRequest()),
        scenario["solution"],
    )


@pytest.fixture
def mock_rpc():
    """Mock only the provider; retain auth, adapter, pipeline and final validators."""
    servers = []

    def call(request, payload):
        assessment = {
            "decision": "GENERATE_SOLUTION",
            "reason": "Synthetic sufficient context",
            "problem_evidence_sufficient": True,
            "missing_operational_information": [],
            "relevant_business_constraints": [],
            "additional_question_material": False,
            "repeats_existing_information": False,
        }
        factory = Mock()
        factory.return_value.with_structured_output.side_effect = lambda schema, **kw: (
            Mock(
                invoke=Mock(
                    return_value=copy.deepcopy(
                        payload
                        if schema["title"] == "GeneratedSolutionTurn"
                        else assessment
                    )
                )
            )
        )
        service = AtlasAIService()
        service.consultation_pipeline = ConsultationPipeline(
            ContextAssessor(
                LLMSettings(model="synthetic-mock", api_key=SecretStr("test-only")),
                factory,
            )
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2),
            interceptors=[BearerAuthInterceptor("test-only")],
        )
        ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port
        servers.append(server)
        server.start()
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            response = ai_pb2_grpc.AIServiceStub(channel).ProcessConsultationTurn(
                request, timeout=30, metadata=(("authorization", "Bearer test-only"),)
            )
        return response, factory.call_count

    yield call
    for server in servers:
        server.stop(0).wait(timeout=5)


@pytest.mark.parametrize("event", [1, 2, 3, 4, 5])
def test_current_go_request_preserves_sources_and_wire_contract(
    synthetic_turn, mock_rpc, event
):
    request, payload = synthetic_turn
    request.event = event
    before_request = request.SerializeToString()
    before_context = prepare_context(request)
    before_payload = copy.deepcopy(payload)
    response, calls = mock_rpc(request, payload)
    assert response.WhichOneof("result") == "solution"
    assert all(
        getattr(response.solution, f.name) for f in response.solution.DESCRIPTOR.fields
    )
    assert (
        ai_pb2.ProcessConsultationTurnResponse.FromString(response.SerializeToString())
        == response
    )
    assert response.model_id == "synthetic-mock"
    assert response.prompt_version
    assert not response.memory_candidates
    assert calls == (1 if event == ai_pb2.CONSULTATION_EVENT_GENERATE_NOW else 2)
    assert request.SerializeToString() == before_request
    assert prepare_context(request) == before_context
    assert payload == before_payload


@pytest.mark.parametrize("event", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "namespace", ["opinion_ids", "owner_message_ids", "memory_ids"]
)
def test_valid_source_id_cannot_support_a_fabricated_factual_finding(
    synthetic_turn, mock_rpc, event, namespace
):
    request, payload = synthetic_turn
    request.event = event
    context = prepare_context(request)
    registry = {
        "opinion_ids": context.sources.opinions,
        "owner_message_ids": context.sources.owner_messages,
        "memory_ids": context.sources.memories,
    }[namespace]
    source_id = next(iter(registry))
    prefix = {
        "opinion_ids": "Pelanggan menyebut: ",
        "owner_message_ids": "Owner menyatakan: ",
        "memory_ids": "",
    }[namespace]
    payload["findings"][0].update(
        statement=prefix + "Mesin kasir rusak.", references={namespace: [source_id]}
    )
    # The ID is current and belongs to the RIGHT namespace; content proof must block.
    assert source_id in registry
    before = request.SerializeToString()
    with pytest.raises(grpc.RpcError) as error:
        mock_rpc(request, payload)
    assert error.value.code() == grpc.StatusCode.INTERNAL
    assert "findings[0].statement" in error.value.details()
    assert request.SerializeToString() == before


@pytest.mark.parametrize(
    "attack",
    [
        "customer_negation",
        "quote",
        "quote_negation",
        "operational_quantity",
        "sql",
        "wrong_namespace",
        "historical_ai",
        "historical_content_with_current_owner_id",
    ],
)
def test_objective_evidence_attacks_stay_blocking_on_wire(
    synthetic_turn, mock_rpc, attack
):
    request, payload = synthetic_turn
    if attack == "customer_negation":
        evidence = request.context.evidence[0]
        evidence.evidence_text = evidence.full_feedback = "Antreannya tidak lama."
        payload["customer_evidence"][0]["quote"] = evidence.full_feedback
        payload["findings"][0]["statement"] = "Pelanggan menyebut: Antreannya lama."
    elif attack == "quote":
        payload["customer_evidence"][0]["quote"] = "Antrean cepat sekali."
    elif attack == "quote_negation":
        evidence = request.context.evidence[0]
        evidence.evidence_text = evidence.full_feedback = "Antreannya tidak lama."
        payload["findings"][0]["statement"] = (
            "Pelanggan menyebut: Antreannya tidak lama."
        )
        payload["customer_evidence"][0]["quote"] = "lama"
    elif attack == "operational_quantity":
        payload["findings"][0].update(
            statement="Jumlah kasir 3 orang.",
            references={"memory_ids": [request.context.memories[0].id]},
        )
    elif attack == "sql":
        payload["findings"][0]["statement"] = "Sebanyak 99 opini negatif."
    elif attack == "wrong_namespace":
        payload["findings"][0]["references"] = {
            "opinion_ids": [request.context.memories[0].id]
        }
    else:
        ai_id = "20000000-0000-4000-8000-000000000099"
        request.messages.insert(
            0,
            ai_pb2.ConversationMessage(
                id=ai_id,
                role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
                content="Mesin kasir rusak.",
            ),
        )
        payload["findings"][0].update(
            statement="Owner menyatakan: Mesin kasir rusak.",
            references={
                "owner_message_ids": [ai_id]
                if attack == "historical_ai"
                else [request.messages[-1].id]
            },
        )
        assert ai_id not in prepare_context(request).sources.owner_messages
    with pytest.raises(grpc.RpcError) as error:
        mock_rpc(request, payload)
    assert error.value.code() == grpc.StatusCode.INTERNAL


@pytest.mark.parametrize("candidate_source", ["valid_owner", "ai"])
def test_candidate_source_id_does_not_confirm_or_create_a_business_fact(
    synthetic_turn, mock_rpc, candidate_source
):
    request, payload = synthetic_turn
    owner_id = request.messages[-1].id
    ai_id = "20000000-0000-4000-8000-000000000099"
    request.messages.insert(
        0,
        ai_pb2.ConversationMessage(
            id=ai_id,
            role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
            content="Jumlah kasir 3 orang.",
        ),
    )
    payload["memory_candidates"] = [
        {
            "fact_text": "Jumlah kasir 3 orang.",
            "source_owner_message_ids": [
                owner_id if candidate_source == "valid_owner" else ai_id
            ],
        }
    ]
    before = request.SerializeToString()
    response, _ = mock_rpc(request, payload)
    assert response.HasField("solution")
    assert not response.memory_candidates
    assert request.SerializeToString() == before


@pytest.mark.parametrize("event", [1, 2, 3, 4, 5])
def test_nonblocking_relevance_does_not_downgrade_findings_on_wire(
    synthetic_turn, mock_rpc, event
):
    request, payload = synthetic_turn
    request.event = event
    payload["recommendations"][0]["description"] = "Evaluasi orbit satelit."
    response, calls = mock_rpc(request, payload)
    assert response.HasField("solution")
    assert calls == (1 if event == 4 else 2)
    assert response.solution.recommendations[0].description == "Evaluasi orbit satelit."
    payload["findings"][0]["statement"] = "Pelanggan menyebut: Mesin kasir rusak."
    with pytest.raises(grpc.RpcError) as error:
        mock_rpc(request, payload)
    assert error.value.code() == grpc.StatusCode.INTERNAL
    assert "findings[0].statement" in error.value.details()
