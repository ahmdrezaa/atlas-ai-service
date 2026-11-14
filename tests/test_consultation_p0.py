import json
from pathlib import Path
from unittest.mock import Mock

import grpc
import pytest
from google.protobuf.json_format import ParseDict
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import LLMSettings
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.evaluation import load_dataset
from atlas_ai.consultation.grounding import GroundingClaim, evaluate_claim
from atlas_ai.consultation.models import References
from scripts.consultation_live_readiness import run_case

ROOT = Path(__file__).resolve().parents[1]
_, CONTEXT, _ = load_dataset(ROOT / "evaluation/grounding_cases.json")
WAIT = References(opinion_ids=(CONTEXT.representative_evidence[0].opinion_id,))


@pytest.mark.parametrize(
    "text,accepted",
    [
        (
            "Hipotesis: antrean mungkin dipengaruhi alur pesanan; perlu diverifikasi.",
            True,
        ),
        ("Hipotesis: harga Bitcoin mungkin berubah besok.", False),
        ("Hipotesis: antrean mungkin lambat; harga Bitcoin mungkin berubah.", False),
        ("Hipotesis: antrean mungkin panjang karena keuntungan meningkat.", False),
        ("Hipotesis: antrean mungkin panjang dan toko memiliki mesin baru.", False),
        ("Hipotesis: antrean mungkin panjang. Mesin rusak.", False),
        (
            "Hipotesis: antrean mungkin panjang dan penyebabnya terbukti kekurangan staf.",
            False,
        ),
        ("Hipotesis: antrean mungkin panjang karena mesin rusak yang tersedia.", False),
    ],
)
def test_hypothesis_needs_cited_relevance_and_no_new_factual_premise(text, accepted):
    verdict = evaluate_claim(
        GroundingClaim(claim_id="h", kind="hypothesis", text=text, references=WAIT),
        CONTEXT,
    )
    assert (verdict.label == "NOT_APPLICABLE") is accepted


def test_valid_reference_from_irrelevant_source_is_not_hypothesis_relevance():
    bread = References(opinion_ids=(CONTEXT.representative_evidence[1].opinion_id,))
    verdict = evaluate_claim(
        GroundingClaim(
            claim_id="h",
            kind="hypothesis",
            text="Hipotesis: antrean mungkin panjang.",
            references=bread,
        ),
        CONTEXT,
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "kind,text,accepted",
    [
        ("recommendation", "Lakukan pencatatan antrean.", True),
        ("recommendation", "Sebaiknya owner mengamati antrean.", True),
        ("recommendation", "Lakukan pencatatan karena keuntungan meningkat.", False),
        (
            "recommendation",
            "Lakukan pencatatan dengan mesin yang sudah tersedia.",
            False,
        ),
        ("recommendation", "Lakukan pencatatan. Mesin rusak.", False),
        (
            "recommendation",
            "Usulan: lakukan pencatatan karena semua pelanggan puas.",
            False,
        ),
        ("metric", "Catatan durasi transaksi kasir harian", True),
        ("metric", "Waktu antrean per pesanan", True),
        ("metric", "Keuntungan yang telah meningkat", False),
        ("metric", "Antrean berkurang", False),
        ("target", "Usulan: Terkumpulnya data hambatan alur transaksi kasir", True),
        ("target", "Usulan: jumlah transaksi meningkat", True),
        ("target", "Usulan: hasil sudah tercapai", False),
        ("target", "Usulan: saat ini keuntungan besar", False),
        ("target", "Usulan: data terkumpul karena pemasukan meningkat", False),
        ("timeframe", "Usulan: Selama 7 hari pengamatan", True),
        ("timeframe", "Usulan: setelah satu minggu observasi", True),
        ("timeframe", "Usulan: sudah selesai selama 7 hari", False),
        ("evidence_note", "Keluhan antrean.", True),
        ("evidence_note", "Keluhan roti enak.", False),
        ("evidence_note", "Antreannya lama sekali.", True),
        ("evidence_note", "Keluhan pelanggan terkait waktu tunggu antrean", False),
    ],
)
def test_field_roles_allow_proposals_but_not_unsupported_facts(kind, text, accepted):
    decision = evaluate_claim(
        GroundingClaim(claim_id="p", kind=kind, text=text, references=WAIT), CONTEXT
    )
    assert (decision.label in {"SUPPORTED", "NOT_APPLICABLE"}) is accepted


def test_c5_four_false_rejections_fixed_two_notes_remain_unresolved():
    ledger = json.loads(
        (ROOT / "evaluation/functional_live_readiness.json").read_text(encoding="utf-8")
    )
    row = next(r for r in ledger["runs"] if r["id"] == "C5")
    context = prepare_context(
        ParseDict(row["synthetic_request"], ai_pb2.ProcessConsultationTurnRequest())
    )
    from atlas_ai.consultation.grounding import solution_claims
    from atlas_ai.consultation.solution import GeneratedSolution

    result = GeneratedSolution.model_validate_json(
        row["provider_responses"][0]["visible_text"]
    )
    decisions = {
        c.claim_id: evaluate_claim(c, context) for c in solution_claims(result)
    }
    for key in (
        "recommendations[0].description",
        "success_indicators[0].metric",
        "success_indicators[0].target",
        "success_indicators[0].timeframe",
    ):
        assert decisions[key].label == "NOT_APPLICABLE"
    for i in range(2):
        assert (
            decisions[f"customer_evidence[{i}].note"].label == "INSUFFICIENT_EVIDENCE"
        )


def test_development_headroom_is_checked_before_provider_access(monkeypatch):
    factory = Mock(side_effect=AssertionError("No API request without headroom"))
    monkeypatch.setattr(
        "scripts.consultation_live_readiness.Capture.model_factory", factory
    )
    row = run_case(
        "solution",
        LLMSettings(model="mock", api_key=SecretStr("test"), timeout_seconds=45),
        rpc_timeout=50,
    )
    assert row["status"] == "FAILED"
    assert row["grpc_status_equivalent"] == grpc.StatusCode.FAILED_PRECONDITION.name
    factory.assert_not_called()


def test_saved_live_output_revalidation_is_not_reported_as_a_new_live_success():
    from copy import deepcopy

    from scripts.consultation_live_readiness import revalidate_saved_live_solution

    ledger = json.loads(
        (ROOT / "evaluation/p0_solution_live.json").read_text(encoding="utf-8")
    )
    row = next(row for row in ledger["runs"] if row["id"] == "P1")
    before = deepcopy(row)
    gate = revalidate_saved_live_solution([row])
    assert gate["original_status"] == "FAILED"
    assert gate["type"].endswith("NOT_NEW_LIVE_SUCCESS")
    assert gate["used_as_rpc_response"] is False
    assert row == before
    row["provider_responses"][0]["visible_text"] = "{}"
    assert revalidate_saved_live_solution([row]) is None
    row = deepcopy(before)
    row["http_events"] = [{"http_status": 504}]
    assert revalidate_saved_live_solution([row]) is None
