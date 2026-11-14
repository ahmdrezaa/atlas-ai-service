"""P-B1: advice is checked as advice; weakly grounded items are dropped, not fatal."""

from unittest.mock import Mock

import grpc
import pytest

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    AssessmentQuestion,
    ConsultationFailure,
    SufficiencyAssessment,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.consultation.solution import (
    GeneratedSolution,
    prune_solution,
    solution_to_proto,
)


def mapped(request, payload):
    return solution_to_proto(
        GeneratedSolution.model_validate(payload), prepare_context(request)
    )


# Real recommendation wording from the 2026-11-14 live probe; all were rejected
# by the former verb whitelist although none asserts a fact.
@pytest.mark.parametrize(
    "field,text",
    [
        (
            "description",
            "Usulan: Gunakan sendok takar atau timbangan digital kecil untuk memastikan porsi lebih konsisten.",
        ),
        (
            "description",
            "Periksa kembali standar takaran yang digunakan pada setiap pembuatan.",
        ),
        (
            "description",
            "Usulan: Buat lembar panduan takaran bumbu dan garam di area dapur.",
        ),
        (
            "description",
            "Usulan: Meninjau kembali alur pencatatan pesanan di kasir untuk mengurangi waktu tunggu.",
        ),
        (
            "description",
            "Usulan: mengatur jam istirahat kedua kasir agar keduanya tetap berjaga pada jam sibuk.",
        ),
        (
            "description",
            "Mengatur ulang tata letak antrean tanpa penambahan staf.",
        ),
        ("title", "Standarisasi Cita Rasa"),
        ("title", "Gunakan Struk Fisik"),
        ("title", "Penjadwalan Istirahat Staf"),
    ],
)
def test_real_advice_is_accepted(solution_request, solution_payload, field, text):
    solution_payload["recommendations"][0][field] = text
    result = mapped(solution_request, solution_payload)
    assert getattr(result.recommendations[0], field) == text


@pytest.mark.parametrize(
    "field,text",
    [
        ("description", "Usulan: tambah barista baru pada shift sore."),
        ("description", "Usulan: lakukan pengadaan mesin kasir kedua."),
        ("description", "Atur alur antrean karena mesin kasir rusak."),
        ("description", "Atur alur antrean. Pendapatan meningkat."),
        ("description", "Atur alur antrean, hasilnya dijamin lebih cepat."),
        ("description", "Gunakan mesin cadangan yang tersedia."),
        ("description", "Evaluasi alur karena kasir lelah."),
        ("title", "Optimasi Kasir Saat Ini"),
        ("title", "Optimasi Kasir; pelanggan puas"),
    ],
)
def test_advice_still_cannot_assert_facts_resources_or_outcomes(
    solution_request, solution_payload, field, text
):
    # The single recommendation is dropped -> section empty -> turn rejected.
    solution_payload["recommendations"][0][field] = text
    with pytest.raises(ConsultationFailure) as error:
        mapped(solution_request, solution_payload)
    assert error.value.code == grpc.StatusCode.INTERNAL
    assert "recommendations[0]" in str(error.value)


@pytest.mark.parametrize(
    "description",
    [
        # Live probe 2026-11-14 (bakery_filling START): numbered alternatives.
        (
            "Lakukan penimbangan isian coklat pada setiap pembuatan.\n"
            "Usulan alternatif 1: Memeriksa kembali standar resep.\n"
            "Usulan alternatif 2: Meminta masukan staf dapur."
        ),
        # Live probe (cafe_packaging): "rusak" as a risk, not an asserted state.
        "Usulan: uji ketahanan kemasan terhadap risiko terbuka atau rusak selama dibawa pulang.",
        "Usulan: tambahkan stiker segel agar kemasan tidak rusak.",
    ],
)
def test_enumerations_and_risks_are_not_facts(
    solution_request, solution_payload, description
):
    solution_payload["recommendations"][0]["description"] = description
    result = mapped(solution_request, solution_payload)
    assert result.recommendations[0].description == description


def test_hedged_cause_is_allowed_in_a_hypothesis(solution_request, solution_payload):
    # Live probe (owner_constraints): "mungkin disebabkan oleh ...".
    text = (
        "Hipotesis: keterlambatan layanan kasir pada sore hari mungkin disebabkan "
        "oleh beban antrean yang padat."
    )
    solution_payload["hypotheses"][0]["statement"] = text
    assert mapped(solution_request, solution_payload).hypotheses[0].statement == text


@pytest.mark.parametrize(
    "section,field,text",
    [
        ("recommendations", "description", "Usulan: perbaiki mesin kasir yang rusak."),
        ("hypotheses", "statement", "Hipotesis: antrean disebabkan kekurangan staf."),
        ("findings", "statement", "Antrean mungkin disebabkan kekurangan staf."),
    ],
)
def test_hedges_do_not_open_a_hole(
    solution_request, solution_payload, section, field, text
):
    solution_payload[section] = [{**solution_payload[section][0], field: text}]
    with pytest.raises(ConsultationFailure):
        mapped(solution_request, solution_payload)


def test_untentative_hypothesis_is_dropped_when_another_remains(
    solution_request, solution_payload
):
    # Live probe (cafe_packaging): the second hypothesis had no hedge word.
    bad = "Hipotesis: metode penutupan kemasan belum sepenuhnya rapat atau aman."
    solution_payload["hypotheses"].append(
        {**solution_payload["hypotheses"][0], "statement": bad}
    )
    result = mapped(solution_request, solution_payload)
    assert [h.statement for h in result.hypotheses] == [
        solution_payload["hypotheses"][0]["statement"]
    ]


def test_negated_verification_word_is_tentative(solution_request, solution_payload):
    text = (
        "Hipotesis: alur pesanan mungkin belum terverifikasi ulang sebelum disiapkan."
    )
    solution_payload["hypotheses"][0]["statement"] = text
    assert mapped(solution_request, solution_payload).hypotheses[0].statement == text


def test_invalid_note_is_blanked_not_fatal(solution_request, solution_payload):
    solution_payload["customer_evidence"][0]["note"] = (
        "Menyatakan durasi antrean 45 menit."
    )
    result = mapped(solution_request, solution_payload)
    assert result.customer_evidence[0].note == ""
    assert (
        result.customer_evidence[0].quote
        == solution_payload["customer_evidence"][0]["quote"]
    )


def test_unsupported_item_is_dropped_when_others_remain(
    solution_request, solution_payload
):
    solution_payload["findings"].insert(
        0,
        {
            "statement": "Pelanggan menyebut: kasir tidak ramah.",
            "references": solution_payload["findings"][0]["references"],
        },
    )
    pruned, dropped = prune_solution(
        GeneratedSolution.model_validate(solution_payload),
        prepare_context(solution_request),
    )
    statements = [item.statement for item in pruned.findings]
    assert "Pelanggan menyebut: kasir tidak ramah." not in statements
    assert len(statements) == 2
    assert dropped[0][0] == "findings[0].statement"


def test_unsupported_sole_item_rejects_with_claim_id(
    solution_request, solution_payload
):
    solution_payload["hypotheses"][0]["statement"] = (
        "Hipotesis: antrean mungkin lama karena mesin kasir rusak."
    )
    with pytest.raises(ConsultationFailure) as error:
        mapped(solution_request, solution_payload)
    assert "hypotheses[0].statement" in str(error.value)
    assert "hypotheses empty" in str(error.value)


def test_fabrication_still_rejects_the_whole_output(solution_request, solution_payload):
    # A second, valid evidence item does not save an answer that invents a quote.
    solution_payload["customer_evidence"].append(
        dict(solution_payload["customer_evidence"][0])
    )
    solution_payload["customer_evidence"][0]["quote"] = "Pelanggan tidak menulis ini."
    with pytest.raises(ConsultationFailure, match="verbatim"):
        mapped(solution_request, solution_payload)


def invalid_question_assessment():
    return SufficiencyAssessment(
        decision="ASK_QUESTION",
        reason="Gap",
        problem_evidence_sufficient=True,
        missing_operational_information=["Budget"],
        relevant_business_constraints=[],
        additional_question_material=True,
        repeats_existing_information=False,
        # Two questions in one turn: rejected by validate_question.
        question=AssessmentQuestion(
            topic_key="budget", text="Berapa anggaran? Berapa staf?"
        ),
    )


def test_invalid_question_falls_back_to_solution_when_evidence_exists(
    solution_request, solution_assessor, monkeypatch
):
    solution_request.event = ai_pb2.CONSULTATION_EVENT_START
    monkeypatch.setattr(
        solution_assessor, "assess", Mock(return_value=invalid_question_assessment())
    )
    response = ConsultationPipeline(solution_assessor).run(solution_request)
    assert response.WhichOneof("result") == "solution"


def test_invalid_question_without_evidence_still_fails(
    request_factory, solution_assessor, monkeypatch
):
    monkeypatch.setattr(
        solution_assessor, "assess", Mock(return_value=invalid_question_assessment())
    )
    with pytest.raises(ConsultationFailure) as error:
        ConsultationPipeline(solution_assessor).run(request_factory())
    assert error.value.code == grpc.StatusCode.INTERNAL
