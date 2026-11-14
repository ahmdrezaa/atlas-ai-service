"""P-B3: checker false positives found on OpenAI output (2026-11-14 probe).

Each fix accepts one proven, faithful pattern; the matching unsafe variants stay
rejected. Texts are synthetic bakery examples.
"""

import pytest

from atlas_ai.consultation import grounding as g
from atlas_ai.consultation.grounding import GroundingClaim, GroundingLabel
from atlas_ai.consultation.models import References


def fact(text, source_text, kind="customer"):
    claim = GroundingClaim(
        claim_id="findings[0].statement",
        kind="finding",
        text=text,
        references=References(opinion_ids=("o1",)),
    )
    return g._fact(claim, [{"kind": kind, "text": source_text}])


# --- 1. negation never crosses a line/sentence boundary of a source -----------------

JOINED = (
    "Rasa rotinya tidak konsisten\nRasa rotinya tidak konsisten, kadang terlalu asin."
)


def test_exact_quote_after_a_repeated_line_is_supported():
    label, _ = fact(
        "Pelanggan menyebut: rasa rotinya tidak konsisten, kadang terlalu asin.", JOINED
    )
    assert label == GroundingLabel.SUPPORTED


@pytest.mark.parametrize(
    "claim,source",
    [
        # A real negation inside the same sentence still blocks the affirmation.
        ("Pelanggan menyebut: rotinya keras.", "Rotinya tidak keras."),
        # A paraphrase is still not proof.
        (
            "Pelanggan menyebut: pesanan yang datang berbeda dari yang diminta.",
            "Pesanan yang datang berbeda dari yang saya minta.",
        ),
        # Words never written by the customer.
        ("Pelanggan menyebut: rotinya beracun.", JOINED),
    ],
)
def test_unsupported_or_negated_findings_stay_rejected(claim, source):
    label, _ = fact(claim, source)
    assert label != GroundingLabel.SUPPORTED


def test_contradiction_across_lines_is_still_detected():
    # A known opposite pair split over two lines of one source is still a conflict.
    label, reason = fact(
        "Pelanggan menyebut: antreannya cepat.", "Antreannya cepat.\nAntreannya lama."
    )
    assert label != GroundingLabel.SUPPORTED
    assert reason == "conflicting_sources"


# --- 2. "penyebabnya belum diketahui" disclaims a cause --------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "penyebabnya belum diketahui",
        "penyebab pasti masih belum diketahui",
        "penyebabnya tidak jelas",
    ],
)
def test_cause_disclaimers_are_removed_before_certainty_checks(text):
    assert not g._asserts(g._CERTAINTY, g._without_disclaimers(text))


@pytest.mark.parametrize(
    "text",
    [
        "penyebabnya adalah oven yang rusak",
        "penyebabnya sudah diketahui: oven rusak",
        "keluhan ini disebabkan oleh staf baru",
        "terbukti bahwa adonan terlalu lama",
    ],
)
def test_asserted_causes_still_count_as_certainty(text):
    assert g._asserts(g._CERTAINTY, g._without_disclaimers(text))


# --- 3. conditions the owner should CHECK are not asserted facts ----------------------


@pytest.mark.parametrize(
    "body",
    [
        "Catat apakah tutup terbuka atau kemasan rusak setelah dibawa.",
        "Tinjau apakah batch yang terasa terlalu asin memiliki perbedaan takaran.",
        "Gunakan alat takar yang tersedia jika ada.",
        "Pakai timbangan yang tersedia bila memungkinkan.",
        "Periksa agar kemasan tidak rusak saat diserahkan.",
    ],
)
def test_checked_or_conditional_conditions_are_not_asserted(body):
    assert not g._asserts_proposal_fact(body)


@pytest.mark.parametrize(
    "body",
    [
        "Ganti oven yang rusak dengan oven baru.",
        "Gunakan mesin pengemas yang tersedia di dapur.",
        "Karena staf lelah, kurangi jam kerja.",
        # The question ends at the sentence boundary; the next clause asserts.
        "Catat apakah tutup terbuka. Kemasan rusak harus diganti.",
        "Gunakan alat takar yang tersedia, jika perlu tambah porsi.",
    ],
)
def test_asserted_conditions_in_advice_stay_rejected(body):
    assert g._asserts_proposal_fact(body)
