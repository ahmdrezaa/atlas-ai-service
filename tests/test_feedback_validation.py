import re
from uuid import uuid4

import grpc
import pytest

from atlas_ai.feedback.models import DropReason
from atlas_ai.feedback.validation import (
    locate_evidence,
    negation_cropped,
    validate_feedback_request,
    validate_opinions,
)
from atlas_ai.llm import AIFailure

TEXT = "Rotinya empuk, tapi isian coklatnya sedikit."


def opinion(evidence, **overrides):
    item = {
        "opinion_text": "Isian coklat pada roti terlalu sedikit",
        "evidence_text": evidence,
        "sentiment": "NEGATIVE",
        "severity": "NOT_SEVERE",
        "variant": "coklat",
        "issue_title_hint": "Porsi Isian Roti Terlalu Sedikit",
    }
    item.update(overrides)
    return item


# --- FR-01 request validation -------------------------------------------------


def test_valid_request_passes(feedback_request_factory):
    validate_feedback_request(feedback_request_factory(), max_text_chars=5000)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: setattr(r, "shop_id", "bad"), "shop_id"),
        (lambda r: r.ClearField("feedback"), "Missing feedback"),
        (lambda r: r.ClearField("shop"), "Missing shop"),
        (lambda r: setattr(r.feedback, "id", "bad"), "feedback.id"),
        (lambda r: setattr(r.shop, "id", str(uuid4())), "does not match"),
        (lambda r: setattr(r.feedback, "text", " \n "), "empty"),
        (lambda r: setattr(r.feedback, "text", "x" * 5001), "length"),
        (lambda r: setattr(r.feedback, "rating", 0.5), "rating"),
        (lambda r: setattr(r.feedback, "rating", float("nan")), "rating"),
        (lambda r: r.candidate_issues.add(issue_id="bad"), "candidate_issues[0]"),
        (lambda r: r.candidate_vectors.add(opinion_id="bad"), "candidate_vectors[0]"),
        (
            lambda r: r.candidate_vectors.add(opinion_id=str(uuid4()), issue_id="bad"),
            "candidate_vectors[0].issue_id",
        ),
    ],
)
def test_invalid_requests(feedback_request_factory, mutate, message):
    request = feedback_request_factory()
    mutate(request)
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_feedback_request(request, max_text_chars=5000)


def test_duplicate_candidate_ids_rejected(feedback_request_factory):
    request = feedback_request_factory()
    issue_id = str(uuid4())
    request.candidate_issues.add(issue_id=issue_id)
    request.candidate_issues.add(issue_id=issue_id)
    with pytest.raises(ValueError, match="Duplicate candidate issue"):
        validate_feedback_request(request, max_text_chars=5000)

    request = feedback_request_factory()
    opinion_id = str(uuid4())
    request.candidate_vectors.add(opinion_id=opinion_id)
    request.candidate_vectors.add(opinion_id=opinion_id)
    with pytest.raises(ValueError, match="Duplicate candidate vector"):
        validate_feedback_request(request, max_text_chars=5000)


def test_unassigned_vector_issue_id_is_allowed(feedback_request_factory):
    request = feedback_request_factory()
    request.candidate_vectors.add(opinion_id=str(uuid4()), issue_id="")
    validate_feedback_request(request, max_text_chars=5000)


# --- evidence span location -----------------------------------------------------


def span_text(quote, text):
    return [text[a:b] for a, b in locate_evidence(quote, text)]


def test_exact_span():
    assert span_text("isian coklatnya sedikit", TEXT) == ["isian coklatnya sedikit"]


def test_case_and_whitespace_return_original_text():
    text = "Rotinya EMPUK  bgt\ndan wangi"
    assert span_text("rotinya empuk bgt", text) == ["Rotinya EMPUK  bgt"]


def test_quote_not_in_text():
    assert locate_evidence("isian vanilla sedikit", TEXT) == []
    assert locate_evidence("   ", TEXT) == []


def test_word_boundary_required():
    # "enak" is inside "enaknya" and must not be cut out of a word.
    assert locate_evidence("enak", "Rasanya enaknya pas") == []
    assert span_text("enaknya", "Rasanya enaknya pas") == ["enaknya"]


def test_no_fuzzy_or_translation():
    assert locate_evidence("isian coklat sedikit", TEXT) == []
    assert locate_evidence("chocolate filling is small", TEXT) == []


@pytest.mark.parametrize(
    "text, quote",
    [
        ("Rotinya tidak enak", "enak"),
        ("Kopinya nggak terlalu enak", "enak"),
        ("Pelayanannya kurang ramah", "ramah"),
        ("Belum bersih mejanya", "bersih mejanya"),
    ],
)
def test_negation_crop_detected(text, quote):
    ((start, _),) = locate_evidence(quote, text)
    assert negation_cropped(text, start)


def test_negation_in_previous_clause_is_not_a_crop():
    text = "Kopinya tidak pahit. Rotinya enak"
    ((start, _),) = locate_evidence("Rotinya enak", text)
    assert not negation_cropped(text, start)


# --- FR-07a per-opinion validation ------------------------------------------------


def test_valid_opinions_kept_with_original_span():
    outcome = validate_opinions(
        [
            opinion(
                "rotinya empuk",
                opinion_text="Tekstur roti empuk",
                sentiment="POSITIVE",
                variant=None,
                issue_title_hint="Tekstur Roti Empuk",
            ),
            opinion("isian coklatnya sedikit"),
        ],
        TEXT,
        max_opinions=10,
    )
    assert [o.evidence_text for o in outcome.opinions] == [
        "Rotinya empuk",
        "isian coklatnya sedikit",
    ]
    assert outcome.dropped == ()
    assert outcome.raw_count == 2
    for item in outcome.opinions:
        assert item.evidence_text in TEXT


def test_empty_model_output_is_a_valid_empty_result():
    outcome = validate_opinions([], "Saya datang pukul 15.00.", max_opinions=10)
    assert outcome.opinions == () and outcome.raw_count == 0


def test_partial_drop_keeps_the_rest():
    outcome = validate_opinions(
        [
            opinion("isian coklatnya sedikit"),
            opinion("filling is small"),
            opinion(""),
            opinion("isian coklatnya sedikit", sentiment="UNSPECIFIED"),
        ],
        TEXT,
        max_opinions=10,
    )
    assert len(outcome.opinions) == 1
    assert outcome.dropped == (
        DropReason.EVIDENCE_NOT_FOUND,
        DropReason.INVALID_FIELDS,
        DropReason.INVALID_FIELDS,
    )


def test_negation_cropped_opinion_dropped():
    outcome = validate_opinions(
        [
            opinion(
                "enak",
                opinion_text="Rasa roti enak",
                sentiment="POSITIVE",
                issue_title_hint="Rasa Roti Enak",
            ),
            opinion("lama"),
        ],
        "Rotinya nggak enak dan antrenya lama",
        max_opinions=10,
    )
    assert [o.evidence_text for o in outcome.opinions] == ["lama"]
    assert outcome.dropped == (DropReason.NEGATION_CROPPED,)


def test_duplicate_dropped():
    outcome = validate_opinions(
        [opinion("isian coklatnya sedikit"), opinion("Isian coklatnya  sedikit")],
        TEXT,
        max_opinions=10,
    )
    assert len(outcome.opinions) == 1
    assert outcome.dropped == (DropReason.DUPLICATE,)


def test_all_dropped_is_an_error_not_an_empty_result():
    with pytest.raises(AIFailure) as error:
        validate_opinions([opinion("tidak ada di teks")], TEXT, max_opinions=10)
    assert error.value.code == grpc.StatusCode.INTERNAL


def test_over_limit_is_an_error_not_silent_truncation():
    items = [
        opinion(
            "Rotinya empuk",
            opinion_text="Tekstur roti empuk",
            sentiment="POSITIVE",
            issue_title_hint="Tekstur Roti Empuk",
        ),
        opinion("isian coklatnya sedikit"),
    ]
    with pytest.raises(AIFailure) as error:
        validate_opinions(items, TEXT, max_opinions=1)
    assert error.value.code == grpc.StatusCode.INTERNAL


def test_blank_variant_normalized_to_none():
    outcome = validate_opinions(
        [opinion("isian coklatnya sedikit", variant="  ")], TEXT, max_opinions=10
    )
    assert outcome.opinions[0].variant is None
