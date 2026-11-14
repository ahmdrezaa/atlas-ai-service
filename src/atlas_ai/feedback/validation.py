"""Path A request validation (FR-01) and per-opinion output validation (FR-07a)."""

import math
import re
from uuid import UUID

import grpc
from pydantic import ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.feedback.models import (
    DropReason,
    ExtractionOutcome,
    OpinionDraft,
    ValidatedOpinion,
)
from atlas_ai.llm import AIFailure

# Formal and informal Indonesian negators, plus "kurang" ("kurang enak" != "enak").
NEGATORS = frozenset(
    {
        "tidak",
        "tak",
        "gak",
        "ga",
        "nggak",
        "ngga",
        "enggak",
        "engga",
        "gk",
        "tdk",
        "bukan",
        "belum",
        "blm",
        "jangan",
        "jgn",
        "kurang",
    }
)
_CLAUSE_BREAK = re.compile(r"[.!?;,\n]")
_WORD = re.compile(r"\w+")


def _uuid(value: str, field: str) -> None:
    try:
        UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: expected UUID") from exc


def validate_feedback_request(
    request: ai_pb2.ProcessFeedbackRequest, *, max_text_chars: int
) -> None:
    """Structural checks only; Go owns shop authorization and persistence."""
    _uuid(request.shop_id, "shop_id")
    if not request.HasField("feedback"):
        raise ValueError("Missing feedback")
    if not request.HasField("shop"):
        raise ValueError("Missing shop")
    _uuid(request.feedback.id, "feedback.id")
    _uuid(request.shop.id, "shop.id")
    if request.shop.id != request.shop_id:
        raise ValueError("shop.id does not match shop_id")

    text = request.feedback.text
    if not text.strip():
        raise ValueError("Invalid feedback.text: empty")
    if len(text) > max_text_chars:
        raise ValueError("Invalid feedback.text: exceeds configured length")
    rating = request.feedback.rating
    if not math.isfinite(rating) or not 1.0 <= rating <= 5.0:
        raise ValueError("Invalid feedback.rating: expected 1.0-5.0")

    seen: set[str] = set()
    for index, issue in enumerate(request.candidate_issues):
        _uuid(issue.issue_id, f"candidate_issues[{index}].issue_id")
        if issue.issue_id in seen:
            raise ValueError("Duplicate candidate issue ID")
        seen.add(issue.issue_id)

    seen = set()
    for index, vector in enumerate(request.candidate_vectors):
        _uuid(vector.opinion_id, f"candidate_vectors[{index}].opinion_id")
        if vector.opinion_id in seen:
            raise ValueError("Duplicate candidate vector opinion ID")
        seen.add(vector.opinion_id)
        if vector.issue_id:
            _uuid(vector.issue_id, f"candidate_vectors[{index}].issue_id")
    # Embedding model/dimension/finiteness is filtered (not rejected) by matching.


def _folded(text: str) -> tuple[str, list[int]]:
    """Casefold and collapse whitespace, keeping a map back to original indices."""
    chars: list[str] = []
    index: list[int] = []
    previous_space = True  # Also strips leading whitespace.
    for position, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                chars.append(" ")
                index.append(position)
            previous_space = True
            continue
        previous_space = False
        for folded in char.casefold():
            chars.append(folded)
            index.append(position)
    while chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return "".join(chars), index


def _bounded(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before.isalnum() or after.isalnum())


def locate_evidence(quote: str, text: str) -> list[tuple[int, int]]:
    """Return original-text spans for the quote; never rewrites the source.

    Exact match first, then case-insensitive with collapsed whitespace. Spans
    must sit on word boundaries. No fuzzy matching, translation or joining.
    """
    quote = quote.strip()
    if not quote:
        return []
    spans = []
    start = text.find(quote)
    while start != -1:
        if _bounded(text, start, start + len(quote)):
            spans.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    if spans:
        return spans
    folded_text, mapping = _folded(text)
    folded_quote, _ = _folded(quote)
    start = folded_text.find(folded_quote)
    while start != -1:
        original_start = mapping[start]
        original_end = mapping[start + len(folded_quote) - 1] + 1
        if _bounded(text, original_start, original_end):
            spans.append((original_start, original_end))
        start = folded_text.find(folded_quote, start + 1)
    return spans


def negation_cropped(text: str, start: int) -> bool:
    """True when a negator in the same clause, just before the span, was cut off."""
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    previous = _WORD.findall(clause.casefold())[-3:]
    return bool(NEGATORS.intersection(previous))


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def validate_opinions(
    items: list[dict], text: str, *, max_opinions: int
) -> ExtractionOutcome:
    """FR-07a: fix what is safe (span location), drop what is wrong, fail if nothing is left."""
    kept: list[ValidatedOpinion] = []
    dropped: list[DropReason] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        try:
            draft = OpinionDraft.model_validate(item)
        except ValidationError:
            dropped.append(DropReason.INVALID_FIELDS)
            continue
        spans = locate_evidence(draft.evidence_text, text)
        if not spans:
            dropped.append(DropReason.EVIDENCE_NOT_FOUND)
            continue
        faithful = [span for span in spans if not negation_cropped(text, span[0])]
        if not faithful:
            dropped.append(DropReason.NEGATION_CROPPED)
            continue
        start, end = faithful[0]
        evidence = text[start:end]
        key = (_key(draft.opinion_text), _key(evidence), draft.sentiment)
        if key in seen:
            dropped.append(DropReason.DUPLICATE)
            continue
        seen.add(key)
        kept.append(
            ValidatedOpinion(
                opinion_text=draft.opinion_text,
                evidence_text=evidence,
                sentiment=draft.sentiment,
                severity=draft.severity,
                variant=(draft.variant or "").strip() or None,
                issue_title_hint=draft.issue_title_hint,
            )
        )
    if items and not kept:
        raise AIFailure(
            grpc.StatusCode.INTERNAL,
            "Invalid feedback extraction: no extracted opinion passed validation",
        )
    if len(kept) > max_opinions:
        raise AIFailure(
            grpc.StatusCode.INTERNAL,
            "Invalid feedback extraction: opinion count exceeds configured limit",
        )
    return ExtractionOutcome(
        opinions=tuple(kept), dropped=tuple(dropped), raw_count=len(items)
    )
