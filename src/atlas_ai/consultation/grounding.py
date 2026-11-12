"""Conservative source verification, not a general entailment model.

4A structural validation remains the prerequisite. Unresolved language fails closed;
only bounded extractive/attribution proofs and premise-free proposals are accepted.
"""

import re
import time
import unicodedata
from collections.abc import Callable
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Literal

import grpc
from pydantic import Field

from atlas_ai.consultation.models import ConsultationContext, ContextModel, References

if TYPE_CHECKING:
    from atlas_ai.consultation.solution import GeneratedSolution

STAT_LABELS = {
    "unique_feedback_count": "jumlah feedback unik",
    "opinion_count": "jumlah opini",
    "average_rating": "rata-rata rating",
    "negative_count": "opini negatif",
    "positive_count": "opini positif",
    "neutral_count": "opini netral",
    "severe_count": "opini severe",
    "not_severe_count": "opini tidak severe",
}
STAT = re.compile(r"\[\[stat:([a-z_]+)=(-?\d+(?:\.\d+)?)\]\]")
GROUNDING_RULE_VERSION = "grounding-rules-v2"


class GroundingLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


ACCEPTED_LABELS = {GroundingLabel.SUPPORTED, GroundingLabel.NOT_APPLICABLE}


class GroundingClaim(ContextModel):
    claim_id: str
    kind: Literal[
        "finding",
        "hypothesis",
        "recommendation",
        "evidence_note",
        "metric",
        "target",
        "timeframe",
    ]
    text: str
    references: References = Field(default_factory=References)


class GroundingDecision(ContextModel):
    claim_id: str
    label: GroundingLabel
    reason: str = Field(min_length=1, max_length=500)


def cited_sources(
    claim: GroundingClaim, context: ConsultationContext
) -> list[dict[str, str]]:
    """Only cited CURRENT sources; no history, uncited context, or normalized AI opinions."""
    result = []
    for ids, registry, kind in (
        (claim.references.opinion_ids, context.sources.opinions, "customer"),
        (claim.references.owner_message_ids, context.sources.owner_messages, "owner"),
        (claim.references.memory_ids, context.sources.memories, "memory"),
    ):
        for source_id in ids:
            source = registry.get(source_id)
            if source is None:
                continue  # Structural validator rejects these before production checks.
            text = (
                f"{source.evidence_text}\n{source.full_feedback}"
                if kind == "customer"
                else source.content
                if kind == "owner"
                else source.fact_text
            )
            result.append({"source_id": source_id, "kind": kind, "text": text})
    return result


_SMALL = {
    "nol": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
    "sepuluh": 10,
    "sebelas": 11,
}
NUMBER_WORDS = re.compile(
    r"\b(?:(?:"
    + "|".join(word for word, number in _SMALL.items() if 2 <= number <= 9)
    + r")\s+puluh(?:\s+(?:"
    + "|".join(word for word, number in _SMALL.items() if 1 <= number <= 9)
    + r"))?|(?:"
    + "|".join(word for word, number in _SMALL.items() if 2 <= number <= 9)
    + r")\s+belas|(?:"
    + "|".join(_SMALL)
    + r"))\b"
)


def _number_word(match: re.Match) -> str:
    words = match[0].split()
    number = _SMALL[words[0]]
    if len(words) > 1:
        number = number * 10 if words[1] == "puluh" else number + 10
    if len(words) > 2:
        number += _SMALL[words[2]]
    return str(number)


def _normal(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    # Deliberately small, reviewable surface equivalences; not fuzzy entailment.
    for pattern, replacement in (
        (r"\b(antreannya|antrian|antriannya)\b", "antrean"),
        (r"\b(pelayanannya|layanan)\b", "pelayanan"),
        (r"\brotinya\b", "roti"),
        (r"\blezat\b", "enak"),
        (r"\blambat\b", "lama"),
    ):
        text = re.sub(pattern, replacement, text)
    text = NUMBER_WORDS.sub(_number_word, text)
    return " ".join(re.findall(r"\w+", text))


_ATTRIBUTION = re.compile(
    r"^(?:(pelanggan|customer|owner|pemilik)\s+(?:menyebut|menilai|melaporkan|menyatakan|mengatakan|mengeluhkan)"
    r"|menurut\s+(pelanggan|customer|owner|pemilik))\s*[:,]?\s*",
    re.IGNORECASE,
)


def attributed_text(text: str) -> str:
    """Strip only the explicit reporting prefix, without modifying source content."""
    return _ATTRIBUTION.sub("", text, count=1)


def verbatim_quote_supported(quote: str, evidence: str, feedback: str) -> bool:
    """Allow shorter exact excerpts, but not word/negation cropping.

    No rewriting or normalization of the returned quote. This is a bounded context
    check, not a proof of every possible pragmatic meaning of a review.
    """
    if not quote.strip():
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(quote) + r"(?!\w)")

    def faithful(text: str) -> bool:
        for match in pattern.finditer(text):
            # Stay within the current clause; a previous sentence's negation is
            # not a modifier of this excerpt. Preserve modifiers before the quote.
            prefix = re.split(r"[.!?;,\n]", text[: match.start()])[-1]
            if not {"tidak", "bukan", "jangan", "belum"}.intersection(
                _normal(prefix).split()[-3:]
            ):
                return True
        return False

    # If full feedback contains the excerpt, its wider context is authoritative;
    # a cropped evidence_text cannot erase negation visible in that feedback.
    if quote in feedback:
        return faithful(feedback)
    return faithful(evidence)


# Deliberately bounded operational propositions, NOT number-bag matching. The
# predicate, unit, order and values must all agree with a cited owner/memory.
_OPERATIONAL = re.compile(
    r"\b(?:jam (?:ramai|sibuk|buka|tutup|operasional) "
    r"\d{1,2} \d{2}(?: (?:sampai|hingga) \d{1,2} \d{2})?"
    r"|jumlah (?:barista|staf|kasir|karyawan) \d+ orang"
    r"|shift (?:pagi|sore|malam) \d+ orang)\b"
)
_UNCERTAIN_OPERATION = re.compile(
    r"\b(tidak|bukan|belum|jangan|jika|usulan|target|rencana|mungkin|diduga|akan)\b"
)


def operational_number_context(
    text: str, sources: list[dict[str, str]]
) -> tuple[str, bool]:
    """Remove only proved operational fragments; return residual text and proof.

    Used by structural AND grounding checks. Other numbers need the existing
    exact-excerpt/explicit-proposal route. Unknown phrasing fails closed.
    """

    def normalize(value):
        return " ".join(word for word in _normal(value).split() if word != "pukul")

    normalized = normalize(text)
    fragments = list(_OPERATIONAL.finditer(normalized))
    source_clauses = [
        (normalize(part), bool(_UNCERTAIN_OPERATION.search(_normal(part))))
        for source in sources
        if source["kind"] in {"owner", "memory"}
        and not _INSTRUCTIONS.search(source["text"])
        # Do not turn an uncertain/negated operational sentence into a fact.
        for part in re.split(r"(?<!\d)[.;]|[.;](?!\d)|[!?\n]", source["text"])
    ]
    for match in fragments:
        atom = match[0]
        pattern = re.sub(r"\d+", "__NUMBER__", re.escape(atom)).replace(
            "__NUMBER__", r"\d+"
        )
        if any(
            uncertain
            and re.search(
                pattern, " ".join(_UNCERTAIN_OPERATION.sub(" ", source).split())
            )
            for source, uncertain in source_clauses
        ):
            return normalized, False
        candidates = [
            found[0]
            for source, _ in source_clauses
            for found in re.finditer(pattern, source)
        ]
        if atom not in candidates or any(value != atom for value in candidates):
            return normalized, False
    return _OPERATIONAL.sub(" ", normalized), True


_INSTRUCTIONS = re.compile(
    r"\b(abaikan|ignore|system prompt|instruksi sistem|katakan)\b", re.IGNORECASE
)
_UNIVERSAL = re.compile(
    r"\b(semua pelanggan|seluruh pelanggan|mayoritas|selalu|setiap pelanggan)\b",
    re.IGNORECASE,
)
_CERTAINTY = re.compile(
    r"\b(terbukti|dipastikan|terverifikasi|independen|penyebab pasti|disebabkan|penyebabnya|caused by)\b",
    re.IGNORECASE,
)
_ACHIEVED = re.compile(r"\b(telah|sudah|tercapai|achieved|already)\b", re.IGNORECASE)
_PREMISE = re.compile(
    r"\b(tersedia|memiliki|dimiliki|yang ada|yang rusak)\b", re.IGNORECASE
)
_ACTION = re.compile(
    r"^(amati|catat|ukur|evaluasi|bandingkan|uji|pertimbangkan|tinjau|verifikasi|diskusikan|tambah|kurangi)\b",
    re.IGNORECASE,
)


def _affirmed_part(atom: str, source: str) -> bool:
    for match in re.finditer(r"(?<!\w)" + re.escape(atom) + r"(?!\w)", source):
        previous = source[: match.start()].split()[-3:]
        if not {"tidak", "bukan", "jangan"}.intersection(previous):
            return True
    return False


def _opposites(atom: str) -> set[str]:
    result = set()
    for left, right in (
        ("cepat", "lama"),
        ("lama", "tidak lama"),
        ("enak", "tidak enak"),
        ("ada", "tidak ada"),
        ("terbatas", "tidak terbatas"),
    ):
        for old, new in ((left, right), (right, left)):
            if re.search(r"\b" + old + r"\b", atom):
                result.add(re.sub(r"\b" + old + r"\b", new, atom))
    if "tidak " in atom:
        result.add(atom.replace("tidak ", "", 1))
    if "bukan " in atom:
        result.add(atom.replace("bukan ", "", 1))
    # Different numbers with the same surrounding proposition are contradictions.
    return result


def _fact(
    claim: GroundingClaim, sources: list[dict[str, str]]
) -> tuple[GroundingLabel, str]:
    attribution = _ATTRIBUTION.match(claim.text)
    role = (attribution[1] or attribution[2]).casefold() if attribution else ""
    kind = (
        "customer"
        if role in {"pelanggan", "customer"}
        else "owner"
        if role
        else "memory"
    )
    body = attributed_text(claim.text)
    if claim.kind == "evidence_note":
        kind = "customer"
        body = re.sub(r"^keluhan\s+", "", body, flags=re.IGNORECASE)
    eligible = [
        source
        for source in sources
        if source["kind"] == kind and not _INSTRUCTIONS.search(source["text"])
    ]
    if not eligible:
        return (
            GroundingLabel.INSUFFICIENT_EVIDENCE,
            "no_cited_source_with_required_provenance",
        )
    atoms = [
        _normal(part)
        for part in re.split(r"[.;]|\s+dan\s+", body, flags=re.IGNORECASE)
        if part.strip()
    ]
    if not atoms:
        return GroundingLabel.INSUFFICIENT_EVIDENCE, "empty_claim"
    for atom in atoms:
        positive = negative = False
        for source in eligible:
            text = _normal(source["text"])
            positive |= _affirmed_part(atom, text)
            negative |= any(
                _affirmed_part(opposite, text) for opposite in _opposites(atom)
            )
            if re.search(r"\d", atom):
                pattern = re.sub(r"\d+", r"__NUMBER__", re.escape(atom)).replace(
                    "__NUMBER__", r"\d+"
                )
                negative |= any(
                    match[0] != atom for match in re.finditer(pattern, text)
                )
        if negative:
            return (
                (GroundingLabel.INSUFFICIENT_EVIDENCE, "conflicting_sources")
                if positive
                else (GroundingLabel.CONTRADICTED, "cited_source_contradicts_clause")
            )
        if not positive:
            return (
                GroundingLabel.INSUFFICIENT_EVIDENCE,
                "no_bounded_source_proof_for_clause",
            )
    return GroundingLabel.SUPPORTED, "all_clauses_supported_with_provenance"


_METRICS = [
    (r"(?:jumlah )?feedback unik", "unique_feedback_count"),
    (r"rata rata rating", "average_rating"),
    (r"(?:jumlah )?opini tidak severe", "not_severe_count"),
    (r"(?:jumlah )?opini negatif", "negative_count"),
    (r"(?:jumlah )?opini positif", "positive_count"),
    (r"(?:jumlah )?opini netral", "neutral_count"),
    (r"(?:jumlah )?opini severe", "severe_count"),
    (r"(?:jumlah )?opini", "opinion_count"),
]


def _statistics(text: str, context: ConsultationContext) -> tuple[GroundingLabel, str]:
    # Marker values/fields have already passed 4A. Here check surrounding meaning.
    matches = list(STAT.finditer(text))
    residual = _normal(STAT.sub(" ", text))
    if not matches or context.aggregate_statistics is None:
        return GroundingLabel.INSUFFICIENT_EVIDENCE, "no_verified_sql_marker"
    cursor = 0
    for match in matches:
        prefix = _normal(text[cursor : match.start()])
        for pattern, field in _METRICS:
            if re.search(pattern, prefix):
                if field != match[1]:
                    return GroundingLabel.CONTRADICTED, "sql_marker_metric_relabelled"
                break
        cursor = match.end()
    for pattern, field in _METRICS:
        number_pattern = r"\b(\d+) " + pattern + r"\b"
        for match in re.finditer(number_pattern, residual):
            if Decimal(match[1]) != Decimal(
                str(getattr(context.aggregate_statistics, field))
            ):
                return GroundingLabel.CONTRADICTED, "spelled_quantity_contradicts_sql"
        residual = re.sub(number_pattern, " ", residual)
        if re.search(pattern, residual):
            if len(matches) != 1 or field != matches[0][1]:
                return GroundingLabel.CONTRADICTED, "sql_metric_name_conflict"
            residual = re.sub(pattern, " ", residual)
    allowed = {
        "snapshot",
        "mencatat",
        "statistik",
        "sql",
        "yaitu",
        "dan",
        "total",
        "menunjukkan",
        "jumlah",
    }
    if set(residual.split()) - allowed:
        return (
            GroundingLabel.INSUFFICIENT_EVIDENCE,
            "sql_marker_does_not_prove_surrounding_prose",
        )
    return GroundingLabel.SUPPORTED, "bounded_sql_claim"


def evaluate_claim(
    claim: GroundingClaim, context: ConsultationContext
) -> GroundingDecision:
    """Classify structurally valid text; unknown language is not accepted as proven."""
    text = claim.text.strip()
    sources = cited_sources(claim, context)
    label, reason = GroundingLabel.INSUFFICIENT_EVIDENCE, "unresolved_claim"
    if _UNIVERSAL.search(text) or _CERTAINTY.search(text):
        reason = "unproven_scope_causality_or_independent_verification"
    elif claim.kind == "finding" and STAT.search(text):
        label, reason = _statistics(text, context)
    elif claim.kind == "hypothesis":
        clauses = [
            part.strip()
            for part in re.split(r"[.;]", text.removeprefix("Hipotesis:"))
            if part.strip()
        ]
        if (
            clauses
            and all(
                re.search(
                    r"\b(mungkin|diduga|belum diketahui|perlu diverifikasi)\b",
                    part,
                    re.IGNORECASE,
                )
                for part in clauses
            )
            and not _PREMISE.search(text)
        ):
            label, reason = (
                GroundingLabel.NOT_APPLICABLE,
                "tentative_hypothesis_not_established_cause",
            )
    elif claim.kind in {"recommendation", "metric", "target", "timeframe"}:
        body = text.removeprefix("Usulan:").strip()
        # Explicit target/timeframe fields are proposals, not current resources.
        # Check the existing operational premises embedded in actions instead.
        operations_supported = True
        if claim.kind == "recommendation":
            _, operations_supported = operational_number_context(body, sources)
        action, *premise = re.split(
            r"\s+(?:karena|sebab|mengingat|because)\s+",
            body,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        premise_ok = True
        if premise:
            verdict = evaluate_claim(
                GroundingClaim(
                    claim_id=claim.claim_id,
                    kind="finding",
                    text=premise[0],
                    references=claim.references,
                ),
                context,
            )
            premise_ok = verdict.label == GroundingLabel.SUPPORTED
        metric = re.match(
            r"^(durasi|jumlah|rata-rata|persentase)\b", action, re.IGNORECASE
        )
        if (
            not _ACHIEVED.search(body)
            and not _PREMISE.search(action)
            and operations_supported
            and premise_ok
            and (_ACTION.match(action) or metric)
        ):
            label, reason = (
                GroundingLabel.NOT_APPLICABLE,
                "proposal_with_no_unproved_factual_premise",
            )
        else:
            reason = "proposal_contains_unproved_premise_or_unresolved_language"
    else:
        label, reason = _fact(claim, sources)
    return GroundingDecision(claim_id=claim.claim_id, label=label, reason=reason)


def solution_claims(solution: "GeneratedSolution") -> list[GroundingClaim]:
    """Keep component boundaries; evaluate each factual clause and proposal premise."""
    claims = []
    for section, kind, fields in (
        ("findings", "finding", ("statement",)),
        ("hypotheses", "hypothesis", ("statement",)),
        ("recommendations", "recommendation", ("title", "description")),
    ):
        for index, item in enumerate(getattr(solution, section)):
            for field in fields:
                text = getattr(item, field)
                if text.strip():
                    claims.append(
                        GroundingClaim(
                            claim_id=f"{section}[{index}].{field}",
                            kind=kind,
                            text=text,
                            references=item.references,
                        )
                    )
    for index, item in enumerate(solution.customer_evidence):
        if item.note.strip():
            claims.append(
                GroundingClaim(
                    claim_id=f"customer_evidence[{index}].note",
                    kind="evidence_note",
                    text=item.note,
                    references=References(opinion_ids=(item.opinion_id,)),
                )
            )
    for index, item in enumerate(solution.success_indicators):
        for field in ("metric", "target", "timeframe"):
            text = getattr(item, field)
            if text.strip():
                claims.append(
                    GroundingClaim(
                        claim_id=f"success_indicators[{index}].{field}",
                        kind=field,
                        text=text,
                    )
                )
    return claims


def validate_grounding(
    solution: "GeneratedSolution",
    context: ConsultationContext,
    *,
    time_remaining: float | None = None,
    is_active: Callable[[], bool] | None = None,
) -> list[GroundingDecision]:
    """No model invocation or repair. Fail the whole solution on an unresolved claim."""
    from atlas_ai.consultation.assessment import ConsultationFailure

    deadline = time.monotonic() + time_remaining if time_remaining is not None else None
    decisions = []
    for claim in solution_claims(solution):
        if deadline is not None and time.monotonic() >= deadline:
            raise ConsultationFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                "Solution grounding deadline exceeded",
            )
        if is_active is not None and not is_active():
            raise ConsultationFailure(
                grpc.StatusCode.CANCELLED, "Consultation cancelled"
            )
        decision = evaluate_claim(claim, context)
        if decision.label not in ACCEPTED_LABELS:
            raise ConsultationFailure(
                grpc.StatusCode.INTERNAL,
                f"Solution grounding rejected {decision.claim_id}: {decision.label.value}",
            )
        decisions.append(decision)
    if deadline is not None and time.monotonic() >= deadline:
        raise ConsultationFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, "Solution grounding deadline exceeded"
        )
    return decisions
