"""Conservative source verification, not a general entailment model.

4A structural validation remains the prerequisite. Factual source proof blocks;
pure lexical relevance is diagnostic. Mixed factual/scope guards remain bounded
and conservative where they cannot safely be separated from language judgments.
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
GROUNDING_RULE_VERSION = "grounding-rules-v8"
TENTATIVE = re.compile(
    r"\b(mungkin|diduga|belum diketahui|perlu diverifikasi|berpotensi dipengaruhi|"
    r"ada kemungkinan|perlu ditelusuri apakah)\b",
    re.IGNORECASE,
)


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
    # Internal field linkage set only for Solution recommendation titles. The
    # description is a proposal to validate, never an additional factual source.
    recommendation_description: str | None = None


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
    r"^(?:(pelanggan|customer|owner|pemilik)\s+(?:menyebut|menilai|melaporkan|menyatakan|"
    r"mengatakan|mengeluhkan|menulis|menjelaskan|menceritakan|menginformasikan|mengonfirmasi)"
    r"|menurut\s+(pelanggan|customer|owner|pemilik))\s*[:,]?\s*",
    re.IGNORECASE,
)
# A confirmed-memory label; the claim is then checked against memory sources.
_MEMORY_PREFIX = re.compile(
    r"^(?:fakta|memori|catatan)\s+bisnis(?:\s+(?:yang\s+)?(?:terkonfirmasi|dikonfirmasi))?\s*:\s*",
    re.IGNORECASE,
)


def attributed_text(text: str) -> str:
    """Strip only the explicit reporting prefix, without modifying source content."""
    return _MEMORY_PREFIX.sub("", _ATTRIBUTION.sub("", text, count=1), count=1)


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
    r"|jumlah (?:barista|staf|kasir|karyawan|baker) \d+ orang"
    r"|shift (?:pagi|sore|malam) \d+ orang)\b"
)
_UNCERTAIN_OPERATION = re.compile(
    r"\b(tidak|bukan|belum|jangan|jika|usulan|target|rencana|mungkin|diduga|akan)\b"
)
_ACTION_WINDOW = re.compile(
    r"\b(?:pada jam|jam|pukul) (\d{1,2} \d{2} sampai \d{1,2} \d{2})\b"
)


def operational_number_context(
    text: str, sources: list[dict[str, str]], *, action_window: bool = False
) -> tuple[str, bool]:
    """Remove only proved operational fragments; return residual text and proof.

    Used by structural AND grounding checks. Other numbers need the existing
    exact-excerpt/explicit-proposal route. Unknown phrasing fails closed.
    """

    def normalize(value):
        # Range punctuation equivalence only between complete clock values.
        # This does not normalize UUIDs, quotes, SQL metrics or wire text.
        value = re.sub(
            r"(?<!\d)(\d{1,2}[.:]\d{2})\s*[-–—]\s*(\d{1,2}[.:]\d{2})(?!\d)",
            r"\1 sampai \2",
            value,
        )
        value = re.sub(r"\bhingga\b", "sampai", _normal(value))
        return re.sub(
            r"\b(jam (?:ramai|sibuk|buka|tutup|operasional)) pukul\b", r"\1", value
        )

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
                pattern,
                normalize(" ".join(_UNCERTAIN_OPERATION.sub(" ", source).split())),
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
    residual = _OPERATIONAL.sub(" ", normalized)
    # An action may reuse a cited clock window without asserting a new business
    # predicate. Opening hours and busy hours remain different factual claims.
    if action_window and not text.startswith("Usulan:"):
        for window in list(_ACTION_WINDOW.finditer(residual)):
            supported = False
            for clause, uncertain in source_clauses:
                clean = normalize(
                    " ".join(_UNCERTAIN_OPERATION.sub(" ", clause).split())
                )
                for fact in _OPERATIONAL.finditer(clean):
                    if fact[0].startswith("jam ") and window[1] in fact[0]:
                        if uncertain:
                            return residual, False
                        supported = True
            if not supported:
                return residual, False
            residual = residual.replace(window[0], " ", 1)
    return residual, True


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
_NEGATORS = {"belum", "tidak", "bukan", "tak"}


def _asserts(pattern: re.Pattern, text: str) -> bool:
    """A match counts unless directly negated ("belum terverifikasi", "tidak disebabkan")."""
    for match in pattern.finditer(text):
        if not _NEGATORS.intersection(_normal(text[: match.start()]).split()[-2:]):
            return True
    return False


_PREMISE = re.compile(
    r"\b(tersedia|memiliki|dimiliki|yang ada|yang rusak)\b", re.IGNORECASE
)
_ACTION = re.compile(
    r"^(amati|catat|ukur|evaluasi|bandingkan|uji|pertimbangkan|tinjau|verifikasi|diskusikan|tambah|kurangi|lakukan)\b",
    re.IGNORECASE,
)
_CURRENT_ASSERTION = re.compile(
    r"\b(saat ini|sekarang|kenyataannya|faktanya|tercatat)\b",
    re.IGNORECASE,
)
_TOPIC_STOP = {
    "menyatakan",
    "belum",
    "hipotesis",
    "pada",
    "itu",
    "agar",
    "pelanggan",
    "ini",
    "karena",
    "saat",
    "oleh",
    "owner",
    "alur",
    "ke",
    "tidak",
    "dari",
    "diverifikasi",
    "atau",
    "pemilik",
    "setiap",
    "customer",
    "mengatakan",
    "telah",
    "jumlah",
    "waktu",
    "jam",
    "mungkin",
    "proses",
    "menurut",
    "orang",
    "ada",
    "di",
    "lama",
    "dengan",
    "menyebut",
    "perlu",
    "hari",
    "untuk",
    "yang",
    "dalam",
    "dan",
    "sudah",
    "diduga",
    "penyebab",
    "tanpa",
    "bukan",
    "sekali",
}


def _related_hypothesis(text: str, sources: list[dict[str, str]]) -> bool:
    """Necessary lexical relevance to cited context, NOT semantic entailment.

    A tentative cause need not be true, but must anchor to substantive cited words.
    No history, uncited issue title or generic uncertainty word can supply a proof.
    """

    def words(value):
        return {
            word
            for word in _normal(value).split()
            if len(word) > 2 and not word.isdigit() and word not in _TOPIC_STOP
        }

    anchors = set().union(
        *(
            words(source["text"])
            for source in sources
            if not _INSTRUCTIONS.search(source["text"])
        )
    )
    return bool(words(text) & anchors)


def _action_framing(text: str) -> bool:
    # Imperative, action nominal, or explicit advisory modality. These establish
    # proposal intent only; factual premises are still verified separately.
    return bool(
        _ACTION.match(text)
        or re.match(
            r"^(?:sebaiknya|disarankan|pertimbangan untuk|pemantauan|pencatatan|pengukuran|observasi)\b",
            text,
            re.IGNORECASE,
        )
    )


# Action-label intent is necessary, never sufficient for accepting a heading.
# Descriptions keep the existing action/premise guards; the complete heading's
# subject must also be bounded by that description AND its actual cited sources.
_LABEL_INTENTS = (
    (r"(?:optimasi|optimalisasi|perbaikan)", r"\b(?:evaluasi|tinjau|uji|tata ulang)\b"),
    (r"penataan", r"\b(?:evaluasi|tinjau|tata ulang)\b"),
    (r"pembagian", r"\b(?:evaluasi|pembagian|tata ulang)\b"),
)
_LABEL_ASSERTION = re.compile(
    r"\b(pasti|dijamin|menjamin|jaminan|garansi|guaranteed|guarantee|"
    r"rusak|kekurangan|kurang|adalah|berjumlah|tersedia|memiliki|"
    r"meningkat|berkurang|naik|turun|semua|seluruh|karena|sebab|mengingat)\b",
    re.IGNORECASE,
)


def _label_subject_words(text: str) -> set[str]:
    # Work/tasks equivalence is for topic linkage only, never factual entailment.
    text = re.sub(r"\btugas\b", "kerja", _normal(text))
    return set(text.split()) - {
        "alur",
        "proses",
        "dan",
        "untuk",
        "di",
        "saat",
        "pada",
        "jam",
        "yang",
        "tanpa",
        "dengan",
    }


def _recommendation_title(
    claim: GroundingClaim, context: ConsultationContext, *, semantic_checks: bool = True
) -> GroundingDecision:
    text = claim.text.strip()
    standalone = claim.model_copy(update={"recommendation_description": None})
    verdict = evaluate_claim(standalone, context, semantic_checks=semantic_checks)
    # Do not let even an imperative heading independently promise outcomes or
    # assert an operational condition. Descriptions cannot launder such claims.
    if (
        _LABEL_ASSERTION.search(text)
        or _ACHIEVED.search(text)
        or _CURRENT_ASSERTION.search(text)
    ):
        return GroundingDecision(
            claim_id=claim.claim_id,
            label=GroundingLabel.INSUFFICIENT_EVIDENCE,
            reason="title_asserts_fact_or_outcome",
        )
    if verdict.reason == "bounded_nominal_proposal":
        description = claim.model_copy(
            update={
                "text": claim.recommendation_description or "",
                "recommendation_description": None,
            }
        )
        desc_verdict = evaluate_claim(
            description, context, semantic_checks=semantic_checks
        )
        common = (
            set(_normal(text).split())
            & set(_normal(description.text).split())
            & _source_words(cited_sources(claim, context))
        )
        if desc_verdict.label not in ACCEPTED_LABELS or (
            semantic_checks and len(common) < 2
        ):
            return GroundingDecision(
                claim_id=claim.claim_id,
                label=GroundingLabel.INSUFFICIENT_EVIDENCE,
                reason="nominal_title_description_mismatch",
            )
    if verdict.label in ACCEPTED_LABELS:
        return verdict  # Preserve already-supported imperative/advisory titles.
    description = (claim.recommendation_description or "").strip()
    action = claim.model_copy(
        update={"text": description, "recommendation_description": None}
    )
    action_verdict = evaluate_claim(action, context, semantic_checks=semantic_checks)
    sources = cited_sources(claim, context)
    words = text.split()
    if (
        not sources
        or not description
        or not 2 <= len(words) <= 8
        or not re.fullmatch(r"[A-Za-z ]+", text)
        or action_verdict.label != GroundingLabel.NOT_APPLICABLE
        or (semantic_checks and not _related_hypothesis(description, sources))
        or not any(
            re.fullmatch(head, words[0], re.IGNORECASE)
            and (not semantic_checks or re.search(intent, description, re.IGNORECASE))
            for head, intent in _LABEL_INTENTS
        )
    ):
        return verdict
    subject = _label_subject_words(" ".join(words[1:]))
    description_words = _label_subject_words(description)
    source_words = set().union(
        *(
            _label_subject_words(s["text"])
            for s in sources
            if not _INSTRUCTIONS.search(s["text"])
        )
    )
    if (
        subject
        # A proposal can name an activity, but cannot supply new facts such as
        # 'new machine' or 'tired cashier'. Other subject words must come from
        # the cited sources, never merely from generated description prose.
        and subject
        <= (
            source_words
            | (
                description_words
                & {
                    "kerja",
                    "pembagian",
                    "pencatatan",
                    "pengukuran",
                    "pengamatan",
                    "observasi",
                }
            )
        )
        and (not semantic_checks or subject & description_words)
        and subject & source_words
        and not NUMBER_WORDS.search(text.casefold())
    ):
        return GroundingDecision(
            claim_id=claim.claim_id,
            label=GroundingLabel.NOT_APPLICABLE,
            reason="action_label_linked_to_validated_description_and_cited_subject",
        )
    return verdict


def _source_words(sources: list[dict[str, str]]) -> set[str]:
    return set().union(
        *(
            set(_normal(s["text"]).split())
            for s in sources
            if not _INSTRUCTIONS.search(s["text"])
        )
    )


def _tentative_mechanism(text: str, sources: list[dict[str, str]]) -> bool:
    """Bounded possibility: cited symptom + operational subject, never a new fact."""
    clauses = [p.strip() for p in text.removeprefix("Hipotesis:").split(";")]
    if any(
        _normal(p)
        not in {
            "perlu diverifikasi",
            "hal ini perlu diperiksa",
            "hal ini perlu ditelusuri",
        }
        for p in clauses[1:]
    ):
        return False
    body = re.sub(
        r"^(?:ada kemungkinan|perlu ditelusuri apakah)\s+",
        "",
        clauses[0],
        flags=re.IGNORECASE,
    ).rstrip(".")
    match = re.fullmatch(
        r"(.+?)\s+(?:mungkin(?: terjadi)? karena|(?:mungkin |diduga )?(?:terkait|berkaitan dengan)|berpotensi dipengaruhi)\s+(.+)",
        body,
        re.IGNORECASE,
    )
    if not match or not TENTATIVE.search(text):
        return False
    symptom, mechanism = map(_normal, match.groups())
    if not re.fullmatch(
        r"(?:(?:lamanya|keluhan|durasi) )?(?:antrean|pelayanan)(?: lama)?", symptom
    ):
        return False
    subject = set(mechanism.split()) - {"dan", "dengan"}
    allowed = {
        "koordinasi",
        "pelayanan",
        "kasir",
        "barista",
        "alur",
        "pesanan",
        "pembagian",
        "tugas",
        "kerja",
    }
    cited = _source_words(sources)
    symptom_anchor = set(symptom.split()) & cited - _TOPIC_STOP
    symptom_proof = symptom.replace("lamanya antrean", "antrean lama").replace(
        "lamanya pelayanan", "pelayanan lama"
    )
    if not any(
        s["kind"] == "customer"
        and not _INSTRUCTIONS.search(s["text"])
        and _affirmed_part(symptom_proof, _normal(s["text"]))
        for s in sources
    ):
        return False
    mechanism_anchor = (
        subject
        & _source_words([s for s in sources if s["kind"] in {"owner", "memory"}])
        - _TOPIC_STOP
    )
    return bool(
        symptom_anchor
        and mechanism_anchor
        and len(symptom_anchor | mechanism_anchor) >= 2
        and len(subject) >= 2
        and subject <= allowed
    )


def _nominal_proposal(text: str, sources: list[dict[str, str]]) -> bool:
    """Action nominal morphology and a bounded object, not a new verb list."""
    match = re.fullmatch(r"pe[a-z]+an\s+([a-z ]+)\.?", text, re.IGNORECASE)
    if not match:
        return False
    # Resource commitments need a separate constraint proof; this bounded
    # observation/workflow path does not infer permission from a resource name.
    if re.match(
        r"(?:penambahan|pengadaan|pembelian|pengurangan)\b", text, re.IGNORECASE
    ):
        return False
    subject = set(_normal(match[1]).split()) - {"dan", "di", "pada", "terhadap"}
    cited = _source_words(sources)
    activities = {
        "alur",
        "pembagian",
        "tugas",
        "kerja",
        "pelayanan",
        "pencatatan",
        "observasi",
        "antrean",
    }
    if not subject & {"alur", "pembagian", "tugas", "pencatatan", "observasi"}:
        return False
    return bool(
        len(subject & cited) >= 2
        and subject <= cited | activities
        and not _LABEL_ASSERTION.search(text)
    )


def _source_bound_nominal(text: str, sources: list[dict[str, str]]) -> bool:
    """A nominal proposal with a fully cited object, not a domain vocabulary.

    Retain resource/predicate guards: a grammatical noun form alone cannot prove
    existing equipment, availability or outcomes. The source text, never another
    generated field, must supply the complete object phrase in one source.
    """
    match = re.fullmatch(r"pe[a-z]+an\s+([a-z ]+)\.?", text, re.IGNORECASE)
    if not match or _LABEL_ASSERTION.search(text):
        return False
    if re.match(
        r"(?:penambahan|pengadaan|pembelian|pengurangan)\b", text, re.IGNORECASE
    ):
        return False
    subject = _normal(match[1])
    return bool(subject) and any(
        not _INSTRUCTIONS.search(source["text"])
        and _affirmed_part(subject, _normal(source["text"]))
        for source in sources
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


def _independent_actions(text: str) -> list[str]:
    """Separate sentences and independently framed coordinated actions only.

    Do not split shared noun phrases ("jumlah dan konsistensi isian roti") or
    infer that arbitrary noun/adjective fragments are independent propositions.
    This prevents cross-action borrowing; it does not prove semantic relevance
    inside a single action. Returned slices are for checking, never output repair.
    """
    parts = []
    for sentence in re.split(r"(?<!\d)[.;]|[.;](?!\d)", text):
        start = 0
        for match in re.finditer(r"\s+(?:dan|serta)\s+", sentence, re.IGNORECASE):
            # Action nominals can share a trailing object ("pemantauan dan
            # pencatatan ..."). Only a repeated imperative establishes this
            # bounded coordination boundary; do not guess the nominal scope.
            if _ACTION.match(sentence[match.end() :].strip()):
                parts.append(sentence[start : match.start()].strip())
                start = match.end()
        parts.append(sentence[start:].strip())
    return [part for part in parts if part]


def _action_before_purpose(text: str) -> str:
    """Bound the action-object span for relevance, not for factual validation.

    An intended purpose is generated reasoning, never evidence connecting an
    otherwise unrelated action to cited sources. Keep the full original text
    for premise, quantity, outcome and namespace checks. This bounded surface
    distinction is not a general Indonesian dependency or semantic parser.
    """
    return re.split(r"\s+(?:untuk|agar)\s+", text, maxsplit=1, flags=re.IGNORECASE)[0]


def evaluate_claim(
    claim: GroundingClaim, context: ConsultationContext, *, semantic_checks: bool = True
) -> GroundingDecision:
    """Diagnose claims; evidence-only checks never waive factual source proof.

    The default retains historical semantic diagnostics for offline review.
    Production uses semantic_checks=False: lexical relevance and safely bounded
    nominal wording do not block. Mixed factual/scope guards remain conservative.
    """
    text = claim.text.strip()
    sources = cited_sources(claim, context)
    label, reason = GroundingLabel.INSUFFICIENT_EVIDENCE, "unresolved_claim"
    if _UNIVERSAL.search(text) or (
        _asserts_with(_CERTAINTY, text, _HYPOTHESIS_HEDGES)
        if claim.kind == "hypothesis"
        else _asserts(_CERTAINTY, text)
    ):
        reason = "unproven_scope_causality_or_independent_verification"
    elif (
        claim.kind == "recommendation" and claim.recommendation_description is not None
    ):
        return _recommendation_title(claim, context, semantic_checks=semantic_checks)
    elif claim.kind == "finding" and STAT.search(text):
        label, reason = _statistics(text, context)
    elif claim.kind == "hypothesis":
        # Contrast introduces a separate proposition: uncertainty in the first
        # clause does not make an asserted second clause tentative. Check facts
        # using the existing provenance rules, not generated text as evidence.
        contrast = re.split(
            r"\s+(?:tetapi|namun|padahal|sedangkan)\s+",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(contrast) == 2:
            left = evaluate_claim(
                claim.model_copy(update={"text": contrast[0]}),
                context,
                semantic_checks=semantic_checks,
            )
            right = contrast[1]
            # "Owner menyatakan: mungkin ..." asserts what the owner said;
            # uncertainty inside that attribution cannot waive source proof.
            tentative = bool(TENTATIVE.search(right)) and not _ATTRIBUTION.match(right)
            other = evaluate_claim(
                claim.model_copy(
                    update={
                        "kind": "hypothesis" if tentative else "finding",
                        "text": "Hipotesis: " + right if tentative else right,
                    }
                ),
                context,
                semantic_checks=semantic_checks,
            )
            return GroundingDecision(
                claim_id=claim.claim_id,
                label=GroundingLabel.NOT_APPLICABLE
                if left.label in ACCEPTED_LABELS and other.label in ACCEPTED_LABELS
                else GroundingLabel.INSUFFICIENT_EVIDENCE,
                reason="hypothesis_contrast_checked_independently",
            )
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
            # "saat ini" inside an all-tentative hypothesis names the current
            # practice ("metode penyegelan saat ini mungkin ..."), not a new fact.
            and not _ACHIEVED.search(text)
            and (not semantic_checks or _related_hypothesis(text, sources))
            and all(
                _normal(part) in {"perlu diverifikasi", "belum diketahui"}
                or not semantic_checks
                or _related_hypothesis(part, sources)
                for part in clauses
            )
            and not _cause_asserts_fact(text)
        ):
            label, reason = (
                GroundingLabel.NOT_APPLICABLE,
                "tentative_hypothesis_not_established_cause",
            )
        elif (
            not _PREMISE.search(text)
            and not _CURRENT_ASSERTION.search(text)
            and not _ACHIEVED.search(text)
            and _tentative_mechanism(text, sources)
        ):
            label, reason = GroundingLabel.NOT_APPLICABLE, "bounded_tentative_mechanism"
    elif claim.kind in {"recommendation", "metric", "target", "timeframe"}:
        body = text.removeprefix("Usulan:").strip()
        if claim.kind == "recommendation":
            parts = _independent_actions(body)
            if len(parts) > 1:
                # Each independent action must pass the SAME checks with the
                # SAME actual citations. No source comes from a sibling action.
                prefix = "Usulan: " if text.startswith("Usulan:") else ""
                decisions = [
                    evaluate_claim(
                        claim.model_copy(update={"text": prefix + part}),
                        context,
                        semantic_checks=semantic_checks,
                    )
                    for part in parts
                ]
                return GroundingDecision(
                    claim_id=claim.claim_id,
                    label=GroundingLabel.NOT_APPLICABLE
                    if all(d.label in ACCEPTED_LABELS for d in decisions)
                    else GroundingLabel.INSUFFICIENT_EVIDENCE,
                    reason="independent_actions_checked_separately",
                )
        # Explicit target/timeframe fields are proposals, not current resources.
        # Check the existing operational premises embedded in actions instead.
        operations_supported = True
        if claim.kind == "recommendation":
            _, operations_supported = operational_number_context(
                text, sources, action_window=True
            )
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
        # Typed indicator fields define a variable, goal or evaluation window.
        # They do not need an imperative verb. Explicit proposed framing and the
        # absence of current/historical assertions matter, not their first word.
        indicator = claim.kind in {"metric", "target", "timeframe"}
        framing = (
            claim.kind == "metric" or text.startswith("Usulan:")
            if indicator
            else _action_framing(action)
            or (not semantic_checks and _source_bound_nominal(action, sources))
        )
        # Separate sentences cannot hide a new fact behind an initial imperative.
        independent_parts = re.split(r"(?<!\d)[.;]|[.;](?!\d)", action)
        clauses_ok = indicator or all(
            _action_framing(part.strip())
            or (not semantic_checks and _source_bound_nominal(part.strip(), sources))
            for part in independent_parts
            if part.strip()
        )
        if (
            not _ACHIEVED.search(body)
            and not _CURRENT_ASSERTION.search(body)
            and not (
                claim.kind == "metric"
                and re.search(
                    r"\b(meningkat|bertambah|berkurang|naik|turun)\b",
                    body,
                    re.IGNORECASE,
                )
            )
            and not _PREMISE.search(action)
            and operations_supported
            and premise_ok
            and framing
            and clauses_ok
        ):
            label, reason = (
                GroundingLabel.NOT_APPLICABLE,
                "proposal_with_no_unproved_factual_premise",
            )
        else:
            reason = "proposal_contains_unproved_premise_or_unresolved_language"
        if claim.kind == "recommendation":
            if (
                not _PREMISE.search(body)
                and not _ACHIEVED.search(body)
                and not _CURRENT_ASSERTION.search(body)
                and _nominal_proposal(body, sources)
            ):
                label, reason = (
                    GroundingLabel.NOT_APPLICABLE,
                    "bounded_nominal_proposal",
                )
            action_object = _action_before_purpose(action)
            if (
                semantic_checks
                and not _related_hypothesis(action_object, sources)
                and not re.search(
                    r"\b(alur|pesanan|antrean|pencatatan|shift|durasi|transaksi)\b",
                    action_object,
                    re.IGNORECASE,
                )
            ):
                label, reason = (
                    GroundingLabel.INSUFFICIENT_EVIDENCE,
                    "action_has_no_cited_subject",
                )
    else:
        label, reason = _fact(claim, sources)
    return GroundingDecision(claim_id=claim.claim_id, label=label, reason=reason)


_GUARANTEE = re.compile(
    r"\b(pasti|dijamin|menjamin|jaminan|garansi|guaranteed|guarantee)\b", re.IGNORECASE
)
# Asserted conditions/resources that advice must not introduce as facts.
_PROPOSAL_FACT = re.compile(
    r"\b(rusak|lelah|tersedia|ketersediaan|memiliki|dimiliki)\b", re.IGNORECASE
)
_RESOURCE = (
    r"(staf|karyawan|pegawai|barista|kasir|mesin|alat|peralatan|aplikasi|sistem|"
    r"cabang|kendaraan)"
)
# Committing to resources the owner never confirmed (budget/staff/equipment).
_RESOURCE_COMMIT = re.compile(
    r"\b(penambahan|pengadaan|pembelian|membeli|beli|merekrut|rekrut|perekrutan|"
    r"menambah|tambah|tambahkan)\s+(?:\w+\s+){0,2}?" + _RESOURCE + r"\b",
    re.IGNORECASE,
)
_NEW_RESOURCE = re.compile(
    r"\b" + _RESOURCE + r"\s+(baru|tambahan|otomatis|modern|digital)\b", re.IGNORECASE
)
_OUTCOME = re.compile(
    r"\b(meningkat|naik|turun|menurun|berkurang|bertambah|puas)\b", re.IGNORECASE
)
_PURPOSE = re.compile(
    r"\b(untuk|agar|supaya|diharapkan|harapannya|target|sasaran|guna)\b", re.IGNORECASE
)
_PROPOSAL_KINDS = {"recommendation", "metric", "target", "timeframe"}
# In advice, "sudah" usually states a condition ("pastikan roti sudah dingin sebelum
# dikemas"); only an achieved RESULT is blocked ("sudah meningkat", "tercapai").
_ACHIEVED_RESULT = re.compile(
    r"\b(tercapai|achieved|already)\b"
    r"|\b(telah|sudah)\s+(?:\w+\s+)?(meningkat\w*|naik\w*|turun\w*|menurun\w*|"
    r"berkurang|mengurangi\w*|bertambah|menambah\w*|membaik|memperbaiki|berhasil|"
    r"teratasi|terselesaikan|efektif|terbukti|cepat|lancar|bagus|optimal|maksimal)\b",
    re.IGNORECASE,
)


def _asserted_outcome(text: str) -> bool:
    """An outcome stated as a result ("Pendapatan meningkat"), not as a purpose."""
    for clause in re.split(r"[.;!?]", text):
        for match in _OUTCOME.finditer(clause):
            if not _PURPOSE.search(clause[: match.start()]):
                return True
    return False


def _asserts_with(
    pattern: re.Pattern, text: str, negators: set[str], window: int = 2
) -> bool:
    for match in pattern.finditer(text):
        before = _normal(text[: match.start()]).split()[-window:]
        if not negators.intersection(before):
            return True
    return False


# Words that turn a condition into a risk/goal rather than an asserted state
# ("risiko ... rusak", "agar tidak rusak", "mencegah kemasan rusak").
_RISK_HEDGES = _NEGATORS | {
    "risiko",
    "potensi",
    "kemungkinan",
    "mencegah",
    "menghindari",
    "agar",
    "supaya",
    "jika",
    "bila",
    "apabila",
}
# A hedged cause in a hypothesis ("mungkin disebabkan oleh ...") is tentative.
_HYPOTHESIS_HEDGES = _NEGATORS | {"mungkin", "diduga", "kemungkinan", "bisa"}


def _cause_asserts_fact(text: str) -> bool:
    """A tentative hypothesis may name a cause, but the cause must not be a new fact."""
    _, *cause = re.split(
        r"\s+(?:karena|sebab|mengingat|because)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if not cause:
        return False
    return bool(
        _asserts_with(_PROPOSAL_FACT, cause[0], _NEGATORS)
        or _PREMISE.search(cause[0])
        or _asserted_outcome(cause[0])
    )


def evaluate_proposal(
    claim: GroundingClaim, context: ConsultationContext
) -> GroundingDecision:
    """Advice is checked as advice, not as a fact (production path, P-B1).

    No verb whitelist: any actionable wording is allowed. Still blocked: promised
    outcomes, already-achieved results, universal or causal assertions, asserted
    broken equipment, unproven operational numbers, and unsupported "karena" premises.
    Numbers/ID namespaces are also checked structurally in solution.validate_solution.
    """

    def decision(label: GroundingLabel, reason: str) -> GroundingDecision:
        return GroundingDecision(claim_id=claim.claim_id, label=label, reason=reason)

    text = claim.text.strip()
    body = text.removeprefix("Usulan:").strip()
    if not body:
        return decision(GroundingLabel.INSUFFICIENT_EVIDENCE, "empty_proposal")
    if _GUARANTEE.search(body) or _ACHIEVED_RESULT.search(body):
        return decision(
            GroundingLabel.INSUFFICIENT_EVIDENCE, "proposal_promises_outcome"
        )
    if _UNIVERSAL.search(body) or _asserts(_CERTAINTY, body):
        return decision(
            GroundingLabel.INSUFFICIENT_EVIDENCE, "proposal_asserts_scope_or_cause"
        )
    if claim.kind in {"target", "timeframe"}:
        # Future goals ("tidak ada kemasan rusak") are not factual premises.
        return decision(
            GroundingLabel.NOT_APPLICABLE, "proposal_without_unproven_assertion"
        )
    if claim.kind == "metric":
        # A metric names what to measure ("Keluhan kemasan rusak"); it may name a
        # condition but must not state a result.
        if _OUTCOME.search(body):
            return decision(
                GroundingLabel.INSUFFICIENT_EVIDENCE, "metric_asserts_outcome"
            )
        return decision(
            GroundingLabel.NOT_APPLICABLE, "proposal_without_unproven_assertion"
        )
    if _asserts_with(_PROPOSAL_FACT, body, _RISK_HEDGES, window=4):
        return decision(GroundingLabel.INSUFFICIENT_EVIDENCE, "proposal_asserts_fact")
    if _asserts_with(_RESOURCE_COMMIT, body, _NEGATORS | {"tanpa"}) or _asserts_with(
        _NEW_RESOURCE, body, _NEGATORS | {"tanpa"}
    ):
        return decision(
            GroundingLabel.INSUFFICIENT_EVIDENCE,
            "proposal_assumes_unconfirmed_resource",
        )
    if claim.kind == "metric" and _OUTCOME.search(body):
        return decision(GroundingLabel.INSUFFICIENT_EVIDENCE, "metric_asserts_outcome")
    if claim.kind == "recommendation" and _asserted_outcome(body):
        return decision(
            GroundingLabel.INSUFFICIENT_EVIDENCE, "proposal_asserts_outcome"
        )
    if claim.kind == "recommendation":
        # A recommendation TITLE is a label: it must not assert a fact either.
        if claim.recommendation_description is not None and (
            _LABEL_ASSERTION.search(body)
            or _CURRENT_ASSERTION.search(body)
            # A short title has no room for a condition: "sudah" claims a state.
            or _ACHIEVED.search(body)
        ):
            return decision(
                GroundingLabel.INSUFFICIENT_EVIDENCE, "title_asserts_fact_or_outcome"
            )
        sources = cited_sources(claim, context)
        _, operations_supported = operational_number_context(
            text, sources, action_window=True
        )
        if not operations_supported:
            return decision(
                GroundingLabel.INSUFFICIENT_EVIDENCE, "operational_number_unproven"
            )
        _, *premise = re.split(
            r"\s+(?:karena|sebab|mengingat|because)\s+",
            body,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
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
            if verdict.label != GroundingLabel.SUPPORTED:
                return decision(
                    GroundingLabel.INSUFFICIENT_EVIDENCE, "proposal_premise_unsupported"
                )
    return decision(
        GroundingLabel.NOT_APPLICABLE, "proposal_without_unproven_assertion"
    )


def production_verdict(
    claim: GroundingClaim, context: ConsultationContext
) -> GroundingDecision:
    """The blocking check used for returned Solutions."""
    if claim.kind in _PROPOSAL_KINDS:
        return evaluate_proposal(claim, context)
    return evaluate_claim(claim, context, semantic_checks=False)


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
                            recommendation_description=item.description
                            if kind == "recommendation" and field == "title"
                            else None,
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
    """Block evidence failures; return semantic diagnostics without promoting facts.

    Diagnostics may contain INSUFFICIENT_EVIDENCE even for a returned Solution.
    This never labels an unresolved semantic judgment as factual support. No
    source text or diagnostic is logged or added to the public protobuf response.
    """
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
        evidence = production_verdict(claim, context)
        if evidence.label not in ACCEPTED_LABELS:
            raise ConsultationFailure(
                grpc.StatusCode.INTERNAL,
                f"Solution grounding rejected {evidence.claim_id}: {evidence.label.value}",
            )
        decisions.append(evaluate_claim(claim, context))
    if deadline is not None and time.monotonic() >= deadline:
        raise ConsultationFailure(
            grpc.StatusCode.DEADLINE_EXCEEDED, "Solution grounding deadline exceeded"
        )
    return decisions
