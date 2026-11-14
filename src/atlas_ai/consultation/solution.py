"""Typed solution generation, deterministic structural checks and protobuf mapping."""

import logging
import re
import time
from collections.abc import Callable
from typing import Annotated, Literal

import grpc
from pydantic import BeforeValidator, Field, StringConstraints, WithJsonSchema

from atlas.ai.v1 import ai_pb2
from atlas_ai import observability
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    compact_provider_schema,
    invoke_structured,
)
from atlas_ai.consultation.grounding import (
    ACCEPTED_LABELS,
    NUMBER_WORDS,
    TENTATIVE,
    GroundingClaim,
    attributed_text,
    cited_sources,
    operational_number_context,
    production_verdict,
    solution_claims,
    validate_grounding,
    verbatim_quote_supported,
)
from atlas_ai.consultation.memory import MemoryProposals, ProposedMemoryCandidate
from atlas_ai.consultation.models import (
    ConsultationContext,
    ContextModel,
    Finding,
    Hypothesis,
    Recommendation,
    References,
    SolutionEvidence,
    SuccessIndicator,
)
from atlas_ai.consultation.prompts import SOLUTION_PROMPT
from atlas_ai.consultation.relevance import (
    RELEVANCE_PROMPT_VERSION,
    RelevanceSettings,
    check_relevance,
)

logger = logging.getLogger(__name__)

Text = Annotated[str, StringConstraints(min_length=1, max_length=3000)]
Label = Annotated[
    Literal["", "low", "medium", "high", "rendah", "sedang", "tinggi"],
    BeforeValidator(lambda value: "" if value is None else value),
    # Provider enums require nonempty strings. Provider null means unspecified;
    # normalize to the existing legal protobuf empty label before local validation.
    WithJsonSchema(
        {
            "anyOf": [
                {
                    "type": "string",
                    "enum": ["low", "medium", "high", "rendah", "sedang", "tinggi"],
                },
                {"type": "null"},
            ]
        }
    ),
]


class GeneratedFinding(Finding):
    statement: Text
    references: References


class GeneratedEvidence(SolutionEvidence):
    quote: Text
    note: str = Field(default="", max_length=3000)


class GeneratedHypothesis(Hypothesis):
    statement: Text
    confidence: Label
    references: References


class GeneratedRecommendation(Recommendation):
    title: Text
    description: str = Field(max_length=3000)
    priority: Label
    references: References


class GeneratedIndicator(SuccessIndicator):
    metric: Text
    target: str = Field(max_length=3000)
    timeframe: str = Field(max_length=3000)


class GeneratedSolution(ContextModel):
    # Separate constraints from permissive historical models: history isn't output.
    findings: list[GeneratedFinding] = Field(min_length=1, max_length=20)
    customer_evidence: list[GeneratedEvidence] = Field(min_length=1, max_length=20)
    hypotheses: list[GeneratedHypothesis] = Field(min_length=1, max_length=20)
    recommendations: list[GeneratedRecommendation] = Field(min_length=1, max_length=3)
    success_indicators: list[GeneratedIndicator] = Field(min_length=1, max_length=20)


class GeneratedSolutionTurn(GeneratedSolution):
    """Flat LLM turn output; metadata is excluded from the five-field wire Solution."""

    memory_candidates: MemoryProposals = Field(default_factory=list)

    def solution_output(self) -> GeneratedSolution:
        return GeneratedSolution.model_validate(
            self.model_dump(exclude={"memory_candidates"})
        )


def solution_provider_schema() -> dict:
    return compact_provider_schema(GeneratedSolutionTurn)


NUMBER = re.compile(r"\d")
ENUMERATOR = re.compile(
    r"(?:\b(?:usulan\s+)?(?:alternatif|opsi|pilihan|langkah|tahap)\s+\d{1,2}\b\s*:?)"
    r"|(?m:^\s*\d{1,2}[.)]\s+)",
    re.IGNORECASE,
)
SQL_QUANTITY = re.compile(r"\b(opini|feedback|rata[- ]rata rating)\b", re.IGNORECASE)
CERTAINTY = re.compile(
    r"\b(terbukti|dipastikan|penyebab pasti|proven|confirmed cause)\b", re.IGNORECASE
)


def _reject(message: str) -> None:
    raise ConsultationFailure(
        grpc.StatusCode.INTERNAL, f"Invalid solution output: {message}"
    )


def _sources(refs: References, context: ConsultationContext) -> list[dict[str, str]]:
    for ids, registry in (
        (refs.opinion_ids, context.sources.opinions),
        (refs.owner_message_ids, context.sources.owner_messages),
        (refs.memory_ids, context.sources.memories),
    ):
        for source_id in ids:
            if source_id not in registry:
                _reject("reference ID is unavailable in its source namespace")
    return cited_sources(
        GroundingClaim(claim_id="sources", kind="finding", text="", references=refs),
        context,
    )


def _check_numbers(
    text: str,
    sources: list[dict[str, str]],
    context: ConsultationContext,
    *,
    proposal=False,
) -> None:
    if "[[stat:" in text:
        _reject("aggregate statistics are context-only in MVP Solution output")
    # List numbering in advice ("Usulan alternatif 1:", "1. ...") is not a fact.
    rest = ENUMERATOR.sub(" ", text) if proposal else text
    is_proposal = proposal and text.startswith("Usulan:")
    if (
        not is_proposal
        and SQL_QUANTITY.search(rest)
        and NUMBER_WORDS.search(rest.casefold())
    ):
        _reject(
            "IssueStats aggregate quantities are context-only in MVP Solution output"
        )
    if NUMBER.search(rest):
        if SQL_QUANTITY.search(rest) and not is_proposal:
            _reject(
                "IssueStats aggregate quantities are context-only in MVP Solution output"
            )
        exact_excerpt = any(
            attributed_text(text).strip() in source["text"] for source in sources
        )
        if exact_excerpt and not proposal:
            # Faithful attributed denials/uncertain owner reports may contain
            # numbers too; grounding still checks attribution and every clause.
            return
        residual, operational_supported = operational_number_context(
            rest, sources, action_window=proposal
        )
        if not operational_supported:
            _reject("operational number lacks matching cited owner/memory provenance")
        if not NUMBER.search(residual):
            return
        if is_proposal:
            return
        if not exact_excerpt:
            _reject(
                "unverified numeric factual text; cite matching operational provenance or an exact source excerpt"
            )


def validate_solution(
    solution: GeneratedSolution, context: ConsultationContext, *, notes: bool = True
) -> None:
    """Raise on the first structural problem (all items must be valid).

    notes=False is the production structural stage: it skips the optional
    customer_evidence.note (blanked) and hypothesis framing (dropped), which
    prune_solution handles item by item. Fabrication checks always raise.
    """
    for item in (*solution.findings, *solution.hypotheses, *solution.recommendations):
        _check_claim_item(item, context, framing=notes)
    for item in solution.customer_evidence:
        _check_quote(item, context)
        if notes:
            _check_note(item, context)
    for item in solution.success_indicators:
        _check_indicator(item, context)


def check_fabrication(
    solution: GeneratedSolution, context: ConsultationContext
) -> None:
    """Invented sources: reference IDs outside their namespace, non-verbatim quotes."""
    for item in (*solution.findings, *solution.hypotheses, *solution.recommendations):
        _sources(item.references, context)
    for item in solution.customer_evidence:
        _check_quote(item, context)


def _check_claim_item(
    item, context: ConsultationContext, *, framing: bool = True
) -> None:
    sources = _sources(item.references, context)
    text = item.title if isinstance(item, GeneratedRecommendation) else item.statement
    if not text.strip():
        _reject("required text is blank")
    if not sources:
        _reject("claim/action has no current sources")
    _check_numbers(
        text, sources, context, proposal=isinstance(item, GeneratedRecommendation)
    )
    if isinstance(item, GeneratedRecommendation):
        # Local output policy, not a protobuf/historical-model constraint.
        # Keep the provider schema and existing application error boundary.
        if not item.description.strip():
            _reject("recommendation description is blank")
        _check_numbers(item.description, sources, context, proposal=True)
    if isinstance(item, GeneratedFinding) and re.search(
        r"\b(disebabkan|penyebabnya|caused by|because of)\b", text, re.IGNORECASE
    ):
        _reject("finding asserts a causal explanation")
    if isinstance(item, GeneratedHypothesis) and framing:
        _check_hypothesis_framing(item)


def _check_hypothesis_framing(item) -> None:
    """Wording, not fabrication: production drops a badly framed hypothesis."""
    text = item.statement
    if (
        not text.startswith("Hipotesis:")
        or CERTAINTY.search(text)
        or not TENTATIVE.search(text)
    ):
        _reject("hypothesis must be explicitly tentative")


def _check_quote(item, context: ConsultationContext) -> None:
    source = context.sources.opinions.get(item.opinion_id)
    if source is None or not verbatim_quote_supported(
        item.quote, source.evidence_text, source.full_feedback
    ):
        _reject("customer quote is not verbatim in the cited source")


def _check_note(item, context: ConsultationContext) -> None:
    _check_numbers(item.note, [], context)


def _check_indicator(item, context: ConsultationContext) -> None:
    if not item.metric.strip():
        _reject("success metric is blank")
    _check_numbers(item.metric, [], context)
    for value in (item.target, item.timeframe):
        if value and (
            not value.startswith("Usulan:") or not value.removeprefix("Usulan:").strip()
        ):
            _reject("success target/timeframe must be explicitly proposed")
        if "[[stat:" in value:
            _reject("statistic markers are not proposed targets")
        if re.search(
            r"\b(telah|sudah|tercapai|achieved|already)\b", value, re.IGNORECASE
        ):
            _reject("proposed success indicator claims a historical achievement")


SECTIONS = (
    "findings",
    "customer_evidence",
    "hypotheses",
    "recommendations",
    "success_indicators",
)


def prune_solution(
    solution: GeneratedSolution, context: ConsultationContext
) -> tuple[GeneratedSolution, list[tuple[str, str]]]:
    """Drop weakly grounded items instead of failing the whole turn (P-B1).

    Policy:
    - Fabrication (a non-verbatim quote, an invented or wrong-namespace reference
      ID) REJECTS the whole output (checked in check_fabrication): a model that
      invents sources is not trusted for the rest of that answer.
    - Any other failing item is DROPPED: unverifiable numbers or statistics,
      asserted causes in findings, untentative hypotheses, unproven advice, blank
      or uncited text. Every remaining item passes the same checks as before and
      nothing is rewritten. A dropped item never reaches the output.
    - An invalid optional customer_evidence.note is blanked.
    - Fails when a section would be empty (all five are required by Go).
    """
    dropped: list[tuple[str, str]] = []
    claims: dict[str, list] = {}
    for claim in solution_claims(solution):
        claims.setdefault(claim.claim_id.rsplit(".", 1)[0], []).append(claim)

    def grounding_problem(key: str) -> tuple[str, str] | None:
        for claim in claims.get(key, []):
            verdict = production_verdict(claim, context)
            if verdict.label not in ACCEPTED_LABELS:
                return claim.claim_id, f"{verdict.label.value}:{verdict.reason}"
        return None

    kept: dict[str, list] = {name: [] for name in SECTIONS}
    for name in SECTIONS:
        for index, item in enumerate(getattr(solution, name)):
            key = f"{name}[{index}]"
            if name == "customer_evidence":
                _check_quote(item, context)  # Fabrication: always fatal.
            else:
                try:
                    if name == "success_indicators":
                        _check_indicator(item, context)
                    else:
                        _check_claim_item(item, context, framing=False)
                except ConsultationFailure as exc:
                    if "namespace" in str(exc):
                        raise  # Invented/wrong-namespace reference ID.
                    dropped.append(
                        (key, str(exc).removeprefix("Invalid solution output: "))
                    )
                    continue
            if name == "hypotheses":
                try:
                    _check_hypothesis_framing(item)
                except ConsultationFailure:
                    dropped.append((f"{key}.statement", "not_tentative"))
                    continue
            if name == "customer_evidence":
                note_problem = None
                try:
                    _check_note(item, context)
                except ConsultationFailure as exc:
                    note_problem = str(exc)
                note_problem = note_problem or grounding_problem(key)
                if note_problem:
                    dropped.append((f"{key}.note", "blanked"))
                    item = item.model_copy(update={"note": ""})
                kept[name].append(item)
                continue
            problem = grounding_problem(key)
            if problem:
                dropped.append(problem)
                continue
            kept[name].append(item)

    empty = [name for name in SECTIONS if not kept[name]]
    if empty:
        reasons = "; ".join(
            f"{key}: {why}" for key, why in dropped if key.split("[")[0] in empty
        )
        raise ConsultationFailure(
            grpc.StatusCode.INTERNAL,
            f"Solution grounding rejected {reasons} ({', '.join(empty)} empty)",
        )
    return GeneratedSolution.model_validate(
        {name: [item.model_dump() for item in kept[name]] for name in SECTIONS}
    ), dropped


def solution_to_proto(
    solution: GeneratedSolution,
    context: ConsultationContext,
    *,
    time_remaining: float | None = None,
    is_active: Callable[[], bool] | None = None,
    timings: dict[str, float] | None = None,
    refine: Callable[[GeneratedSolution, float | None], GeneratedSolution]
    | None = None,
) -> ai_pb2.Solution:
    """Validate and map a generated Solution.

    ``refine`` (optional) runs after pruning and before the final grounding pass,
    with the remaining budget; production uses it for the B2 relevance check.
    """
    started = time.monotonic()
    stage_started = time.monotonic() if timings is not None else 0
    try:
        # Fabrication (non-verbatim quotes, invented reference IDs) rejects the output.
        with observability.span("validation.fabrication"):
            check_fabrication(solution, context)
    finally:
        if timings is not None:
            timings["solution.structural"] = time.monotonic() - stage_started
    stage_started = time.monotonic() if timings is not None else 0
    try:
        # Drop weakly grounded items; the remainder must then pass every check.
        with observability.span("validation.grounding"):
            solution, dropped = prune_solution(solution, context)
        if dropped:
            # Claim IDs and reason codes only; never generated or source text.
            logger.info(
                "solution_pruned dropped=%s",
                ",".join(f"{key}={why[:80]}" for key, why in dropped),
            )
        if refine is not None:
            solution = refine(
                solution,
                None
                if time_remaining is None
                else time_remaining - (time.monotonic() - started),
            )
        validate_grounding(
            solution,
            context,
            time_remaining=None
            if time_remaining is None
            else time_remaining - (time.monotonic() - started),
            is_active=is_active,
        )
    finally:
        if timings is not None:
            timings["solution.grounding"] = time.monotonic() - stage_started
    stage_started = time.monotonic() if timings is not None else 0
    data = solution.model_dump(mode="json")
    result = ai_pb2.Solution(**data)
    if timings is not None:
        timings["solution.protobuf_mapping"] = time.monotonic() - stage_started
    return result


class SolutionGenerator:
    def __init__(
        self,
        assessor: ContextAssessor,
        *,
        timings: dict[str, float] | None = None,
        relevance: RelevanceSettings | None = None,
    ):
        self.assessor = assessor
        self.timings = timings
        self.relevance = relevance or RelevanceSettings.from_env()

    def _relevance(self, context, is_active):
        def refine(solution, remaining):
            started = time.monotonic()
            solution, status, off_topic = check_relevance(
                solution,
                context,
                self.assessor.settings,
                self.assessor.model_factory,
                self.relevance,
                time_remaining=remaining,
                is_active=is_active,
                timings=self.timings,
            )
            if self.timings is not None:
                self.timings["solution.relevance"] = time.monotonic() - started
            # Item IDs and status only; never generated or source text.
            logger.info(
                "solution_relevance version=%s status=%s off_topic=%s",
                RELEVANCE_PROMPT_VERSION,
                status,
                ",".join(off_topic) or "-",
            )
            return solution

        return refine

    def generate(
        self,
        context: ConsultationContext,
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
    ) -> tuple[ai_pb2.Solution, list[ProposedMemoryCandidate]]:
        if not any(
            item.evidence_text.strip() or item.full_feedback.strip()
            for item in context.representative_evidence
        ):
            raise ConsultationFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Solution requires verifiable customer evidence; no-evidence policy needs Go agreement",
            )
        started = time.monotonic()
        result = invoke_structured(
            context,
            self.assessor.settings,
            self.assessor.model_factory,
            SOLUTION_PROMPT,
            GeneratedSolutionTurn,
            stage="Solution",
            time_remaining=time_remaining,
            is_active=is_active,
            provider_schema=solution_provider_schema(),
            timings=self.timings,
        )
        solution = result.solution_output()
        response = solution_to_proto(
            solution,
            context,
            time_remaining=None
            if time_remaining is None
            else time_remaining - (time.monotonic() - started),
            is_active=is_active,
            timings=self.timings,
            refine=self._relevance(context, is_active),
        )
        if time_remaining is not None and time.monotonic() - started >= time_remaining:
            raise ConsultationFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Solution deadline exceeded"
            )
        if is_active is not None and not is_active():
            raise ConsultationFailure(
                grpc.StatusCode.CANCELLED, "Consultation cancelled"
            )
        return response, result.memory_candidates
