"""Typed solution generation, deterministic structural checks and protobuf mapping."""

import re
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Literal

import grpc
from pydantic import BeforeValidator, Field, StringConstraints, WithJsonSchema

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    invoke_structured,
)
from atlas_ai.consultation.grounding import (
    NUMBER_WORDS,
    STAT,
    STAT_LABELS,
    GroundingClaim,
    attributed_text,
    cited_sources,
    operational_number_context,
    validate_grounding,
    verbatim_quote_supported,
)
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

Text = Annotated[str, StringConstraints(min_length=1, max_length=3000)]
Label = Annotated[
    Literal["", "low", "medium", "high", "rendah", "sedang", "tinggi"],
    BeforeValidator(lambda value: "" if value is None else value),
    # Gemini enums require nonempty strings. Provider null means unspecified;
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
    note: str = Field(max_length=3000)


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


def solution_provider_schema() -> dict:
    """Small native JSON-schema dialect; full Pydantic constraints remain local.

    Inline references and omit annotations/bounds in the provider grammar only.
    This never changes the typed output model, reference checks or grounding.
    """
    schema = GeneratedSolution.model_json_schema()
    definitions = schema.get("$defs", {})
    allowed = {
        "type",
        "properties",
        "items",
        "required",
        "enum",
        "anyOf",
        "additionalProperties",
    }

    def compact(value):
        if isinstance(value, list):
            return [compact(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return compact(definitions[value["$ref"].split("/")[-1]])
        return {
            key: (
                {name: compact(item) for name, item in part.items()}
                if key == "properties"
                else compact(part)
            )
            for key, part in value.items()
            if key in allowed
        }

    return {"title": "GeneratedSolution", **compact(schema)}


NUMBER = re.compile(r"\d")
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
    matches = list(STAT.finditer(text))
    for match in matches:
        field, value = match.groups()
        if context.aggregate_statistics is None or field not in STAT_LABELS:
            _reject("statistic is unavailable")
        expected = getattr(context.aggregate_statistics, field)
        if Decimal(value) != Decimal(str(expected)):
            _reject("statistic contradicts IssueStats")
    rest = STAT.sub("", text)
    if "[[stat:" in rest:
        _reject("malformed statistic marker")
    is_proposal = proposal and text.startswith("Usulan:") and not matches
    if (
        not matches
        and not is_proposal
        and SQL_QUANTITY.search(rest)
        and NUMBER_WORDS.search(rest.casefold())
    ):
        _reject("actual statistical quantities require verified IssueStats markers")
    if NUMBER.search(rest):
        if SQL_QUANTITY.search(rest) and not is_proposal:
            _reject("actual statistical quantities require verified IssueStats markers")
        exact_excerpt = any(
            attributed_text(text).strip() in source["text"] for source in sources
        )
        if exact_excerpt and not proposal:
            # Faithful attributed denials/uncertain owner reports may contain
            # numbers too; grounding still checks attribution and every clause.
            return
        residual, operational_supported = operational_number_context(rest, sources)
        if not operational_supported:
            _reject("operational number lacks matching cited owner/memory provenance")
        if not NUMBER.search(residual):
            return
        if is_proposal:
            return
        if not exact_excerpt:
            _reject(
                "unverified numeric factual text; use SQL markers or a cited exact excerpt"
            )


def validate_solution(
    solution: GeneratedSolution, context: ConsultationContext
) -> None:
    for item in (*solution.findings, *solution.hypotheses, *solution.recommendations):
        sources = _sources(item.references, context)
        text = (
            item.title if isinstance(item, GeneratedRecommendation) else item.statement
        )
        if not text.strip():
            _reject("required text is blank")
        if not sources and not (
            isinstance(item, GeneratedFinding) and STAT.search(text)
        ):
            _reject("claim/action has no current sources")
        _check_numbers(
            text, sources, context, proposal=isinstance(item, GeneratedRecommendation)
        )
        if isinstance(item, GeneratedRecommendation):
            _check_numbers(item.description, sources, context, proposal=True)
        if isinstance(item, GeneratedFinding) and re.search(
            r"\b(disebabkan|penyebabnya|caused by|because of)\b", text, re.IGNORECASE
        ):
            _reject("finding asserts a causal explanation")
        if isinstance(item, GeneratedHypothesis) and (
            not text.startswith("Hipotesis:")
            or CERTAINTY.search(text)
            or not re.search(
                r"\b(mungkin|diduga|belum diketahui|perlu diverifikasi)\b",
                text,
                re.IGNORECASE,
            )
        ):
            _reject("hypothesis must be explicitly tentative")
    for item in solution.customer_evidence:
        source = context.sources.opinions.get(item.opinion_id)
        if source is None or not verbatim_quote_supported(
            item.quote, source.evidence_text, source.full_feedback
        ):
            _reject("customer quote is not verbatim in the cited source")
        _check_numbers(item.note, [], context)
    for item in solution.success_indicators:
        if not item.metric.strip():
            _reject("success metric is blank")
        _check_numbers(item.metric, [], context)
        for value in (item.target, item.timeframe):
            if value and (
                not value.startswith("Usulan:")
                or not value.removeprefix("Usulan:").strip()
            ):
                _reject("success target/timeframe must be explicitly proposed")
            if "[[stat:" in value:
                _reject("statistic markers are not proposed targets")
            if re.search(
                r"\b(telah|sudah|tercapai|achieved|already)\b", value, re.IGNORECASE
            ):
                _reject("proposed success indicator claims a historical achievement")


def solution_to_proto(
    solution: GeneratedSolution,
    context: ConsultationContext,
    *,
    time_remaining: float | None = None,
    is_active: Callable[[], bool] | None = None,
) -> ai_pb2.Solution:
    started = time.monotonic()
    validate_solution(solution, context)
    validate_grounding(
        solution,
        context,
        time_remaining=None
        if time_remaining is None
        else time_remaining - (time.monotonic() - started),
        is_active=is_active,
    )
    # A SQL marker may be semantically verifiable without opinion references,
    # but Go's current wire validator still requires a reference on every item.
    # Fail explicitly instead of inventing a citation just to satisfy Go.
    for item in (*solution.findings, *solution.hypotheses, *solution.recommendations):
        if not (
            item.references.opinion_ids
            or item.references.owner_message_ids
            or item.references.memory_ids
        ):
            _reject("Go requires at least one current reference per claim/action")
    data = solution.model_dump(mode="json")

    # Replace verified internal markers only in prose; never modify quotes or UUIDs.
    def render(text: str) -> str:
        return STAT.sub(lambda m: f"{STAT_LABELS[m[1]]}: {m[2]}", text)

    for section, fields in (
        ("findings", ("statement",)),
        ("hypotheses", ("statement",)),
        ("recommendations", ("title", "description")),
        ("success_indicators", ("metric",)),
        ("customer_evidence", ("note",)),
    ):
        for item in data[section]:
            for field in fields:
                item[field] = render(item[field])
    return ai_pb2.Solution(**data)


class SolutionGenerator:
    def __init__(self, assessor: ContextAssessor):
        self.assessor = assessor

    def generate(
        self,
        context: ConsultationContext,
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
    ) -> ai_pb2.Solution:
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
            GeneratedSolution,
            stage="Solution",
            time_remaining=time_remaining,
            is_active=is_active,
            provider_schema=solution_provider_schema(),
        )
        response = solution_to_proto(
            result,
            context,
            time_remaining=None
            if time_remaining is None
            else time_remaining - (time.monotonic() - started),
            is_active=is_active,
        )
        if time_remaining is not None and time.monotonic() - started >= time_remaining:
            raise ConsultationFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Solution deadline exceeded"
            )
        if is_active is not None and not is_active():
            raise ConsultationFailure(
                grpc.StatusCode.CANCELLED, "Consultation cancelled"
            )
        return response
