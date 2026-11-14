"""Issue matching for ProcessFeedback (Path A, P-A5).

E5 narrows the candidates, an LLM judge decides the merge. A wrong merge is the most
harmful error, so E5 similarity alone never merges (opposite-direction pairs can
score higher than same-problem pairs, see evaluation/feedback_p_a2_*). Any judge
failure falls back to a new issue; matching never fails the feedback.

Only NEGATIVE opinions form planets in Go (PRD ADR-A08), so only they are matched.
"""

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Literal

import grpc
import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.feedback.embeddings import E5Embedder, filter_candidate_vectors
from atlas_ai.feedback.models import ValidatedOpinion
from atlas_ai.feedback.prompts import MATCH_PROMPT
from atlas_ai.llm import (
    MIN_CALL_SECONDS,
    AIFailure,
    LLMSettings,
    compact_provider_schema,
    invoke_llm_structured,
)

logger = logging.getLogger(__name__)
NONE = "NONE"


class MatchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # Provisional values, NOT calibrated: E5 cosine is not a probability. In bakery
    # text different problems score ~0.855-0.865 and the same problem ~0.89-0.90
    # (2026-11-14), so the gate is set for recall. It only decides whether the
    # LLM judge is consulted; it never merges by itself.
    threshold: float = Field(default=0.85, ge=0, le=1)
    top_k: int = Field(default=3, ge=1, le=10)
    # One small call; 10 s leaves room for the one bounded retry on a provider 5xx.
    # The RPC deadline still caps the actual call.
    judge_timeout_seconds: float = Field(default=10, ge=1, le=20)
    judge_enabled: bool = True

    @classmethod
    def from_env(cls) -> "MatchSettings":
        try:
            return cls(
                threshold=os.getenv("FEEDBACK_MATCH_THRESHOLD", "0.85"),
                top_k=os.getenv("FEEDBACK_MATCH_TOP_K", "3"),
                judge_timeout_seconds=os.getenv("FEEDBACK_MATCH_TIMEOUT_SECONDS", "10"),
                judge_enabled=os.getenv("FEEDBACK_MATCH_JUDGE", "true"),
            )
        except ValidationError:
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION, "Invalid matching configuration"
            ) from None


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    issue_id: str
    title: str
    score: float


class MatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    opinion_index: int
    issue_id: str


class MatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decisions: list[MatchDecision]


class ProviderMatchOutput(BaseModel):
    """Provider grammar only."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    decisions: list[MatchDecision]


class MatchStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exact_title: int = 0
    judged: int = 0
    merged_by_judge: int = 0
    judge_status: Literal[
        "not_needed", "ok", "disabled", "skipped_budget", "failed"
    ] = "not_needed"
    skipped_vectors: int = 0


def title_key(title: str) -> str:
    return " ".join(title.casefold().split())


class TitleEmbeddingCache:
    """Bounded in-process cache of candidate-title embeddings, keyed by (issue_id, title)."""

    def __init__(self, max_entries: int = 20000):
        self._data: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get_many(
        self, embedder: E5Embedder, items: Sequence[tuple[str, str]], **kwargs
    ) -> dict[tuple[str, str], np.ndarray]:
        with self._lock:
            found = {key: self._data[key] for key in items if key in self._data}
            for key in found:
                self._data.move_to_end(key)
        missing = [key for key in items if key not in found]
        if missing:
            vectors = embedder.embed([title for _, title in missing], **kwargs)
            with self._lock:
                for key, vector in zip(missing, vectors, strict=True):
                    found[key] = self._data[key] = np.asarray(vector, dtype=np.float32)
                while len(self._data) > self._max:
                    self._data.popitem(last=False)
        return found


def retrieve(
    opinion_vectors: Sequence[Sequence[float]],
    request: ai_pb2.ProcessFeedbackRequest,
    embedder: E5Embedder,
    cache: TitleEmbeddingCache,
    settings: MatchSettings,
    **embed_kwargs,
) -> tuple[list[list[Candidate]], int]:
    """Top-k candidate issues per opinion: max(cosine vs issue's vectors, vs its title)."""
    titles: dict[str, str] = {}
    for issue in request.candidate_issues:
        if issue.title.strip():
            titles.setdefault(issue.issue_id, issue.title)
    if not titles or not opinion_vectors:
        return [[] for _ in opinion_vectors], 0

    kept, skipped = filter_candidate_vectors(request.candidate_vectors)
    by_issue: dict[str, list] = {}
    for vector in kept:
        if vector.issue_id in titles:  # Unknown issues have no title to judge.
            by_issue.setdefault(vector.issue_id, []).append(vector.embedding)
    issue_ids = list(titles)
    title_vectors = cache.get_many(
        embedder,
        [(issue_id, titles[issue_id]) for issue_id in issue_ids],
        **embed_kwargs,
    )
    title_matrix = np.stack([title_vectors[(i, titles[i])] for i in issue_ids])
    title_matrix /= np.linalg.norm(title_matrix, axis=1, keepdims=True)
    groups = {
        issue_id: _normalize(np.asarray(values, dtype=np.float32))
        for issue_id, values in by_issue.items()
    }

    results = []
    for raw in opinion_vectors:
        query = _normalize(np.asarray(raw, dtype=np.float32)[None, :])[0]
        scores = title_matrix @ query
        ranked = []
        for index, issue_id in enumerate(issue_ids):
            score = float(scores[index])
            if issue_id in groups:
                score = max(score, float((groups[issue_id] @ query).max()))
            if score >= settings.threshold:
                ranked.append(
                    Candidate(issue_id=issue_id, title=titles[issue_id], score=score)
                )
        ranked.sort(key=lambda item: item.score, reverse=True)
        results.append(ranked[: settings.top_k])
    return results, sum(skipped.values())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class MergeJudge:
    """One small LLM call per feedback; failures never fail the feedback."""

    def __init__(
        self,
        settings: LLMSettings,
        match: MatchSettings,
        model_factory: Callable = ChatOpenAI,
    ):
        self.settings = settings.model_copy(
            update={"timeout_seconds": match.judge_timeout_seconds}
        )
        self.model_factory = model_factory

    def decide(
        self,
        items: list[dict],
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
        timings: dict[str, float] | None = None,
    ) -> dict[int, str]:
        """items: [{opinion_index, opinion_text, sentiment, variant, candidates}]."""
        output = invoke_llm_structured(
            self.settings,
            self.model_factory,
            MATCH_PROMPT,
            MatchOutput,
            stage="Match",
            label="Feedback",
            config_names="FEEDBACK_MODEL and OPENAI_API_KEY",
            input_data={"opinions": items},
            time_remaining=time_remaining,
            is_active=is_active,
            provider_schema=compact_provider_schema(ProviderMatchOutput),
            timings=timings,
        )
        allowed = {
            item["opinion_index"]: {c["issue_id"] for c in item["candidates"]}
            for item in items
        }
        result: dict[int, str] = {}
        for decision in output.decisions:
            choices = allowed.get(decision.opinion_index)
            # Only an ID from THAT opinion's candidates counts; anything else is NONE.
            if choices is not None and decision.issue_id in choices:
                result.setdefault(decision.opinion_index, decision.issue_id)
        return result


def assign_issues(
    opinions: Sequence[ValidatedOpinion],
    vectors: Sequence[Sequence[float]],
    request: ai_pb2.ProcessFeedbackRequest,
    *,
    embedder: E5Embedder,
    cache: TitleEmbeddingCache,
    settings: MatchSettings,
    judge: MergeJudge | None,
    remaining: Callable[[], float | None],
    is_active: Callable[[], bool] | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[list[ai_pb2.IssueAssignment], MatchStats]:
    stats = MatchStats()
    existing: dict[int, str] = {}

    exact: dict[str, str] = {}
    for issue in request.candidate_issues:
        exact.setdefault(title_key(issue.title), issue.issue_id)
    pending = []
    for index, opinion in enumerate(opinions):
        if opinion.sentiment != "NEGATIVE":
            continue
        issue_id = exact.get(title_key(opinion.issue_title_hint))
        if issue_id:
            existing[index] = issue_id
            stats.exact_title += 1
        else:
            pending.append(index)

    if pending:
        candidates, stats.skipped_vectors = retrieve(
            [vectors[i] for i in pending],
            request,
            embedder,
            cache,
            settings,
            time_remaining=remaining(),
            is_active=is_active,
        )
        items = [
            {
                "opinion_index": index,
                "opinion_text": opinions[index].opinion_text,
                "sentiment": opinions[index].sentiment,
                "variant": opinions[index].variant,
                "candidates": [
                    {"issue_id": c.issue_id, "title": c.title} for c in found
                ],
            }
            for index, found in zip(pending, candidates, strict=True)
            if found
        ]
        if items:
            stats.judged = len(items)
            budget = remaining()
            if judge is None or not settings.judge_enabled:
                stats.judge_status = "disabled"
            elif budget is not None and budget < MIN_CALL_SECONDS + 0.5:
                # Not even one useful call fits: keep a new planet.
                stats.judge_status = "skipped_budget"
            else:
                try:
                    decided = judge.decide(
                        items,
                        time_remaining=budget,
                        is_active=is_active,
                        timings=timings,
                    )
                    existing.update(decided)
                    stats.merged_by_judge = len(decided)
                    stats.judge_status = "ok"
                except AIFailure as exc:
                    # Cancellation must still stop the RPC; everything else falls back.
                    if exc.code == grpc.StatusCode.CANCELLED:
                        raise
                    stats.judge_status = "failed"
                    logger.warning(
                        "feedback_match judge fallback code=%s", exc.code.name
                    )

    assignments = [
        ai_pb2.IssueAssignment(
            existing_issue_id=existing.get(index, ""),
            new_issue_title=opinion.issue_title_hint,
        )
        for index, opinion in enumerate(opinions)
    ]
    return assignments, stats
