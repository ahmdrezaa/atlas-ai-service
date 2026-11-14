"""ProcessFeedback orchestration (Path A).

validate + LLM extraction -> per-opinion validation -> E5 embeddings ->
issue matching -> protobuf. Matching (P-A5): exact title match first, then E5
retrieval narrows candidates and an LLM judge decides the merge; any judge
failure falls back to a new issue (see feedback/matching.py).
"""

import logging
import threading
import time
from collections.abc import Callable

import grpc

from atlas.ai.v1 import ai_pb2
from atlas_ai import observability
from atlas_ai.feedback.embeddings import (
    EMBEDDING_MODEL_ID,
    E5Embedder,
    EmbeddingSettings,
)
from atlas_ai.feedback.extractor import FeedbackExtractor, FeedbackLimits
from atlas_ai.feedback.matching import (
    MatchSettings,
    MatchStats,
    MergeJudge,
    TitleEmbeddingCache,
    assign_issues,
)
from atlas_ai.feedback.prompts import FEEDBACK_PROMPT_VERSION, MATCH_PROMPT_VERSION
from atlas_ai.feedback.validation import validate_feedback_request
from atlas_ai.llm import AIFailure

logger = logging.getLogger(__name__)

# Finish before the caller's deadline so Go receives our status, not its own timeout.
DEADLINE_MARGIN_SECONDS = 0.5

SENTIMENTS = {
    "POSITIVE": ai_pb2.SENTIMENT_POSITIVE,
    "NEGATIVE": ai_pb2.SENTIMENT_NEGATIVE,
    "NEUTRAL": ai_pb2.SENTIMENT_NEUTRAL,
}
SEVERITIES = {
    "SEVERE": ai_pb2.SEVERITY_SEVERE,
    "NOT_SEVERE": ai_pb2.SEVERITY_NOT_SEVERE,
}


class FeedbackPipeline:
    """Components load lazily (cheap construction); the server calls warm_up()."""

    def __init__(
        self,
        extractor: FeedbackExtractor | None = None,
        embedder: E5Embedder | None = None,
        judge: MergeJudge | None = None,
        match_settings: MatchSettings | None = None,
    ):
        self._extractor = extractor
        self._embedder = embedder
        self._judge = judge
        self._match_settings = match_settings
        self._title_cache = TitleEmbeddingCache()
        self._lock = threading.Lock()

    def _components(self):
        with self._lock:
            if self._extractor is None:
                self._extractor = FeedbackExtractor.from_env()
            if self._embedder is None:
                self._embedder = E5Embedder(EmbeddingSettings.from_env())
            if self._match_settings is None:
                self._match_settings = MatchSettings.from_env()
            if self._judge is None:
                # Same model/key as extraction, with its own short timeout.
                self._judge = MergeJudge(
                    self._extractor.settings,
                    self._match_settings,
                    self._extractor.model_factory,
                )
            return self._extractor, self._embedder, self._judge, self._match_settings

    def warm_up(self) -> None:
        """Load the E5 model and configuration once, before serving."""
        self._components()

    def run(
        self,
        request: ai_pb2.ProcessFeedbackRequest,
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
        timings: dict[str, float] | None = None,
    ) -> ai_pb2.ProcessFeedbackResponse:
        started = time.monotonic()
        # Malformed input is INVALID_ARGUMENT even when components are unavailable.
        limits = (
            self._extractor.limits if self._extractor else FeedbackLimits.from_env()
        )
        validate_feedback_request(request, max_text_chars=limits.max_text_chars)
        extractor, embedder, judge, match_settings = self._components()
        deadline = (
            None
            if time_remaining is None
            else started + time_remaining - DEADLINE_MARGIN_SECONDS
        )

        def remaining() -> float | None:
            return None if deadline is None else deadline - time.monotonic()

        with observability.span("extraction"):
            outcome = extractor.extract(
                request,
                time_remaining=remaining(),
                is_active=is_active,
                timings=timings,
            )
        if not outcome.opinions:
            self._log(request, outcome, MatchStats(), started)
            return ai_pb2.ProcessFeedbackResponse()

        with observability.span("embedding"):
            vectors = embedder.embed(
                [item.opinion_text for item in outcome.opinions],
                time_remaining=remaining(),
                is_active=is_active,
                timings=timings,
            )
        with observability.span("matching"):
            assignments, stats = assign_issues(
                outcome.opinions,
                vectors,
                request,
                embedder=embedder,
                cache=self._title_cache,
                settings=match_settings,
                judge=judge,
                remaining=remaining,
                is_active=is_active,
                timings=timings,
            )
        opinions = [
            ai_pb2.ExtractedOpinion(
                opinion_text=item.opinion_text,
                evidence_text=item.evidence_text,
                sentiment=SENTIMENTS[item.sentiment],
                severity=SEVERITIES[item.severity],
                embedding=vector,
                embedding_model=EMBEDDING_MODEL_ID,
                issue=assignment,
            )
            for item, vector, assignment in zip(
                outcome.opinions, vectors, assignments, strict=True
            )
        ]
        response = ai_pb2.ProcessFeedbackResponse(opinions=opinions)
        current = remaining()
        if current is not None and current <= 0:
            raise AIFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Feedback deadline exceeded"
            )
        if is_active is not None and not is_active():
            raise AIFailure(grpc.StatusCode.CANCELLED, "Feedback cancelled")
        self._log(request, outcome, stats, started)
        return response

    @staticmethod
    def _log(request, outcome, stats: MatchStats, started: float) -> None:
        # IDs, counts and reason codes only; never feedback text or names.
        logger.info(
            "process_feedback feedback_id=%s prompt_version=%s match_version=%s "
            "opinions=%d exact_title=%d judged=%d merged=%d judge=%s "
            "skipped_vectors=%d dropped=%s latency_ms=%d",
            request.feedback.id,
            FEEDBACK_PROMPT_VERSION,
            MATCH_PROMPT_VERSION,
            len(outcome.opinions),
            stats.exact_title,
            stats.judged,
            stats.merged_by_judge,
            stats.judge_status,
            stats.skipped_vectors,
            ",".join(reason.value for reason in outcome.dropped) or "-",
            (time.monotonic() - started) * 1000,
        )
