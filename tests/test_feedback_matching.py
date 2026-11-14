from unittest.mock import Mock
from uuid import uuid4

import grpc
import numpy as np
import pytest
from pydantic import SecretStr

from atlas.ai.v1 import ai_pb2
from atlas_ai.feedback.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL_ID
from atlas_ai.feedback.matching import (
    MatchSettings,
    MergeJudge,
    TitleEmbeddingCache,
    assign_issues,
    retrieve,
)
from atlas_ai.feedback.models import ValidatedOpinion
from atlas_ai.feedback.prompts import MATCH_PROMPT
from atlas_ai.llm import AIFailure, LLMSettings

SETTINGS = MatchSettings(threshold=0.8, top_k=2, judge_timeout_seconds=10)
GEMINI = LLMSettings(model="mock-gemini", api_key=SecretStr("test-only"))


def unit(*components):
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for index, value in components:
        vector[index] = value
    return (vector / np.linalg.norm(vector)).tolist()


# Deterministic "semantic" space for tests: dimension 0 = queue, 1 = taste, 2 = parking.
SPACE = {
    "Antrean Terlalu Lama": unit((0, 1.0)),
    "Antrean Kasir Terlalu Lama": unit((0, 1.0), (1, 0.1)),
    "Rasa Terlalu Asin": unit((1, 1.0)),
    "Parkiran Sempit": unit((2, 1.0)),
}


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [SPACE.get(text, unit((3, 1.0))) for text in texts]


def opinion(text, sentiment="NEGATIVE", title="Judul Baru"):
    return ValidatedOpinion(
        opinion_text=text,
        evidence_text=text,
        sentiment=sentiment,
        severity="NOT_SEVERE",
        variant=None,
        issue_title_hint=title,
    )


def request_with(issues, vectors=()):
    request = ai_pb2.ProcessFeedbackRequest(shop_id=str(uuid4()))
    for issue_id, title in issues:
        request.candidate_issues.add(issue_id=issue_id, title=title)
    for issue_id, values, model in vectors:
        request.candidate_vectors.add(
            opinion_id=str(uuid4()),
            issue_id=issue_id,
            embedding=values,
            embedding_model=model,
        )
    return request


def judge_returning(decisions=None, exc=None):
    factory = Mock()
    invoke = factory.return_value.with_structured_output.return_value.invoke
    if exc is not None:
        invoke.side_effect = exc
    else:
        invoke.return_value = {"decisions": decisions or []}
    return MergeJudge(GEMINI, SETTINGS, factory), factory


def run(opinions, request, judge, *, budget=20.0, settings=SETTINGS, embedder=None):
    embedder = embedder or FakeEmbedder()
    vectors = embedder.embed([o.opinion_text for o in opinions])
    return assign_issues(
        opinions,
        vectors,
        request,
        embedder=embedder,
        cache=TitleEmbeddingCache(),
        settings=settings,
        judge=judge,
        remaining=lambda: budget,
    )


QUEUE_ID, TASTE_ID, PARKING_ID = str(uuid4()), str(uuid4()), str(uuid4())
ISSUES = [
    (QUEUE_ID, "Antrean Terlalu Lama"),
    (TASTE_ID, "Rasa Terlalu Asin"),
    (PARKING_ID, "Parkiran Sempit"),
]


# --- stage 1: retrieval ---------------------------------------------------------------


def test_retrieval_uses_titles_and_respects_threshold_and_top_k():
    embedder = FakeEmbedder()
    query = SPACE["Antrean Kasir Terlalu Lama"]
    found, skipped = retrieve(
        [query], request_with(ISSUES), embedder, TitleEmbeddingCache(), SETTINGS
    )
    assert [c.issue_id for c in found[0]] == [QUEUE_ID]  # taste/parking below 0.8
    assert skipped == 0


def test_retrieval_uses_vectors_of_the_issue():
    # Title is unrelated, but a stored opinion vector of the issue is identical.
    request = request_with(
        [(PARKING_ID, "Parkiran Sempit")],
        vectors=[(PARKING_ID, SPACE["Antrean Terlalu Lama"], EMBEDDING_MODEL_ID)],
    )
    found, _ = retrieve(
        [SPACE["Antrean Terlalu Lama"]],
        request,
        FakeEmbedder(),
        TitleEmbeddingCache(),
        SETTINGS,
    )
    assert found[0][0].issue_id == PARKING_ID
    assert found[0][0].score == pytest.approx(1.0, abs=1e-5)


def test_retrieval_skips_other_embedding_spaces_and_unknown_issues():
    request = request_with(
        [(PARKING_ID, "Parkiran Sempit")],
        vectors=[
            (PARKING_ID, SPACE["Antrean Terlalu Lama"], "other-model"),
            (str(uuid4()), SPACE["Antrean Terlalu Lama"], EMBEDDING_MODEL_ID),
        ],
    )
    found, skipped = retrieve(
        [SPACE["Antrean Terlalu Lama"]],
        request,
        FakeEmbedder(),
        TitleEmbeddingCache(),
        SETTINGS,
    )
    assert found == [[]]
    assert skipped == 1


def test_title_embeddings_are_cached():
    embedder, cache = FakeEmbedder(), TitleEmbeddingCache()
    request = request_with(ISSUES)
    for _ in range(2):
        retrieve([SPACE["Antrean Terlalu Lama"]], request, embedder, cache, SETTINGS)
    assert sum(len(call) for call in embedder.calls) == len(ISSUES)


# --- stage 2: decisions -------------------------------------------------------------


def test_judge_merges_paraphrased_problem():
    judge, factory = judge_returning([{"opinion_index": 0, "issue_id": QUEUE_ID}])
    assignments, stats = run(
        [opinion("Antrean Kasir Terlalu Lama", title="Antrean Kasir Terlalu Lama")],
        request_with(ISSUES),
        judge,
    )
    assert assignments[0].existing_issue_id == QUEUE_ID
    assert assignments[0].new_issue_title == "Antrean Kasir Terlalu Lama"
    assert stats.judge_status == "ok" and stats.merged_by_judge == 1
    # Only that opinion's top-k candidates are shown to the judge.
    prompt = factory.return_value.with_structured_output.return_value.invoke.call_args[
        0
    ][0]
    human = prompt.to_messages()[-1].content
    assert QUEUE_ID in human and PARKING_ID not in human


def test_judge_none_creates_new_issue():
    judge, _ = judge_returning([{"opinion_index": 0, "issue_id": "NONE"}])
    assignments, stats = run(
        [opinion("Antrean Kasir Terlalu Lama")], request_with(ISSUES), judge
    )
    assert assignments[0].existing_issue_id == ""
    assert stats.judge_status == "ok" and stats.merged_by_judge == 0


@pytest.mark.parametrize(
    "decision",
    [
        {"opinion_index": 0, "issue_id": PARKING_ID},  # not in its candidates
        {"opinion_index": 0, "issue_id": str(uuid4())},  # invented
        {"opinion_index": 5, "issue_id": QUEUE_ID},  # wrong opinion
    ],
)
def test_judge_cannot_pick_outside_candidates(decision):
    judge, _ = judge_returning([decision])
    assignments, _ = run(
        [opinion("Antrean Kasir Terlalu Lama")], request_with(ISSUES), judge
    )
    assert assignments[0].existing_issue_id == ""


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), TimeoutError()],
)
def test_judge_failure_falls_back_to_new_issue(exc):
    judge, _ = judge_returning(exc=exc)
    assignments, stats = run(
        [opinion("Antrean Kasir Terlalu Lama")], request_with(ISSUES), judge
    )
    assert assignments[0].existing_issue_id == ""
    assert stats.judge_status == "failed"


def test_invalid_judge_output_falls_back():
    factory = Mock()
    factory.return_value.with_structured_output.return_value.invoke.return_value = {
        "wrong": []
    }
    assignments, stats = run(
        [opinion("Antrean Kasir Terlalu Lama")],
        request_with(ISSUES),
        MergeJudge(GEMINI, SETTINGS, factory),
    )
    assert assignments[0].existing_issue_id == "" and stats.judge_status == "failed"


def test_cancellation_is_not_swallowed():
    judge = Mock()
    judge.decide.side_effect = AIFailure(
        grpc.StatusCode.CANCELLED, "Feedback cancelled"
    )
    with pytest.raises(AIFailure):
        run([opinion("Antrean Kasir Terlalu Lama")], request_with(ISSUES), judge)


def test_insufficient_budget_skips_judge():
    judge, factory = judge_returning([{"opinion_index": 0, "issue_id": QUEUE_ID}])
    assignments, stats = run(
        [opinion("Antrean Kasir Terlalu Lama")], request_with(ISSUES), judge, budget=1.2
    )
    assert assignments[0].existing_issue_id == ""
    assert stats.judge_status == "skipped_budget"
    factory.assert_not_called()


def test_judge_disabled():
    judge, factory = judge_returning([{"opinion_index": 0, "issue_id": QUEUE_ID}])
    settings = SETTINGS.model_copy(update={"judge_enabled": False})
    _, stats = run(
        [opinion("Antrean Kasir Terlalu Lama")],
        request_with(ISSUES),
        judge,
        settings=settings,
    )
    assert stats.judge_status == "disabled"
    factory.assert_not_called()


def test_exact_title_needs_no_judge():
    judge, factory = judge_returning()
    assignments, stats = run(
        [opinion("Antre lama", title="  antrean TERLALU lama ")],
        request_with(ISSUES),
        judge,
    )
    assert assignments[0].existing_issue_id == QUEUE_ID
    assert stats.exact_title == 1 and stats.judge_status == "not_needed"
    factory.assert_not_called()


def test_non_negative_opinions_are_never_matched():
    judge, factory = judge_returning()
    assignments, _ = run(
        [
            opinion(
                "Antrean Terlalu Lama",
                sentiment="POSITIVE",
                title="Antrean Terlalu Lama",
            ),
            opinion("Antrean Terlalu Lama", sentiment="NEUTRAL"),
        ],
        request_with(ISSUES),
        judge,
    )
    assert [a.existing_issue_id for a in assignments] == ["", ""]
    assert all(a.new_issue_title for a in assignments)
    factory.assert_not_called()


def test_no_candidates_no_judge():
    judge, factory = judge_returning()
    _, stats = run([opinion("Antrean Terlalu Lama")], request_with([]), judge)
    assert stats.judge_status == "not_needed"
    factory.assert_not_called()


def test_one_judge_call_for_several_opinions():
    judge, factory = judge_returning(
        [
            {"opinion_index": 0, "issue_id": QUEUE_ID},
            {"opinion_index": 1, "issue_id": TASTE_ID},
        ]
    )
    assignments, stats = run(
        [
            opinion("Antrean Kasir Terlalu Lama"),
            opinion("Rasa Terlalu Asin", title="X"),
        ],
        request_with(ISSUES),
        judge,
    )
    assert [a.existing_issue_id for a in assignments] == [QUEUE_ID, TASTE_ID]
    assert (
        factory.return_value.with_structured_output.return_value.invoke.call_count == 1
    )
    assert stats.judged == 2


def test_judge_uses_its_own_short_timeout():
    judge, factory = judge_returning([])
    run([opinion("Antrean Kasir Terlalu Lama")], request_with(ISSUES), judge)
    assert factory.call_args.kwargs["timeout"] <= SETTINGS.judge_timeout_seconds


def test_match_settings_from_env(monkeypatch):
    for name in (
        "FEEDBACK_MATCH_THRESHOLD",
        "FEEDBACK_MATCH_TOP_K",
        "FEEDBACK_MATCH_TIMEOUT_SECONDS",
        "FEEDBACK_MATCH_JUDGE",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = MatchSettings.from_env()
    assert (settings.threshold, settings.top_k, settings.judge_timeout_seconds) == (
        0.85,
        3,
        10,
    )
    assert settings.judge_enabled is True
    monkeypatch.setenv("FEEDBACK_MATCH_TOP_K", "zero")
    with pytest.raises(AIFailure) as error:
        MatchSettings.from_env()
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION


def test_match_prompt_treats_input_as_data():
    messages = MATCH_PROMPT.format_prompt(context_json="{}").to_messages()
    assert messages[-1].type == "human"
    assert "UNTRUSTED_MATCH_JSON" in messages[-1].content
    assert "NONE" in messages[0].content
