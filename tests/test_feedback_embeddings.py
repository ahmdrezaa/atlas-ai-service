import math
import os
from pathlib import Path
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from atlas.ai.v1 import ai_pb2
from atlas_ai.feedback.embeddings import (
    DEFAULT_MODEL_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    E5Embedder,
    EmbeddingSettings,
    SkipReason,
    filter_candidate_vectors,
    mean_pool_normalize,
)
from atlas_ai.llm import AIFailure


class FakeTokenizer:
    def __init__(self):
        self.seen = []

    def encode_batch(self, texts):
        self.seen.extend(texts)
        # Text i gets i+1 real tokens, padded to the longest.
        width = len(texts)
        return [
            SimpleNamespace(
                ids=[5] * (i + 1) + [1] * (width - i - 1),
                attention_mask=[1] * (i + 1) + [0] * (width - i - 1),
            )
            for i in range(len(texts))
        ]


class FakeSession:
    def __init__(self, output=None, inputs=("input_ids", "attention_mask")):
        self.output = output
        self.inputs = inputs
        self.feeds = None

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.inputs]

    def run(self, _, feeds):
        self.feeds = feeds
        if self.output is not None:
            return [self.output(feeds)]
        batch, width = feeds["input_ids"].shape
        hidden = np.zeros((batch, width, EMBEDDING_DIM), dtype=np.float32)
        for row in range(batch):
            hidden[row, :, row % EMBEDDING_DIM] = 1.0  # distinct direction per row
            hidden[row, :, (row + 1) % EMBEDDING_DIM] = np.arange(
                width
            )  # padding-sensitive
        return [hidden]


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


def make(model_dir, session=None, tokenizer=None):
    session = session or FakeSession()
    tokenizer = tokenizer or FakeTokenizer()
    embedder = E5Embedder(
        EmbeddingSettings(model_dir=str(model_dir)),
        runtime_loader=lambda path, threads: (session, tokenizer),
    )
    return embedder, session, tokenizer


def test_query_prefix_and_shape(model_dir):
    embedder, session, tokenizer = make(model_dir)
    vectors = embedder.embed(["Es kopi susu mantul", "Parkiran sempit"])
    assert tokenizer.seen == ["query: Es kopi susu mantul", "query: Parkiran sempit"]
    assert len(vectors) == 2 and all(len(v) == EMBEDDING_DIM for v in vectors)
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-5)
    assert "token_type_ids" not in session.feeds


def test_token_type_ids_supplied_when_model_needs_them(model_dir):
    session = FakeSession(inputs=("input_ids", "attention_mask", "token_type_ids"))
    embedder, _, _ = make(model_dir, session=session)
    embedder.embed(["a"])
    assert not session.feeds["token_type_ids"].any()


def test_mean_pooling_ignores_padding():
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    pooled = mean_pool_normalize(hidden, mask)
    assert np.allclose(pooled, [[1.0, 0.0]])  # the padded 100s are ignored


def test_batch_order_preserved(model_dir):
    embedder, _, _ = make(model_dir)
    vectors = np.array(embedder.embed(["a", "b", "c"]))
    # FakeSession gives row i its strongest component at dimension i.
    assert vectors.argmax(axis=1).tolist() == [0, 1, 2]


def test_empty_input_returns_empty(model_dir):
    embedder, session, _ = make(model_dir)
    assert embedder.embed([]) == []
    assert session.feeds is None


@pytest.mark.parametrize(
    "output",
    [
        lambda feeds: np.full((1, feeds["input_ids"].shape[1], EMBEDDING_DIM), np.nan),
        lambda feeds: np.zeros((1, feeds["input_ids"].shape[1], EMBEDDING_DIM)),
        lambda feeds: np.ones((1, feeds["input_ids"].shape[1], 10)),
        lambda feeds: np.ones((1, EMBEDDING_DIM)),
    ],
    ids=["nan", "zero_norm", "wrong_dimension", "wrong_rank"],
)
def test_invalid_model_output_is_internal(model_dir, output):
    embedder, _, _ = make(model_dir, session=FakeSession(output=output))
    with pytest.raises(AIFailure) as error:
        embedder.embed(["a"])
    assert error.value.code == grpc.StatusCode.INTERNAL


def test_inference_exception_is_sanitized(model_dir):
    def boom(feeds):
        raise RuntimeError("internal onnx detail")

    embedder, _, _ = make(model_dir, session=FakeSession(output=boom))
    with pytest.raises(AIFailure) as error:
        embedder.embed(["a"])
    assert error.value.code == grpc.StatusCode.INTERNAL
    assert "onnx detail" not in str(error.value)


def test_missing_model_files_is_failed_precondition(tmp_path):
    with pytest.raises(AIFailure) as error:
        E5Embedder(EmbeddingSettings(model_dir=str(tmp_path / "absent")))
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION


def test_loader_failure_is_failed_precondition(model_dir):
    def broken(path, threads):
        raise OSError("corrupt file at /secret/path")

    with pytest.raises(AIFailure) as error:
        E5Embedder(EmbeddingSettings(model_dir=str(model_dir)), runtime_loader=broken)
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "/secret/path" not in str(error.value)


def test_deadline_and_cancellation_skip_inference(model_dir):
    embedder, session, _ = make(model_dir)
    with pytest.raises(AIFailure) as error:
        embedder.embed(["a"], time_remaining=0)
    assert error.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
    with pytest.raises(AIFailure) as error:
        embedder.embed(["a"], is_active=lambda: False)
    assert error.value.code == grpc.StatusCode.CANCELLED
    assert session.feeds is None


def test_settings_from_env(monkeypatch):
    monkeypatch.delenv("FEEDBACK_EMBEDDING_MODEL_DIR", raising=False)
    monkeypatch.delenv("FEEDBACK_EMBEDDING_THREADS", raising=False)
    settings = EmbeddingSettings.from_env()
    assert settings.model_dir == DEFAULT_MODEL_DIR and settings.threads == 2
    monkeypatch.setenv("FEEDBACK_EMBEDDING_THREADS", "zero")
    with pytest.raises(AIFailure) as error:
        EmbeddingSettings.from_env()
    assert error.value.code == grpc.StatusCode.FAILED_PRECONDITION


def test_model_id_encodes_preprocessing_version():
    assert EMBEDDING_MODEL_ID == "intfloat/multilingual-e5-small@query-l2-v1"
    assert EMBEDDING_DIM == 384


# --- candidate vector filtering ----------------------------------------------------


def vector(model=EMBEDDING_MODEL_ID, values=None):
    values = [0.0] * EMBEDDING_DIM if values is None else values
    if values and len(values) == EMBEDDING_DIM and not any(values):
        values = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    return ai_pb2.CandidateVector(
        opinion_id="o", issue_id="i", embedding=values, embedding_model=model
    )


def test_filter_candidate_vectors():
    good = vector()
    candidates = [
        good,
        vector(model="intfloat/multilingual-e5-small"),  # different preprocessing
        vector(model="other-model"),
        vector(values=[1.0] * 10),
        vector(values=[float("nan")] + [0.0] * (EMBEDDING_DIM - 1)),
        ai_pb2.CandidateVector(
            opinion_id="z",
            embedding=[0.0] * EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL_ID,
        ),
    ]
    kept, skipped = filter_candidate_vectors(candidates)
    assert kept == [good]
    assert skipped == {
        SkipReason.MODEL_MISMATCH: 2,
        SkipReason.BAD_DIMENSION: 1,
        SkipReason.NON_FINITE: 1,
        SkipReason.ZERO_NORM: 1,
    }


# --- real model (only inside the Docker image, where the model is bundled) ----------

REAL_DIR = Path(os.getenv("FEEDBACK_EMBEDDING_MODEL_DIR", DEFAULT_MODEL_DIR))
real_model = pytest.mark.skipif(
    not (REAL_DIR / "model.onnx").is_file(),
    reason="E5 model not bundled in this environment",
)


@pytest.fixture(scope="module")
def real_embedder():
    return E5Embedder(EmbeddingSettings(model_dir=str(REAL_DIR)))


@real_model
def test_real_model_output_contract(real_embedder):
    texts = ["Es kopi susu mantul", "Nunggu pesanan lama banget", "Parkiran sempit"]
    batch = np.array(real_embedder.embed(texts))
    assert batch.shape == (3, EMBEDDING_DIM)
    assert np.all(np.isfinite(batch))
    assert np.allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
    single = np.array([real_embedder.embed([text])[0] for text in texts])
    assert np.allclose(batch, single, atol=1e-5)
    again = np.array(real_embedder.embed(texts))
    assert np.array_equal(batch, again)  # deterministic


@real_model
def test_real_model_ranks_same_problem_higher(real_embedder):
    # Informational ordering check only; NOT a merge threshold.
    a, b, c = np.array(
        real_embedder.embed(
            ["Kopi susunya kemanisan", "Es kopi terlalu manis", "Parkirannya sempit"]
        )
    )
    assert a @ b > a @ c
