"""Multilingual E5 Small embeddings via ONNX Runtime (Path A, P-A2).

Official ONNX export of intfloat/multilingual-e5-small, bundled in the image at
build time. Input "query: <text>", attention-mask mean pooling, L2 normalization,
matching the model card / sentence-transformers configuration.
"""

import math
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path

import grpc
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.llm import AIFailure

# The suffix versions preprocessing ("query:" prefix + L2). Change it whenever the
# prefix, pooling or normalization changes so vector spaces are never mixed.
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small@query-l2-v1"
EMBEDDING_DIM = 384
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
QUERY_PREFIX = "query: "
MAX_TOKENS = 512
DEFAULT_MODEL_DIR = "/opt/models/multilingual-e5-small"


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_dir: str = DEFAULT_MODEL_DIR
    # Provisional; ONNX Runtime intra-op threads per process.
    threads: int = Field(default=2, ge=1, le=32)

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        try:
            return cls(
                model_dir=os.getenv("FEEDBACK_EMBEDDING_MODEL_DIR", DEFAULT_MODEL_DIR),
                threads=os.getenv("FEEDBACK_EMBEDDING_THREADS", "2"),
            )
        except ValidationError:
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Invalid embedding configuration",
            ) from None


def _load_runtime(model_dir: Path, threads: int):
    """Real ONNX session + tokenizer; imported lazily so tests can inject fakes."""
    # onnxruntime >=1.31 ships Microsoft 1DS usage telemetry; the service sends
    # nothing outward except the LLM call. The variable must be set before import.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from tokenizers import Tokenizer

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise AIFailure(
            grpc.StatusCode.FAILED_PRECONDITION, "Embedding tokenizer has no pad token"
        )
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
    return session, tokenizer


class E5Embedder:
    """Loads once; embed() is thread-safe (ONNX Runtime sessions allow concurrent run)."""

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        *,
        runtime_loader: Callable = _load_runtime,
    ):
        self.settings = settings or EmbeddingSettings()
        model_dir = Path(self.settings.model_dir)
        if (
            not (model_dir / "model.onnx").is_file()
            or not (model_dir / "tokenizer.json").is_file()
        ):
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Embedding model files are not available",
            )
        try:
            self.session, self.tokenizer = runtime_loader(
                model_dir, self.settings.threads
            )
        except AIFailure:
            raise
        except Exception:  # noqa: BLE001 - never surface runtime internals
            raise AIFailure(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Embedding model could not be loaded",
            ) from None
        self.input_names = {item.name for item in self.session.get_inputs()}

    def embed(
        self,
        texts: Sequence[str],
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        if time_remaining is not None and time_remaining <= 0:
            raise AIFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Feedback deadline exceeded"
            )
        if is_active is not None and not is_active():
            raise AIFailure(grpc.StatusCode.CANCELLED, "Feedback cancelled")
        started = time.monotonic()
        encodings = self.tokenizer.encode_batch([QUERY_PREFIX + t for t in texts])
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        try:
            hidden = self.session.run(None, feeds)[0]
        except Exception:  # noqa: BLE001
            raise AIFailure(
                grpc.StatusCode.INTERNAL, "Embedding inference failed"
            ) from None
        vectors = mean_pool_normalize(np.asarray(hidden), mask)
        if timings is not None:
            timings["embedding.inference"] = time.monotonic() - started
        if time_remaining is not None and time.monotonic() - started >= time_remaining:
            raise AIFailure(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Feedback deadline exceeded"
            )
        if is_active is not None and not is_active():
            raise AIFailure(grpc.StatusCode.CANCELLED, "Feedback cancelled")
        if vectors.shape != (len(texts), EMBEDDING_DIM):
            raise AIFailure(grpc.StatusCode.INTERNAL, "Invalid embedding dimension")
        return [row.tolist() for row in vectors]


def mean_pool_normalize(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Attention-mask mean pooling + L2 normalization, with strict validity checks."""
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
        raise AIFailure(grpc.StatusCode.INTERNAL, "Invalid embedding output shape")
    weights = mask[..., None].astype(np.float32)
    summed = (hidden.astype(np.float32) * weights).sum(axis=1)
    counts = np.clip(weights.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.all(np.isfinite(pooled)) or np.any(norms <= 1e-12):
        raise AIFailure(grpc.StatusCode.INTERNAL, "Invalid embedding values")
    normalized = pooled / norms
    if not np.all(np.isfinite(normalized)):
        raise AIFailure(grpc.StatusCode.INTERNAL, "Invalid embedding values")
    return normalized


class SkipReason(str, Enum):
    MODEL_MISMATCH = "MODEL_MISMATCH"
    BAD_DIMENSION = "BAD_DIMENSION"
    NON_FINITE = "NON_FINITE"
    ZERO_NORM = "ZERO_NORM"


def filter_candidate_vectors(
    vectors: Sequence[ai_pb2.CandidateVector],
) -> tuple[list[ai_pb2.CandidateVector], Counter]:
    """Keep only vectors from the same embedding space; never compare across spaces."""
    kept: list[ai_pb2.CandidateVector] = []
    skipped: Counter = Counter()
    for vector in vectors:
        values = vector.embedding
        if vector.embedding_model != EMBEDDING_MODEL_ID:
            skipped[SkipReason.MODEL_MISMATCH] += 1
        elif len(values) != EMBEDDING_DIM:
            skipped[SkipReason.BAD_DIMENSION] += 1
        elif not all(math.isfinite(value) for value in values):
            skipped[SkipReason.NON_FINITE] += 1
        elif math.sqrt(sum(value * value for value in values)) <= 1e-12:
            skipped[SkipReason.ZERO_NORM] += 1
        else:
            kept.append(vector)
    return kept, skipped
