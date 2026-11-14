"""P-A2 E5 embedding benchmark (CPU, ONNX Runtime). Local only, no network calls.

Measures cold load, warm latency per batch size and peak process memory. The
similarity table is informational: it is NOT a merge threshold or accuracy score.

Usage (inside the Docker image, where the model is bundled):
    python scripts/feedback_embedding_benchmark.py --image-size-bytes <n>
"""

import argparse
import json
import os
import platform
import resource
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from atlas_ai.feedback.embeddings import (
    EMBEDDING_MODEL_ID,
    MODEL_REVISION,
    E5Embedder,
    EmbeddingSettings,
)

ROOT = Path(__file__).resolve().parents[1]
TEXTS = [
    "Es kopi susu mantul",
    "Nunggu pesanan lama banget, hampir 20 menit",
    "Nasi goreng nggak pedas sama sekali padahal pesan level 3",
    "Kuah bakso keasinan",
    "Pelayanan mbaknya ga ramah",
    "Harganya nggak mahal buat porsi segini",
    "Semoga ada pilihan susu oat",
    "Ayam goreng masih merah di dalam",
]
# (a, b, relation) — relation is the author's intent, not a gold label.
PAIRS = [
    ("Kopi susunya kemanisan", "Es kopi terlalu manis", "same_problem"),
    ("Kopi susunya kemanisan", "Kopi susunya kurang manis", "opposite_direction"),
    ("Antrean kasir lama banget", "Minuman lama banget dibuatnya", "different_stage"),
    ("Antrean kasir lama banget", "Nunggu di kasir lama", "same_problem"),
    ("Isian coklat dikit", "Isian vanilla pelit", "same_problem_other_variant"),
    ("Tempat nyaman buat nugas", "Tempatnya kotor", "opposite_direction"),
    ("Kuah bakso keasinan", "Parkiran sempit", "unrelated"),
]


def peak_rss_mb() -> float:
    # Linux ru_maxrss is in kilobytes.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size-bytes", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(ROOT / "evaluation" / "feedback_p_a2_embedding_benchmark.json"),
    )
    args = parser.parse_args()

    rss_before = peak_rss_mb()
    started = time.monotonic()
    embedder = E5Embedder(EmbeddingSettings.from_env())
    cold_load = time.monotonic() - started
    started = time.monotonic()
    embedder.embed(TEXTS[:1])
    first_call = time.monotonic() - started

    latency = {}
    for size in (1, 3, 8):
        batch = TEXTS[:size]
        samples = []
        for _ in range(args.repeats):
            started = time.monotonic()
            embedder.embed(batch)
            samples.append(time.monotonic() - started)
        samples.sort()
        latency[str(size)] = {
            "median_ms": round(statistics.median(samples) * 1000, 2),
            "p95_ms": round(samples[int(0.95 * (len(samples) - 1))] * 1000, 2),
        }

    pairs = []
    for a, b, relation in PAIRS:
        va, vb = np.array(embedder.embed([a, b]))
        pairs.append(
            {"a": a, "b": b, "relation": relation, "cosine": round(float(va @ vb), 4)}
        )

    report = {
        "milestone": "P-A2",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "local CPU benchmark; similarity table is informational only",
        "not_a_threshold_or_accuracy_score": True,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": "onnxruntime CPUExecutionProvider, fp32 onnx/model.onnx",
        "threads": embedder.settings.threads,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "image_size_bytes": args.image_size_bytes,
        "cold_load_seconds": round(cold_load, 3),
        "first_call_seconds": round(first_call, 3),
        "warm_latency_by_batch_size": latency,
        "repeats": args.repeats,
        "peak_rss_mb": {"before_load": rss_before, "after": peak_rss_mb()},
        "similarity_pairs": pairs,
        "parity_check": (
            "One-off throwaway-container comparison with sentence-transformers on 5 "
            "texts: min cosine 0.99999999, max abs diff 1.2e-7 (2026-11-14)."
        ),
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "cold_load_seconds",
                    "warm_latency_by_batch_size",
                    "peak_rss_mb",
                )
            }
        )
    )
    for pair in pairs:
        print(pair["relation"], pair["cosine"])


if __name__ == "__main__":
    main()
