"""Per-RPC operational metrics: latency spans, LLM tokens, retries, cost.

One collector per RPC (ContextVar, so pipelines need no extra parameters). At the end
of the RPC a single structured JSON log line is written; optionally the same record is
appended to a local JSONL trace file (ATLAS_TRACE_FILE) for evaluation/waterfalls.

Only names, IDs, durations, token counts and status codes are recorded: never prompt,
feedback, owner or model text. External tracing (LangSmith/OTel) stays off.
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("atlas_ai.rpc")
_current: ContextVar["RequestRecord | None"] = ContextVar("atlas_rpc", default=None)
_file_lock = threading.Lock()


@dataclass
class RequestRecord:
    rpc: str
    request_id: str = ""
    subject_id: str = ""  # feedback_id or session_id (an ID, never content)
    status: str = "UNKNOWN"
    started: float = field(default_factory=time.monotonic)
    spans: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)

    def offset_ms(self, moment: float) -> float:
        return round((moment - self.started) * 1000, 1)

    def summary(self) -> dict:
        input_tokens = sum(c.get("input_tokens") or 0 for c in self.llm_calls)
        output_tokens = sum(c.get("output_tokens") or 0 for c in self.llm_calls)
        return {
            "event": "rpc",
            "rpc": self.rpc,
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "status": self.status,
            "latency_ms": self.offset_ms(time.monotonic()),
            "llm_calls": self.llm_calls,
            "llm_retries": sum(c.get("retries", 0) for c in self.llm_calls),
            "tokens": {"input": input_tokens, "output": output_tokens},
            "cost_usd": total_cost(self.llm_calls),
            "spans": self.spans,
        }


def total_cost(calls: list[dict]) -> float | None:
    """Sum of per-call costs (each priced with its own model's configured official
    per-1M-token prices); None if any call that used tokens has no configured price."""
    total = 0.0
    for call in calls:
        cost = call.get("cost_usd")
        if cost is None:
            if call.get("input_tokens") or call.get("output_tokens"):
                return None
            continue
        total += cost
    return round(total, 8)


def current() -> RequestRecord | None:
    return _current.get()


@contextmanager
def request(rpc: str, request_id: str = "", subject_id: str = ""):
    record = RequestRecord(rpc=rpc, request_id=request_id, subject_id=subject_id)
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)
        data = record.summary()
        logger.info(json.dumps(data, ensure_ascii=False))
        path = os.getenv("ATLAS_TRACE_FILE", "").strip()
        if path:
            try:
                with _file_lock, open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(data, ensure_ascii=False) + "\n")
            except OSError:
                logger.warning("trace file not writable")


@contextmanager
def span(name: str):
    """Record a named stage with start offset and duration (no-op outside an RPC)."""
    record = _current.get()
    started = time.monotonic()
    try:
        yield
    finally:
        if record is not None:
            record.spans.append(
                {
                    "name": name,
                    "start_ms": record.offset_ms(started),
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                }
            )


def record_llm_call(info: dict) -> None:
    record = _current.get()
    if record is not None:
        record.llm_calls.append(info)
