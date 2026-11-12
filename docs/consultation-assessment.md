# Guided Consultation assessment — Milestone 3

Historical milestone description: Milestone 4A now implements the solution branch;
see [consultation-solution.md](consultation-solution.md). The assessment prompt is
now `consultation-assessment-v2` and both stages share one total timeout budget.

The pipeline prepares the official Go protobuf request using the existing context
builder. GENERATE_NOW and question_count >=4 bypass Gemini. These paths, and a model
decision GENERATE_SOLUTION, return UNIMPLEMENTED until Milestone 4. No solution or
memory candidate is fabricated; Python never resets counters or persists state.

An eligible turn uses one Gemini invocation for a typed assessment and optional
question. Review `consultation/prompts.py` (base/assessment prompts and version
`consultation-assessment-v1`) and `SufficiencyAssessment` in `assessment.py`.
Internal reasons are neither logged nor sent to Go. Native `method="json_schema"`
is used, without agent tools:
[LangChain API reference](https://reference.langchain.com/python/langchain-google-genai/chat_models/ChatGoogleGenerativeAI/with_structured_output).

Copy `.env.example` locally. Set GEMINI_API_KEY and GEMINI_MODEL to a model enabled
for your account with native structured output. No default model is assumed.
Compose forwards the variables; native Python reads process environment, not `.env`
automatically. Never commit credentials. Provider timeout defaults to 30 seconds
(maximum configurable 55s), zero retries, reduced to the remaining gRPC deadline
with a 250ms response margin. Cancellation is checked before and after the synchronous
call; an in-flight HTTP call may continue until its timeout.

All source text remains untrusted data in one human message. SQL statistics,
sample evidence, owner statements, confirmed memories and historical AI output
remain attributed. Only the redundant source registry is excluded from the prompt;
original source sections and UUIDs are preserved. The complete UTF-8 prompt is limited
to 120000 bytes by default. This is a byte bound, not a token estimate. Oversized
context is rejected without truncation, summarization, or extra provider calls.
Model context-limit errors are explicit. LangSmith tracing is disabled around
invocation regardless of ambient tracing flags; formatting is untraced and caching
is disabled. This implementation does not log raw text, keys or provider exceptions.

Deterministic guards reject used/skipped/history topics, exact/near lexical repeats,
invalid question shape and obvious compound questions. The assessment also checks
known information and semantic repetition. These measures cannot prove semantic
non-repetition or resistance to every prompt injection. Paraphrases and relevance
need real-model evaluation; claim-level grounding is not implemented. Invalid output
is rejected without a repair retry.

| Failure | gRPC status |
| --- | --- |
| Invalid Go request | INVALID_ARGUMENT |
| Missing/invalid configuration | FAILED_PRECONDITION |
| Invalid structured output or rejected question | INTERNAL |
| Context too large/model context overflow | RESOURCE_EXHAUSTED |
| Timeout/deadline | DEADLINE_EXCEEDED |
| Cancellation | CANCELLED |
| Other provider failures, including rate limits | UNAVAILABLE |
| Solution branch | UNIMPLEMENTED |

Suthe integration blocker: Go currently sends a cumulative question_count while the
PRD wants four questions per cycle. Go must define/reset counter and topic scope or
provide explicit cycle metadata. Until then CONTINUE_DISCUSSION at count >=4 cannot
ask again. Empty evidence is legal input; Milestone 4 must resolve the Go requirement
for nonempty customer_evidence before a no-evidence solution can be returned.

Run `python -m pytest -q -p no:cacheprovider` and Ruff on changed modules/tests.
Tests inject mocked models and do not call Gemini. Real-model smoke testing and
Go-backend integration remain separate checks; mocks establish neither model quality
nor semantic grounding.
