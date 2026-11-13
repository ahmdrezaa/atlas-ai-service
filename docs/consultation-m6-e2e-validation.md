# Milestone 6 — Real-provider Consultation validation

**Decision: BLOCKED.** One fresh authenticated gRPC START returned a valid
NextQuestion. No fresh M6 Solution passed the complete pipeline. Production Gemini
returned HTTP 504; the explicit comparator also produced local validation
rejections and an HTTP 503. These are separate failure classes. The successful P0
Solution and M5 mock response are historical evidence, not new M6 successes.

Run date: 2026-11-13, timestamps in UTC. All provider inputs were synthetic.
Full results: [`evaluation/m6_consultation_e2e.json`](../evaluation/m6_consultation_e2e.json).

## A. Workspace and contract audit

Base commit `c3e1e589fca7333dc8189bd71ea308877a1ae682`, branch `main`.
The working tree already contained uncommitted P0/M5 changes: consultation
assessment, prompts, solution, grounding, pipeline, memory.py, adapter/server,
tests, scripts, and evaluation artifacts. `nlp/` was already deleted and
`feedback/` untracked following the user's relocation. M6 preserves that state.
The adapter currently imports `atlas_ai.feedback.pipeline`; importing the real
service and running native gRPC tests succeeded. ProcessFeedback was not changed.

Inspected the actual PRD (header v2.1 Audited), official ai.proto, server,
grpc_service, consultation pipeline/validation/context/models/assessment/solution/
grounding/memory/prompts, tests, runners, P0/4C diagnostics, business-memory-candidates.md,
and both M5 mock/live ledgers. No AGENTS.md was found in the workspace.

Current architecture is intact: request validation through prepare_context,
separate source namespaces, untrusted context prompts, Pydantic structured output,
deterministic question/grounding/memory guards, then official protobuf mapping.
Memory proposals share assessment/solution calls; Python never saves or confirms
them. No LLM verifier, tracing of real data, database client, or new retrieval.

Read-only Go audit through the GitHub connector:

- [Official proto](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/proto/atlas/ai/v1/ai.proto),
  blob `22500463be61dce1c2764a3fcc7329e71aea4c67`. Local text is identical after
  normalizing CRLF and the final newline; local proto/stubs were not changed.
- [AI client](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/aiclient/client.go),
  blob `043268da183955bd3cc1b6e3df8b81f90a8744b7`: consultation context.WithTimeout,
  with any earlier parent deadline still applying.
- [Configuration](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/config/config.go),
  blob `91af27c36fc0e08039f677d1c6e0bf6f3c561a7d`: consultation default 60s,
  HTTP write default 90s. Deployed Go environment was not inspected.
- [Response validation](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/service/consultation_validate.go),
  blob `6600f5df31ca2a91400de8e3459b6ccf2ff9b363`: five nonempty sections,
  current references for findings/hypotheses/recommendations, OWNER-only memory
  candidates, and one result. This was source inspection, not execution of Go.

## B. Verification matrix

| Category | Actual result | Limits of the evidence |
| --- | --- | --- |
| A: full offline regression | **899 passed in 6.42s**; baseline 873 passed in 10.85s | Docker Python 3.11, network disabled; no Gemini calls |
| B: authenticated native gRPC, mock provider | PASS, including 16 new M6 RPC tests within the 26 new tests | Actual server, interceptor, adapter and serializers; model boundary mocked |
| C: authenticated native gRPC, real Gemini | START NextQuestion PASS; other attempted RPCs failed | No passing M6 Solution or owner-memory response |
| Ruff | All checks passed with `--ignore EXE002` | Windows bind mount presents Python files as executable |

The full suite covers all five events, both result paths and oneof, source
namespace/conflicting duplicates, quotes, unsupported numbers, memory provenance,
conflicts/dedup/injection, zero additional extraction calls, invalid-token
UNAUTHENTICATED, unavailable/deadline mappings, shared deadlines, no-evidence, the
50 grounding cases and P0 regressions. New M6 tests also exercise failed-RPC
timing, HTTP 503/504/429 versus client timeout, local Pydantic rejection, optional
empty memory, provider responsiveness gates and historical-input boundaries.

Initial Ruff attempts failed because its cache was on the read-only mount and
EXE002 interpreted synthetic executable bits as missing shebangs. Final check uses
no cache and excludes only EXE002; application checks remain enabled. UTC/unused
imports found while developing the runner were corrected.

## C. Live execution ledger

Six real authenticated RPCs plus one small structured preflight; **9 actual SDK
generate_content invocations**, no automatic retries or correction invocations.
The cap was seven attempts / eleven possible invocations. No further calls after
the final failure. Rows and per-invocation UTC timestamps were recorded by the
runner, with actual HTTP response hooks and native gRPC status codes.

| Attempt / event | Model | SDK calls | RPC seconds* | HTTP | Actual gRPC | Outcome |
| --- | --- | ---: | ---: | --- | --- | --- |
| M6-P / preflight | 3.8 Flash | 1 | 8.552* | 504 | N/A | Provider response before parsing; not an RPC |
| M6-D-PROD / GENERATE_NOW + F | 3.8 Flash | 1 | 28.654 | 504 | DEADLINE_EXCEEDED | No model output/local validation |
| M6-D-COMP / GENERATE_NOW + F | 3.6 Flash | 1 | 5.579 | 200 | INTERNAL | Schema/structure pass; hypothesis grounding rejected |
| M6-A-COMP / START | 3.6 Flash | 1 | 3.768 | 200 | OK | Valid NextQuestion, metadata and protobuf round trip |
| M6-B-COMP / OWNER_REPLY | 3.6 Flash | 2 | 17.742 | 200 / 200 | INTERNAL | Schema/structure pass; recommendation title grounding rejected |
| M6-C-COMP / SKIP_QUESTION | 3.6 Flash | 2 | 24.050 | 200 / 200 | INTERNAL | Schema pass; numeric metric structural rejection |
| M6-E-COMP / CONTINUE_DISCUSSION | 3.6 Flash | 1 | 13.889 | 503 | UNAVAILABLE | Assessment provider failure; no solution invocation |

*Preflight is total diagnostic time, not gRPC latency. Other times measure the
actual stub call, including failures. HTTP 200 alone never counts as a valid turn.

Production settings remained `gemini-3.8-flash`, 30s total turn budget, 2,048 output
tokens, SDK-default API version (observed v1beta), provider-default thinking
(no explicit thinking config). Comparator overrides were explicit CLI arguments:
`gemini-3.6-flash`, v1beta, LOW, 45s turn / 60s RPC. No .env/model/dependency changes.
Comparator warned that its temperature parameter is ignored; observed effective
temperature was null. No deterministic model-output guarantee is inferred.

Installed runtime: google-genai 3.2.0, langchain-google-genai 4.5.0,
langchain-core 1.8.0, Pydantic 2.14.3, grpcio 1.85.1, protobuf 7.36.2.
All six successful provider responses ended STOP; token totals were genuinely
reported: D comparator 3,513; A 2,371; B 3,053 + 3,441; C 2,513 + 3,314.
Sum **18,205 reported tokens**. Failed invocations had no usage; this is not a
billing total. Null thoughts usage does not establish zero thinking tokens.

Both production failures had actual HTTP 504 responses. They were not local
parsing errors or exhaustion of the 60s gRPC caller deadline. The 49-byte
preflight also failed with a 256-token budget, so large ATLAS prompts / 2,048
output tokens are not necessary for this failure. This does not identify the
provider's internal cause, capacity, account routing, or model availability.
Comparator HTTP 200 proves this environment can generate, but its later HTTP 503
and rejected outputs do not establish reliability or production 3.8 readiness.

## D. Functional observations and unresolved validation alignment

- **START:** Gemini chose ASK_QUESTION. Returned topic `peak_hours_staff_count`,
  text `Berapa jumlah staf yang bertugas melayani kasir pada jam sibuk tersebut?`,
  model_id 3.6, prompt_version consultation-assessment-v4. Actual Pydantic and
  question guards passed, oneof/protobuf round trip passed. Memory [] was valid
  because this scenario had no OWNER history.
- **OWNER_REPLY:** latest OWNER appears once in conversation history and once in
  the OWNER registry; its text appears in a supported returned finding proposal.
  `latest_owner_message` is also a separate context view of that same message,
  so the prompt is not claimed to contain the text only once. Assessment moved
  to solution without re-asking staffing. The complete turn still failed.
- **SKIP_QUESTION:** staffing was included in asked/skipped topics, cumulative
  count 1. Assessment moved to solution rather than repeating a question. The
  complete turn failed; this is not proof of a passing live skip response.
- **GENERATE_NOW:** captured SDK sequence contains only GeneratedSolutionTurn,
  no assessment. Comparator output had section sizes 2/2/1/1/1 and passed full
  Pydantic and structural checks. Grounding rejected hypotheses[0].statement,
  reason `unproven_scope_causality_or_independent_verification`. This row retained
  an output digest and verdicts, not the original hypothesis wording; a more
  precise lexical root cause cannot be recovered from it. Later diagnostic rows
  preserve parsed synthetic output to avoid that loss of diagnostic detail.
- **CONTINUE_DISCUSSION:** previous Solution was read from P0 P3 as historical
  AI input only, with a new revision OWNER message. Context tags and exclusion
  of AI IDs from current OWNER sources were verified locally. The new assessment
  failed HTTP 503, so use of the revision by a successful live response is unverified.
- **Memory F:** D included `Pesanan dicatat manual di kasir.` from OWNER UUID
  `20000000-0000-4000-8000-000000000002`. Neither D returned a response. B proposed
  an owner-sourced staffing candidate, but grounding failed before production
  memory validation/mapping. **Owner-memory response demonstration remains
  UNVERIFIED.** Native mock RPC tests returned a valid owner-sourced candidate,
  discarded an AI-ID candidate, and accepted empty metadata while preserving
  Solution. Candidates were never saved or added to confirmed snapshot memories.

The B title `Optimasi Alur Kerja Kasir` failed
`proposal_contains_unproved_premise_or_unresolved_language` because the bounded
action recognizer does not recognize this nominal heading. That verdict does not
prove the heading contains a false fact. It exposes a language-coverage limitation
requiring policy review; M6 did not expand the recognizer to pass this sample.

C's metric `Waktu tunggu antrean kasir pada jam 17.00-19.00` triggered
`unverified numeric factual text`. The hours exist in a confirmed memory, but
SuccessIndicator has no reference field and the current numeric output policy
cannot establish that provenance in this metric. This is an output/validator
alignment issue, not HTTP failure. Grounding was not reached. B and C are retained
as rejected outputs; no rewriting, revalidation-as-live-success, validator bypass,
or follow-up generation to repair them.

Grounding remains a conservative deterministic recognizer, not general semantic
entailment. Lexical false rejections and semantically repeated topic aliases remain
risks; LLM verification stayed disabled.

## E. Minimal changes

Only diagnostic/test/report files changed during M6:

1. `scripts/gemini_live_diagnostics.py`: per-actual-SDK-invocation timestamps,
   schema stage, duration, exception type/code and HTTP statuses; no raw error body.
2. `scripts/consultation_live_readiness.py`: reuse existing authenticated server;
   accept explicit synthetic request / allowed result kinds / config overrides;
   measure RPC duration in finally on failure; classify known sanitized wire
   failures; retain parsed synthetic claims for rejection diagnosis.
3. `scripts/consultation_m6_e2e.py`: A–E fixtures, F combined with D, bounded
   ledger, explicit comparator overrides and provider responsiveness gate, native
   response/stage checks, memory provenance, historical-input handling.
4. `tests/test_consultation_m6_e2e.py`: 26 new offline tests, including 16 native
   authenticated gRPC tests with mocked model boundaries.
5. This report and `evaluation/m6_consultation_e2e.json`.

Before: failed RPCs lost latency; START was assumed to require a question.
After: failures retain actual latency/status, and START accepts either valid
official result. Provider HTTP 200 is a responsiveness gate, not an acceptance
gate. Empty optional memory no longer fails the M6 diagnostic (the old M5 smoke
mode is preserved). No production pipeline, validator, prompt, auth, feedback,
proto/stub, lifecycle, dependency, or timeout changes. Historical ledgers are intact.

## F. Deadline and Go integration limits

One shared pipeline budget B = min(configured Python budget, incoming RPC
remaining), measured from before context preparation. Each LangChain/SDK call
receives the remaining shared time minus a 0.25s reserve; solution does not obtain
a new full budget after assessment. Local guards/mapping consume the same budget.
Python server has no independent fixed consultation timeout beyond this budget
and the caller's deadline. gRPC receives cancellation/deadline through the adapter.

Observed B comparator SDK limits: 44.749s assessment, 37.728s generation;
C: 44.749s then 35.567s. This corroborates the shared-budget regression test.
Production D SDK limit 29.749s, provider HTTP 504 after 28.547s, native RPC 28.654s.
Comparator entry caller remaining was about 60s with about 15s headroom; native
transport/pipeline overheads are recorded per row, not assumed zero. The runner
requires at least 10s entry headroom for development budgets >=40s.

The 45s comparator setting is a diagnostic-only development choice under the
observed 60s caller. No production timeout increase is recommended from these
samples. Suthe must verify deployed AI timeout, earlier parent deadlines, HTTP
write timeout, network overhead and concurrent load. Go's question_count remains
cumulative; Python conservatively stops asking at >=4 and does not reset it.

Protobuf compatibility and native Python transport passed offline. No Go client,
SQL/PostgreSQL lifecycle, tenant authorization, deployment TLS, save/confirm or
optimistic locking integration was executed. Before a real Go demo: resolve the
production provider errors, review the demonstrated output/validator alignment,
then obtain a fresh valid Solution and OWNER-memory response with the agreed
caller budget. Comparator NextQuestion success is insufficient.

## G. Commands and acceptance

Executed final offline commands (workspace mounted read-only):

```powershell
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m pytest -q -p no:cacheprovider
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m ruff check --no-cache --config pyproject.toml --ignore EXE002 src/atlas_ai tests scripts
```

Live commands used this Docker prefix, with network enabled and only evaluation
output writable. The .env was read inside the container, never printed:

```powershell
$m6Docker = @('run', '--rm', '--mount', 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly', '--mount', 'type=bind,source=D:\atlas-ai-service\evaluation,target=/results', '--workdir', '/workspace', '--env', 'PYTHONPATH=/workspace/src', '--env', 'PYTHONDONTWRITEBYTECODE=1', 'atlas-ai-service:dev')
docker @m6Docker python -m scripts.consultation_m6_e2e --output /results/m6_consultation_e2e.json --id M6-P --scenario preflight
docker @m6Docker python -m scripts.consultation_m6_e2e --output /results/m6_consultation_e2e.json --id M6-D-PROD --scenario D
```

Comparator calls used the same prefix/runner/output with
`--model gemini-3.6-flash --api-version v1beta --thinking low --turn-budget 45`,
in order D, A, B, C, E, IDs M6-D-COMP through M6-E-COMP. E additionally used
`--previous-ledger /workspace/evaluation/p0_solution_live.json --previous-run-id P3`.
The existing ledger refuses repeat IDs and has exhausted its seven-attempt cap.

Acceptance 1–3 and 8–9 have evidence. Criterion 4 (fresh valid Solution) is not met;
5 is partial; 6 (OWNER-memory live response) is unverified; 7 is not met for the
rejected outputs. They were correctly prevented from reaching protobuf responses.
**M6 full acceptance is not met.** No production readiness or Go end-to-end
readiness claim; no commit/push and no Milestone 7 work.
