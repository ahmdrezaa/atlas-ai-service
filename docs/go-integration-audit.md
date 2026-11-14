# Go ↔ Python integration audit (read-only)

**Date:** 2026-11-14  ·  **Go repo:** `Elioth-Gray/atlas-backend-core` @ `2bf7f083e1d6dee37271382f938dc336dfdbfbbd`
(cloned to `D:\atlas-backend-core`, no changes made; `git status` clean after the audit)
**Python repo:** this repo, P-A3 working tree (`ProcessFeedback` implemented, not yet committed).

Earlier concerns G1–G9 were written from the Go PRD text. This document checks them against the actual Go code.

## 1. Contract

| Item | Result |
| --- | --- |
| `proto/atlas/ai/v1/ai.proto` | **Identical** to ours (only a trailing newline differs). Last Go change: commit `1e6a4eb4`, 2026-11-12. |
| Connection | Go: `AI_GRPC_ADDR`, `AI_GRPC_TOKEN` (sent as `authorization: Bearer …`), `AI_GRPC_TLS=false` default, `x-request-id` on every call (`internal/grpcclient/client.go`). Python reads `GRPC_AUTH_TOKEN` — names differ, **values must match**. |
| Timeouts | `AI_FEEDBACK_TIMEOUT=30s`, `AI_CONSULTATION_TIMEOUT=60s` (`internal/config/config.go:150-155`). Python leaves 0.5 s margin before the feedback deadline. |
| Local run | `compose.dev.yaml` runs Go + `pgvector/pgvector:0.8.0-pg17` + auto-migrations, with a `fakeai` stub by default. Set `DOCKER_AI_GRPC_ADDR=host.docker.internal:50051` in Go's `.env` to use the real Python service. **No Go code change needed to integrate.** |

## 2. Path A — `ProcessFeedback` (needed for the first integration)

| # | Concern | Verdict | Evidence |
| --- | --- | --- | --- |
| A1 | Go retries only `UNAVAILABLE`/`DEADLINE_EXCEEDED`, so `FAILED_PRECONDITION`/`INTERNAL` would lose feedback | **Not an issue.** The worker retries **any** error up to `FEEDBACK_JOB_MAX_ATTEMPTS=8`, backoff 10 s → 30 min with ±20 % jitter, then marks the job `FAILED` with `last_error` (a dead-letter). | `internal/worker/feedback.go:133-157`, `internal/config/config.go:162-168` |
| A2 | `opinions=[]` makes Go reprocess forever (`HasOpinions`) | **Not an issue.** Empty result → `Process` returns nil → job `DONE`; never re-run. | `internal/service/feedback.go:150-203`, `internal/worker/feedback.go:134` |
| A3 | Embedding model/dimension not checked by Go | **Confirmed, acceptable.** Go only requires non-empty text, a valid sentiment, a non-empty embedding and model name; `opinion_vectors.embedding` has no fixed dimension. Python guarantees 384-D, finite, L2-normalized, model ID `intfloat/multilingual-e5-small@query-l2-v1`. | `feedback.go:290-305`, `migrations/000002…up.sql:103-107` |
| A4 | Candidate selection | **Confirmed as documented in the PRD:** top 200 issues by `opinion_count`, 200 most recent vectors (not nearest-neighbour). Matters for P-A5, not for the first integration. | `internal/repository/feedback.go` (`ListCandidateIssues`, `ListCandidateVectors`) |
| A5 | **Only NEGATIVE opinions get an issue (planet)** | **Confirmed; product decision needed.** `resolveIssue` returns no issue for POSITIVE/NEUTRAL; they are stored with `issue_id = NULL`. Insight queries (Galaxy, evidence, stats) also filter to NEGATIVE. This conflicts with PRD Path A ADR-A08 ("positive planets allowed"). | `internal/service/feedback.go:253-255`; `internal/repository/insight.go:126,199,257,274` |
| A6 | Invalid `existing_issue_id` | **Handled by Go:** an issue of another shop is never trusted; Go creates a new issue from `new_issue_title` instead. Same new titles within one response share one issue (case-insensitive). | `feedback.go:250-286` |

**Recommendation for A5:** align the Python PRD to Go (planets = complaints only) for the MVP. This requires **no code change** on either side: Python keeps sending a title for every opinion, and Go simply ignores it for non-negative opinions. If positive planets are wanted later, that is a product + Go change (resolveIssue and the insight queries), to be agreed with Suthe and Sakha.

**Conclusion for Path A: nothing in Go blocks the first integration.**

## 3. Path B — Consultation (for the later integration)

| # | Concern | Verdict | Evidence |
| --- | --- | --- | --- |
| G1 | `skipped_topics` updated after the AI call / with the wrong topic | **Not an issue.** On `SKIP_QUESTION` Go adds the last question's topic to `skipped_topics` **before** the call. | `internal/service/consultation.go:290-295` |
| G2 | `question_count` not reset on `CONTINUE_DISCUSSION` | **Confirmed.** It only increments (`update.QuestionCount++`). With Python's cumulative cap of 4, a discussion after 4 questions can never ask again. | `internal/service/consultation_snapshot.go:129-142` |
| G3 | Snapshot with no evidence while Go requires `customer_evidence` | **Low risk.** Evidence comes from the issue's NEGATIVE opinions (an issue always has ≥1), ordered severe → negative → newest. Risk remains only if all `evidence_text`/`full_feedback` are empty. Go's validator still requires non-empty `customer_evidence`. | `consultation_snapshot.go:186`, `internal/repository/insight.go:271-280`, `consultation_validate.go:78-90` |
| G5 | Session stuck (`409`) when the AI call fails | **Not an issue.** Go releases the lock after an AI error **and** after a rejected AI response; a failed START deletes the empty session so a retry starts clean; stale locks also expire. | `consultation_snapshot.go:103-112`, `consultation.go:149-162,193-199` |
| G6 | Snapshot size unbounded | **Bounded:** `evidenceLimit = 20` opinions; owner messages ≤ 4000 chars. Long sessions can still grow message history; Python rejects prompts > 120 KB. Low risk for MVP. | `cmd/api/main.go:31-32`, `consultation.go:21` |
| G7 | `prior_issue_solution` only a summary | **Not an issue.** Go sends the full stored `SolutionData` of the latest archived session. | `internal/repository/consultation.go:243-270` |
| G9 | Circular FK sessions ↔ messages | **Not an issue.** Added with a separate `ALTER TABLE` after both tables exist. | `migrations/000003…up.sql:58-60` |
| — | `consultation_sessions.shop_id` | Present; the Go PRD keeps it. PRD AI §0/§10 "Decision C" is resolved. | `migrations/000003…up.sql:5` |

**Only G2 needs a decision** (reset per discussion cycle in Go, or accept a session-wide cap of 4). It does not block Path A.

## 4. Proposed next steps

1. **Path A integration (no code changes):** run Go's `compose.dev.yaml` with `DOCKER_AI_GRPC_ADDR=host.docker.internal:50051`, matching `AI_GRPC_TOKEN` (Go) = `GRPC_AUTH_TOKEN` (Python), Python with `FEEDBACK_GEMINI_MODEL=gemini-3.5-flash-lite`. Test: QR feedback → job → `ProcessFeedback` → opinions/vectors/issues in Postgres → issues API; plus a forced AI failure → retry → `FAILED` after max attempts.
2. **Decide A5** (complaint-only planets) and update PRD Path A ADR-A08 / FR-10 accordingly.
3. **Path B later:** decide G2 with Suthe; reduce Python's fail-closed rejections before integrating consultation.

## 5. Path A integration run (2026-11-14) — PASSED

**Setup (no code changes on either side):** Go `compose.dev.yaml` (`api` + `db`, auto-migrations) at `2bf7f083` with a
compose override kept outside the Go repo (`AI_GRPC_ADDR=atlas-ai:50051`, shared random `AI_GRPC_TOKEN`, dev
`JWT_SECRET`); Python `atlas-ai-service:dev` on the same Docker network (alias `atlas-ai`) with
`FEEDBACK_GEMINI_MODEL=gemini-3.5-flash-lite`. Go logged `ai service configured addr=atlas-ai:50051`.

**Flow:** owner signup (with shop) → 5 feedbacks via `POST /api/v1/public/shops/:shopId/feedbacks` (all `201`) →
worker → `ProcessFeedback` → Postgres → `GET /api/v1/owner/shops/:shopId/issues` (`200`) and
`GET /api/v1/owner/issues/:issueId` (`200`).

| Feedback | Result in Postgres |
| --- | --- |
| `Rotinya enak tapi antriannya lama banget, hampir setengah jam` | 2 opinions: `Roti enak` Positive (no issue), `Antrean lama banget, hampir setengah jam` Negative → issue `Antrean Terlalu Lama` |
| `Antre di kasir lama sekali, cuma satu kasir yang buka` | Negative → issue `Antrean Kasir Terlalu Lama` |
| `Pelayanannya ramah, tempatnya bersih` | 2 Positive opinions, no issue (A5 as designed) |
| `Ada rambut di dalam roti saya!` | Negative, **Severe** → issue `Benda Asing di Dalam Roti` |
| `Mampir sebentar beli dua roti` | No opinions; job `DONE` |

- All jobs `DONE` on attempt 1; Python RPC latency 1.2–2.3 s; every vector 384-D with model
  `intfloat/multilingual-e5-small@query-l2-v1`; every `evidence_text` is a verbatim substring.
- **Failure/recovery:** with the AI container stopped, a new feedback was still accepted (`201`); the job stayed
  `PENDING` with `last_error` "AI service is temporarily unavailable"; after restarting the AI it completed on
  attempt 2 (`Kopi dingin waktu disajikan` → `Suhu Kopi Dingin Saat Disajikan`).
- **Known gap (expected):** `Antrean Terlalu Lama` and `Antrean Kasir Terlalu Lama` are two planets for the same
  problem. P-A3 only reuses an issue on an exact title match; this fragmentation is what P-A5 (E5 retrieval +
  Gemini merge decision) addresses.
- Not covered in this run: multi-shop isolation with real data, concurrent duplicate-planet race, Path B.

## 6. P-A5 matching run (2026-11-14) — PASSED

Same setup as §5, fresh shop. Seed feedbacks created 3 planets; then 6 test feedbacks:

| Feedback | Expected | Result | How |
| --- | --- | --- | --- |
| `Ngantri bayarnya lama bgt tadi` | merge | → `Antrean Kasir Terlalu Lama` | E5 retrieval + Gemini judge |
| `Kopinya udah dingin pas dianter ke meja` | merge | → `Suhu Kopi Terlalu Dingin` | E5 retrieval + Gemini judge |
| `Nemu rambut lagi di roti, jijik` | merge | → `Benda Asing di Dalam Roti` | exact title (no judge call) |
| `Minumannya lama banget dibuatnya` | new (different stage) | new `Waktu Pembuatan Minuman Terlalu Lama` | judge: NONE |
| `Rotinya keasinan` | new | new `Roti Keasinan` | judge: NONE |
| `Kopinya kepanasan banget sampai lidah kebakar` | new (opposite direction) | new `Suhu Minuman Terlalu Panas` | judge: NONE |

6/6 as expected; all jobs `DONE` on attempt 1. A small synthetic run, not an accuracy score.

**Finding fixed during the run:** Gemini rejects request deadlines under 10 s with HTTP 400
("Minimum allowed deadline is 10s"). The first run used a 5 s judge timeout, so every judge call failed and
safely fell back to new planets. Fix in the shared `src/atlas_ai/llm.py` (affects Path B too): budgets under
10 s now fail fast with `DEADLINE_EXCEEDED` without a request, at least 10 s is always sent, and provider HTTP
400 maps to `INTERNAL` (was hidden as `UNAVAILABLE`). Judge timeout default is now 10 s.

## 7. Path B consultation run (2026-11-14) — PASSED

After the P-B1 fix (`docs/consultation-p-b1-reliability.md`), same compose environment, real Go API:
3 queue complaints → 1 planet `Antrean Kasir Terlalu Lama` → `POST /owner/issues/:id/consultations` (START,
question "Berapa jumlah total staf yang bertugas untuk melayani kasir pada jam sibuk tersebut?") →
`OWNER_REPLY` "Kasir kami hanya 1 orang. Jam ramai pukul 17.00 sampai 19.00. Kami belum bisa menambah staf."
→ `SOLUTION_READY` (2 findings, 3 evidence, 1 hypothesis, 3 recommendations, 1 indicator) → same
`request_id` again → `replayed=true` (Go idempotency, no second AI call) → `POST …/save` → `ARCHIVED` with
2 owner-confirmed memories (`Kasir kami hanya 1 orang.`, `Jam ramai pukul 17.00 sampai 19.00.`).
G5 confirmed in practice: an earlier failed reply released the session lock and the retry succeeded.

## 8. Fase 1 run (2026-11-14) — current build (G2, retry, observability, B2, bakery prompts)

Real Go API, same compose environment. Per-RPC records come from the new structured log
(`atlas_ai.rpc`, one JSON line per RPC: request_id from Go `x-request-id`, status, spans, Gemini tokens, retries).

| RPC (request_id) | Result | Gemini calls (stage, attempts, status, tokens in/out, ms) |
| --- | --- | --- |
| ProcessFeedback `job-affa68ff` | OK, new planet | extraction 1 OK 1894/79 1565 |
| ProcessFeedback `job-49e3c7d9` | OK, merged | extraction 1 OK 1893/74 1522; match 1 OK 411/70 1119 |
| ProcessFeedback `job-2d645fd7` | OK, merged | extraction 1 OK 1889/91 1468; match 1 OK 409/52 894 |
| START | OK (question) | assessment **2** OK 2084/163 5183 — rescued by the retry |
| OWNER_REPLY ×2 | UNAVAILABLE | assessment 2 × Google 503 (provider overload) |
| GENERATE_NOW | OK, `SOLUTION_READY` | solution **2** OK 2551/1095 9892 (rescued by the retry); relevance 1 OK 445/64 4385 |
| Save | `ARCHIVED` | no AI call (by design) |

- 3 stale-bread complaints → **1 planet** (judge merged both; judge timeout now 20 s so its retry fits).
- Bakery solution: 3 verbatim customer findings, tentative hypothesis (cooling/packing moisture), relevant
  recommendation (check bread temperature at packing, seal tightness), proposed indicator.
- Failures were provider 503s after the one bounded retry; Go answered "temporarily unavailable" and the session
  stayed retryable (G5). No memory candidates this run because the owner reply never succeeded.
- Earlier the same day the OWNER_REPLY → solution → idempotent replay → Save + 2 memories path passed (§7).
