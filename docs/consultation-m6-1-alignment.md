# Milestone 6.1 — Solution validation alignment

**Decision: BLOCKED by provider rate limiting.** Offline alignment is verified,
but no fresh Solution or OWNER-memory response was returned live. The single
fresh authenticated RPC received HTTP 429 → actual gRPC UNAVAILABLE. No retry,
repair invocation, second test, production-model change, commit or push.

Baseline: `350d305270831d98b1d696fb0a87db23d3afa7a8`, initially clean working tree.
Date: 2026-11-13; live timestamps are UTC. Evidence:
[`evaluation/m6_1_alignment.json`](../evaluation/m6_1_alignment.json).

## A. Root causes and exact replay

Inspected the actual consultation modules, ai.proto, M6 runner, M6/P0/M5 ledgers,
M6/P0/memory reports and existing tests. No local AGENTS.md was found. All replay
and adversarial testing used synthetic data and disabled external networking.
Source texts, references and field verdicts are recorded under `before` and
`after` in the new ledger. No historical ledger was edited.

| Case | Actual failed field / stage | Root cause | Correction and observed replay |
| --- | --- | --- | --- |
| M6-B-COMP | `recommendations[0].title`: `Optimasi Alur Kerja Kasir`; grounding `INSUFFICIENT_EVIDENCE`, `proposal_contains_unproved_premise_or_unresolved_language` | A short action label was evaluated as a standalone imperative claim, despite its valid action description and cited subject | Validate eligible descriptive labels with their independently validated description and actual citations. Exact unchanged output now passes structure, grounding and protobuf mapping |
| M6-C-COMP | `success_indicators[0].metric`: `Waktu tunggu antrean kasir pada jam 17.00-19.00`; structural rejection `unverified numeric factual text` | Clock window is present in memory, but metric has no references field. No valid output-field provenance policy authorizes those numbers | Prompt v7 requires generic metric and explicitly proposed measurement schedule. Original recorded output remains rejected; numeric validator is unchanged |
| M6-D-COMP | `hypotheses[0].statement`; grounding reason `unproven_scope_causality_or_independent_verification` | Original wording and references are missing; only verdicts and output digest remain | Not exactly replayable; false-positive classification UNKNOWN. Hypothesis validator is unchanged. Added focused positive/negative tests and clearer prompt wording |

B references are the original OWNER message
`20000000-0000-4000-8000-000000000003`:
`Saat jam ramai, dua barista menangani pesanan.`, and confirmed memory
`30000000-0000-4000-8000-000000000004`:
`Owner ingin mengurangi antrean dengan memperbaiki alur kasir.`
The description is unchanged:
`Evaluasi dan tata ulang pembagian tugas barista saat pelayanan jam ramai.`
The title describes the proposed work; it does not assert an established cause,
resource availability or guaranteed outcome. Its prior rejection was a field-role
false rejection. Replay uses identical generated text and IDs, never a repaired
live response.

C's time window is in confirmed memory
`30000000-0000-4000-8000-000000000001`:
`Jam ramai pukul 17.00 sampai 19.00 setiap hari.` Metric references remain absent,
as required by proto. The existing rejection is justified under the current
conservative output policy. A prospective test fixture using generic metric plus
an `Usulan:` schedule passes that numeric check. Changing the metric alone does
not make the original C Solution valid: posthoc offline checks also reject its
causal hypothesis and unresolved title scope. These later checks are not claimed
to have been reached by the original M6 RPC.

Repository search found no additional original D output. Its checksum cannot
recover the sentence; no hypothetical reconstruction is treated as evidence.

## B. Changes and safeguards

Only two production modules changed:

- `grounding.py`: internal `GroundingClaim.recommendation_description` links a
  recommendation title to its description. `solution_claims()` sets this field
  only on titles. `_recommendation_title()` retains existing independent
  action/advisory handling and adds a bounded descriptive-label path. Rule version
  is `grounding-rules-v4`.
- `prompts.py`: `consultation-solution-v7` clarifies title/description roles,
  forbids numbers/clock ranges in metric definitions, puts new proposed schedules
  in timeframe, and clarifies tentative versus verified/causal/population claims.
  Examples are instructions about format, never evidence. Assessment prompt and
  memory extraction policy are unchanged.

The label path is **not unconditional acceptance of `Optimasi` or arbitrary noun
headings**. It requires a short action-family label with an intent matching its
description, a description passing existing action/premise validation, substantive
relevance to current cited sources, and subject linkage to both description and
citations. Every subject token must come from those cited sources except a small
set of planned activity nouns present in the description. `tugas`/`kerja` are
equivalent only for topic linkage. Neither generated description nor historical
AI output becomes factual evidence. New resources/modifiers cannot be supplied
solely by generated prose. Assertions of current conditions, achieved results,
causes, guarantees and unsupported numbers remain rejected.

Final adversarial review caught five unsupported modifiers accepted by the initial
linkage draft (`lelah`, new equipment/staff, automatic/modern cashier). The five
tests first failed with DID NOT RAISE, then passed after restricting description
contributions to planned activity nouns. This tightening happened after the HTTP
429 attempt. That attempt retains its actual code hashes; the final revision is
offline-verified and has no successful live validation.

No changes to solution.py, numeric/quote/source-ID validators, hypothesis logic,
memory.py, assessment.py, pipeline.py, authentication, feedback, proto/stubs, Go,
dependencies, production environment or session lifecycle. The caller still
receives exactly one official result and optional turn-level memory metadata.

Other new files:

- `scripts/consultation_m6_1_alignment.py`: exact offline replay and reuse of the
  existing authenticated gRPC runner; requires recorded offline checks, permits
  at most two total live GENERATE_NOW calls, requires a reason for a second, and
  refuses further calls after a successful response. No automatic fallback/retry.
- `tests/test_consultation_m6_1_alignment.py`: 66 focused tests, including five
  authenticated native gRPC mock tests for all events.
- This report and the new M6.1 ledger.

## C. Regression and negative coverage

Final full suite: **965 passed in 6.05s**, Docker Python 3.11, network disabled.
Pre-live suite: 960 passed in 13.76s. All original 899 tests remain passing.

Coverage includes descriptive labels with paired actions; unknown heading words;
current-condition and guaranteed-outcome titles; unsupported description premises
and modifiers; irrelevant-but-valid citations; fake quotes; wrong namespaces;
fabricated SQL numbers; generic metrics versus unreferenced clocks/staff counts;
tentative evidence-related hypotheses versus proven causes/unrelated speculation;
P0 P3 replay and the 50-case grounding regression; M5 provenance/conflicts/dedup,
optional malformed metadata and existing no-extra-call behavior; NextQuestion and
all five events. The authenticated mock RPCs returned the exact B Solution and
OWNER candidate with unchanged request snapshots, correct oneof, metadata and
protobuf round trip. GENERATE_NOW used one mocked boundary call, other generating
events two; no dedicated memory call.

Ruff full application/tests/scripts lint passed, with the same documented EXE002
Windows bind-mount exception as M6. Format check passed for all four changed/new
Python files. `git diff --check` passed. Historical SHA-256 checks preserve all
22 pre-existing evaluation JSON files, including P0, M5 and M6.

## D. Memory provenance

Exact B candidate:

```json
{
  "fact_text": "Saat jam ramai, dua barista menangani pesanan.",
  "source_owner_message_ids": ["20000000-0000-4000-8000-000000000003"]
}
```

It matches the OWNER message verbatim. Original snapshot has no equivalent
confirmed staffing memory, so it survives. Tests verify repeated proposals merge,
equivalent confirmed memory suppresses it, contradictory OWNER staffing suppresses
it, and an AI message ID cannot support it. These are offline/native mock results.
Python never confirms, saves or appends it to snapshot memories. Live owner-memory
response remains **UNVERIFIED**, because the provider produced no output.

## E. Fresh live gRPC attempt

| Field | Actual observation |
| --- | --- |
| Attempt | M61-L1; new authenticated GENERATE_NOW, synthetic D + durable OWNER fact |
| Model | Explicit diagnostic override gemini-3.6-flash; production remains gemini-3.8-flash |
| API / thinking / output budget | v1beta / LOW / 2,048 tokens |
| UTC start / end | 2026-11-13T00:58:39.343435+00:00 / 00:58:39.985012+00:00 |
| Actual invocation count | 1 GeneratedSolutionTurn; assessment 0; dedicated extraction 0; retries 0 |
| Native RPC latency | 0.630898 seconds |
| SDK / LangChain boundary | SDK 0.484s; invocation/error handling 0.503811s |
| Prompt / schema UTF-8 bytes | 10,671 / 2,685 |
| HTTP / actual gRPC | **429 / UNAVAILABLE**; sanitized category PROVIDER_RATE_LIMIT |
| Local schema / structure / grounding / memory | All NOT_REACHED |
| Returned Solution / memory | None; no successful protobuf response |
| Token usage | Unavailable, not assumed zero |

An observed HTTP 429 is distinct from a client timeout, HTTP 503/504, parsing or
local validation failure. Existing application mapping to UNAVAILABLE is preserved.
No raw provider error body, API key or actual user data is recorded. The specific
quota dimension/reset time is unknown; there is no basis to blame prompt size,
thinking, output budget or validators. A second request was not justified after
this rate-limit response and was not sent.

Development budget remained 45s with a native caller deadline 60s. Server entry
remaining was 59.998816s; SDK got 44.749345s. This failure was far before either
deadline. Production Python remains 30s. Go source's default 60s is not proof of
its deployed deadline; no Go + database integration was executed.

## F. Commands and remaining blockers

Final offline commands:

```powershell
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m pytest -q -p no:cacheprovider
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m ruff check --no-cache --config pyproject.toml --ignore EXE002 src/atlas_ai tests scripts
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m ruff format --check src/atlas_ai/consultation/grounding.py src/atlas_ai/consultation/prompts.py scripts/consultation_m6_1_alignment.py tests/test_consultation_m6_1_alignment.py
```

Replay uses the same read-only workspace plus writable evaluation mount, with
network disabled: `python -m scripts.consultation_m6_1_alignment --phase before`
then `--phase after`, each with `--output /results/m6_1_alignment.json`.
Actual live command (network enabled):

```powershell
docker run --rm --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --mount 'type=bind,source=D:\atlas-ai-service\evaluation,target=/results' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m scripts.consultation_m6_1_alignment --phase live --output /results/m6_1_alignment.json --id M61-L1 --model gemini-3.6-flash --api-version v1beta --thinking low --turn-budget 45
```

Before Go integration, resolve provider availability/rate limits and obtain a new
fully validated Solution RPC on the final revision. An owner-memory response is
still unverified. Production 3.8 remains unvalidated; comparator success, if later
obtained, will only establish diagnostic-model functionality. The title recognizer
remains deliberately bounded and can abstain on other valid language; lexical
topic linkage is not complete semantic entailment. Numeric metric provenance,
Go's cumulative question counter and deployment caller deadlines remain explicit
integration constraints. No Milestone 7 work was started.
