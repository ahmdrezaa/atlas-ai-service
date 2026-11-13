# M7.1 — Gemini-first prompt alignment and consultation quality

**PROMPT ALIGNMENT: OFFLINE PASS; LIVE GENERATION QUALITY: NOT VERIFIED.**

Baseline `d8328e618b15f8c5276a4ea07916860e75cfd10b`, initially clean working tree.
The only production change is the Solution system prompt and its version,
`consultation-solution-v8`. The structured-output invocation, evidence guards,
provider schema, protobuf mapping, deadlines and event policy are unchanged.
No provider calls, retries, repair calls, dependencies, commits or pushes.

## What changed from v7

V7 already distinguished symptoms from unproved causes and required valid
references. V8 preserves those requirements; this is not a claim that v7 required
proof of every proposed cause. The rewrite consolidates overlapping rules and
places responsibility for useful advice with Gemini rather than a lexical checker.

| Concern | V7 | V8 and rationale |
| --- | --- | --- |
| Consultation task | Contract-led MVP instructions | Starts with understanding the specific issue and practical Indonesian MSME advice. |
| Sources | Detailed source constraints, some repeated | One reasoning paragraph distinguishes period/SQL scale, sampled opinions/sentiment, OWNER, confirmed memory and historical AI. |
| Hypotheses | Uncertainty plus a queue example and a list of forbidden causal/verification terms | Explicitly says references ground symptoms/context, not proof of the cause. A bakery example contrasts a valid tentative mechanism with an invented broken machine. No new banned-word list. |
| Recommendations | Multiple framing/title examples around validator behavior | Concrete, testable, issue-relevant actions, real constraints, coherent natural nominal/compound wording, one primary and up to two alternatives. |
| Findings/quotes | Extractive attribution and exact quotation rules | Retains actual cited-content proof, short faithful attributed excerpts, exact quote/negation/context protection, and preferably empty notes. |
| Numbers | Long overlapping operational/metric instructions | Consolidates SQL context-only policy, existing operational provenance, numeric action proposals, generic metrics and required target/timeframe proposal framing. |
| Memory/injection | Repeated source exclusions | Refers to the separate unchanged OWNER-only policy and base untrusted-data boundary while retaining essential exclusions. |

The complete prompt is reviewable in `src/atlas_ai/consultation/prompts.py`.
Solution-specific text decreased from **675 to 605 whitespace-delimited words**
and **4891 to 4462 UTF-8 bytes**. This is not a token count or a measured latency
improvement. The seven complete rendered prompts, including unchanged base/memory
messages and all supplied context, are **10540–10686 bytes**. No source is truncated.

The example uses customer text `Rotinya empuk, tapi isian coklatnya sedikit.`,
a faithful finding excerpt, and `Hipotesis: takaran isian coklat mungkin belum
konsisten selama produksi.` A broken filling machine is explicitly not established.
Examples are illustrative, not additional evidence or fixed output templates.

## Actual runtime compatibility

All five arrays remain required and nonempty; recommendations remain limited to
three. Descriptions must be nonblank. Hypotheses retain the required `Hipotesis:`
prefix and explicit uncertainty. Confidence/priority remain qualitative; provider
null for an unspecified label is still adapted locally to the legal empty label.

Every finding, hypothesis and recommendation needs current, correctly namespaced
source IDs. Findings still need actual cited content; merely valid IDs are not
enough. Quotes must be faithful exact substrings. SQL aggregate output remains
excluded from MVP Solution claims even when aggregates inform the reasoning.
Existing operational numbers preserve their predicate, unit, value and time scope.
Metrics have no reference field and remain generic without numbers/clock ranges.
Nonempty targets and timeframes start `Usulan:`; new numeric plans in actions are
also explicitly proposed. Candidates stay separate response metadata, not a sixth
Solution component or confirmed memory.

Current fingerprints were captured before the prompt edit and verified unchanged:

| Artifact | SHA-256 |
| --- | --- |
| Provider schema | `2e79359e0584ab9e01f04fb90368ab7b540e24d027ba4e4e425d950de07d3805` |
| Full local Pydantic schema | `b0f9b46a99ac2d9f93a073bfaf810968bb388849a31591ca1277994b05ccaaac` |
| Official protobuf descriptor | `1675910827873f8c8423fa7a8f9a443b15139575dcc0fcd86df4965d31e521ad` |

The existing provider-compatible compact schema remains in use with full local
validation afterward. Base, Assessment and Memory prompt hashes are unchanged.
All five event tests verify v8 reaches the actual formatted provider boundary and
the authenticated gRPC response metadata. GENERATE_NOW invokes only generation;
the mocked assessment-to-solution paths retain two provider invocations, including
exactly one generation invocation. No extra invocation was added to production.

## Complete synthetic scenarios

The focused runner owns reproducible, predefined fixtures. Each request uses
official protobuf types, stable UUIDs, an issue/shop, period, aggregate statistics,
two representative opinions, OWNER history, a confirmed memory, previous Solution
and archived Solution. The deliberately unverified historical AI claims are never
current evidence. Aggregate opinion count is 40 while the evidence sample has two
opinions; no population inference is made from those two excerpts.

| Scenario | Customer evidence | OWNER / confirmed context | Relevant action | Example forbidden finding |
| --- | --- | --- | --- | --- |
| Bakery filling | Soft bread but little chocolate filling | Filling measured manually; production in shop kitchen | Evaluate filling portions across batches | Broken filling machine |
| Bakery quality | Inconsistent taste, sometimes too salty | Taste checked before display; made in shop kitchen | Compare taste across batches with simple inspection notes | Removing the customer's negation and calling taste consistent |
| Cafe queues | Good coffee but long queue | Manual order recording; two cashiers confirmed | Observe order intake and record queue duration | Broken cashier machine |
| Cafe takeaway | Packaging damaged on the journey home | Orders wrapped before handover; takeaway service confirmed | Simulate takeaway and test packaging closure | Supplier delivered defective packaging |
| Pricing | Price feels high for portion size | Price list shown; owner sets prices | Collect and compare price/portion perceptions | All customers reject shop prices |
| Order mistakes | Received order differs from request | Manual records; orders accepted at cashier | Review recording and test order confirmation | Cashier intentionally ignored orders |
| OWNER constraints | Long afternoon cashier queue | Limited budget, cannot add staff, supplied hours; two cashiers confirmed | Manual duration records without buying equipment | Unlimited repair budget |

Each case in `evaluation/m7_1_consultation_quality.json` includes full customer
text, owner context, memory, factual boundaries, a plausible hypothesis, a relevant
and an irrelevant recommendation, the complete official request, and a clearly
labelled **predefined output, not model-generated evidence**.

## Deterministic acceptance results

**42/42 checks passed:** seven valid five-section Solutions and 35 intentionally
invalid variants. Every scenario tests fabricated factual content with actual
current source IDs, invented quotes, wrong source namespaces, fabricated SQL
quantities and operational numbers without matching provenance.

All seven valid outputs pass Pydantic, structural evidence validation, blocking
factual grounding and official protobuf response round trip. The constraints case
also returns one eligible verbatim OWNER hours candidate with its actual owner ID;
other cases return an empty candidate list. No candidate is inserted into memories.

These outputs were authored as fixtures. Their acceptance establishes compatibility,
not that Gemini generated them, preferred them, or improved relative to v7.
No real-model A/B comparison, semantic score, or prompt-injection resistance result
is inferred from the mock provider or from prompt-string assertions.

## Generation-quality review rubric

The eight rubric dimensions are recorded in the evaluation JSON and review bundle:

| Dimension | Reviewer evidence required |
| --- | --- |
| Factual accuracy and attribution (critical) | Match each finding to its cited source text; check sample scope, claims and attribution. |
| Hypothesis uncertainty (critical) | Identify tentative scope and confirm no proposed cause became an asserted operating condition. |
| Recommendation relevance | Explain how the actual action addresses the issue, beyond incidental shared words. |
| Operational usefulness | Identify an executable/investigable action and manual observation or measure, without guaranteed impact. |
| Constraint awareness | Check supplied staff, budget, equipment and capacity limits; flag invented resources. |
| Indonesian clarity | Judge concise, approachable, specific language and coherent compounds; flag jargon or repeated generic disclaimers. |
| Reference faithfulness (critical) | Check actual current source content, ID namespaces, exact quotes/negation, AI-history exclusion and memory provenance. |
| Generation/validation success | Use actual capture/status/timing to establish valid output within the existing turn budget without repair calls. |

Score each dimension **0** for substantive failure, **1** for partial satisfaction
or insufficient review evidence, **2** for fully meeting it with identified output
and source evidence. A quality pass requires actual provider evidence plus local
validation, at least **14/16**, no zero, and every critical dimension at 2. Record
the reviewer and criterion-specific reasons; do not calculate these scores from
fixture acceptance or lexical diagnostics. All delivered scorecards remain null.

## Reproducible offline harness and pending live review

Run in Docker `atlas-ai-service:dev` with the repository mounted at `/workspace`,
`PYTHONPATH=/workspace/src`, and a new writable `/results` directory:

```text
python -m scripts.consultation_m7_1_quality --output /results/new-quality.json --review-bundle /results/new-review-bundle.json
```

The runner refuses existing output paths, including historical ledgers.
The delivered `evaluation/m7_1_quality_review_bundle.json` contains seven exact
rendered LangChain prompts, official requests, current provider schema, the rubric,
blank reviewer scores and capture-metadata fields. Per-case prompt hashes bind a
future capture to the supplied synthetic context/version. No fixture output is
injected into the human prompt as a proposed answer.

When live access is authorized and restored, use those official requests through
the existing service and existing invocation path. Record actual model ID, v8,
request ID, prompt hash, elapsed time, status, usage when available, invocation
count and raw structured output. Do not add an LLM evaluator or automatic repair.
The current milestone performs no such requests: recovery from the prior provider
blocker has not been established. It does not assert that HTTP 429 still occurs now.

For local validation of externally collected output, `--captures` accepts a JSON
object mapping one or more known scenario IDs to their typed structured outputs:

```text
python -m scripts.consultation_m7_1_quality --output /results/new-capture-validation.json --captures /results/captures.json
```

This validates against the same source IDs/context and supplies blank quality
scorecards. The script explicitly cannot verify a file's provider origin; even
successful supplied captures leave live-quality status NOT VERIFIED until actual
provider evidence and human review are recorded. Its reconstructed protobuf
response is an offline mapping result, not proof of a live gRPC round trip.

## Full verification and limitations

- Full pytest: **1264 passed, 25 xfailed in 9.53s**. All existing historical strict
  xfails remain visible; none were removed or added for this milestone.
- The 62 new prompt/fixture/regression tests cover seven contexts, all six variants,
  human-review exports, untrusted-data placement, provider/local/protobuf schema
  compatibility, unchanged base/assessment/memory prompts and five authenticated
  native mock gRPC events with prompt-version metadata.
- Existing M6.1–M6.4 replays, Gemini-first factual safeguards, valid/invalid source
  references, quotes, numbers, OWNER-only memory, NextQuestion, deadlines and
  provider-failure behavior pass in the full suite.
- Ruff lint across source/tests/scripts: **All checks passed!**, using the existing
  Windows Docker executable-bit exception `--ignore EXE002`.
- Ruff format: **4 files already formatted**; `git diff --check` passes.

The prompt now asks Gemini to make better relevance/usefulness judgments, but
real-model compliance is unmeasured. Irrelevant proposals may still pass the
nonblocking semantic policy. Mixed causal/factual and nominal-modifier guards
retain known conservative rejections; natural language that passes the schema
may still fail those guards. The existing bakery `mungkin ... karena ...` example
remains an expected limitation; v8's illustrative tentative explanation is actually
tested through mapping. No deterministic grammar rules were added to hide this.

Model variability, prompt injection, latency/output-token sufficiency, and generation
quality require actual provider captures and review. Offline success is not a claim
of production readiness or improved real-model accuracy.

Files changed: `prompts.py`, the current-version assertion in
`tests/test_consultation_m6_1_alignment.py`, new `tests/test_consultation_m7_1_prompt.py`,
`scripts/consultation_m7_1_quality.py`, the two new M7.1 evaluation artifacts, and
this report. Grounding, Solution schema/mapping, context preparation, assessment,
memory, auth, proto/stubs, Go, session/database behavior, model configuration,
shared deadlines, question limit and Feedback/NLP remain unchanged. Historical
evaluation hashes and protected-file checks are recorded in the new report metadata.
