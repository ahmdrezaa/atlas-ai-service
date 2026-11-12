# Milestone 4B — semantic grounding and evaluation

Latest bounded numeric provenance and quote-context changes are documented in
[Milestone 4C.1](solution-validation-alignment.md), with a new report on the same
50-case dataset. The original evaluation below remains historical evidence.

## Audit of the 4A baseline, before implementation

Baseline execution: **448 tests passed**. The existing `validate_solution` checks
schema shape, nonempty arrays, reference namespaces, exact customer quote substrings,
numeric marker field/value equality, a few causal/uncertainty expressions, qualitative
labels, and proposed target/timeframe prefixes. It does not establish relevance or
entailment. A bread review can have a valid ID yet not support an antrean claim.
An exact quote can contain a malicious command rather than a factual observation.
A correct `opinion_count` marker can be described as unique feedback or sentiment.

No structural validator is duplicated in 4B. `validate_solution` still runs first;
its schema, reference and quote checks remain authoritative. `STAT`/`STAT_LABELS`
were moved, not copied, to the grounding module so both layers use the same syntax.
One small numeric policy extension allows explicitly attributed exact source excerpts
(e.g. `Owner menyatakan: Shift sore 2 orang.`); the full numerical body still has to
be in the cited source. Stored quotes and UUIDs are never normalized or rewritten.

### Prompt-version audit

The v1 assessment base prompt prohibited solution generation in all uses. 4A removed
that assessment-specific sentence from the shared base so it could also serve the
solution path. Assessment-specific instructions still prohibit solutions. Hence the
assessment metadata changed to `consultation-assessment-v2`; this did not add an RPC,
reset counters, modify archived messages, or change the protobuf schema. Go should
store this as version metadata, not assume only v1 is valid; that Go behavior has not
been tested here. Existing source-selection/trust boundaries remain in the base.

4B keeps assessment v2 and increments only the solution prompt to
`consultation-solution-v2`, reflecting more conservative output/attribution guidance.

### Marker and frontend audit

Markers bind the SQL field and Decimal value, not arbitrary surrounding language.
4B accepts a bounded SQL sentence form, rejects common relabelled metric phrases and
unproved prose attached to a marker, and checks small Indonesian number words (0–99).
Unknown expressions are rejected rather than inferred. This is not unrestricted
natural-language numeric reasoning, a percentage calculator or a statistical test.

`Usulan:` is an **internal text policy**, not a proto field or a verified frontend
requirement. It is preserved on the wire and in any subsequently stored string, so
the frontend may display it literally, duplicate its own heading, or wrap differently.
Removing it now could obscure the distinction between a proposed target and actual
historical results. Keep it for compatibility with the existing validator; review
rendering/wording with Sakha and Suthe. No frontend source is available here, and no
frontend rendering compatibility is claimed. Likewise, `Hipotesis:` is textual
framing, not evidence that a cause was established.

## Meaning of the four labels

- **SUPPORTED**: the intended meaning is that every factual clause is supported by
  the cited current sources with the appropriate provenance. The checker issues this
  judgment after bounded matching; it is fallible, not a formal proof or independent
  confirmation of truth.
- **CONTRADICTED**: a recognized proposition, polarity, quantity or SQL metric
  description conflicts with its cited source/statistic.
- **INSUFFICIENT_EVIDENCE**: irrelevant/missing support, unresolved language,
  unsupported population/causal conclusions, lost attribution, conflicting cited
  sources, or an unproved premise. It includes abstention, not only known falsehood.
- **NOT_APPLICABLE**: an explicitly tentative hypothesis, measurement definition or
  normative proposal with no unproved factual premise. It does not certify that the
  proposed action will work or that a hypothesized cause is likely.

Customer material establishes what the customer reported, not population prevalence.
Owner material supports an attributed owner statement, not independent verification.
Confirmed memories are treated as owner-confirmed business context. SQL statistics
are a separate aggregate source, never a count of the representative evidence sample.
Historical AI messages/solutions and normalized AI opinion paraphrases are not used
as independent semantic proofs. The verifier sees raw evidence/full feedback instead.

## Dataset created before the verifier

`evaluation/grounding_cases.json` has **50 synthetic, developer-labelled cases**:
16 SUPPORTED, 7 CONTRADICTED, 16 INSUFFICIENT_EVIDENCE and 11 NOT_APPLICABLE. Thus 27
cases are acceptable and 23 should fail. Each includes the claim, source aliases
mapped to real-format synthetic UUIDs, the expected result and a review rationale.
The shared input is parsed through the official request protobuf/context builder.

Cases include narrow and wider paraphrases, valid-but-irrelevant references,
negation/antonyms, sample overgeneralization, causality, hypotheses-as-facts, renamed
SQL metrics, number words, owner attribution, historical AI, prompt injection,
unsupported recommendation premises, proposed targets, and multi-source conjunctions.
These are proposed labels **for human review**, not independently adjudicated ground
truth or a held-out benchmark. The rules were developed against this small set;
reported performance must not be generalized to production language distributions.

## Strategies and selected production behavior

| Strategy | Reliability evidence | Complexity/cost/latency | Decision |
| --- | --- | --- | --- |
| A. Existing structural checks only | 20 false acceptances on 50 cases | No verifier invocation, cheap | Insufficient alone |
| B. Structured LLM claim/source verification | Live attempts produced no completed judgments | Extra network/model cost; failed call observed at 28.747s | Evaluation-only, not deployed |
| C. Structural checks then bounded source verification, abstain on unknowns | 0 false acceptances, 2 false rejections on the same developmental set | No additional model invocation; measured rule CPU 2.998ms for 50 small claims | Selected conservative implementation |

The selected rules check explicit source attribution, small reviewable surface
equivalences, each conjunct, recognized opposite/negated/quantified propositions,
SQL meaning around markers, scope/causal overstatement, and normative premises.
They are not a general entailment model. The two positive abstentions are preserved:
`wider_positive_paraphrase` and `polite_action`. Do not remove them or relabel them as
negative to improve the score. This trade-off deliberately sacrifices some legitimate
outputs rather than silently accepting unsupported facts.

`solution_to_proto` runs the existing structural checks and then this source check.
Any CONTRADICTED/INSUFFICIENT_EVIDENCE result fails the **whole** solution with INTERNAL,
without raw source text in the error. No partial solution, invented reference, silent
rewrite or repair retry is returned. Invalid Go input remains INVALID_ARGUMENT;
provider errors/timeouts retain their distinct existing statuses. Deadlines and
cancellation are checked during grounding, using the remaining shared turn budget.
NextQuestion does not invoke this solution validator; all five events retain their
existing deterministic question/counter policies. No business memory extraction,
session state, agents, retrieval, embeddings, database or proto changes were added.

## LLM verifier and live observations

`evaluation.py` contains the separate, versioned `grounding-evaluation-v1` prompt and
Pydantic batch result. It is **not imported by the production pipeline**. It receives
only claims, their current cited raw sources and the SQL snapshot, not archived AI
outputs or expected labels. Current evaluation wiring uses opaque claim IDs so
descriptive dataset IDs do not leak label hints. Every input claim needs exactly one
output ID; missing/duplicate/unknown IDs invalidate the batch. Source UUIDs stay intact.
The initial failed live attempts preceded that opaque-ID hardening; no judgments
were obtained from those attempts. Mock tests exercise wiring, not semantic accuracy.

As explicitly requested, local `.env` now selects **gemini-3.8-flash**. No key value
was printed. All generation inputs were synthetic. Model metadata lookup succeeded.
Generation evaluation did not complete:

| Attempt | Input | Observation |
| --- | --- | --- |
| Full-set evaluation, first batch | 10 claims; later batches skipped | DEADLINE_EXCEEDED after 28.747s |
| Smaller evaluation | 4 claims, timeout cap 55s | UNAVAILABLE after 5.104s |
| Diagnostic evaluation | 4 claims, timeout cap 55s | SDK GoogleAPIError / HTTP 504; latency not recorded |

There were three bounded generation attempts, no automatic retries, and one separate
model-metadata request. **Zero live verifier decisions** can be compared to labels.
Provider failures and skipped cases are unscored; they must not be reported as model
accuracy, semantic false rejection or evidence that the model is always wrong.
Reports keep error coverage separately and use null scores when nothing was scored.
No fallback mock is substituted for a live judgment. A future unresolved-claims LLM
hybrid is represented only as an offline candidate; these failures provide no basis
for enabling it in the production hot path.

The measured deterministic CPU overhead is for tiny synthetic sources in one local
container run, not a production p95/SLA. The selected path adds **zero provider calls**.
An additional verifier call could consume most/all of the default 30s shared turn
budget; observed failing-call latency alone was 28.747s. Successful-verifier latency,
token use, cost, quality and end-to-end Go behavior remain unproven.

## Evaluation results and reproduction

| Expected label | Cases | Accepted by structural baseline | Accepted by selected rules | Correct rule label |
| --- | ---: | ---: | ---: | ---: |
| SUPPORTED | 16 | 16 | 15 | 15 |
| CONTRADICTED | 7 | 7 | 0 | 7 |
| INSUFFICIENT_EVIDENCE | 16 | 13 | 0 | 16 |
| NOT_APPLICABLE | 11 | 11 | 10 | 10 |

Structural baseline: false acceptance **20/23 unsafe cases**, false rejection **0/27**.
Selected checks: false acceptance **0/23**, false rejection **2/27**, exact label
agreement **48/50**. These are deterministic results against proposed synthetic
labels, **not Gemini accuracy**. Full confusion matrices, per-case reasons and dataset
SHA-256 are in `evaluation/grounding_rules_report.json`.

After installing the existing project/dev dependencies, from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m atlas_ai.consultation.evaluation --output evaluation/grounding_rules_report.json
# Explicit opt-in network evaluation using synthetic data only:
python -m atlas_ai.consultation.evaluation --live --env-file .env --output evaluation/grounding_live_report.json
python -m pytest -q -p no:cacheprovider
ruff check src/atlas_ai/consultation src/atlas_ai/grpc_service.py tests
```

Windows Application Control previously blocked native gRPC DLLs. Tests/evaluation
for this milestone run in the existing Python 3.11 Docker image with the repository
mounted read-only, except the evaluation artifact directory. The `.env` file is only
read for explicitly opted-in live evaluation. Offline tests/evaluation use no network.

## Remaining risks and next evidence needed

This is **not production-ready unrestricted semantic grounding**. The controlled
language checker may reject valid paraphrases, mixed-origin sentences, advanced number
words, complex negation, comparisons or causal language. It may miss more subtle
premises/presuppositions or attacks outside the small tested vocabulary. Opposing
uncited sources and real-world truth are not comprehensively verified. Source reviews
can themselves be wrong; confirmed memories can become stale. NOT_APPLICABLE is not
a guarantee of feasibility, safety, calibration or future business impact.

Before widening acceptance, have humans adjudicate/extend the labels, collect a
held-out set of real-style anonymized cases, red-team source conflicts and hidden
premises, test customer/owner attribution with the frontend, and obtain completed
live verifier evaluations under an explicit latency/cost budget. Do not deploy a
mandatory second invocation just because a mocked test returns SUPPORTED. The existing
Go no-evidence and cumulative-counter integration decisions also remain unresolved.

## Final verification

- Baseline before implementation: 448 passed in 5.11s.
- Final full suite: **526 passed in 4.89s**, Python 3.11 Docker, network disabled.
- Ruff lint and formatting checks pass; Compose configuration validation passes.
- Tests cover all existing events/question paths, source-label decisions, full-solution
  rejection, contradictory sources, numeric words/metric swaps, injection boundaries,
  deadline/cancellation and verifier failure/coverage handling.
- No successful live verifier judgment or Go-backend end-to-end result is claimed.
