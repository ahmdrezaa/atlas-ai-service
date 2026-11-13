# Gemini-first grounding policy with blocking evidence safeguards

Baseline: `fd2e416a0e779dfcf9283aa1078cc80ef43c1be8`. The user explicitly
authorized retaining factual guards and making language plausibility/relevance
heuristics nonblocking. This supersedes earlier M6.4 assertions that every
semantically irrelevant proposal must be rejected at runtime. Historical corpus
labels and evaluation ledgers remain unchanged.

## Runtime behavior

`validate_grounding()` now evaluates two concerns separately using the same
checker, with no extra model invocation:

1. `evaluate_claim(..., semantic_checks=False)` is the runtime evidence gate.
   A rejected finding, unsupported factual premise, or other retained evidence
   guard still aborts the entire Solution with the existing INTERNAL error.
2. The default `evaluate_claim()` remains the historical language diagnostic.
   Its original label and reason are returned from `validate_grounding()` even
   when the diagnostic is negative and the evidence gate permits the Solution.

A negative language diagnostic is never changed to SUPPORTED. The returned
diagnostic list is an internal inspection/evaluation result; `solution_to_proto()`
does not put it into the public response, log it, or persist it. Callers needing
quality diagnostics must inspect that internal result or the offline evaluator.
There is no new public field or production logging of source content.

Pure lexical relevance for explicitly tentative hypotheses and framed actions
does not block. For example, `Evaluasi orbit satelit.` can be returned with
`action_has_no_cited_subject` as a negative diagnostic. This means the pipeline
does not guarantee recommendation quality or relevance; Gemini is responsible
for producing useful reasoning under the existing prompt. An ID or a diagnostic
does not turn the suggestion into a factual finding.

Title/description consistency is also diagnostic when the title's object is
independently bounded by actual cited sources and its description passes the
evidence gate. Existing factual modifier/source checks on titles remain blocking.
`Optimasi Kasir` with an unrelated proposed action can be returned with a warning;
`Optimasi Kasir Lelah` remains blocked when no source supports the tired condition.

Nominal wording can pass without the old workflow vocabulary when its complete
object phrase is affirmed in one actual cited source. Examples include
`Penyempurnaan kualitas produk.`, `Pemeriksaan kemasan.`, and `Pemeriksaan isian
roti.` The full object must occur in one source; words from separate propositions
cannot be recombined into a new condition. Resource-changing heads and factual
predicate/outcome guards remain conservative. This reuses the existing nominal
morphology rather than adding new domain or action word lists.

## Evidence remains blocking

The following implementations and contracts are unchanged:

- `_fact()` and its attribution, clause, negation and contradiction checks;
  findings and customer evidence notes still require actual cited content.
- Current source registries, namespace validation, historical AI exclusion and
  Go-owned ContextSnapshot preparation.
- Exact quotation verification, wider negation handling, numeric/statistical
  restrictions, operational predicate/value/unit/window provenance and typed
  output validation.
- OWNER-only optional memory filtering, source-content proof, confirmed-memory
  deduplication/conflicts, and Go-owned confirmation/persistence.
- Public endpoints, services, protobuf definitions/stubs, request/response types,
  five events, cumulative question limit, auth, deadlines and error mappings.
- Production Gemini model/configuration, prompt version and provider schema.

In particular, a current valid opinion, OWNER or memory ID cannot support a
fabricated finding. A relevance error cannot mask an unsupported `karena` premise,
invented available equipment, an achieved outcome, a factual second sentence,
or a false source claim. Tests exercise both nonblocking and blocking failures
in the same full Solution.

## Guards deliberately retained

Some existing checks combine grammar interpretation with factual-premise safety.
They are not globally disabled. The user's safeguard 9 takes precedence where a
safe separation is unavailable:

- Independent sentence and contrast clauses still require their own evidence or
  valid tentative/proposal framing. No blanket field-level or label-level bypass.
- The existing cause/premise guard and bounded tentative mechanism fallback remain.
  The natural bakery case `isian roti mungkin sedikit karena takaran belum
  konsisten` is still rejected; distinguishing modal scope from an asserted causal
  operating condition is not yet reliable enough to remove that guard generally.
- Unknown nominal/title modifiers and resource-changing proposals still abstain
  when their factual implications cannot be proved. Arbitrary Indonesian grammar
  is not claimed to be understood by these rules.
- Known weaknesses in coordinated factual/modal scopes from prior milestones are
  not repaired by this refactor. The 25 historical strict expected failures remain
  visible. They include quality-diagnostic limitations as well as retained scope
  limitations; they are not functional successes or evidence of semantic correctness.

## Measurement against unchanged historical labels

The new runner reads immutable baseline results from `evaluation/m6_4_grounding.json`,
verifies each frozen corpus SHA-256, then runs full current structured, evidence,
diagnostic and protobuf checks. Results are in `evaluation/gemini_first_grounding.json`.
This reports a policy change, not a claimed improvement in semantic accuracy.

| Corpus / historical label outcome | Baseline | Current |
| --- | ---: | ---: |
| M6.2 valid accepted / rejected | 17 / 0 | 17 / 0 |
| M6.2 invalid accepted / rejected | 0 / 29 | 1 / 28 |
| M6.3 valid accepted / rejected | 34 / 1 | 34 / 1 |
| M6.3 invalid accepted / rejected | 3 / 21 | 8 / 16 |
| M6.3 ambiguous accepted / rejected | 1 / 1 | 1 / 1 |
| M6.4 valid accepted / rejected | 24 / 7 | 24 / 7 |
| M6.4 invalid accepted / rejected | 14 / 22 | 29 / 7 |

The 21 additional returned cases across these corpora are historical negatives
for pure action relevance. They remain quality negatives and retain negative
diagnostics. Objective factual cases keep their rejection outcomes. No corpus
label, expected source, or historical ledger was rewritten to improve counts.
Tests assert the exact authorized differences and continue asserting rejection
of all other previously blocked cases.

## Verification

Docker `atlas-ai-service:dev`, Python 3.11, `--network none`, synthetic data and
mock provider only. No live Gemini call or semantic verifier was enabled.

- Full pytest: **1202 passed, 25 xfailed in 9.19s**.
- Ruff lint: **All checks passed!** across source/tests/scripts, with the existing
  Windows bind-mount exception `--ignore EXE002`.
- Ruff format: **7 files already formatted**.
- Current policy ledger matched the final validators before adding verification
  metadata; all original corpus hashes were checked by the runner.
- `git diff --check` passes. Historical evaluation and protected integration files
  have no diff; their hashes/checks are recorded in the new ledger metadata.

Fifty new tests are supplied in two files: 35 authenticated native mock gRPC
scenarios and 15 direct policy/evidence tests. The wire cases cover all five
events and all three valid source namespaces, factual-content fabrication,
quote/negation attacks, numerical/SQL attacks, historical-content laundering,
OWNER-only memory, and the same request with a harmless language warning versus
a blocking factual fabrication. Existing NextQuestion, deadline, cancellation,
provider-failure, memory, P0 and historical replay regressions also pass.

Old tests that demanded semantic blocking now explicitly assert successful mapping
and a negative diagnostic for the authorized cases. They were not removed or
converted into additional xfails. Existing historical xfails remain unchanged.

Reproduction in the image with `/workspace` mounted and `PYTHONPATH=/workspace/src`:

```text
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache --config pyproject.toml --ignore EXE002 src/atlas_ai tests scripts
python -m scripts.consultation_diagnostic_policy --output /results/new-policy-ledger.json
```

The evaluator refuses an existing output path. Use a new path rather than
overwriting any delivered historical evidence.

## Files

- `src/atlas_ai/consultation/grounding.py`: evidence/diagnostic separation,
  recursive propagation, source-bound nominal wording; `grounding-rules-v8`.
- `tests/test_consultation_m6_2_reliability.py`, `test_consultation_m6_3_safety.py`,
  `test_consultation_m6_4_atomic.py`: explicit authorized semantic-policy assertions.
- `tests/test_consultation_evidence_safeguards.py`: authenticated wire regressions.
- `tests/test_consultation_diagnostic_policy.py`: direct mixed-failure and wording tests.
- `scripts/consultation_diagnostic_policy.py`, `evaluation/gemini_first_grounding.json`:
  reproducible policy measurement.
- This report and the safeguard inventory.

The requested policy is implemented within the safe separation boundary described
above. This is not a resolution of the earlier six-case M6.4 semantic objective,
nor a production-readiness claim. Runtime quality can include irrelevant proposals;
provider reliability, Go end-to-end deployment, and the remaining mixed-scope
limitations require separate work. No commit, push, schema change or extra LLM call.
