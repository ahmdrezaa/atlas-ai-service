# M6.4 — Atomic grounding and semantic relevance

**Decision: PARTIAL.** Audited baseline
`55b30e37b79ef48a9eaa9889c09d8d32df4d06a6`, initially clean working tree.
Two of the six original defects are corrected. Four remain unresolved. The change
adds no false acceptance or false rejection on the three frozen corpora. This is
an offline result; it does not establish production or live-provider readiness.

## Evaluation-first evidence

Before production edits, the runner replayed all six original cases and both
ambiguous M6.3 cases through full Pydantic, structural, grounding and protobuf
mapping. Claim-level labels and reason codes are recorded in the new ledger.
The original M6.3 labels, fixtures, and historical ledgers were not edited.

The independent extension contains 67 cases across bakery filling, product
quality, packaging, queues, pricing, order recording and stock recording. It
includes valid/invalid purpose pairs with identical grammar, independent asserted
and tentative coordination, shared noun objects, tentative causes, source-ID,
number, SQL, negation and attribution checks. Unseen unrelated action objects
include satellite orbits and currency derivatives; no topic blacklist is used.
Ten provenance/attribution controls reuse original fixtures explicitly. The other
57 cases are new linguistic variations. Fixtures contain complete official
requests and five-section Solutions. Each run uses current source IDs only.

The extension was frozen before validator edits. SHA-256:

`16d4cdbb90d9ddabfcd904faf081ea75be413ddbc8353bae59abba901348c3d5`

The before/after runner rejects changed corpus hashes or duplicate phases. Its
outputs include each field's decision even when an earlier structural failure
prevents full mapping. Those diagnostic decisions are not substitutes for the
whole-Solution outcome.

## Root causes and original-case verdicts

| Original case | Before | After | Root cause / earliest responsible decision |
| --- | --- | --- | --- |
| A1 `a_independent_coordinate_fact` | Accepted, wrong | Accepted, wrong | Legacy `evaluate_claim()` hypothesis branch applies a modal marker to the full unsplit coordination; `_related_hypothesis()` provides one shared lexical anchor. |
| A2 `a_two_speculations` | Accepted, wrong | Accepted, wrong | Same branch checks relevance over the whole clause, allowing the unrelated second modal proposition to borrow the first subject. |
| A3 `a_bakery_causal` | Rejected, wrong | Rejected, wrong | Structural checks pass; legacy causal guard rejects `karena`; `_tentative_mechanism()` cannot prove this wording through its bounded queue/workflow path. |
| B1 `b_original` | Accepted, wrong | Accepted, wrong | `_independent_actions()` intentionally preserves shared noun coordination; the recommendation relevance check then accepts a cited noun anywhere in the combined object. |
| B2 `b_purpose_overlap` | Accepted, wrong | Rejected, correct | Recommendation relevance used the complete body, allowing the intended queue improvement to supply relevance for an unrelated action object. |
| B3 `b_repeated_word_attack` | Accepted, wrong | Rejected, correct | The same whole-body check accepted bread in a purpose phrase as relevance for an unrelated financial action. |

Before B2/B3 reason code was `proposal_with_no_unproved_factual_premise`;
after it is `action_has_no_cited_subject`, with INSUFFICIENT_EVIDENCE and whole
Solution rejection. A1/A2 remain `tentative_hypothesis_not_established_cause`;
B1 remains `proposal_with_no_unproved_factual_premise`; A3 remains unresolved.
The ledger includes the exact text, reason and label for every claim.

The original equipment hypothesis remains ambiguous and accepted; the explicit
equipment inquiry remains ambiguous and rejected. Neither was relabelled as a
clear positive or negative to improve the confusion matrix.

## Revised policy and implementation

The new `_action_before_purpose()` identifies the action-object span before the
first explicit `untuk` or `agar` phrase. Only that span can establish proposal
relevance using existing source checks and the existing process fallback. The
purpose describes generated intent and cannot prove that the action itself is
connected to the issue.

`evaluate_claim()` uses this span for recommendation relevance after its existing
independent-action, premise and factual validation. The full original text still
undergoes quantity, operational-window, availability, achieved-outcome and
provenance checks. No output is rewritten or removed. No new acceptance path is
introduced. Titles still pass the existing independently validated description
and cited-subject rules. M6.3 contrast and independent-action handling is retained.
Grounding rule version is now `grounding-rules-v7`.

This is a small role distinction between action and purpose, not an Indonesian
dependency parser or a general semantic proof. `untuk` can also attach to nominal
complements; those constructions can remain unresolved. A relevant word in the
action itself still does not establish complete semantic relevance. A purpose
could also itself be an unreasonable proposal; this change does not prove its
feasibility or connection to the action.

The four remaining original defects require more reliable scope and object
interpretation. Splitting every `dan` breaks shared objects such as `Evaluasi
jumlah dan konsistensi isian roti.` A causal-word relaxation would allow unknown
operational predicates to masquerade as hypothetical mechanisms. Neither change
has a defensible general correction within the existing bounded deterministic
rules. The extension reproduces these limitations in seven domains rather than
adding bakery-specific wording or domain exceptions. They remain explicit defects.

## Results

| Corpus / result | Before | After |
| --- | ---: | ---: |
| M6.2 valid accepted | 17 | 17 |
| M6.2 valid rejected | 0 | 0 |
| M6.2 invalid accepted | 0 | 0 |
| M6.2 invalid rejected | 29 | 29 |
| M6.3 valid accepted | 34 | 34 |
| M6.3 valid rejected | 1 | 1 |
| M6.3 invalid accepted | 5 | 3 |
| M6.3 invalid rejected | 19 | 21 |
| M6.3 ambiguous accepted / rejected | 1 / 1 | 1 / 1 |
| M6.4 valid accepted | 24 | 24 |
| M6.4 valid rejected | 7 | 7 |
| M6.4 invalid accepted | 29 | 14 |
| M6.4 invalid rejected | 7 | 22 |
| M6.4 ambiguous accepted / rejected | 0 / 0 | 0 / 0 |

All seven new causal variants were already rejected before implementation. All
14 residual invalid acceptances are independent coordinated factual/speculative
clauses. No baseline accepted valid case becomes rejected; no baseline rejected
invalid case becomes accepted. Finite curated counts do not prove unrestricted
Indonesian semantic correctness.

Accepted natural examples include `Evaluasi jumlah dan konsistensi isian roti.`,
`Evaluasi kemasan agar memahami keluhan pelanggan.`, nominal recommendations from
M6.2/M6.3, and the historical C5 coordinated monitoring/recording action.
`Evaluasi orbit satelit untuk memperbaiki kemasan.` and `Evaluasi derivatif valuta
untuk usaha roti.` now fail. Existing unsupported numeric quantities, false SQL,
false owner attribution and wrong-namespace cases remain rejected.

## Tests and compatibility

Docker `atlas-ai-service:dev`, Python 3.11, `--network none`; no Gemini credentials
passed to test containers and zero provider requests.

- Full suite: **1152 passed, 25 xfailed in 19.60s**. All 1103 previously passing
  tests still pass. Two old strict xfail markers were removed after the unchanged
  whole-Solution rejection assertions passed. Four original strict xfails remain;
  21 new strict xfails expose the extension's existing semantic defects. They do
  not count as functional successes. An unexpected fix becomes a strict XPASS.
- All 46 M6.2 cases retain their outcomes. M6.1 exact title/Solution replay, M6-C
  numeric rejection, P0 P3 replay, OWNER-only memory, all five consultation events,
  authenticated mock gRPC Solution/memory and NextQuestion tests pass in the full
  suite. Historical replays are offline, not fresh live successes.
- Ruff lint across source/tests/scripts: **All checks passed!**, retaining the
  existing `--ignore EXE002` Windows bind-mount executable-bit exception.
- Ruff format: **4 files already formatted**; `git diff --check` passes.
- Tracked historical evaluation files have no diff. Their SHA-256 hashes are
  included in the new ledger.

Reproduction in the Docker image with the repository mounted at `/workspace`,
`PYTHONPATH=/workspace/src`, and an independently writable `/results` directory:

```text
python -m scripts.consultation_m6_4_evaluation --phase before --output /results/m6_4_grounding.json
python -m scripts.consultation_m6_4_evaluation --phase after --output /results/m6_4_grounding.json
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache --config pyproject.toml --ignore EXE002 src/atlas_ai tests scripts
python -m ruff format --check --no-cache src/atlas_ai/consultation/grounding.py scripts/consultation_m6_4_evaluation.py tests/test_consultation_m6_4_atomic.py tests/test_consultation_m6_3_safety.py
```

For a true before run, use baseline validator code plus the new frozen evaluator
and corpus. Do not overwrite delivered ledgers or run both phases on final code
and call that baseline evidence.

Changed files: `grounding.py`, the two removed xfail entries in
`tests/test_consultation_m6_3_safety.py`, new `tests/test_consultation_m6_4_atomic.py`,
`scripts/consultation_m6_4_evaluation.py`, `evaluation/m6_4_cases.json`,
`evaluation/m6_4_grounding.json`, and this report. No other production function
was modified. The runner reuses M6.3 request/payload construction and the original
corpus evaluators, preserving the historical fixtures and measurements.

Proto/stubs, schema, findings, quotes, SQL, numeric facts, source registries,
confirmed memories, OWNER extraction, Go, Feedback, auth, session state, assessment,
question cap, production model, timeout and prompts are unchanged. No dependency,
provider invocation, retry, semantic verifier, automatic repair, commit or push.
The historical provider HTTP 429 blocker was not probed and is independent of the
four original defects still preventing OFFLINE PASS.
