# Milestone 6.2 — Field-aware grounding reliability

**Decision: OFFLINE PASS.** Baseline `beac6361fe40468e55fad3e04fed66833ecd22e3`,
initial working tree clean. No Gemini requests, dependencies, state changes,
commit or push. Live provider restoration is unknown after M6.1 HTTP 429;
production gemini-3.8-flash and its timeout remain unchanged.

## Measurement

Frozen corpus: `evaluation/m6_2_cases.json`, version `m6.2-frozen-id-v1`.
It contains 46 synthetic whole-Solution variations based on M6-B, covering
findings, hypotheses, actions, titles, metrics, targets, schedules, quotations,
negation, mixed language, incorrect namespaces, missing IDs and AI source attacks.
Expected labels and file bytes were fixed before production changes. Both runs
record the same SHA-256 in `evaluation/m6_2_grounding_reliability.json`.
The evaluator executes full Pydantic, structural, grounding and protobuf mapping;
it does not evaluate isolated regex matches or call a model.

| Expected / result | Before | After |
| --- | ---: | ---: |
| Valid accepted | 11 | 17 |
| Valid rejected | 6 | 0 |
| Invalid accepted | 1 | 0 |
| Invalid rejected | 28 | 29 |

Acceptance of valid cases increases from 64.7% to 100% on this corpus. There are
no newly accepted negatives; the existing unrelated recommendation false acceptance
is also rejected. These are curated regression counts, not statistical estimates
of Indonesian language accuracy or guarantees over arbitrary model output.

## Root causes and changes

Hypothesis tentativeness was checked independently in structural and grounding
validation with narrower lexical rules. Every clause needed a recognized marker,
so a harmless uncertainty closure could fail. Causal connectors were rejected
even when introducing an explicitly tentative, neutral mechanism. Recommendation
intent depended on action framing; noun forms failed, while a familiar imperative
could allow an unrelated object.

`grounding.py` now provides a shared `TENTATIVE` detector for structural validation
in `solution.py`. Rule version is `grounding-rules-v5`. The existing strict path
remains; a bounded additional path accepts a tentative relation between a proved
customer symptom and a neutral workflow mechanism. It requires:

- Verbatim-normalized, affirmative symptom proof in cited customer text, including
  negation checks; no arbitrary premise appended to the symptom.
- An operational subject anchored to cited OWNER/memory text and at least two
  distinct source anchors across symptom and mechanism.
- A bounded neutral mechanism vocabulary, without invented quantities, equipment
  states, resource availability or independent factual clauses.
- Tentative framing and only nonfactual verification closures after semicolons.

Thus `berpotensi dipengaruhi`, `ada kemungkinan`, `perlu ditelusuri apakah`, and a
relevant `mungkin terjadi karena` can pass without asserting the cause is true.
Causal language is not globally whitelisted. Examples newly accepted include:

- `Hipotesis: antrean berpotensi dipengaruhi koordinasi kasir dan barista.`
- `Hipotesis: ada kemungkinan antrean berkaitan dengan koordinasi kasir dan barista.`
- `Hipotesis: Lamanya antrean mungkin berkaitan dengan koordinasi pelayanan kasir dan barista; hal ini perlu diperiksa.`

Recommendation descriptions additionally recognize nominal action morphology for
bounded workflow objects, such as `Penataan alur pelayanan kasir.` This path needs
multiple cited object anchors, a workflow object, and no unsupported modifier or
current factual premise. Resource-changing nominals abstain rather than inferring
permission from a resource name. Newly accepted nominal titles require a consistent,
independently validated description. `_ACTION` and `_LABEL_INTENTS` were not expanded.
An imperative about unrelated objects cannot establish relevance merely by its verb.

Existing process-only observation/scheduling instructions retain their prior path;
source proof for actual clock windows is still checked structurally. Early testing
exposed 15 scheduling regressions from overly strict relevance handling; the final
policy preserves those existing observation cases, and all original tests pass.

## Policy by field

| Field | Final treatment |
| --- | --- |
| Findings | Existing strict attributed clause/source proof; no keyword-overlap replacement |
| Customer quote / note | Existing verbatim, namespace, negation and numerical checks; unsupported notes still fail |
| Hypothesis | Existing path plus bounded tentative mechanism interpretation; verified causes, unrelated speculation and new operational facts fail |
| Recommendation title | M6.1 paired-label proof retained; new nominal titles also need description consistency |
| Recommendation description | Proposed workflow role distinguished from factual premises; additional nominal path needs multiple anchors and bounded objects |
| Metric | Existing generic definition policy; unreferenced operational numbers still fail |
| Target / timeframe | Existing explicit future proposal policy; historical achievement assertions still fail |
| Memory | Entire extraction/filtering implementation unchanged; OWNER proof, conflicts, dedup and confirmation ownership remain intact |

Still rejected: invented findings, negation flips, fake quotes, source namespace
attacks, proven machine-failure explanations, unsupported staff counts, unrelated
Bitcoin speculation, guaranteed revenue increases, broken-machine action premises,
and clock ranges in metrics despite their existence elsewhere in memory.

All-or-nothing validation was reviewed. A protobuf evidence note can be absent,
but automatic omission would change the model response and obscure unsupported
content. **No optional-field degradation or repair is implemented.** The existing
prompt already prefers empty notes. Required fields are never rewritten or removed.

## Tests and compatibility

Final Docker Python 3.11 run: **1023 passed in 6.12s**, network disabled. This includes
all original 965 tests and 58 new cases/checks. The old 50-case grounding set, P0 P3,
M5 memory invariants, exact M6-B replay success, historical M6-C numeric rejection,
unresolved M6-D classification, protobuf mapping, and native authenticated mock
gRPC across all five events remain passing. There is no dedicated memory call.
New tests cover every frozen case, additional factual/resource attacks, paired
nominal title consistency, and a customer-negation attack against the new hypothesis
path. The final evaluator was rerun and matched the recorded after-state exactly.

Ruff lint: **All checks passed**, retaining the documented Windows bind-mount
`EXE002` exclusion. Ruff format: **4 files already formatted**. Commands inside
the existing `atlas-ai-service:dev` container, with read-only workspace and
`--network none`, `PYTHONPATH=/workspace/src`, `PYTHONDONTWRITEBYTECODE=1`:

```text
python -m scripts.consultation_m6_2_evaluation --phase before --output /results/m6_2_grounding_reliability.json
python -m scripts.consultation_m6_2_evaluation --phase after --output /results/m6_2_grounding_reliability.json
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache --config pyproject.toml --ignore EXE002 src/atlas_ai tests scripts
python -m ruff format --check src/atlas_ai/consultation/grounding.py src/atlas_ai/consultation/solution.py scripts/consultation_m6_2_evaluation.py tests/test_consultation_m6_2_reliability.py
```

Only evaluator results use a writable `/results` bind mount. Host
`git diff --exit-code -- evaluation` confirms no tracked historical JSON changes;
their SHA-256 hashes are included in the new report. The container has no Git,
so historical Git checks were executed on the host.

Changed files: `grounding.py`, `solution.py`, the new evaluator script, frozen
corpus, new tests, new result ledger and this report. Proto/stubs, Go, auth,
feedback, prompts, settings, memory, context preparation and pipeline remain
unchanged. Source/quote/numeric validation is still performed before grounding;
no implicit stats_id or extra references are introduced.

## Limits and live status

This is intentionally not a general semantic verifier. New safe paths cover a
bounded queue/service/workflow domain; other symptoms, mechanisms, noun forms,
causal paraphrases (including forms blocked by existing certainty rules), and
ambiguous operational changes can still abstain. Existing legacy paths remain
bounded lexical checks and retain their limitations; arbitrary semantics or all
business constraints are not proven by this finite corpus. Further expansion
needs additional frozen evidence and adversarial evaluation, not more unconditional
word exceptions. No LLM judge, embeddings, NLP dependency or extra model call.

Gemini live status: **NOT ATTEMPTED**, because restored rate-limit access is not
known. OFFLINE PASS does not establish live Solution/memory readiness or production
3.8 readiness. Provider blocking remains distinct from grounding implementation
quality. Go/database integration and Feedback work were not started.
