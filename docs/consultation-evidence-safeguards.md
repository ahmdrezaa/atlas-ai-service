# Consultation evidence and integration safeguards

Baseline inspected: `fd2e416a0e779dfcf9283aa1078cc80ef43c1be8`.
This follow-up adds regression protection for the user's non-negotiable evidence
and Go/Python compatibility requirements. The subsequent authorized language-policy implementation is documented in
`consultation-gemini-first-grounding.md`. The previous M6.4 result remains PARTIAL.

## Blocking evidence boundary

| Invariant | Current implementation | Required treatment during any language refactor |
| --- | --- | --- |
| Official request, response and five events | `ai.proto`, generated stubs, adapter and `ConsultationPipeline.run()` | Preserve wire types, oneof result, event behavior and invocation paths. |
| Go-owned context and source namespaces | `prepare_context()`, `SourceRegistry`, `solution._sources()` | Preserve current opinion/OWNER/memory registries; historical AI cannot enter a registry. |
| Actual factual support | `grounding._fact()`, called by `evaluate_claim()` and enforced by `validate_grounding()` | Findings and factual evidence notes must remain blocking. A valid ID does not prove its claim. |
| Customer quotes and negation | `verbatim_quote_supported()`, `_affirmed_part()`, `_opposites()` | Preserve exact-source verification and wider negation context. |
| Operational quantities and time scopes | `_check_numbers()`, `operational_number_context()` | Retain matching predicate, value, unit and existing owner/memory provenance. |
| Aggregate SQL facts | Existing structural aggregate-output restrictions | No SQL laundering through an unrelated opinion, owner or memory ID. |
| OWNER-only Business Memory | `memory_candidates_to_proto()` | Preserve conservative extraction, content proof, deduplication and conflict handling. No confirmation or persistence in Python. |
| Session, auth, errors and deadlines | Existing server interceptor, adapter and shared pipeline budget | Preserve unchanged. Invalid model output remains an application failure, not a malformed-request error. |

Hypotheses and recommendations can contain factual assertions as well as tentative
reasoning or proposed actions. Field kind alone cannot determine whether a rejected
claim is safe to downgrade. For example, a recommendation failure may concern
invented resources or an unverified premise rather than stylistic action framing.
Likewise an unresolved hypothesis can mix a tentative clause and an unsupported
assertion. Downgrading every rejection for these two fields, or every
INSUFFICIENT_EVIDENCE result, would remove evidence protection.

Any future relaxation must isolate pure plausibility/relevance/grammar diagnostics
after objective source, quote, numeric and factual-premise checks. Where the
current checker cannot safely separate those concerns, retain its guard and
document the limitation. Diagnostics must not mark an unverified fact SUPPORTED,
and generated titles, descriptions or historical solutions cannot become evidence.

## New regression evidence

`tests/test_consultation_evidence_safeguards.py` adds 35 tests, all with a mocked
Gemini boundary and real authenticated native Python gRPC server/client:

- Five accepted official requests, one for each event, verify the five-section
  Solution, protobuf round trip, metadata, expected provider invocation count,
  unchanged request bytes, unchanged prepared context and unchanged model payload.
- Fifteen fabricated-finding cases, across all five events and each of the three
  correct source namespaces, cite current valid IDs. Content is absent from the
  actual cited source. Every response is INTERNAL at the factual grounding stage.
- Eight objective attacks cover factual negation, invented quotes, negation
  cropping, false operational quantity, false SQL, namespace mismatch, AI message
  IDs, and historical AI content laundered through an otherwise valid OWNER ID.
- Two optional memory attacks use either a current OWNER ID without matching
  source content, or an AI ID. Candidates are omitted while the valid Solution
  remains returnable; supplied request/session data stays unchanged.
- Five further event scenarios accept a nonblocking relevance warning, then reject
  a fabricated finding on the same input without downgrading the evidence gate.

The model factory is the only mocked boundary. Auth, generated serializers,
adapter error mapping, context preparation, source registries, structural
validation, grounding, memory filtering and final mapping all execute normally.
No provider credentials, real customer data or live Gemini calls are used.

Full suite in Docker `atlas-ai-service:dev`, Python 3.11, `--network none`:

```text
python -m pytest -q -p no:cacheprovider
1202 passed, 25 xfailed in 9.19s
```

All existing passing tests remain passing. The 25 strict expected failures are the
previous unresolved semantic cases; they do not count as functional successes.
Existing NextQuestion, memory-provenance, event, auth, provider-failure and deadline
regressions are included in the full run.

Ruff lint across `src/atlas_ai tests scripts`: **All checks passed!**, using the
existing Windows bind-mount executable-bit exception `--ignore EXE002`.
Ruff format on all changed Python files: **7 files already formatted**. `git diff --check`
passes. Protected integration/proto/evaluation/config files have no diff;
`grounding.py` implements the authorized policy separation.

## Scope clarification

The user clarified: retain factual guards and make plausibility/relevance language
heuristics nonblocking. That policy is implemented where separable; mixed factual
guards remain conservative under safeguard 9. See the companion policy report for
the exact boundary, retained limitations and changed historical acceptance counts.

No endpoints, public service interfaces, schemas, generated stubs, context supply,
auth, session lifecycle, database behavior, deadlines, error handling, production
model, historical ledgers, or NLP/Feedback code were changed. No commit or push.
