# Milestone 4C — Functional Live Readiness

Follow-up: [Milestone 4C.1 validation alignment](solution-validation-alignment.md)
fixes the numeric false rejection and adds a final Go-reference guard. Its new live
attempt returned HTTP 504; the original runs below are preserved unchanged.

**Result: partial, not full functional readiness.** START produced a live, valid
NextQuestion. GENERATE_NOW eventually produced a typed five-section response from
Gemini, but it failed local structural validation. Consequently no live Solution
was returned and the conditional live gRPC test was skipped. Mocked regression tests
are not substituted for those missing live successes.

Production model selection remains `gemini-3.8-flash`. Every live run in this milestone
explicitly overrides it with `gemini-3.6-flash`, v1beta and thinking LOW, in the test
runner only. No API key or real customer data was printed or sent as prompt content.

## End-to-end deadline audit

The Go source was read, not assumed from the PRD, at commit
`002472705b76f1f86e0b6628fc07bd6a92fe478d`:

- [aiclient/client.go](https://github.com/Elioth-Gray/atlas-backend-core/blob/002472705b76f1f86e0b6628fc07bd6a92fe478d/internal/aiclient/client.go#L53)
  calls context.WithTimeout, so an earlier parent deadline still wins.
- [config/config.go](https://github.com/Elioth-Gray/atlas-backend-core/blob/002472705b76f1f86e0b6628fc07bd6a92fe478d/internal/config/config.go#L155)
  defaults AI_CONSULTATION_TIMEOUT to 60s and HTTP_WRITE_TIMEOUT to 90s. Validation
  requires AI timeout + 15s consultation overhead to fit inside HTTP write timeout.

These are repository defaults, **not verified deployment environment values**. An
HTTP proxy, browser or middleware may impose an earlier deadline. See the pinned
source metadata in `evaluation/functional_live_readiness_deadlines.json`.

Python's shared budget B is min(GEMINI_TIMEOUT_SECONDS, incoming gRPC time_remaining).
The timer starts before preparation. The existing default remains 30s, maximum 55s.

| Path | Model calls | Effective accounting |
| --- | ---: | --- |
| Assessment -> NextQuestion | 1 | Prepare + assessment + question validation/mapping must fit B |
| Assessment -> GENERATE_SOLUTION | 2 | Generation gets only B minus time already spent, not a fresh B |
| GENERATE_NOW / cumulative cap -> Solution | 1 | Assessment bypassed; generation and all validators share B |

The shared helper sets LangChain timeout to min(configured timeout, remaining minus
250ms), subtracts prompt preparation, and the SDK receives milliseconds. Retry count
is zero. Python gRPC has no independent larger per-request deadline: it uses the
caller's context and the internal budget. Synchronous SDK cancellation may wait for
the HTTP timeout; the request still fails rather than returning a late success.

One audit fix adds a final budget/cancellation check after question validation and
response mapping. Previously the question validator could consume extra CPU after
the helper's final deadline check. This does not change the budget or question rules.
The solution path likewise checks the final response boundary. Deadline regression
tests simulate the question check exhausting the budget without any live request.

The live test budget is **55s only as an override**. Direct calls supplied
time_remaining=60; the planned actual gRPC client timeout was 60s, but that conditional
live gRPC test was not run because the Solution prerequisite failed.
Production timeout/configuration is unchanged. Increasing production limits would
require checking actual Go and upstream settings, not merely changing Python.
Two-call turns can still fail if assessment consumes almost the entire shared budget.

## Single-question analysis

The previous diagnostic assessment listed peak hours and staffing as separate missing
facts, then combined both in one sentence. One question mark did not make it one
information request; the unchanged guard correctly rejected `dan berapa`.

Assessment prompt v3 now asks the model to rank gaps and select one information
variable/topic, with a focused staffing example and an explicit ban on bundled topic
keys. The synthetic START fixture supplies busy hours, budget constraints, observation
window and objective; only staffing is missing. Prompt clarification and fixture
completeness both changed, so one success does not prove the prompt alone caused an
improvement or guarantees future single questions. No correction invocation or repair
retry was added. Guard code is unchanged.

Live C1 returned:

> Berapa staf yang melayani kasir pada jam sibuk tersebut?

topic_key is `cashier_staff_count`. It passed typed assessment, deterministic question
validation and protobuf NextQuestion mapping. The final boundary budget guard was
added after that live run and verified offline; it does not change the question or
business-validation policy. No second live START was run just to repeat the sample.

## Live ledger: five generation requests, zero retries/corrections

| ID | Case / schema variant | HTTP | Seconds | Actual prompt text / native schema bytes | Result |
| --- | --- | ---: | ---: | --- | --- |
| C1 | START / assessment v3 | 200 | 2.130 | 5,913 / 1,446 | Valid NextQuestion |
| C2 | GENERATE_NOW / full Pydantic schema | 400 | 1.460 | 7,812 / 3,359 | Provider rejected arguments before output |
| C3 | One instrumented reproduction | 400 | 1.692 | 7,812 / 3,359 | Sanitized error: Request contains an invalid argument |
| C4 | Nullable-label provider representation | 400 | 1.534 | 7,959 / 3,413 | Enum change alone did not fix request |
| C5 | Compact native provider schema | 200 | 7.430 | 7,959 / 2,409 | Typed five-section output; local structural rejection |

All calls used 55s shared test budget, approximately 54.75s effective SDK timeout,
thinking LOW, and the same **2,048 output token** budget. Prompt bytes count actual
system/human text in the SDK request; JSON schema bytes are separate. These are not
token estimates. Full telemetry and synthetic inputs/outputs are in
`evaluation/functional_live_readiness.json`.

C1 used 1,708 input + 207 output = 1,915 reported tokens, provider duration 2.060s.
C5 used 2,229 input + 1,115 output = 3,344 reported tokens, provider duration 7.353s.
Thinking token counts were not reported; this is not evidence that thinking was off.
Failed HTTP 400 calls returned no token usage. C5 finished STOP, not MAX_TOKENS.
Thus the observations do not justify blaming the 2,048 budget for earlier HTTP 504s.
Earlier 503/504 findings remain separate from the newly reproduced schema rejection.

## Provider schema compatibility versus local validation

The full generated schema used an empty-string enum alternative for unspecified
confidence/priority. Google documents nonempty string enum elements in its structured
output subset ([reference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/control-generated-output)).
A provider null -> existing protobuf empty-label adapter avoids that representation,
while local literals still reject numerical probabilities/priorities. However C4
still returned 400, so empty enums are **not proven to be the sole cause**.

C5 keeps the same prompt text as C4 and uses a smaller native schema: inline references,
required objects/properties, string/array types, nonempty qualitative enums and null.
Annotations, defaults and grammar bounds are omitted from the provider request.
Exact rejecting keywords cannot be identified from Google's generic 400 response.
The changed representation was accepted, but no claim is made that one specific
keyword caused the rejection. The API supports a subset of JSON Schema:
[Google structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output).

Crucially the original GeneratedSolution Pydantic model is **always** applied after
the native JSON parser, including min/max array counts, text lengths, extra-field
rejection and label restrictions. Structural source/quote/numeric checks and semantic
grounding are unchanged and still mandatory. No fake solution, partial protobuf,
source substitution or disabled validator is used. The solution prompt version is
v3 to document the provider null representation. Assessment remains separate v3.

## Actual Solution validation outcome

C5 parsed through GeneratedSolution with findings=3, customer_evidence=2, hypotheses=1,
recommendations=1 and success_indicators=1. That proves only schema/shape validity.
The validator rejected the recommendation description:

> Lakukan pemantauan dan pencatatan manual waktu pelayanan transaksi kasir pada jam ramai 17.00 sampai 19.00 untuk mengidentifikasi titik hambatan tanpa melakukan pembelian alat baru.

Those hours do occur in the synthetic confirmed memory. The rejection is the existing
numeric-prose policy: a proposed numerical recommendation must start `Usulan:`, or
the complete numerical statement must be an exact cited excerpt. This description
meets neither condition. It is not evidence that the hours were hallucinated.

The first failure occurs in structural validation, before the quote-loop completion
and semantic grounding stage. Grounding status is therefore **NOT_REACHED**, not
passed or failed on semantic entailment. No offline edited/corrected response is
counted as live success. There may be further wording/grounding rejections after
this one is resolved, particularly under the conservative 4B language checker.

Because no direct Solution passed, the conditional gRPC live request is **SKIPPED**.
The runner enforces this before reading credentials or sending a request. The
existing mocked/in-process gRPC suite still passes but does not fill that live gap.

## Minimal changes / configuration separation

- `prompts.py`: assessment v3 single-gap instructions; solution v3 null-label wording.
- `solution.py`: provider-only schema adapter; original local bounds/grounding kept.
- `assessment.py`: optional native schema parameter, then mandatory Pydantic validation.
- `pipeline.py`: final shared deadline/cancellation guard without budget increases.
- Diagnostic scripts: synthetic fixture, actual request byte/latency capture,
  redacted HTTP 400 details, post-hoc inspection of saved real output and gRPC gate.
- Tests cover schema adaptation without weakening local limits, single-gap fixtures,
  probability rejection, redaction, final-budget handling and conditional gRPC blocking.

No changes to ai.proto, NLP Ucan, authentication, Go lifecycle, production model or
production timeout. No LLM semantic verifier, Business Memory Extraction or agent tools.

### Demo recommendation (not production configuration)

For a limited NextQuestion demonstration, use the runner's explicit 3.6/v1beta/LOW
override, 2,048 tokens and 55s shared budget only when the actual caller allows 60s
and upstream HTTP allows AI timeout + Go overhead. This is an opt-in test factory,
not an automatically applied production fallback or an unsupported thinking env var.
Show real failure states. A full live Solution demo is **not yet validated**.

Production remains 3.8 and the configured/default 30s shared budget. One fast sample
does not establish p95 reliability. Before changing production, verify deployment
caller deadlines, provider availability and the interaction between generated wording
and structural/grounding policies. Future work should evaluate prompt compliance with
numeric proposal formatting and bounded grounding without relaxing guards just to
pass a demo. No mandatory correction/verification invocation was introduced.

## Reproduction

With existing project dependencies installed and PYTHONPATH=src, the runner is invoked
as a module. It uses synthetic data only and refuses duplicate run IDs. Its six-run
cap includes a reserved conditional gRPC test; five were used and no sixth was sent.

```powershell
$env:PYTHONPATH = "src"
python -m scripts.consultation_live_readiness --case start --id unique-start --output evaluation/new-4c-run.json --env-file .env
python -m scripts.consultation_live_readiness --case solution --id unique-solution --output evaluation/new-4c-run.json --env-file .env
# Only permitted after both direct paths passed:
python -m scripts.consultation_live_readiness --case grpc --id unique-grpc --output evaluation/new-4c-run.json --env-file .env
```

Recorded runs use the existing Python 3.11 Docker image with read-only source and
only the evaluation artifact mount writable. No live calls occur in regression tests.

Final verification: **548 passed in 2.57s**; Ruff lint and formatting checks pass.
Those tests include all previous event/question/solution regressions and do not
replace the missing successful live Solution or live gRPC round trip.
