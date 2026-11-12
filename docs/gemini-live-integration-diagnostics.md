# ATLAS Gemini Live Integration Diagnostics

Date: 12 November 2026 (Asia/Jakarta). All content is synthetic. Production selection
remains `GEMINI_MODEL=gemini-3.8-flash`; no production settings or application rules
were changed. No API key, request headers, raw exception messages or URL queries
are included in the diagnostic log.

## Findings

**At least one real invocation is proven successful**: direct Google SDK plain-text
returned `ATLAS_OK` (HTTP 200), and LangChain native JSON schema returned
`DiagnosticReply(status="ATLAS_OK")` (HTTP 200). Both successes used the explicitly
selected diagnostic comparator `gemini-3.6-flash`, not an automatic production fallback.

The failures are not merely local gRPC/client timeouts. HTTP response hooks observed
actual 503/504 responses at `generativelanguage.googleapis.com`, and the SDK raised
ServerError. These occurred with an 11-token plain-text prompt, without LangChain,
Pydantic, ATLAS prompts, source registries, grounding or large evaluation batches.
Therefore those layers are not required for the observed failures to occur.

This isolates the failure to the generation endpoint's server-error boundary. It does
not reveal Google's internal root cause. Model serving load, routing/capacity or other
upstream failures remain possibilities, not proven diagnoses. Google documents 503
as temporary unavailability/possible overload and 504 as deadline exceeded:
[GenerateContent API errors](https://ai.google.dev/gemini-api/docs/generate-content/api-errors).
Successful generation on 3.6 also proves the tested credential/environment can perform
generation; it does not establish every model's access, availability, quota or SLA.

## Audited configuration

| Item | Actual observation |
| --- | --- |
| Production model | `gemini-3.8-flash`, unchanged in local `.env` |
| Python runtime | Existing Python 3.11 Docker image |
| `langchain-google-genai` | 4.5.0 |
| `google-genai` | 3.2.0 |
| `langchain-core` | 1.8.0 |
| Pydantic / HTTPX | 2.14.3 / 0.28.1 |
| SDK default API | `v1`, verified in client options and actual request path |
| LangChain default API | `v1beta`, selected by the installed integration's `_default_api_version` |
| Backend | Gemini Developer API (`vertexai=False`) |
| ATLAS timeout | Default 30s, configurable up to 55s; shared turn budget; 250ms response margin |
| Diagnostic timeout | 30s for every request; ATLAS stage receives approximately 29.75s |
| Production output budget | 2,048 tokens; earlier 4B verifier evaluation explicitly used 8,192 |
| Diagnostic output budget | 2,048 for every request, unchanged between stages |
| Production thinking | Not explicitly configured; provider chooses its default |
| Diagnostic thinking | Provider default in D1/D2; explicit `LOW` in D3–D7 |
| Sampling | ATLAS passes temperature=0; installed integration drops it for fixed-sampling 3.6, observed `effective_temperature=null` |
| Retry / tracing / cache | Zero automatic retries; LangSmith tracing and model cache disabled |

The SDK API default changed to v1 in google-genai 3.x; LangChain 4.5.0 explicitly
selects v1beta for its Developer API default. This was audited in the installed code,
not inferred from metadata availability. Both API versions produced server errors on
3.8, while v1beta worked on 3.6, so an API-version mismatch alone is not a demonstrated
fix. No dependency downgrade/upgrade or API-version production change is justified
by these observations.

Google documents configurable thinking levels and warns that a small output budget
can truncate thinking/answers; lowering thinking is a latency control, not an outage
fix: [Thinking documentation](https://ai.google.dev/gemini-api/docs/thinking).
The 3.8 low-thinking plain-text control still returned 504. Thus default thinking or
large output budgets cannot alone explain all failures. No successful response here
ended with MAX_TOKENS; failed responses did not return usage/finish metadata. Sampling
warnings are not generation failures: D5 succeeded after the integration dropped the
unsupported explicit temperature. [LangChain parameter reference](https://reference.langchain.com/python/langchain-google-genai/chat_models/ChatGoogleGenerativeAI).

## Actual invocation ledger

Every row is exactly one generation request; no automatic retries occurred. All
HTTP statuses below were recorded by response hooks, not guessed from gRPC codes.
gRPC columns are equivalent codes/current application exception mappings: these
diagnostics call the SDK/application components directly, not a live Go gRPC client.

| ID | Stage | Model | API / thinking | Duration | HTTP | Result / gRPC equivalent | Tokens: input / output / thoughts / total |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| D1 | SDK plain-text | 3.8 Flash | v1 / default | 8.588s | 503 | Provider error / UNAVAILABLE | not returned |
| D2 | SDK plain-text comparator | 3.7 Flash | v1 / default | 28.476s | 504 | Provider deadline / DEADLINE_EXCEEDED | not returned |
| D3 | SDK plain-text control | 3.8 Flash | v1beta / low | 28.506s | 504 | Provider deadline / DEADLINE_EXCEEDED | not returned |
| D4 | SDK plain-text comparator | 3.6 Flash | v1beta / low | 7.302s | 200 | Exact ATLAS_OK / OK | 11 / 4 / 85 / 100 |
| D5 | LangChain + simple Pydantic | 3.6 Flash | v1beta / low | 21.554s | 200 | Schema parsed, ATLAS_OK / OK | 11 / 8 / unreported / 19 |
| D6 | Real ATLAS assessment | 3.6 Flash | v1beta / low | 11.938s | 200 | Typed schema valid; question guard rejects / INTERNAL | 1,114 / 192 / unreported / 1,306 |
| D7 | Isolated ATLAS solution generator | 3.6 Flash | v1beta / low | 29.246s | 504 | Provider deadline before parsing / DEADLINE_EXCEEDED | not returned |

Token counts are from actual SDK usage_metadata. Missing counts mean **unknown**,
not zero; no cost or billing total is inferred. Durations are single end-to-end
samples, not controlled performance benchmarks. D4 versus D5 does not prove a fixed
LangChain latency penalty: provider conditions, schemas and prompts differ.

The complete ledger includes effective API settings, provider model version, finish
reasons, safe exception chains and synthetic visible responses:
`evaluation/gemini_live_diagnostics.json`. Hidden thinking text is not recorded.

### Assessment failure is separate from provider/parsing failures

D6 returned an ASK_QUESTION assessment that passed SufficiencyAssessment. The actual
synthetic question was:

> Apakah antrean panjang ini terutama terjadi pada jam-jam tertentu, dan berapa jumlah staf yang melayani kasir saat itu?

This combines peak hours and staffing. The existing deterministic question validator
correctly rejected the compound `dan berapa` construction. No rule was relaxed and
no invalid NextQuestion was returned. The saved live output was revalidated offline
to confirm schema success and the separate question-guard failure.

Because the assessment invocation/schema succeeded, D7 isolated the solution generator
with synthetic GENERATE_NOW context. It did not bypass the failed question into an
accepted consultation response. D7 received 504 before there was any solution output,
so schema parsing, structural grounding and semantic checks could not be evaluated.

## Error separation and remaining uncertainty

- **Provider HTTP 503/504:** observed directly; SDK ServerError/GoogleAPIError,
  with HTTP status recorded. This reproduces the error boundary of earlier 4B failures.
- **Client timeout:** HTTPX/TimeoutError without a server response; distinct classifier
  and tests exist, but no such failure was observed in these seven live calls.
- **Structured parsing:** Pydantic/OutputParserException is a separate diagnostic
  category; no parsing error was observed live. D5 and D6 demonstrate valid parsing.
- **Rate limit:** HTTP 429 is classified separately; none was observed. This does not
  establish the absence of all internal provider quota/capacity issues.
- **Application validation:** D6 is a real example after HTTP 200, not a provider error.
  The diagnostics preserve the existing application's gRPC mapping if it supplies one.
- **Local application deadline:** separately classified; not equated with a provider
  504 just because both may map to DEADLINE_EXCEEDED.

No evidence shows API key invalidity as the common cause, a required SDK upgrade,
parsing as the cause of the 503/504 failures, or a grounding bug in D7. Actual internal
Google capacity/routing state cannot be proven from these responses. Model metadata
success is not used as proof of successful generation or production readiness.

## Minimal changes and reproduction

Only an opt-in diagnostic script, its unit tests, this report and the JSON ledger
were added. Production model selection, dependencies, timeouts, prompts, grounding,
protobuf, NLP and Go session lifecycle remain unchanged. The diagnostic comparator
is passed with `--model`, never written back to `.env`.

Install existing project/dev dependencies, set PYTHONPATH=src and run a single stage:

```powershell
$env:PYTHONPATH = "src"
python scripts/gemini_live_diagnostics.py --stage sdk_plain --id diagnostic-unique-id --env-file .env --output evaluation/gemini_live_diagnostics.json
# Explicit comparator/structured stage; still does not change production selection:
python scripts/gemini_live_diagnostics.py --stage lc_simple --model gemini-3.6-flash --api-version v1beta --thinking low --id another-unique-id --env-file .env --output evaluation/gemini_live_diagnostics.json
```

The runner refuses duplicate IDs and limits a ledger to eight requests. Seven were
used; no large evaluation was run and no additional live requests are needed to
support the findings above. Existing ledger IDs D1–D7 prevent accidental duplicate
execution. For a separate later diagnostic run, choose a new report path explicitly.
All invocations here used the existing Python 3.11 Docker image and read-only source
mount, with only the evaluation directory writable for artifacts.

The conditional assessment/solution gates were respected: plain-text first, then
simple structured output, then actual assessment; solution isolated only after the
assessment invocation/schema had succeeded. The new script tests distinguish transport,
rate-limit, parser and application errors and check that URL keys/headers never enter
the ledger. They do not pretend mocked tests prove live provider reliability.

Final offline regression run in the Python 3.11 container: **539 passed in 2.88s**.
Ruff lint and formatting checks pass for the diagnostic script, tests and existing
consultation modules. No additional live requests were made during regression tests.

## Not established / next decisions

There is no successful live end-to-end consultation solution or Go integration result.
The production 3.8 model still failed both minimal controls. 3.6 succeeded on small
requests but its solution call also failed; it is not justified to call it a reliable
production fallback based on this sample. Production model changes need an explicit
decision, not silent substitution. Provider incident/account diagnostics and later
small, bounded rechecks may be needed. Assessment also needs evidence that single
questions are consistently generated; this report does not change that policy.
