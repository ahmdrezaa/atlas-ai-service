# M7.2 — Controlled live Gemini evaluation with LangSmith

Status: **LIVE PARTIAL / BLOCKED**, 14 November 2026 (Asia/Jakarta).

The external evaluation workflow is implemented. LangSmith contains the seven approved synthetic examples and the failed smoke run. One actual Gemini invocation returned provider HTTP 504; it produced no model output. The six remaining cases were not attempted. Consultation quality, successful structured generation, evidence acceptance, and live protobuf mapping therefore remain unverified in this experiment. No human quality scores have been supplied.

## Baseline and scope

Local `main`, `origin/main`, and GitHub `main` were checked at commit [`002b1e14e4810db0ba67909c27ce7c10cb599314`](https://github.com/ahmdrezaa/atlas-ai-service/commit/002b1e14e4810db0ba67909c27ce7c10cb599314). The workflow uses the existing `ConsultationPipeline.run()`, Gemini settings, Solution provider schema, `consultation-solution-v8`, `grounding-rules-v8`, memory safeguards, and protobuf mapping. Every fixture uses `CONSULTATION_EVENT_GENERATE_NOW`: this evaluates one Solution invocation per case, **not Assessment or multi-turn question quality**.

No production source, public contract, generated stub, provider dependency, model selection, deadline, authentication, tracing policy, Feedback processing, or historical evaluation artifact was changed. In particular, production `tracing_context(enabled=False)` remains in place. There is no judge model, repair invocation, fallback model, or automatic provider retry.

The installed Python 3.11 Docker environment supplied:

| Package | Version |
| --- | --- |
| `langsmith` | 0.14.17 |
| `langchain-google-genai` | 4.5.0 |
| `google-genai` | 3.2.0 |
| `pydantic` | 2.14.3 |

SDK signatures and current official documentation were checked. Dataset examples use supported [`create_example(reference_outputs=...)`](https://reference.langchain.com/python/langsmith/client/Client/create_example). Experiment runs are explicit [`create_run`](https://reference.langchain.com/python/langsmith/client/Client/create_run) writes associated with a reference example and a project referencing the dataset. Run feedback uses [`create_feedback`](https://reference.langchain.com/python/langsmith/client/Client/create_feedback) with `run_id` and **`session_id`**; passing `project_id` together with `run_id` is rejected by this installed SDK. A regression test exercises the actual SDK argument validation and HTTP serialization with transport mocked.

`read_run` and the optional `get_run_url` fallback use the installed SDK's still-supported synchronous compatibility API. The official [runs migration guide](https://docs.langchain.com/langsmith/smithdb-sdk-migration-runs) describes their scheduled removal on 31 January 2027. This runner will need migration to the new runs API before that removal. No obsolete feedback lookup API is used; see also the [feedback migration guide](https://docs.langchain.com/langsmith/smithdb-sdk-migration-feedback).

## Files and design

| File | Purpose |
| --- | --- |
| `scripts/consultation_m7_2_langsmith.py` | Explicit offline preparation, dataset registration, gated live capture, upload recovery, review export, and human-feedback submission |
| `tests/test_consultation_m7_2_langsmith.py` | Mocked provider/LangSmith regression coverage; no live test calls |
| `.env.example` | Evaluation-only credential and endpoint names; production tracing stays disabled |
| `evaluation/m7_2_live_evaluation.json` | Durable manifest, attempt reservations, actual capture, remote identities, and recovery history |
| `evaluation/m7_2_human_review_bundle.json` | Offline review export with actual capture and blank human scores |
| This document | Procedure, evidence, results, and limitations |

The runner reuses the existing request/context construction, production invocation, diagnostics capture, error classification, Pydantic validation, evidence guards, and protobuf mapping. Instrumentation observes the Google SDK entry and its status; it does not change the production request or add an invocation. A prompt hash is checked before SDK invocation. Deterministic offline revalidation is recorded separately from runtime stages: an offline check cannot turn a timed-out pipeline into a successful generation.

## Corpus identity and dataset shape

Both frozen M7.1 JSON files are checked by SHA-256 before any setup or execution:

| File | SHA-256 |
| --- | --- |
| `m7_1_consultation_quality.json` | `e72e98dd39de03d6a7a87c0bf066673d152febb00305d176d2232afa0119f098` |
| `m7_1_quality_review_bundle.json` | `39df1574d7547de8a8b541066e345bb47f3fab44a23763298c6a2023a2982bc3` |

The actual stored fixture identities differ from four labels in the milestone request. The runner preserves original identities and records these aliases rather than renaming fixtures:

| Requested label | Preserved fixture ID |
| --- | --- |
| `bakery_filling` | `bakery_filling` |
| `bakery_quality` | `bakery_quality` |
| `cafe_queues` | `cafe_queue` |
| `cafe_takeaway` | `cafe_packaging` |
| `pricing` | `msme_pricing` |
| `order_mistakes` | `msme_orders` |
| `owner_constraints` | `owner_constraints` |

Each example has `inputs = {scenario_id, request}`. `request` is the unchanged official `ProcessConsultationTurnRequest` JSON representation, including customer evidence, OWNER messages, confirmed memories, statistics, and historical AI material. It is parsed with the official generated protobuf type. The two fixtures must agree on the request, the currently rendered prompt messages must equal the frozen messages, and the provider schema must match the frozen schema.

`reference_outputs` contains factual boundaries, the expected forbidden factual claim, the original rubric, and reviewer guidance. **Predefined M7.1 Solutions are never supplied as golden answers or model inputs.** Multiple recommendations can be valid. Example metadata contains fixture hash, scenario ID, and prompt version. Dataset metadata includes `synthetic_only`, manifest hash, and prompt version.

Example IDs are stable UUID5 values derived from dataset name and fixture ID. Dataset registration first verifies all existing examples, rejecting changed/extra data before adding missing examples. It never overwrites examples. Partial creation can be resumed. Once recorded, remote dataset/project IDs must stay unchanged; a different workspace's lookalike dataset is rejected. Run IDs are stable UUID5 values within the local experiment UUID.

## Configuration and privacy boundary

Required for live execution: `GEMINI_API_KEY`, `GEMINI_MODEL`, and `LANGSMITH_API_KEY`. Remote registration/upload/review only require the LangSmith key. Optional evaluation configuration: `LANGSMITH_ENDPOINT` (default `https://api.smith.langchain.com`) and `LANGSMITH_WORKSPACE_ID` (UUID when needed by the account). Endpoint validation requires HTTPS and rejects embedded credentials, query parameters, or fragments.

Production settings are reused: `GEMINI_TIMEOUT_SECONDS`, `GEMINI_MAX_OUTPUT_TOKENS`, and `CONSULTATION_MAX_CONTEXT_BYTES`. In this actual run they resolved to 30 seconds shared turn budget, 2,048 output tokens, and 120,000 prompt bytes. The configured model was **`gemini-3.8-flash`** with no override. Gemini API version was `v1beta`; temperature was 0 and thinking configuration remained provider default.

The optional `--env-file .env` loader reads only these named Gemini/LangSmith settings, preserves already-set process variables, and ignores authentication tokens and tracing toggles. It never writes `.env`. Only key-presence booleans are recorded. Never paste keys into commands, review reasons, prompts, or reports. Never commit a real `.env` or upload its contents. The checked-in example contains names and empty keys only. Existing `compose.yaml` does not forward evaluation credentials into the production container.

LangSmith writes are manual, restricted to the frozen synthetic inputs, captured outputs, safe metadata, and reviewer feedback. Automatic tracing remains disabled inside the runner. Runtime/environment metadata upload is disabled, SDK logging is suppressed within the isolated CLI boundary, and known API-key values are redacted before persistence/upload. Error telemetry contains fixed categories, exception class names and safe status codes, never raw exception bodies, headers, or provider logs. Transport retries are disabled; feedback writes use one attempt. There are no real customer or OWNER records in this experiment.

## Exact CLI procedure

Run from the repository root with Python 3.11 and the repository dependencies installed. In PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m scripts.consultation_m7_2_langsmith prepare --ledger evaluation/m7_2_live_evaluation.json --env-file .env
python -m scripts.consultation_m7_2_langsmith dataset --ledger evaluation/m7_2_live_evaluation.json --env-file .env --remote-setup
python -m scripts.consultation_m7_2_langsmith live --ledger evaluation/m7_2_live_evaluation.json --env-file .env --phase smoke --live
```

These commands describe initial creation; the checked-in ledger already exists, and `prepare` refuses to overwrite it. `prepare` and `review` have zero Gemini and LangSmith network calls. Remote operations require their explicit flags. Missing configuration fails before client/model construction; values are not printed. Preparation records missing names as presence booleans without requiring credentials for offline use.

Only a fully accepted, uploaded smoke result opens the remaining phase:

```powershell
python -m scripts.consultation_m7_2_langsmith live --ledger evaluation/m7_2_live_evaluation.json --env-file .env --phase remaining --live
```

**Do not run the remaining phase for this captured experiment: its smoke failed.** The persisted stop reason prevents further live execution. Provider authentication failure, 429, request/model/schema rejection, timeout, transport/infrastructure failure, and a failed smoke stop the experiment. No fallback or deadline extension occurs. Subsequent non-smoke model/validation failures are captured for review without a repair call.

Maximum budget: **seven total Gemini provider invocations**, including one smoke. Cases run sequentially, once each. A single-writer lock and an atomic local write-ahead reservation persist before entering the SDK. Attempt count is persisted immediately before the actual provider entry. Reserved or interrupted cases are never automatically regenerated; uncertain interrupted reservations conservatively consume a slot. Do not delete reservations or clear the stop reason to improve results. Another provider experiment requires a separate explicitly authorized decision.

If a remote upload fails after capture, recover only the upload:

```powershell
python -m scripts.consultation_m7_2_langsmith upload --ledger evaluation/m7_2_live_evaluation.json --env-file .env --remote-upload
python -m scripts.consultation_m7_2_langsmith review --ledger evaluation/m7_2_live_evaluation.json --output evaluation/m7_2_human_review_bundle.json
```

Upload checks stable run identity and server-returned input/output/example/project/metadata before creating feedback. Stable feedback IDs and read-before-create allow recovery after acknowledgement loss. A remote content mismatch stops recovery rather than overwriting a run. Review export revalidates captured payloads offline, protects unrelated files and annotated reviews from overwrite, and never invokes Gemini. No subjective scores are inferred from diagnostic labels.

For Docker, the actual evaluation used image `atlas-ai-service:dev`, a read-only workspace mount, and a writable mount for evaluation artifacts. This equivalent PowerShell setup makes the boundary explicit:

```powershell
$evalDocker = @('run', '--rm', '--mount', 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly', '--mount', 'type=bind,source=D:\atlas-ai-service\evaluation,target=/results', '--workdir', '/workspace', '--env', 'PYTHONPATH=/workspace/src', '--env', 'PYTHONDONTWRITEBYTECODE=1', 'atlas-ai-service:dev')
docker @evalDocker python -m scripts.consultation_m7_2_langsmith upload --ledger /results/m7_2_live_evaluation.json --env-file /workspace/.env --remote-upload
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --mount 'type=bind,source=D:\atlas-ai-service\evaluation,target=/results' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m scripts.consultation_m7_2_langsmith review --ledger /results/m7_2_live_evaluation.json --output /results/m7_2_human_review_bundle.json
```

## Actual LangSmith identities and results

- [Dataset: atlas-consultation-m7-1-synthetic-v1](https://smith.langchain.com/o/deb40c4b-0a07-4f51-85b0-8c77adf9efc6/datasets/8230dcdd-81c1-45f9-a9da-7f9a0ff3f5cf), ID `8230dcdd-81c1-45f9-a9da-7f9a0ff3f5cf`: seven examples verified by read-back.
- [Experiment: atlas-m7-2-v8-77ac2690-a1b](https://smith.langchain.com/o/deb40c4b-0a07-4f51-85b0-8c77adf9efc6/projects/p/d687db14-9d73-45ec-b961-f5588076e20d), project ID `d687db14-9d73-45ec-b961-f5588076e20d`.
- [Actual bakery_filling smoke run](https://smith.langchain.com/o/deb40c4b-0a07-4f51-85b0-8c77adf9efc6/projects/p/d687db14-9d73-45ec-b961-f5588076e20d/r/a32cf5b4-e744-5638-be85-0cf5adad4a41?trace_id=a32cf5b4-e744-5638-be85-0cf5adad4a41&start_time=2026-11-14T01:58:32.603570), ID `a32cf5b4-e744-5638-be85-0cf5adad4a41`.

| Scenario | Actual provider attempts | Provider / pipeline result | Validation / review |
| --- | --- | --- | --- |
| `bakery_filling` | 1 | `gemini-3.8-flash`, HTTP 504, 28.538760 s | No output; Pydantic, structural, grounding, protobuf not reached; uploaded; human scores blank |
| `bakery_quality` | 0 | Not attempted: smoke gate closed | Not evaluated |
| `cafe_queue` | 0 | Not attempted: smoke gate closed | Not evaluated |
| `cafe_packaging` | 0 | Not attempted: smoke gate closed | Not evaluated |
| `msme_pricing` | 0 | Not attempted: smoke gate closed | Not evaluated |
| `msme_orders` | 0 | Not attempted: smoke gate closed | Not evaluated |
| `owner_constraints` | 0 | Not attempted: smoke gate closed | Not evaluated |

The smoke began at **08:58:32 WIB on 14 November 2026**. The SDK recorded provider HTTP **504**, with SDK elapsed time **28.475 s** and whole-pipeline latency **28.538760 s**. Effective SDK timeout was **29.749539 s**, derived by the existing pipeline from its 30-second shared deadline. The error chain was `ConsultationFailure → GoogleAPIError(504) → ServerError(504)`; the application mapped it to `DEADLINE_EXCEEDED`. This is an application status equivalent, **not an observed live gRPC wire response**. This milestone invoked the real production pipeline directly; live Python gRPC was not exercised.

Actual provider prompt text was **10,571 UTF-8 bytes**; provider JSON schema was **2,685 bytes**. Prompt hash matched the frozen expected hash `d0d6d63fe0a97977c0f8a9b5936b23b079dcd2bfff1de4c2c5dd72c5686f4ed9`. No model text, parsed payload, returned model identifier, finish reason, or token usage was available. These fields and validation stages remain null, not successful or rejected.

Provider HTTP 504 is proven by the response hook and SDK status. It is distinct from a client-only timeout, rate limit, parsing failure, Pydantic rejection, grounding failure, or mapping failure. Authentication and schema compatibility were not independently proven by this failure. The underlying provider cause remains unknown; no claim is made about quota exhaustion, output-token budget, thinking, or grounding causing the 504.

The initial LangSmith post-create verification failed with a local `EvaluationError`; the initial differing field was not captured. A later read-back matched all compared fields, so its exact original cause is unresolved. A subsequent upload-only recovery reached feedback creation and exposed the SDK's `run_id`/`project_id` argument conflict. Changing the evaluation helper to `session_id` fixed that proven SDK contract error. A further upload-only recovery completed. **None of these recoveries invoked Gemini.** Both workflow errors remain in the local ledger rather than being erased.

Server read-back verified the uploaded run and two code-feedback values: `atlas.provider_response = 0` and `atlas.fully_accepted = 0`. There are no Pydantic/grounding/protobuf scores for stages not reached. These two booleans are operational results, not semantic quality judgments. Human feedback has not been submitted.

Descriptive metrics: **1/7 planned cases attempted**, **0/1 model responses**, **0/1 accepted Solutions**, **1/1 captured attempts uploaded**, **6 not attempted**. Observed latency median/minimum/maximum are all 28.538760 s because the sample contains only one failed attempt. Quality-pass rate is unavailable, not 0/7: no reviewable output exists. This is not a model-quality estimate, performance benchmark, statistically significant comparison, or v7-versus-v8 experiment.

## Human review

The export preserves the exact M7.1 rubric IDs and criteria. Scores are integers 0–2; reviewer identity and a reason for every supplied score are required. The eight dimensions are:

| Dimension ID | Review focus | Critical |
| --- | --- | --- |
| `factual_accuracy_attribution` | Findings entailed by actual cited current sources; attribution and sample scope | Yes |
| `hypothesis_uncertainty` | Explicitly tentative causes without hidden factual premises | Yes |
| `recommendation_relevance` | Actions substantively address the issue | No |
| `operational_usefulness` | Feasible actions and observable measures | No |
| `constraint_awareness` | Supplied staffing, budget, process and capacity limits | No |
| `indonesian_clarity` | Clear, concrete Indonesian | No |
| `reference_faithfulness` | Correct namespaces, relevant sources, exact quotes, no historical-AI laundering | Yes |
| `generation_validation_success` | Actual five-section output accepted within deadline without repair | No |

A quality pass requires all eight scores, total **at least 14/16**, **no zero**, **every critical score 2**, actual model output, and successful production validation/protobuf mapping. Diagnostic keyword or relevance labels do not generate human scores or establish factual failure. Passing guards also does not establish business usefulness.

Reviewers should inspect the exact synthetic request, actual captured output, runtime validation, offline diagnostics, and factual boundaries; then enter `reviewer`, `scores`, and `reasons` in a copy of the bundle. Do not score unattempted cases. Without model output only a generation-failure score of 0 can be entered; subjective dimensions stay blank. A failed generation cannot receive a positive generation score or a quality pass. Current exports intentionally have no fabricated reviews.

To submit an authorized human review without rerunning a provider call:

```powershell
python -m scripts.consultation_m7_2_langsmith submit-review --ledger evaluation/m7_2_live_evaluation.json --env-file .env --review-file evaluation/m7_2_human_review_bundle.json --remote-feedback
```

Reviewer decisions are persisted locally before feedback upload. Feedback has stable IDs for safe retry of the same review; different content at the same ID is rejected rather than silently overwriting history. Complete failed reviews are reported as `HUMAN_QUALITY_REJECTION`, independently of provider/validation failures. Partial reviews remain pending.

## Verification

Offline mocked tests ran before inspecting live credential availability or making remote calls. After the SDK feedback fix and final identity guard, the full suite ran again in Docker with network disabled:

```powershell
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 atlas-ai-service:dev python -m pytest -q -p no:cacheprovider
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace atlas-ai-service:dev python -m ruff check --no-cache --config pyproject.toml --ignore EXE002 src/atlas_ai tests scripts
docker run --rm --network none --mount 'type=bind,source=D:\atlas-ai-service,target=/workspace,readonly' --workdir /workspace atlas-ai-service:dev python -m ruff format --check --no-cache scripts/consultation_m7_2_langsmith.py tests/test_consultation_m7_2_langsmith.py
```

Actual final outputs: **`1313 passed, 25 xfailed in 14.59s`**, **`All checks passed!`**, and **`2 files already formatted`**. The new file contributes 49 parameterized test cases. Existing gRPC mock integration/regression tests are included in the full suite. Historical expected failures were retained. `EXE002` is excluded for the Windows executable-bit check, consistently with repository verification practice; no production lint rule/configuration was edited.

Coverage includes frozen identities, privacy-preserving dataset serialization, idempotent/partial setup, no-network offline commands, explicit authorization, missing/invalid configuration, single smoke and seven-call cap, authentication/429 stop gates, interruption/concurrency/resume protection, immutable experiment settings, safe errors, actual SDK feedback serialization, successful and rejected captures, nonblocking diagnostics, upload/acknowledgement recovery without generation, and human-score gating.

## Remaining limitations

The immediate blocker is provider HTTP 504 with the configured model and unchanged deadline. Successful provider-generated output, structured compatibility, evidence acceptance, protobuf mapping, and semantic consultation quality have not been demonstrated in M7.2. Mocked tests do not replace that evidence. Six fixtures remain deliberately unattempted.

The runner supports human review but cannot manufacture one. It does not evaluate Assessment, production throughput, multi-turn behavior, caller-Go transport deadlines, or real customer outcomes. SDK migration is required before the documented compatibility API removal. LangSmith network operations can still fail; captures and immutable identities permit upload recovery. Unknown interrupted reservations require manual audit, not automatic re-execution. Production readiness is not established by dataset access, this failed smoke, or offline regression success.
