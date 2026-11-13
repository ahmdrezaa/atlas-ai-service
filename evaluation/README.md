# Synthetic grounding evaluation artifacts

Milestone 5: `m5_memory_mock_grpc.json` contains a successful authenticated gRPC
response with a mock provider, Solution and owner-sourced candidates. It is NOT live.
`m5_memory_live.json` records two real provider HTTP 503 failures; no live M5 output
was validated. `m5_memory_report.json` distinguishes those results and records source
provenance, call counts and latency limits. See `docs/business-memory-candidates.md`.

P0 fresh Solution + Python gRPC success is recorded in `p0_solution_live.json` (P3),
with all six acceptance stages in `p0_acceptance_report.json`. P1 structural failure
and P2 HTTP 504 remain FAILED; P1 offline revalidation is not a new live success.
`p0_c5_audit.json` records original sources and the six abstention classifications;
`p0_grounding_rules_report.json` reruns the unchanged 50-case dataset under MVP policy.
See `docs/p0-live-solution.md` for timings, request limits and production limitations.

Milestone 4C.1 alignment: `solution_alignment_c5_audit.json` replays the unmodified
C5; `grounding_alignment_rules_report.json` re-runs the same 50 cases.
`solution_alignment_live.json` records one new HTTP 504 (no valid Solution, gRPC
skipped); `solution_alignment_deadlines.json` separates caller defaults from unknown
deployed deadlines. See `docs/solution-validation-alignment.md`. Offline replay and
mocked tests are not successful live responses.

Milestone 4C results are in `functional_live_readiness.json` and its pinned Go/Python
deadline audit in `functional_live_readiness_deadlines.json`. START is live-validated;
Solution is not (structural rejection), so live gRPC was conditionally skipped. See
`docs/functional-live-readiness.md` for exact requests, configuration and limitations.

Live integration isolation after 4B is recorded separately in
`gemini_live_diagnostics.json`; see `docs/gemini-live-integration-diagnostics.md`.
Unlike the failed verifier evaluation, that run contains actual SDK and simple
LangChain structured-output successes on an explicitly selected comparator model.

- `grounding_cases.json`: 50 proposed labels and synthetic sources, created before
  the new source-verification implementation. Labels are reviewable, not human-adjudicated.
- `grounding_rules_report.json`: structural baseline and selected deterministic
  checker, per-case predictions, confusion matrices, dataset hash and CPU timings.
- `grounding_live_report.json`: first live batch failed; later cases were not attempted.
- `grounding_smoke_cases.json` / `grounding_live_smoke_report.json`: four-case smaller
  live attempt, no completed judgments.
- `grounding_live_model_check.json`: configured model metadata was available.
- `grounding_live_diagnostic.json`: whitelisted error metadata (HTTP 504), no raw
  provider message, API key or private business data.

See `docs/consultation-grounding-evaluation.md` for definitions, strategy selection,
compatibility audit, exact results and limitations. Reports distinguish unscored
provider failures from semantic mistakes. Mocked verifier tests are wiring tests,
never evidence of model accuracy. No production LLM verification toggle is enabled.
