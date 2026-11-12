# Synthetic grounding evaluation artifacts

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
