# Guided Consultation Solution Generation — Milestone 4A

For current numeric/quote validation, solution prompt v4 and Go reference alignment,
see [Milestone 4C.1](solution-validation-alignment.md). This document describes the
historical 4A behavior; its empty SQL-reference allowance is now blocked at mapping.

Historical milestone description. Milestone 4B adds bounded source verification and
updates solution prompt metadata to `consultation-solution-v2`; see
[grounding audit and evaluation](consultation-grounding-evaluation.md).

The production pipeline now returns a protobuf Solution after GENERATE_SOLUTION,
GENERATE_NOW or the cumulative four-question cap. Eligible question turns retain
the assessment/NextQuestion path. All Go session state remains read-only in Python.
Memory extraction is not implemented: memory_candidates is empty.

Review `consultation/prompts.py`: the shared base prompt has no assessment-only
prohibition on solutions; step-specific prompts constrain each operation. Assessment
version is now `consultation-assessment-v2`, solution version is
`consultation-solution-v1`. Historical AI output remains data rather than new fact.

`GeneratedSolution` in `consultation/solution.py` strengthens the existing historical
component models instead of changing their permissive input semantics. The generated
schema has exactly the five protobuf Solution fields. All arrays are required and
nonempty, required strings must be nonblank, fields have bounded lengths, confidence
and priority accept qualitative labels only, and extra fields are forbidden. There
are at most three recommendations (one primary followed by at most two alternatives).
The wire contract itself is unchanged.

## Deterministic validation before mapping

- Reference IDs are looked up in separate current opinion/owner-message/memory
  namespaces. AI message IDs and citations from historical solutions do not enter
  the registry. Findings/hypotheses/actions need current sources; a finding based
  solely on SQL statistics can use empty references.
- A customer quote must be a nonblank exact substring of that opinion's evidence_text
  or full_feedback. It is not reconstructed from normalized opinion_text. Neither
  quote whitespace nor UUIDs are rewritten.
- Actual SQL numbers use the internal marker `[[stat:opinion_count=10]]` in generated
  prose. The named metric must exist and the Decimal value must equal IssueStats.
  Average rating, zero values, and large integer counts are checked without float
  conversion of integers. Unknown/malformed markers and free numeric factual prose
  are rejected unless the complete numerical text is an exact excerpt of a cited
  source. This intentionally conservative format avoids guessing the metric from
  unrestricted prose. Verified markers are replaced by Indonesian labels before
  protobuf mapping; users never see internal marker syntax.
- Proposed numeric recommendations must start `Usulan:`. Nonempty success target
  and timeframe also start `Usulan:` and cannot contain common achievement words
  (`telah`, `sudah`, `tercapai`, `achieved`, `already`). Empty target/timeframe remain
  legal as the proto permits. Metrics describe what the owner measures manually.
- Hypotheses start `Hipotesis:` and contain an explicit uncertainty phrase. Common
  certainty phrases are rejected; findings reject common causal formulations.
  These are textual guardrails, not a semantic classifier or calibrated confidence.

`solution_to_proto` validates first, renders verified markers only in prose, then
maps exactly the five arrays to ai_pb2.Solution. The response includes model_id and
the solution prompt version. Invalid output is INTERNAL, never INVALID_ARGUMENT.

## Invocation, deadlines and no-evidence policy

Assessment and generation reuse one typed Gemini invocation helper, configuration,
sanitized error mapping, context byte bound, disabled tracing/cache, and zero retries.
An ordinary generate decision uses assessment then generation. GENERATE_NOW/counter
cap bypass assessment and use one generation call. No extra entailment/repair call
is introduced. The configured timeout is one total turn budget, reduced by elapsed
assessment/preparation time and the gRPC deadline; the generator receives the remainder.
Synchronous HTTP cancellation can still wait for the SDK timeout.

If there is no evidence with nonblank evidence_text/full_feedback, generation fails
before a provider call with FAILED_PRECONDITION. No fake quote/empty successful solution
is returned. Go currently requires customer_evidence; Suthe must decide whether
no-evidence solutions should become legal. Question collection can still operate
with empty evidence. Go's cumulative question_count remains conservatively capped
at four; Python does not reset it for CONTINUE_DISCUSSION.

Gemini settings and Compose variables are unchanged. GEMINI_MODEL must be enabled
for the account and support native JSON schema. GEMINI_MAX_OUTPUT_TOKENS may need
tuning after real-model tests; truncated output fails validation rather than being
repaired or silently accepted. See `.env.example` for configuration.

## Milestone 4B limitations

Valid IDs/quotes are not proof that a claim is entailed by its cited sources.
Structural checks cannot reliably detect unsupported claims expressed without
digits, quantities spelled as words, semantic metric swaps around a correct marker,
subtle causality, implicit historical claims, infeasible recommendations, or clever
prompt injection. Proposal prefixes do not prove feasibility or impact. Reference
relevance, attribution of exact source excerpts, sample/population distinctions,
contradictions, hypothesis calibration and measurement quality need semantic
evaluation in 4B. No full semantic grounding claim is made here.

Tests use mocked Gemini, including the real LangChain parser with the SDK transport
mocked, in-process gRPC round trips, and regression tests for all five events.
Live Gemini was not run: neither GEMINI_API_KEY nor GEMINI_MODEL was configured in
the process/local .env at verification time. No credentials or source content were
printed. Go-backend end-to-end integration also remains untested.
