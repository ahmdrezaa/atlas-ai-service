# P-B1 — Consultation (Path B) reliability fix

**Date:** 2026-11-14 · **Model:** `gemini-3.5-flash-lite` · No new LLM calls added (deterministic changes only).

## Problem (live probe before the fix)
The unchanged pipeline on the 7 synthetic M7.1 fixtures × {GENERATE_NOW, START}: questions 5/5 OK,
**solutions 0/9 OK**. Gemini always answered (1.4–5 s) and every output passed structural checks; all were
rejected by our own grounding checker. Rejected per field: recommendation descriptions 14/17, titles 6/17,
evidence notes 6/6, hypotheses 2/9, findings 2/25.

Root causes: (1) advice was checked like a fact (verb whitelist + resource words); (2) one weak claim failed
the whole turn; (3) keyword false positives ("mungkin **belum terverifikasi**", "mungkin **disebabkan**",
"risiko … **rusak**", "Usulan alternatif **1**"); (4) an invalid question failed the turn; (5) Path B still
defaulted to `gemini-3.8-flash` (HTTP 504) with a 30 s budget while Go allows 60 s.

## Output policy now (`consultation/solution.py`, `consultation/grounding.py`)
| Class | Examples | Effect |
| --- | --- | --- |
| **Fabrication** | non-verbatim quote; invented / wrong-namespace reference ID | **whole turn rejected** (`check_fabrication`) |
| **Unsupported item** | unverifiable number/statistic, causal finding, untentative hypothesis, advice asserting a fact/resource/outcome, uncited or blank text | **item dropped**; turn fails only if a section becomes empty (`prune_solution`) |
| **Optional note** | unverifiable `customer_evidence.note` | **blanked** |
| **Relevance** | off-topic advice ("harga Bitcoin") | **diagnostic only** (project policy since M6.2); semantic check is B2 |

Advice (`evaluate_proposal`) is checked as advice — no verb whitelist. Still blocked (negation/hedge-aware):
guaranteed or achieved outcomes, universal/causal assertions, asserted broken equipment or staff state
("rusak", "lelah"), unknown resources asserted to exist ("tersedia"), committing to new resources
("penambahan barista", "mesin baru"; "tanpa penambahan staf" is allowed), asserted results ("Pendapatan
meningkat"), unproven operational numbers, unsupported "karena" premises. Hypotheses may name a hedged cause
("mungkin … karena/disebabkan …") unless the cause clause asserts a new fact ("karena mesin rusak").
Dropped items are logged as `claim_id=reason` only (no text). An invalid question falls back to a solution
when evidence exists. Defaults: `GEMINI_MODEL=gemini-3.5-flash-lite`, `GEMINI_TIMEOUT_SECONDS=45`.

## Results
- **Offline replay** of every captured Gemini solution: **24/24 accepted** (were 0/9 on the first capture).
- **Live probe after the fix:** solutions **7/8**, questions **6/6**; the remaining rejection (an untentative
  second hypothesis) is now dropped instead of failing (replay passes).
- **Go integration (real API, compose environment):** QR feedback → 1 planet → START (question) →
  OWNER_REPLY with numbers (5-section solution) → same `request_id` replay (`replayed=true`, no AI call) →
  Save (`ARCHIVED`, 2 owner-sourced Business Memories). Earlier the same reply failed once on an unverifiable
  number — the case the item-dropping policy addresses.
- Tests: **1605 passed, 17 xfailed** (8 strict xfails for known false rejections now pass and were removed).
  No case labelled as a fabricated quote/ID/number is newly returned; no valid case is newly rejected
  (frozen M6.2/M6.3/M6.4 corpora). Tests that asserted "raises" for a droppable item now assert the stronger
  invariant "rejected **or** the text is absent from the Solution" (`tests/solution_policy.py`).

## Known gaps / next (B2)
- **Relevance** is not enforced deterministically (would reject valid advice such as "panduan takaran garam"
  for "terlalu asin"). Planned: one batched LLM check for relevance/support of claims, skipped when <10 s remain.
- Memory candidates still cover only 4 fact types; `question_count` never resets in Go (G2).
- Small synthetic runs — not an accuracy score.
