# KazCulture v2: pre-registered test analysis (option-grounded retrieval)

**Branch:** `kazculture-v2-dev`
**Written:** 2026-11-08, after the v2 dev runs and **before any v2 test-set run**.
**Status:** frozen once committed and pushed. Later changes go in a dated
"Amendments" section, with the reason, and must not be motivated by test outcomes.

v1 (`KAZCULTURE_PREREGISTRATION.md`, run `evaluation/final_v1b`) is final and
negative. It is not re-analysed or replaced. v2 is a new question, developed only
on held-out **train** MCQ (`docs/KAZCULTURE_V2_OPTION_GROUNDED.md`).

## 1. Questions

- **Q1 (primary).** Does option-grounded retrieval (S1-OG) improve Qwen3-32B on the
  KazCulture test set over the frozen S1 passage retrieval?
- **Q2 (layers, secondary).** With option-grounded retrieval fixed, do the layered-KG
  labels (values, norm type, norm) add accuracy?

Dev evidence that motivated this design (1,000 held-out train MCQ, own passage and
near-duplicates hidden; `evaluation/v2_dev/a2`, `a3`):

| condition | dev acc | vs S1 |
|---|---|---|
| S1 | 68.9 | n/a |
| S1-OG | 76.6 | +7.7 |
| S1-OGpo | 72.1 | +3.2 |
| S4R-OGF | 71.3 | (−0.8 vs S1-OGpo) |
| S4R-OGFR | 70.5 | (−1.6 vs S1-OGpo) |
| S1-OG+L | 74.5 | (−2.1 vs S1-OG; +31/−52) |
| S4R-OG / S4R-OGR (option-grouped) | 59.3 / 60.0 | dropped: the grouping misattributes passages |

The dev results predict a positive Q1 and a null or negative Q2. Q2 is registered so
that the layers are reported whichever way they go.

## 2. Frozen components

Everything in v1 §2 is unchanged: Qwen3-32B at the pinned revision, non-thinking, the
frozen V2 letter scorer and prompt template, BGE-M3 at the pinned revision, the
strict and full S1 indexes, `top_k=8`, a 5,000-character evidence budget, and the
1,334-question test set. S0 and S1 predictions are copied from `final_v1b`, which is
deterministic and reproduced exactly (552, 831 strict, 1,221 full).

New and frozen at this commit: `src/retrieval/kazculture_option_grounded.py` and
`src/evaluation/kazculture_og_evidence.py` (the same builder the dev runs used).
They have no tuned parameters. The score is z(dense question) + z(BM25 question +
options), with stems of 5-character prefixes. Per option, `PER_OPTION_K=4` and
`TAU_GROUND=1.0` were set before any dev result. S1-OG has no minimum-similarity
cut (the hybrid score has no cosine scale) and uses the frozen S1 packing.

## 3. Conditions (Qwen3-32B, identical prompt and scorer; only evidence varies)

| ID | Settings | Evidence |
|---|---|---|
| S0, S1 | strict, full | copied from v1b |
| **S1-OG** | strict, full | flat S1 blocks, pooled option-grounded ranking |
| S1-OG+L | strict, full | S1-OG plus each passage's own labels (as S1+L) |
| S1-OGpo | strict | flat S1 blocks, per-option ranking interleaved |
| S4R-OGF | strict | S1-OGpo's passages, same order, rendered as layered frames |

OLMo-2-1124-7B-Instruct replication: strict S1-OG. Its S0 and S1 are copied from
`final_v1b_olmo`.

## 4. Statistics

The same as in v1 §5: exact two-sided McNemar on paired outcomes, a paired bootstrap
95% CI (10,000 resamples, seed 0), and significance at p < 0.05 with a CI excluding 0.

- **Primary:** strict S1-OG vs strict S1. The minimum meaningful effect is +2.0 pp.
- **Secondary (one Holm family):** strict S1-OG+L vs strict S1-OG; strict S4R-OGF
  vs strict S1-OGpo; full S1-OG vs full S1; full S1-OG+L vs full S1-OG.
- **Replication:** OLMo strict S1-OG vs OLMo strict S1, same direction (descriptive
  p reported).
- **Descriptive:** the strict near-dedup subset (the 110 flagged questions removed,
  `audit/near_duplicates_v1`); per-category deltas; rendered characters per
  condition (S1-OG vs S1).

## 5. Criteria

1. The primary comparison is significant with an effect of at least +2.0 pp
   (option-grounded retrieval works).
2. The direction holds on the near-dedup subset and replicates with OLMo.
3. The test set is touched once per condition. The v2 design, code and every
   constant come from dev runs on train rows only (git history).
4. **Layers (Q2):** reported as "layers help" only if S1-OG+L > S1-OG or S4R-OGF >
   S1-OGpo, significant after Holm. Otherwise "layers neutral/negative on KazCulture".

There is no fallback analysis: whatever Q2 shows is what gets reported.

## Amendments

(none)
