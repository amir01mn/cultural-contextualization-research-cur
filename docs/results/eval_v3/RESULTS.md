# Eval v3 results: a rebuilt layered KG (KG v3)

Run date: 2026-10-10/11 · Branch: `kg-v3` · Protocol (fixed before any test run): [`docs/EVAL_V3_PLAN.md`](../../EVAL_V3_PLAN.md)

## Summary

Eval v2 showed that the layered triples help but the abstraction→refinement pathways (S4) did not, and
located the cause in the pathway builder. KG v3 rebuilds that layer: one **frame** per CultureBank
descriptor, linked upward to a culture-neutral **norm type** and to **values** from a fixed taxonomy
(Schwartz's 19 values), with cultures connected only through those abstract nodes. At answer time
the system **routes**: the culture's own value → norm → practice chains when they exist, otherwise the
abstraction plus how similar cultures handle it, otherwise nothing.

1. **The rebuilt pathways are sound.** Every record is reachable (August pathways: 38%), the gold record
   is in the top 8 for 94.7% of test questions (August: 41.2%), and 88.5% of frame chains are judged
   coherent (August: 22.5%).
2. **The routed frame KG (S4R) is the best system on every in-corpus test, including against plain
   passages:** multiple choice 92.9% vs 91.4% (+1.5, Holm p = 0.015), hard multiple choice 79.6% vs 76.5%
   (+3.2, p < 0.001), open-ended 92.6% vs 88.8% (+3.8, p = 0.009). The result replicates with a second
   answering model (OLMo-2-7B: +3.4 over passages on multiple choice).
3. **The abstraction route helps the weaker model a lot on NormAd, the stronger one little.** With OLMo,
   S4R gains +10.3 points over country-only and +6.9 over the same system without the abstraction route
   (both Holm p < 0.001), ahead of NormAd's own gold value context. With Qwen the gain is +0.6 (n.s.),
   although S4R is still the only retrieval system above country-only and beats passages by +2.5
   (Holm p = 0.004).
4. **Held-out records are still negative.** When a question's own record is removed, S4R (75.0%) is below
   no evidence (80.0%), like every other system. The routing thresholds were tuned on questions whose
   record exists, so S4R usually takes the refinement route with other, misleading records.
5. **The August triples contained evaluation text** (found during this work): the extraction prompt included
   CultureBank's `eval_whole_desc`. 16% of triples are affected; v3 removes them from all triple systems and
   KG v3 never reads those columns. v2's triple results on CultureBank should be read as upper bounds.

## Setup

- **KG v3:** 22,990 frames (21 unlabeled), 3,077 norm types (1,358 shared by ≥ 2 cultures), 2,159 cultural
  groups. Labels by Qwen2.5-7B (norm, norm type, values only; descriptor fields mapped directly).
- **Answering models:** Qwen2.5-7B-Instruct (primary), OLMo-2-1124-7B-Instruct (replication). vLLM,
  temperature 0, top-k 8. **Judge:** microsoft/phi-4 (not yet human-validated; sheets in
  [`docs/annotation/v3/`](../../annotation/v3/)).
- **Statistics:** McNemar exact tests and paired bootstrap 95% CIs; Holm–Bonferroni over the pre-registered
  family of each evaluation. Routing thresholds tuned on each evaluation's dev split only.
- Same questions, splits and paraphrased options as eval v2. All triple systems drop the 30,130 triples that
  carry evaluation-column wording.

| System | Evidence |
|---|---|
| S0 | none (closed-book) |
| S1 / S1g | raw CultureBank passages / with culture filter |
| S2, S3, S3g | flat triples; raw layered triples (S3 now raw, as designed); with culture filter |
| A3 | refined layered triples |
| S4 | August abstraction→refinement pathways (eval v2's S4, unchanged) |
| **S4L** | record-local pathways over the same triples, with source text (step-2 fix) |
| **S4F** | KG v3 frames, value → norm type → norm → practice, always injected |
| **S4R** | KG v3 with routing (refine / abstract / none) — **ours** |
| S4R_noabs | S4R without the abstraction route (ablation) |

## 1. Is the rebuilt layer sound? (no answering model)

| | August pathways (S4) | Record-local (S4L) | **KG v3 frames** |
|---|---|---|---|
| Records reachable | 38% | 99.9% | **100%** |
| Gold record retrieved, R@8 (test) | 0.412 | 0.799 | **0.947** (S4R) |
| Pathway audit, coherent (phi-4, n=200) | 22.5% | 48.5% | **88.5%** |
| culture-consistent | 19.0% | 72.5% | **92.5%** |
| correct | 17.0% | 48.5% | **88.5%** |

Passages (S1) reach R@8 = 0.875. The frame chains' practice node is the descriptor itself, so "correct" is
easier for frames than for triple-built pathways; "coherent" is the informative criterion.

## 2. CultureBank grounded questions (Qwen2.5-7B)

| System | Multiple choice (n=1,500 × 3 orders) | Hard multiple choice (n=1,500 × 3) | Open-ended, judged (n=500) |
|---|---|---|---|
| S0 | 78.6% | 48.5% | 79.2% |
| S1 passages | 91.4% | 76.5% | 88.8% |
| S1g | 91.1% | 76.8% | — |
| S2 | 83.1% | 58.6% | — |
| S3 (raw) | 82.2% | 57.9% | — |
| S3g | 85.5% | 64.0% | 88.8% |
| A3 (refined) | 82.5% | 58.7% | — |
| S4 (August) | 78.6% | 49.0% | 82.6% |
| S4L | 92.1% | 77.1% | 88.4% |
| S4F | 92.4% | 78.7% | — |
| **S4R** | **92.9%** | **79.6%** | **92.6%** |
| S4R_noabs | 92.9% | 79.6% | — |

Pre-registered comparisons (Holm within each column's family):

| Comparison | Multiple choice | Hard multiple choice | Open-ended |
|---|---|---|---|
| S4R vs S0 | +14.3 (p < 0.001) | +31.2 (p < 0.001) | +13.4 (p < 0.001) |
| **S4R vs S1** | **+1.5** [+0.4, +2.5] (p = 0.015) | **+3.2** [+1.7, +4.6] (p < 0.001) | **+3.8** [+1.4, +6.4] (p = 0.009) |
| S4R vs S3g | +7.4 (p < 0.001) | +15.7 (p < 0.001) | +3.8 (p = 0.009) |
| S4R vs S4 | +14.3 (p < 0.001) | +30.6 (p < 0.001) | +10.0 (p < 0.001) |
| S4F vs S4 | +13.9 (p < 0.001) | +29.7 (p < 0.001) | — |
| S4L vs S4 | +13.6 (p < 0.001) | +28.1 (p < 0.001) | +5.8 (p < 0.001) |
| S4R vs S4R_noabs | 0.0 | 0.0 | — |

On these questions the culture's own record exists, so S4R takes the refinement route for 99.8% of them and
equals its ablation. The hard variant uses the most similar valid distractors (mean similarity to the gold
answer 0.645), which drops the closed-book score from 78.6% to 48.5%.

Evidence size: S4R uses ~722 tokens per question, S1 ~616, S3g ~226. S4R's chains add an LLM-written norm
sentence and value labels to the descriptor.

### Held-out records (n = 300; frame systems use a KG built without the held-out records)

| S0 | S1 | S2 | S3g | S4 | S4L | S4F | **S4R** | S4R_noabs |
|---|---|---|---|---|---|---|---|---|
| 80.0% | 78.3% | 80.0% | 76.7% | 75.7% | 76.7% | 74.7% | **75.0%** | 75.0% |

(near mask; far mask within 0.3 points.) S4R routed 233/300 to refinement with other same-culture records
(74.7% vs S0 81.1% on those), 63 to none, 4 to abstraction.

## 3. NormAd (external; 2,315 test stories, 5.8% covered by CultureBank)

| Condition | Qwen2.5-7B | OLMo-2-7B |
|---|---|---|
| C1 country only | 57.2% | 44.1% |
| C2 + gold value | 61.5% | 49.4% |
| C3 + gold rule-of-thumb (ceiling) | 73.0% | 61.9% |
| S1 passages | 55.3% | — |
| S1g | 57.1% | 48.8% |
| S3g | 57.3% | — |
| S4 (August) | 56.1% | — |
| S4L | 57.3% | — |
| S4F | 55.1% | — |
| **S4R** | **57.8%** | **54.4%** |
| S4R_noabs | 57.5% | 47.5% |

| Comparison (Holm) | Qwen | OLMo |
|---|---|---|
| S4R vs country only | +0.6 (n.s.) | **+10.3** [+8.2, +12.5] (p < 0.001) |
| S4R vs S4R_noabs (abstraction route) | +0.3 (n.s.) | **+6.9** [+5.1, +8.6] (p < 0.001) |
| S4R vs S1 passages | **+2.5** (p = 0.004) | — |
| S4R vs S4 | +1.6 (Holm p = 0.17) | — |

- S4R took the abstraction route for 70% (Qwen) / 75% (OLMo) of stories. On those stories: Qwen 56.6% vs
  country-only 55.8% (n.s.); OLMo 57.0% vs 43.0%, above the gold value (47.6%).
- Part of OLMo's gain is calibration: country-only OLMo answers "neither" 44% of the time; with S4R evidence
  it commits more often. Macro-F1 also rises (0.40 → 0.48).
- Covered subset (n = 134): Qwen S4R +4.5 over country-only (6/0, raw p = 0.03, Holm 0.19); the August S4
  remains highest there (57.5%). OLMo: no change.

## Second answering model: CultureBank multiple choice (OLMo-2-7B)

S0 75.4%, S1 90.0%, S3g 81.8%, S4 75.3%, S4L 89.0%, **S4R 93.4%**. S4R vs S1 +3.4 [+2.4, +4.5],
vs S3g +11.6, vs S0 +18.0 (all Holm p < 0.001).

## What this means

- **The layered KG now works, and works better than passages in-corpus**, across three question formats and
  two answering models. The bottleneck identified in eval v2 (the pathway builder) is fixed.
- **The abstraction layer's value is model-dependent so far:** decisive for OLMo on NormAd, negligible for Qwen.
  This is the part of the hypothesis that still needs strengthening.
- **Generalisation to unseen records is not shown.** The next experiment is to tune the refinement gate on a
  held-out *dev* split (records masked), so that S4R falls back to abstraction when only other records of
  the culture match. That would be a new pre-registered run, not a re-tune of this one.

## Limitations

- phi-4 judge not human-validated (open-ended scores and pathway audit). Sheets ready:
  `docs/annotation/v3/judge_open.csv` (144 answers), `stereotype.csv` (100 triples); keys kept separately.
- One run per NormAd condition; three option orders for multiple choice.
- Norm-type clustering leaves some generic types (e.g. "cultural practice", 782 frames).
- The leakage filter is a lexical proxy; S4 (August) still uses its original, partly leaky index.
- KG v3 labels come from Qwen2.5-7B, the primary answering model.

## Files

| Evaluation | Tables |
|---|---|
| CultureBank (MCQ, held-out, retrieval, open-ended) | [`culturebank/summary.md`](culturebank/summary.md), [`culturebank/open_judge_summary.md`](culturebank/open_judge_summary.md) |
| CultureBank hard MCQ | [`culturebank_hard/summary.md`](culturebank_hard/summary.md) |
| NormAd | [`normad/summary.json`](normad/summary.json), [`normad_olmo/summary.json`](normad_olmo/summary.json) |
| OLMo CultureBank MCQ | [`culturebank_olmo/summary.md`](culturebank_olmo/summary.md) |
| Pathway audits | [`intrinsic/`](intrinsic/) |
| KG v3 build and dev check | [`kg_v3/`](kg_v3/) |

Per-question outputs stay on the cluster under `data/outputs/eval_v3/`; the KG under `data/kg_v3/`.
