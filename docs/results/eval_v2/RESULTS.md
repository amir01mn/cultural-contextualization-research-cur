# Eval v2 results: does a CultureBank knowledge graph help an LLM?

Run date: 2026-09-27 · Branch: `eval-v2` · Protocol (fixed before any run): [`docs/EVAL_V2_PLAN.md`](../../EVAL_V2_PLAN.md)

## Summary

Our earlier headline benchmark, CulturalBench, mostly asks about facts CultureBank does not contain:
even oracle retrieval picks the correct option only 27.9% of the time (chance is 25%). Eval v2 therefore
tests the knowledge graph (KG) where its knowledge is actually needed, plus one external benchmark and a
quality audit of the graph itself.

1. **The KG helps when the questions ask about knowledge it contains.** On CultureBank's own grounded
   questions, passages (+12.8 points), layered triples with a culture filter (+6.7) and flat triples (+4.2)
   all beat the no-evidence baseline on multiple choice, and every KG system beats it on open-ended answers.
2. **The abstraction→refinement pathways (S4) are the weak link.** Pathways reach the correct source record
   only 47% of the time (triples: 82%) and keep 29% of the source facts (triples: 93%). Only 22.5% of
   complete pathways were judged coherent. The problem is the pathway builder, not the extracted triples.
3. **A culture-matching relevance filter makes retrieval safe.** On NormAd, where CultureBank covers only
   5.8% of stories, unfiltered retrieval hurts (−2 points) and filtered retrieval matches the baseline.
4. **One signal for the layered hypothesis:** on the 134 NormAd stories CultureBank does cover, S4 is the
   best system (+7.5 points over country-only), but this is not significant after multiple-comparison
   correction (Holm p = 0.07).
5. **No generalization yet:** when a question's own source record is hidden, no system beats the baseline.

## Setup

- **KG:** the August 2026 full build from CultureBank (22,990 descriptors), used read-only.
- **Answering model:** Qwen2.5-7B-Instruct (vLLM, temperature 0, top-k 8 evidence items).
- **Judge (open-ended and graph audit):** microsoft/phi-4, a different model family. Not yet validated
  against human labels.
- **Statistics:** McNemar exact tests and paired bootstrap 95% CIs; Holm–Bonferroni correction over the
  pre-registered comparisons.

| System | Evidence given to the model |
|---|---|
| S0 | none (closed-book baseline) |
| S1 | raw CultureBank passages |
| S2 | flat triples |
| S3 | layered triples (value / norm / practice labels) |
| S4 | abstraction→refinement pathways |
| S4_triple_retrieval, S4_no_shared_pivot | S4 ablations |
| S1g, S3g, S4g | same as S1/S3/S4 with the relevance filter: evidence must match the question's culture and pass a score threshold tuned on a dev split; otherwise the model answers closed-book |

## 1. CultureBank grounded questions (headline)

Every CultureBank row ships an evaluation question generated from that row (Shi et al. 2024, §6), so the
correct source record is known. Sample: 1,500 test questions stratified by culture (agreement ≥ 0.6), plus
300 dev questions for tuning the filter. Leakage check: **0** evidence strings contained the question or its
source description.

### Multiple choice (n = 1,500; mean over 3 option orders)

Correct option = the row's behaviour; distractors = behaviours of other cultures on the same topic
(MiniLM similarity < 0.75 to every descriptor of the correct culture). All options paraphrased identically.

| System | Accuracy | Δ vs S0 | 95% CI | Holm p |
|---|---|---|---|---|
| S0 | 78.6% | — | — | — |
| **S1** | **91.4%** | **+12.8** | [+11.3, +14.2] | < 0.001 |
| S1g | 91.1% | +12.5 | [+11.0, +14.0] | < 0.001 |
| **S3g** | **85.4%** | **+6.7** | [+5.4, +8.0] | < 0.001 |
| S2 | 82.9% | +4.2 | [+3.0, +5.5] | < 0.001 |
| S4_triple_retrieval | 82.3% | +3.7 | [+2.5, +4.9] | 0.002 |
| S3 | 82.2% | +3.6 | [+2.4, +4.8] | < 0.001 |
| S4 | 78.5% | −0.2 | [−1.4, +1.0] | 1.0 |
| S4g | 78.6% | −0.1 | [−1.3, +1.1] | 1.0 |
| S4_no_shared_pivot | 78.5% | −0.1 | [−1.3, +1.0] | 1.0 |

S4 vs S1: −12.9 points (Holm p < 0.001). The filter improves S3 (82.2% → 85.4%) and leaves S1 and S4 unchanged.

### Open-ended answers (n = 500 questions; phi-4 judges whether the answer agrees with the source description)

| System | Agrees with source | Δ vs S0 | 95% CI | Holm p |
|---|---|---|---|---|
| S0 | 78.8% | — | — | — |
| **S1** | **89.4%** | **+10.6** | [+7.4, +14.0] | < 0.001 |
| **S3** | **86.2%** | **+7.4** | [+4.2, +10.6] | < 0.001 |
| **S4** | **82.8%** | **+4.0** | [+1.0, +7.0] | 0.031 |
| S4g | 82.6% | +3.8 | [+0.8, +6.8] | 0.037 |

S4 vs S1: −6.6 points (Holm p < 0.001).

### Retrieval: is the correct source record retrieved? (test, n = 1,500)

| System | Recall@1 | Recall@8 | MRR | Same-culture precision | ~Evidence tokens / question |
|---|---|---|---|---|---|
| S1 passages | 0.712 | 0.875 | 0.785 | 0.303 | 616 |
| S1g | 0.712 | 0.931 | 0.796 | 1.000 | 343 |
| S2 flat triples | 0.703 | 0.807 | 0.745 | 0.421 | 147 |
| S3 layered triples | 0.702 | 0.819 | 0.748 | 0.431 | 268 |
| S3g | 0.702 | 0.864 | 0.758 | 1.000 | 241 |
| **S4 pathways** | **0.339** | **0.465** | **0.389** | 0.374 | 642 |
| S4g | 0.339 | 0.494 | 0.390 | 0.908 | 483 |

### Held-out records (n = 300; each question's own source record removed from the KG)

| System | S0 | S1 | S2 | S3 | S4 | S1g | S3g | S4g |
|---|---|---|---|---|---|---|---|---|
| near mask | 80.0% | 78.3% | 79.3% | 77.3% | 75.7% | 77.7% | 75.7% | 76.3% |
| far mask (+ near-duplicates removed) | 80.0% | 78.3% | 79.3% | 77.7% | 76.0% | 77.7% | 75.7% | 76.3% |

No system benefits once the exact record is gone, so this KG shows no generalization benefit from abstraction yet.

## 2. NormAd (external benchmark; 2,315 test stories, 75 countries)

NormAd gives each story three kinds of context: country only (C1), country + an abstract value (C2), and
country + a specific rule-of-thumb (C3). Only 5.8% of stories have a matching CultureBank descriptor
(MiniLM similarity ≥ 0.5 to the gold rule-of-thumb).

| Condition | All (n = 2,315) | Covered (n = 134) | Not covered (n = 2,181) |
|---|---|---|---|
| C1 country only (baseline) | 57.5% | 48.5% | 58.0% |
| C2 + gold value | 61.3% | 52.2% | 61.9% |
| C3 + gold rule-of-thumb (ceiling) | 72.7% | 61.2% | 73.4% |
| S1 passages | 55.4% | 53.0% | 55.5% |
| S3 layered triples | 56.9% | 50.7% | 57.3% |
| **S4 pathways** | 55.5% | **56.0%** | 55.4% |
| S1g | 57.3% | 53.0% | 57.6% |
| S3g | 57.3% | 50.0% | 57.8% |
| S4g | 57.3% | 52.2% | 57.6% |

- **All stories:** unfiltered S1 (−2.1, Holm p = 0.03) and S4 (−2.0, Holm p = 0.06) hurt; the filtered
  systems match the baseline (−0.1, n.s.). On uncovered stories, S4g beats S4 by +2.2 (Holm p = 0.02).
- **Covered stories:** S4 +7.5 points over country-only (11 answers fixed, 1 broken; raw p = 0.006,
  Holm p = 0.07), closing 59% of the gap to the gold rule-of-thumb and ahead of the gold value (C2).
  Promising, but the subset is small.

## 3. Graph quality (phi-4 judge)

**Fact retention** (300 records, 1,441 atomic facts from the source descriptions): can the fact be
inferred from the record's graph content?

| Source | Facts retained |
|---|---|
| Raw triples | 92.8% |
| Refined triples | 92.7% |
| Pathways | 28.7% (748 facts had no pathway touching their record) |

**Triple audit** (500 refined triples):

| Criterion | All | Directly extracted | Generated |
|---|---|---|---|
| Faithful to source | 71.8% | 74.8% | 65.4% |
| Layer labels correct | 88.8% | 91.2% | 83.7% |
| Culture-specific | 79.6% | 80.9% | 76.7% |
| Flagged as stereotyping* | 26.6% | 23.5% | 33.3% |

\*Needs a human spot-check; LLM judges tend to flag any generalization about a group.

**Pathway audit** (200 complete pathways): coherent value→norm→practice chain 22.5%, culture-consistent 19.0%, correct 17.0%.

Graph composition: 0% of triples came from failed extraction; 14.3% of raw and 29.7% of refined triples are generated rather than directly extracted.

## 4. CulturalBench, re-scored by CultureBank coverage (secondary; August per-item results, no new run)

Combined accuracy (Easy + Hard). Strata = CultureBank descriptors per country.

| System | < 20 (6 countries) | 20–99 (17) | ≥ 100 (22) | All |
|---|---|---|---|---|
| S0 | 55.4% | 48.8% | 53.6% | 52.2% |
| S1 | 36.4% (−18.9) | 37.2% (−11.6) | 45.0% (−8.6) | 41.4% (−10.8) |
| S3 | 40.4% (−15.0) | 39.5% (−9.3) | 44.8% (−8.8) | 42.5% (−9.7) |
| S4 | 36.8% (−18.6) | 35.2% (−13.6) | 40.8% (−12.9) | 38.4% (−13.8) |

All differences vs S0 have p < 0.001. Losses shrink as coverage grows but never disappear. On Easy
questions in the ≥ 100 stratum, S1 is −2.1 (p = 0.21).

## Limitations

- The phi-4 judge has not yet been validated against human labels (planned: ~150 hand-labelled answers).
- One answering model (Qwen2.5-7B). Multiple choice has 3 option orders; other tests use one run.
- CultureBank "cultures" include non-national groups (e.g. "beachgoers"), so the CultureBank test set
  spans 810 groups, not only countries.
- The NormAd covered subset is small (n = 134).
- Held-out masking removes records from retrieval, but entity resolution and shared pivots were built with
  them present.
- The CultureBank questions often paraphrase the correct behaviour, so S0 is already high (78.6%).

## What this implies

The layered representation is not the problem: layered triples help, and the relevance filter makes them
safe. The pathway builder is: it reaches too few records, drops most facts and mixes cultures. The proposed
next step is to rebuild pathways with one frame per CultureBank descriptor, a fixed value taxonomy, clustered
norm types, and cultures linked only through those abstract nodes, then re-test on the held-out split and
the NormAd covered subset.

## Files

| Evaluation | Full tables |
|---|---|
| CultureBank multiple choice, held-out, retrieval | [`culturebank/summary.md`](culturebank/summary.md) |
| CultureBank open-ended (judge) | [`culturebank/open_judge_summary.md`](culturebank/open_judge_summary.md) |
| NormAd | [`normad/summary.md`](normad/summary.md) |
| Graph quality | [`intrinsic/summary.md`](intrinsic/summary.md) |
| CulturalBench by coverage | [`culturalbench_strata/summary.md`](culturalbench_strata/summary.md) |

Machine-readable versions (`summary.json`, `tau.json`, `leakage_guard.json`, `retrieval_metrics.json`)
sit next to each file. Per-question outputs remain on the cluster under `data/outputs/eval_v2/`.
