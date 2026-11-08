# KazCulture v2: why v1 S4R did not help, and option-grounded retrieval

Status: development on held-out **train** MCQ only (`v2_dev/dev_mcq_v1_cal.jsonl`,
1,000 rows, own passage + BGE-M3 near-duplicates ≥ 0.95 hidden). The test set has
not been touched since the preregistered v1b run.

## 1. Diagnosis (2026-11-07, dev only)

Dev accuracy from `evaluation/v2_dev/a1`: S0 56.6, S1 68.9, S1+L 69.1, S4R 68.1,
S4R-PR 68.8. Adding layer text to the same passages changes nothing, so the
question is what limits S1.

**The reader is not the bottleneck; retrieval is.** An evidence passage "bears the
answer" when ≥ 60% of the gold option's content stems (5-char prefixes of words
with ≥ 4 letters) occur in it and it covers the gold option better than any
distractor. Over the 1,000 dev questions:

| S1 evidence | n | S1 accuracy |
|---|---|---|
| bears the answer | 217 | 93% |
| ambiguous (gold and a distractor both covered) | 182 | 76% |
| no answer-bearing passage | 601 | 58% (≈ S0) |

**The answer is usually in the corpus but not retrieved.** On the same
lexical criterion, some other train passage bears the answer for 73% of dev
questions (400-question sample), but S1's question-only dense query puts one in
its top 8 for 27%. The first answer-bearing passage has median dense rank 29.

KazCulture questions are item/term lookups in Kazakh ("what is a horse driven with
from a sled?" → "with a shybyrtqy"). The answer options name the candidate items,
so they are the strongest retrieval key, and v1 never used them (S1 and S4R
query with the question only).

**Why v1 S4R could not help.** (a) It used the same question-only query, so it
inherited S1's coverage. (b) The v3 layers (values, norm type, norm) describe *why*
a practice exists; KazCulture asks *which* item, so the labels add text without
adding the missing fact. (c) The abstract route never fired, because every
question has some Kazakh frame above τ_abs. (d) Labels take budget, so S4R shows
fewer passages (5.4 vs 6.7 blocks); CTRL-LEN, with more passages, beat it on test.

Retrieval proxies (same answer-bearing criterion, evidence within the 5,000-char budget):

| retrieval | answer-bearing evidence |
|---|---|
| S1 dense, question only (top 8) | 26.7% |
| KG expansion: dense top 8 → norm-type siblings, BM25 rerank | 39.8% |
| pooled hybrid: z(dense q) + z(BM25 q + all options), top 8 | 48.4% |
| per-option hybrid (+ distinctive option stems), interleaved top 8 | 55.1% |

Norm-type expansion does find more candidates (46% contain one), but the KG's
L2/L3 structure ranks them worse than lexical option grounding.

## 2. v2 conditions (dev)

`src/retrieval/kazculture_option_grounded.py`, `scripts/eval_kazculture_v2_dev.py`.
Weights are fixed (sum of z-scores) and not tuned.

| ID | Evidence |
|---|---|
| S1-OG | flat S1 blocks, pooled hybrid ranking |
| S1-OG+L | S1-OG plus each passage's own labels |
| S1-OGpo | flat S1 blocks, per-option ranking interleaved (control for S4R-OG) |
| S4R-OG | per-option grounded frames, grouped under each option, with values / norm type / norm |
| S4R-OGR | S4R-OG with grounding routing: an option whose distinctive stems are not all found in its retrieved practices also gets the abstract route (its norm type plus sibling norms) |

What would support the layered-KG claim, as opposed to option-aware retrieval:
S4R-OG or S4R-OGR beating **S1-OGpo** (same retrieval, flat layout, no layers).
Beating only S1 would show option grounding helps, not the layers.

## 3. Dev results (1,000 rows; jobs 8752862 → `a2`, 8754552 → `a3`)

| condition | acc % | paired vs its control |
|---|---|---|
| S0 | 56.6 | n/a |
| S1 | 68.9 | n/a |
| **S1-OG** | **76.6** | +7.7 vs S1 |
| S1-OG+L | 74.5 | −2.1 vs S1-OG (+31/−52) |
| S1-OGpo | 72.1 | +3.2 vs S1 |
| S4R-OGF | 71.3 | −0.8 vs S1-OGpo (+47/−55) |
| S4R-OGFR | 70.5 | −1.6 vs S1-OGpo (+54/−70) |
| S4R-OG | 59.3 | grouped by option: misattribution, Qwen over-picks A |
| S4R-OGR | 60.0 | same layout |

Reading: the answer options are the missing retrieval key (+7.7 pp). The layered
labels, which helped on CultureBank and NormAd (norm and value questions), are
neutral to negative on KazCulture's item-lookup questions at every retrieval
quality tried (S1+L, S4R-PR, S1-OG+L, S4R-OGF). Test plan:
`KAZCULTURE_PREREGISTRATION_V2.md`.

## 4. Test results (preregistered, run once; `evaluation/analysis_v2/final_analysis_v2.md`)

Jobs 8756287/8/9 (Qwen3-32B) and 8756356 (OLMo), on prereg commit 6e72f21 and
analysis script 8ee8984, both pushed before the runs.

| comparison | Δ pp | 95% CI | McNemar p | Holm p |
|---|---|---|---|---|
| **primary: strict S1-OG vs S1** | **+0.00** (831 vs 831) | [−2.55, +2.62] | 1.0 | n/a |
| strict S1-OG+L vs S1-OG | +0.52 | [−1.20, +2.25] | 0.61 | 1.0 |
| strict S4R-OGF vs S1-OGpo | −0.45 | [−2.25, +1.35] | 0.68 | 1.0 |
| full S1-OG vs full S1 | **+4.20** (95.73 vs 91.53) | [+2.62, +5.77] | 1.5e-7 | 1e-6 |
| full S1-OG+L vs S1-OG | −0.60 | [−1.35, +0.15] | 0.19 | 0.56 |
| OLMo strict S1-OG vs S1 | −0.52 | [−3.30, +2.25] | 0.75 | n/a |

§5: criterion 1 not met, criterion 2 not met, criterion 4 = layers neutral/negative.
**v2 is negative on the preregistered primary.** The only significant gain is in the full setting
(test passages in the index), where option grounding finds the source passage more often.

### Why dev did not transfer (exploratory, after the test; no claim is changed by it)

Lexical proxies as in §1:

| | dev S1 | dev S1-OG | test-strict S1 | test-strict S1-OG |
|---|---|---|---|---|
| answer-bearing passage shown | 23% | 45% | 25% | 41% |
| distractor-bearing passage shown | 23% | 59% | 36% | **72%** |
| accuracy when answer-bearing shown | 91% | 86% | 87% | 79% |

The gold-retrieval gain transferred to test. But the test distractors are real Kazakh items
that the corpus also describes (86% of test distractors vs 79% of dev distractors are
grounded somewhere), so keying retrieval on the options doubles the evidence for
wrong options, and the two effects cancel. The dev MCQ distractors were written by Qwen
and were too easy to reject, so dev overestimated the net effect. A dev proxy for KazCulture
needs the dataset's own distractor style, for example distractors sampled from corpus items
of the same category.
