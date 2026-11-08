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
