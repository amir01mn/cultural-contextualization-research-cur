# KazCulture results

**Branch:** `experiment/kazculture-qwen3-phase1`  
**Last updated:** 2026-11-02

This file is the compact results record for the KazCulture experiments on this branch. For the complete methodological history, implementation details, artifact map, and continuation plan, see [`../KAZCULTURE_BRANCH_HANDOFF.md`](../KAZCULTURE_BRANCH_HANDOFF.md).

## 1. Main benchmark results

All comparable QA results below use the frozen Qwen3-32B V2 scoring protocol:

- model: `Qwen/Qwen3-32B`
- revision: `9216db5781bf21249d130ec9da846c4624c16137`
- test questions: **1,334**
- non-thinking mode
- same question/options and A/B/C/D scoring
- only retrieved evidence changes between S0 and S1

| Condition | Correct | Accuracy | Change vs S0 |
|---|---:|---:|---:|
| S0 closed-book | 552 / 1,334 | **41.38%** | — |
| S1 strict exact-passage-disjoint | 831 / 1,334 | **62.29%** | **+20.91 pp** |
| S1 full | 1,221 / 1,334 | **91.53%** | **+50.15 pp** |
| Published reference target | ~861 / 1,334 | **64.54%** | reference only |

The published 64.54% number is a two-stage fine-tuned result and is not methodologically identical to our retrieval-augmented inference setting. It is used as a reference target, not as a directly comparable system.

---

## 2. Frozen S0 baseline

The first Qwen3 scoring implementation produced **24.96%**, but diagnosis showed severe answer-letter bias caused by the way A/B/C/D continuations were scored.

The corrected V2 protocol:

- forces `Answer:` as an assistant prefix
- scores context-valid `" A"`, `" B"`, `" C"`, `" D"` continuations
- verifies tokenizer behavior
- keeps the same model, questions, and no-retrieval setting

Final frozen closed-book baseline:

> **552 / 1,334 = 41.3793%**

Artifact:

- `data/outputs/kazculture_phase1_v2/predictions.jsonl`

This is the comparator used for all later paired statistics.

---

## 3. Strict S1 result

The strict S1 condition removes every train passage that exactly matches a test source passage before retrieval.

### Index

- full train passages: **6,749**
- exact-overlap train passages removed: **1,262**
- strict index passages: **5,487**

### Frozen retrieval configuration

- retrieval model: `BAAI/bge-m3`
- `top_k = 8`
- `min_similarity = 0.45`
- rendered evidence budget: **5,000 characters**
- train-only tuning questions: **400**
- prompt-visible train self-source recall: **71.25%**
- no test answers used for tuning

### QA result

> **831 / 1,334 = 62.2939%**

Difference from S0:

> **+20.9145 percentage points**

Paired statistics:

- paired bootstrap 95% delta: **+17.8411 to +24.1379 pp**
- exact McNemar p-value: **9.22e-37**
- S0 errors rescued by evidence: **394**
- S0-correct questions harmed by evidence: **115**
- net additional correct answers: **+279**
- no-evidence questions: **15**
- mean rendered retrieval units: **6.5127**

### Strict category accuracy

| Category | Correct / N | Accuracy |
|---|---:|---:|
| Clothing | 229 / 360 | **63.61%** |
| Craftsmanship | 5 / 11 | **45.45%** |
| Cuisine | 224 / 362 | **61.88%** |
| Household | 252 / 403 | **62.53%** |
| Livestock | 22 / 41 | **53.66%** |
| Musical instruments | 52 / 100 | **52.00%** |
| National games | 7 / 10 | **70.00%** |
| Superstitions | 4 / 4 | **100%** |
| Traditions and customs | 36 / 43 | **83.72%** |

### Interpretation

This is the primary generalization-oriented S1 result.

Even without the exact train source passage, related retrieved cultural evidence raises Qwen3 from **41.38% to 62.29%**.

Strict S1 is only about **30 net correct answers** below the 64.54% reference target.

---

## 4. Full S1 result

The full S1 index contains all **6,749** unique train passages, including passages that exactly overlap test sources.

The first attempt stalled on node `ng10706` and hit the three-hour wall without producing predictions. A retry excluding that node completed successfully:

- Slurm job: `7692307`
- node: `ng10803`
- runtime: **27:52**
- state: **COMPLETED 0:0**

### QA result

> **1,221 / 1,334 = 91.5292%**

Difference from S0:

> **+50.1499 percentage points**

Paired statistics:

- paired bootstrap 95% delta: **+47.4513 to +53.0735 pp**
- exact McNemar p-value: **2.81e-168**
- S0 errors rescued by evidence: **697**
- S0-correct questions harmed by evidence: **28**
- net additional correct answers: **+669**
- no-evidence questions: **3**
- mean rendered retrieval units: **6.9018**

### Exact-source retrieval

Among test questions whose exact source passage exists in the train index:

- overlap-eligible test questions: **1,263**
- exact source retrieved: **1,074**
- exact-source Recall@k: **85.04%**

This is the strongest evidence that the normal/full result is heavily influenced by train/test source overlap.

### Full category accuracy

| Category | Correct / N | Accuracy |
|---|---:|---:|
| Clothing | 336 / 360 | **93.33%** |
| Craftsmanship | 8 / 11 | **72.73%** |
| Cuisine | 345 / 362 | **95.30%** |
| Household | 367 / 403 | **91.07%** |
| Livestock | 25 / 41 | **60.98%** |
| Musical instruments | 86 / 100 | **86.00%** |
| National games | 10 / 10 | **100%** |
| Superstitions | 3 / 4 | **75.00%** |
| Traditions and customs | 41 / 43 | **95.35%** |

---

## 5. Full vs strict S1

The full-vs-strict gap is:

> **91.53% - 62.29% = 29.24 percentage points**

or:

> **390 additional correct questions in full S1**

This means two things must be reported separately.

### Finding A: retrieval helps even without the exact source

Strict S1 still gains **+20.91 points** over closed-book Qwen.

Therefore the RAG gain is not purely an exact-source lookup effect.

### Finding B: exact source overlap massively inflates the normal/full result

Full S1 gains another **+29.24 points** over strict S1.

Therefore **91.53% must not be described as unseen-source generalization**.

The correct framing is:

- **91.53%** = normal/full benchmark RAG with substantial train/test source overlap
- **62.29%** = exact-passage-disjoint retrieval condition and the more informative generalization-oriented baseline

---

## 5.1 Full-vs-strict question-level decomposition

The completed paired analysis over all 1,334 questions gives:

- **both correct:** 815
- **full correct / strict wrong:** 406
- **strict correct / full wrong:** 16
- **both wrong:** 97

This decomposition is important for the next phase:

- the **406 full-only** cases are the clearest evidence that access to the normal/full corpus, including exact-overlap sources, materially changes success;
- the **16 strict-only** cases show that additional full-corpus evidence can occasionally hurt;
- the **97 both-wrong** cases are the highest-priority test-side diagnostic set for understanding what raw S1 evidence still fails to solve.

### Accuracy conditional on exact-source retrieval

Among the 1,263 overlap-eligible test questions:

- exact source retrieved: **1,074**
- full S1 accuracy when exact source was retrieved: **97.86%**
- exact source available but missed: **189**
- full S1 accuracy when exact source was missed: **55.56%**

This is a very large conditional gap and confirms that exact-source retrieval is the dominant driver of the 91.53% full result.

Among the **71** test questions with no exact matching train passage:

- full S1 accuracy: **91.55%**

This small non-overlap subset is encouraging, but it should not be interpreted as a clean unseen-source benchmark. Exact passage disjointness does not remove paraphrases, related passages, or other near-duplicate cultural evidence. The strict 62.29% condition remains the more conservative generalization-oriented benchmark.

### Category full-minus-strict gaps

- Clothing: **+29.72 pp**
- Craftsmanship: **+27.27 pp**
- Cuisine: **+33.43 pp**
- Household: **+28.54 pp**
- Livestock: **+7.32 pp**
- Musical instruments: **+34.00 pp**
- National games: **+30.00 pp**
- Superstitions: **-25.00 pp** (n=4; do not over-interpret)
- Traditions and customs: **+11.63 pp**

The largest overlap-related gains occur in Cuisine, Musical instruments, Clothing, Household, and National games.

---

## 5.2 What remains inside the 97 both-wrong cases

The diagnostic export of the 97 questions missed by both full and strict S1 shows:

- Household: **32**
- Clothing: **19**
- Cuisine: **15**
- Musical instruments: **13**
- Livestock: **13**
- Craftsmanship: **3**
- Traditions and customs: **2**

Source-overlap breakdown:

- exact train-source overlap exists: **92/97**
- exact source was actually retrieved in full S1: **18/97**
- exact source existed but was missed by full retrieval: **74/97**
- no exact train-source overlap: **5/97**

This means most of the remaining full-S1 failures are primarily **retrieval failures** rather than cases where Qwen saw the exact source and still answered incorrectly. The **18** both-wrong cases where full S1 did retrieve the exact source are especially valuable for studying whether compact S2 factual structure or S4R structure can improve evidence use/reasoning. The **74** overlap-eligible cases where the exact source was missed are more directly a retrieval problem and should not automatically be attributed to the absence of KG reasoning.

The exported diagnostic file is:

- `data/kazculture_pipeline/evaluation/s1_both_wrong_97.csv`

This test-side analysis is for diagnosis. Do not repeatedly redesign the extraction prompt against these gold test outcomes and then report the same test set as untouched.

---

## 6. KG extraction results so far

A full final KG over all 6,749 passages has **not** yet been produced.

### V1 pilot

750 passages:

- covered passages: **382 / 750**
- candidate triples: **773**

Problems found:

- relation-direction errors
- source/title entities
- unsupported norms
- forced norm/value pathways
- object construction and game rules treated as normative behavior

### V2 first 150

- covered: **109 / 150 = 72.67%**
- candidate triples: **347**
- proposed full norm/value pathways: **40**
- truncations: **33**
- invalid JSON: **33**
- quote mismatches: **58**

### V2.1 first 150

- covered: **119 / 150 = 79.33%**
- candidate triples: **260**
- proposed pathways: **58**
- truncations: **0**
- invalid JSON: **2**
- invalid predicates: **21**
- quote mismatches: **60**
- orphan pathways: **31**

### Held-out V2.1

- passages: 150
- covered: **114**
- coverage: **76.0%**
- candidate triples: **247**
- proposed Schwartz pathways: **62**
- invalid quotes: **75**
- invalid predicates: **6**
- orphan pathways: **28**
- truncations: **0**
- invalid JSON: **0**

### Held-out V2.2

- passages: 150
- covered: **98**
- coverage: **65.33%**
- candidate triples: **162**
- proposed Schwartz pathways: **29**
- norm-only candidates: **3**
- invalid quotes: **30**
- invalid predicates: **1**
- orphan pathways: **0**
- truncations: **0**
- invalid JSON: **0**

Interpretation:

- V2.1 has better coverage.
- V2.2 is more conservative and mechanically cleaner.
- neither can be declared semantically superior until independent human review is completed.

---

## 7. Held-out retrieval diagnostics

On 300 train questions linked to 150 held-out train sources, raw S1 retrieval was very strong:

- Recall@1: **90.00%**
- Recall@3: **95.67%**
- Recall@5: **95.67%**
- Recall@8: **97.00%**
- MRR@8: **92.94%**

Structured representations were much sparser.

### V2.1 S2

- source availability: **77.33%**
- Recall@5: **68.33%**
- conditional Recall@5 where S2 exists: **88.36%**

### V2.2 S2

- source availability: **70.00%**
- Recall@5: **62.00%**
- conditional Recall@5: **88.57%**

### V2.1 S3

- source availability: **33.00%**
- Recall@5: **29.00%**
- conditional Recall@5: **87.88%**

### V2.2 S3

- source availability: **15.67%**
- Recall@5: **15.67%**
- conditional Recall@5: **100%**

Main conclusion:

> **S1 should be the retrieval backbone. Structured evidence should be attached as source-local enrichment rather than replacing S1.**

---

## 8. Source-first enrichment diagnostic

With S1 rankings frozen:

### V2.1

- S2 gold availability: **77.33%**
- S2 enrichment conditional on S1 hit@5: **77.70%**
- S4R gold availability: **33.00%**
- S4R enrichment conditional on S1 hit@5: **33.80%**

### V2.2

- S2 gold availability: **70.00%**
- S2 enrichment conditional on S1 hit@5: **71.08%**
- S4R gold availability: **15.67%**
- S4R enrichment conditional on S1 hit@5: **16.38%**

This is why the current working architecture is:

```
question
  -> S1 passage retrieval
  -> optional same-source S2 factual enrichment
  -> optional same-source S4R norm/value enrichment
  -> Qwen3-32B
```

---

## 9. Exploratory 17-question Qwen source-first pilot

A small overlap-only pilot selected 17 test questions whose exact sources were among the held-out 150 train sources.

Frozen S0 on this subset:

- **5/17 = 29.41%**

S1:

- **17/17 = 100%**

All S1+structured conditions also scored:

- **17/17 = 100%**

Therefore this pilot showed that S1 could rescue the subset but could not measure incremental S2/S4 value because S1 already saturated the task.

This is **not** the final benchmark result.

---

## 10. Human validation status

A 100-row blinded audit exists:

- 50 V2.1 candidates
- 50 V2.2 candidates

AI-assisted screening has been performed for prioritization and error diagnosis, but:

> **AI screening is not human validation.**

The official human fields must remain independent.

A Kazakh-literate human audit is still required before claiming:

- factual triple precision
- norm precision
- Schwartz-value precision
- final KG semantic quality

---

## 11. Current result hierarchy

### Strongest normal/full benchmark result

> **S1 full = 91.53%**

Use this to show how strong ordinary source retrieval can be when overlapping train sources are available.

### Strongest generalization-oriented result

> **S1 strict = 62.29%**

Use this as the primary baseline for testing whether structured cultural evidence adds value beyond related-passage retrieval.

### Current unanswered question

> Can S1 + S2 and/or S1 + S4R improve the **62.29% strict S1 baseline**?

This is the main next experiment.

---

## 12. Planned next result table

The final comparison should eventually include:

| Condition | Strict | Full |
|---|---:|---:|
| S0 | **41.38%** | **41.38%** |
| S1 | **62.29%** | **91.53%** |
| S2 standalone | TBD | TBD |
| S4R standalone | TBD | TBD |
| S1 + S2 | TBD | TBD |
| S1 + S4R | TBD | TBD |
| S1 + S2 + S4R | TBD | TBD |

Primary scientific comparison:

> **strict S1 vs strict S1 + structured evidence**

Secondary comparison:

> full S1 vs full S1 + structured evidence

---

## 13. Result artifacts

Frozen baseline:

- `data/outputs/kazculture_phase1_v2/predictions.jsonl`

Strict S1:

- `data/kazculture_pipeline/evaluation/s1_strict/S1.jsonl`
- `data/kazculture_pipeline/evaluation/s1_strict/S1_report.json`
- `data/kazculture_pipeline/evaluation/s1_strict/comparison.json`

Full S1:

- `data/kazculture_pipeline/evaluation/s1_full/S1.jsonl`
- `data/kazculture_pipeline/evaluation/s1_full/S1_report.json`
- `data/kazculture_pipeline/evaluation/s1_full/comparison.json`

Frozen retrieval config:

- `data/kazculture_pipeline/s1_full_dev_thresholds_v3.json`

Full-vs-strict analysis helper:

- `scripts/analyze_kazculture_s1_full_vs_strict.py`

Full KazCulture handoff:

- `docs/KAZCULTURE_BRANCH_HANDOFF.md`

