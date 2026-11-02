# KazCulture partner takeover — start here

**Branch:** `experiment/kazculture-qwen3-phase1`  
**Purpose:** shortest operational handoff for the next person working on KazCulture.  
**Detailed history:** [`KAZCULTURE_BRANCH_HANDOFF.md`](KAZCULTURE_BRANCH_HANDOFF.md)  
**Compact results:** [`results/KAZCULTURE_RESULTS.md`](results/KAZCULTURE_RESULTS.md)

---

## 1. Current status in one minute

The KazCulture work has reached the point where the **S0/S1 baseline and retrieval story are complete**, while the **structured KG phase still needs human validation and full-scale execution**.

Frozen Qwen3-32B V2 baseline:

- S0 closed-book: **552/1334 = 41.38%**

Full-scale raw-passage retrieval:

- S1 strict exact-passage-disjoint: **831/1334 = 62.29%**
- S1 full: **1221/1334 = 91.53%**

The strict condition removes **1,262 unique train passages** that exactly overlap test sources. The 29.24-point full-vs-strict gap shows that exact source overlap has a very large effect, while the strict +20.91-point improvement over S0 shows that related cultural evidence is still strongly useful without the exact source.

The current research question is therefore:

> **Can source-grounded structured evidence (S2/S4R) improve the strong 62.29% strict S1 baseline?**

---

## 2. First commands to run

On Narval:

~~~bash
cd ~/projects/def-enaskt/divc/kazculture-restart
source .venv/bin/activate
git checkout experiment/kazculture-qwen3-phase1
git pull --ff-only
git status
~~~

Run the relevant regression tests before modifying the pipeline:

~~~bash
python -m pytest -q   tests/test_kazculture_phase1_v2.py   tests/test_kazculture_pipeline.py   tests/test_kazculture_kg_v2.py   tests/test_kazculture_kg_v21.py   tests/test_kazculture_kg_v22.py   tests/test_kazculture_retrieval_diagnostic.py
~~~

---

## 3. What is already finished

### Baseline/scoring

- Original V1 scoring diagnostic: **24.96%**, rejected as the scientific baseline because of answer-letter scoring bias.
- Corrected frozen V2 S0 baseline: **41.38%**.
- Frozen Qwen revision:
  `9216db5781bf21249d130ec9da846c4624c16137`.

### Corpus

- 14,803 train QA rows.
- 6,749 unique train source passages.
- 1,334 test questions.
- train/test source-overlap manifest built and frozen.

### KG development

Completed pilot/refinement stages:

- V1
- V2
- V2.1
- V2.2
- matched held-out 150-source comparison
- blinded 100-row human-review package

### Retrieval

- BGE-M3 staged and tested.
- S1 identified as the strongest retrieval backbone.
- source-first design adopted: retrieve S1, then optionally attach same-source S2/S4R.
- S1 full and strict indexes complete.
- train-only retrieval tuning complete and frozen.

Frozen S1 config:

- `top_k = 8`
- `min_similarity = 0.45`
- evidence budget = 5,000 chars
- train-dev visible source recall = **71.25%**

### Full S1 benchmark

S1 strict:

- accuracy **62.29%**
- +20.91 pp vs S0
- 394 S0 errors rescued
- 115 S0-correct questions harmed

S1 full:

- accuracy **91.53%**
- +50.15 pp vs S0
- exact source recalled on 1,074 / 1,263 overlap-eligible test questions
- 697 S0 errors rescued
- 28 S0-correct questions harmed

Full-vs-strict paired decomposition:

- both correct: **815**
- full only: **406**
- strict only: **16**
- both wrong: **97**

Among the 97 both-wrong cases:

- exact train-source overlap exists: **92**
- exact source actually retrieved in full: **18**
- exact source existed but full retrieval missed it: **74**
- no exact train-source overlap: **5**

This tells us most of the remaining full-S1 failures are retrieval failures, not simply failures to reason over an exact source.

---

## 4. Human audit: the immediate blocker

The original blind-review file is still **completely unannotated**:

- `data/kazculture_pipeline/heldout/blind_review_100.csv`
- 100 rows
- all official human-review fields blank

The unblinding key is separate:

- `data/kazculture_pipeline/heldout/blind_review_100_key.json`

**Do not give the key to the reviewer.**

A reviewer should ideally be comfortable reading the Kazakh source passage and should fill:

- `factual_triple_correct_0_or_1` — required for all 100 rows
- `norm_warranted_by_source_0_or_1` — required for all 100 rows
- `proposed_norm_correct_0_or_1_if_present` — only when a norm was proposed
- `proposed_values_correct_0_or_1_if_present` — only when values were proposed
- `reviewer_notes` — optional but strongly encouraged

Important:

- AI-assisted screening already exists, but it is **not human validation**.
- Never copy AI screening labels into the official human columns.
- Preserve the blank original audit and work on a reviewed copy.

After the reviewer returns a completed file:

~~~bash
python scripts/score_kazculture_heldout_audit.py   --audit data/kazculture_pipeline/heldout/blind_review_100_reviewed.csv   --mapping data/kazculture_pipeline/heldout/blind_review_100_key.json   --out data/kazculture_pipeline/heldout/blind_review_100_scored.json

cat data/kazculture_pipeline/heldout/blind_review_100_scored.json
~~~

The scorer intentionally refuses to run while required human fields are blank.

---

## 5. How to choose the final extractor

Do this **after** the human audit.

Current tradeoff:

### V2.1

- held-out coverage: 114/150
- triples: 247
- proposed pathways: 62
- more coverage
- more structural/pathway errors

### V2.2

- held-out coverage: 98/150
- triples: 162
- proposed pathways: 29
- much more conservative
- fewer invalid quotes/predicates
- zero orphan pathways
- much lower coverage

Decision rule:

1. If V2.1 retains acceptable human factual/pathway precision, scale V2.1.
2. If V2.2 is substantially more precise and coverage remains useful, scale V2.2.
3. Create V2.3 only if the human audit identifies a concrete failure that neither existing version handles acceptably.

Do **not** create a new version merely to chase test accuracy.

---

## 6. Before full 6,749-passage KG extraction

Run a larger **train-only** source-first QA experiment with actual headroom.

The old 17-question overlap pilot was not discriminative because S1 scored 17/17.

Use train-development questions where S1 is imperfect and compare:

- S1
- S1 + S2
- S1 + S4R
- S1 + S2 + S4R

This should be done before spending GPU time on a full structured extraction.

The goal is to verify that structured evidence has a plausible downstream signal without adapting the extraction prompt to gold test outcomes.

---

## 7. Then scale the final KG

Once the extractor is frozen:

1. extract over all **6,749 unique train passages**
2. write to a new versioned directory
3. never overwrite V1/V2/V2.1/V2.2 artifacts
4. build structured retrieval units
5. preserve passage/triple/path provenance
6. build strict and full versions as required

Primary source-first architecture:

~~~text
question
  -> S1 raw-passage retrieval
  -> attach same-source S2 factual evidence when available
  -> attach same-source S4R norm/value evidence when available
  -> Qwen3-32B frozen V2 scorer
~~~

Structured evidence should not silently rerank/remove the S1 source in the primary source-first experiment.

---

## 8. Final experiments to run

### Primary: strict exact-passage-disjoint

| Condition | Accuracy |
|---|---:|
| S0 | **41.38%** |
| S1 | **62.29%** |
| S1 + S2 | TBD |
| S1 + S4R | TBD |
| S1 + S2 + S4R | TBD |

The primary goal is to determine whether structured evidence adds a statistically meaningful gain over **62.29%**.

The published 64.54% result corresponds to about 861 correct answers, so strict S1 is about **30 net correct questions short**.

### Secondary: full

| Condition | Accuracy |
|---|---:|
| S0 | **41.38%** |
| S1 | **91.53%** |
| S1 + S2 | TBD |
| S1 + S4R | TBD |
| S1 + S2 + S4R | TBD |

Full S1 is already close to saturation and is strongly affected by exact source overlap. Treat it as a secondary benchmark result, not unseen-source generalization.

---

## 9. Important files

### Baseline

- `data/outputs/kazculture_phase1_v2/predictions.jsonl`

### Corpus

- `data/kazculture_pipeline/corpus/passages.jsonl`
- `data/kazculture_pipeline/corpus/train_dev_qa.jsonl`
- `data/kazculture_pipeline/corpus/test_overlap.json`

### Held-out audit

- `data/kazculture_pipeline/heldout/blind_review_100.csv`
- `data/kazculture_pipeline/heldout/blind_review_100_key.json`

### S1 indexes

- `data/kazculture_pipeline/index/s1_full/`
- `data/kazculture_pipeline/index/s1_strict/`

### S1 results

- `data/kazculture_pipeline/evaluation/s1_full/`
- `data/kazculture_pipeline/evaluation/s1_strict/`
- `data/kazculture_pipeline/evaluation/s1_full_vs_strict_analysis.json`
- `data/kazculture_pipeline/evaluation/s1_full_vs_strict_cases.jsonl`
- `data/kazculture_pipeline/evaluation/s1_both_wrong_97.csv`

The test-case CSV/JSONL files above live on Narval and should not be committed to Git.

### Frozen retrieval config

- `data/kazculture_pipeline/s1_full_dev_thresholds_v3.json`

---

## 10. Rules that should not be broken

- Never use test answers to tune retrieval thresholds.
- Never use test answers in KG extraction.
- Never call AI screening human validation.
- Never fill human audit fields automatically.
- Never treat quote presence as proof of semantic correctness.
- Never force every fact into a norm/value pathway.
- Never overwrite previous versioned KG outputs.
- Never report 91.53% as source-disjoint or unseen-source generalization.
- Always keep the Qwen scoring protocol fixed when comparing evidence systems.
- Use strict S1 = **62.29%** as the primary structured-evidence baseline.
- If a method change is motivated by gold test failures, do not treat the same test set as untouched validation afterward.

---

## 11. Recommended task order

1. **Get the 100-row human audit completed.**
2. Score the audit and compare V2.1 vs V2.2.
3. Freeze the final extractor (or justify one final V2.3).
4. Run a larger train-only S1/S2/S4 source-first QA development experiment.
5. Extract the frozen structured representation over all 6,749 train passages.
6. Build the final structured indexes.
7. Run strict S1 + S2 / S4R / S2+S4R on all 1,334 test questions.
8. Run paired statistics and category/error analysis.
9. Run corresponding full conditions as secondary results.
10. Finish paper-quality validation and reporting.

---

## 12. Where to read more

For implementation history, exact metrics, rationale, failures, artifacts, and detailed next steps:

- [`KAZCULTURE_BRANCH_HANDOFF.md`](KAZCULTURE_BRANCH_HANDOFF.md)

For the compact experimental record:

- [`results/KAZCULTURE_RESULTS.md`](results/KAZCULTURE_RESULTS.md)

For stage-specific older design notes:

- `KAZCULTURE_PHASE1.md`
- `KAZCULTURE_PHASES_2_TO_5.md`
- `KAZCULTURE_KG_V2_PILOT.md`
- `KAZCULTURE_KG_V21_REFINEMENT.md`
- `KAZCULTURE_PHASE2_HELDOUT_QUALITY_GATE.md`
- `KAZCULTURE_PHASE2_RETRIEVAL_DIAGNOSTIC.md`

If an older status line conflicts with this takeover document, prefer this document plus the actual current manifests/results on Narval.
