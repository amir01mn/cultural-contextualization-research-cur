# KazCulture branch handoff: complete research history, current state, and next steps

**Branch:** `experiment/kazculture-qwen3-phase1`  
**Repository:** `amir01mn/cultural-contextualization-research-cur`  
**Working checkout on Narval:** `~/projects/def-enaskt/divc/kazculture-restart`  
**Last updated:** 2026-11-02

This document is the handoff for **all KazCulture work on this branch**. It is intended to let another researcher understand the scientific goal, reproduce the current state, know why each design decision was made, find every important artifact, and continue the work without reconstructing the project from chat history.

The most important rule when reading this document is to distinguish four kinds of evidence:

1. **Frozen benchmark results**: full Qwen3-32B evaluations under a fixed scoring protocol.
2. **Engineering/extraction diagnostics**: coverage, quote validity, parse failures, orphan pathways, etc. These do not establish semantic correctness.
3. **Exploratory retrieval/QA diagnostics**: useful for choosing architecture, but not necessarily final benchmark evidence.
4. **Human validation**: still incomplete. AI-assisted screening is not a replacement for an independent Kazakh-literate human audit.

---

## 1. Overall research goal

The KazCulture extension asks whether the cultural knowledge representation and retrieval ideas developed in the CultureBank project can improve Qwen3-32B on the KazCulture multiple-choice benchmark.

The reference numbers motivating the work are:

- published Qwen3-32B zero-shot result: approximately **39.51%**
- published two-stage fine-tuned result: approximately **64.54%**
- our frozen independent Qwen3-32B closed-book baseline after fixing the scoring protocol: **552/1334 = 41.3793%**

The practical target has therefore been to move from roughly **41.38% toward or beyond 64.54%**, while maintaining a defensible evaluation protocol and determining whether structured cultural knowledge provides value beyond ordinary passage retrieval.

The work has evolved from a graph-first design into a **source-first design**:

```
question
   |
   v
retrieve raw Kazakh source passage(s) with BGE-M3   <-- S1 backbone
   |
   +---- optionally attach source-grounded factual structure (S2)
   |
   +---- optionally attach source-grounded norm/value structure (S4R)
   |
   v
Qwen3-32B answer scoring
```

This change was evidence-driven. Raw passages retrieve source information much more reliably than the sparse KG representations, while structured evidence appears potentially useful as an enrichment layer after the source has already been found.

---

## 2. Frozen models and environment

### Qwen answering/extraction model

All comparable KazCulture Qwen runs use:

- model: `Qwen/Qwen3-32B`
- revision: `9216db5781bf21249d130ec9da846c4624c16137`
- non-thinking mode
- deterministic scoring/extraction settings as appropriate
- no test-answer content in retrieval or KG construction

Do not silently change the Qwen revision for a continuation experiment.

### Retrieval model

The current multilingual retrieval model is:

- `BAAI/bge-m3`
- cached Narval snapshot used in the diagnostic work:
  `5617a9f61b028005a4858fdac845db406aefb181`

BGE-M3 was selected because the evidence and questions are primarily Kazakh/multilingual. It has now been empirically tested on the KazCulture train-source retrieval problem.

### Narval

Typical project settings:

- project directory: `~/projects/def-enaskt/divc/kazculture-restart`
- virtual environment: `.venv`
- GPU account: `def-enaskt_gpu`
- CPU account: `def-enaskt_cpu`
- common GPU partition: `gpubase_bygpu_b1`
- common CPU partition: `cpubase_bycore_b1`
- B1 GPU jobs have a 3-hour wall-time limit
- Qwen3-32B jobs normally request 2 x A100 GPUs
- compute-node jobs use the offline HF cache configured through `slurm/env_hf.sh`

Normal setup:

~~~bash
cd ~/projects/def-enaskt/divc/kazculture-restart
source .venv/bin/activate
git checkout experiment/kazculture-qwen3-phase1
git pull --ff-only
~~~

The dataset and model weights are intentionally **not committed to Git**.

---

## 3. Dataset staging and train/test separation

KazCulture contains:

- **14,803 train QA rows**
- **1,334 test questions**
- **6,749 unique train source passages** after exact normalized passage deduplication

The central data-design decision is that **questions and answers are separated from source evidence**.

The corpus staging script is:

- `scripts/prepare_kazculture_corpus.py`

It produces:

- `data/kazculture_pipeline/corpus/passages.jsonl`  
  6,749 unique train passages used for retrieval/KG construction.
- `data/kazculture_pipeline/corpus/pilot_passages.jsonl`  
  deterministic 750-passage pilot.
- `data/kazculture_pipeline/corpus/train_dev_qa.jsonl`  
  train questions stored separately for retrieval diagnostics/tuning.
- `data/kazculture_pipeline/corpus/test_overlap.json`  
  exact source-passage overlap manifest.
- `data/kazculture_pipeline/corpus/manifest.json`

### Leakage rule

KG extraction receives train source text, source/category metadata required by the protocol, and no test answers. It must not receive:

- test `question`
- test `answer`
- test `answer_label`
- train QA answers as graph evidence
- any evaluation-only label as extraction input

The train QA file exists only for train-only retrieval calibration and diagnostics.

### Important overlap finding

KazCulture contains extensive exact source overlap between train and test. The current strict S1 index removes **1,262 unique train passage IDs** that exactly match test passages:

- full S1 index: **6,749 passages**
- strict exact-passage-disjoint S1 index: **5,487 passages**
- excluded from strict index: **1,262 unique train passages**

Do not confuse 1,262 with a count of test questions. It is a count of unique train passage IDs removed from the index. Use `test_overlap.json` for the exact row-level overlap accounting.

This overlap is why all important RAG results should be reported in both a normal/full setting and an exact-passage-disjoint setting.

---

## 4. Phase 1: establishing the Qwen3 closed-book baseline

Relevant files:

- `docs/KAZCULTURE_PHASE1.md`
- `scripts/stage_kazculture_phase1.py`
- `scripts/eval_kazculture_phase1.py`
- `scripts/eval_kazculture_phase1_v2.py`
- `scripts/diagnose_kazculture_phase1.py`
- `src/evaluation/kazculture_hf_scorer.py`
- `src/evaluation/kazculture_hf_scorer_v2.py`
- `src/evaluation/kazculture_phase1.py`
- `tests/test_kazculture_phase1.py`
- `tests/test_kazculture_phase1_v2.py`

### 4.1 Initial V1 scoring result

The first full independent Qwen3-32B run completed all 1,334 examples and scored:

- **333/1334 = 24.9625%**
- job: `7466192`

This was much lower than the published ~39.51%.

The predictions were pathologically label-biased:

- A: 85
- B: 4
- C: 502
- D: 743

The gold labels were essentially balanced:

- A: 334
- B: 334
- C: 333
- D: 333

This indicated a scoring/serialization problem rather than a credible model capability measurement.

### 4.2 Why V1 was wrong

V1 placed `Answer:` in the user message and then scored bare uppercase letter tokens at the start of an assistant response.

The model's token distribution showed that this context was not treating A/B/C/D symmetrically. In particular, the candidate-token boundaries and assistant-message continuation context produced a severe letter bias.

We did **not** overwrite the old result. V1 remains a diagnostic artifact.

### 4.3 V2 scoring correction

V2 introduced a separately versioned protocol:

- force `Answer:` as an assistant prefix
- score context-valid continuations `" A"`, `" B"`, `" C"`, `" D"`
- verify tokenizer prefix invariance
- score one next-token log probability for each candidate
- keep Qwen3 non-thinking
- keep the same test set and frozen model revision
- keep retrieval disabled
- preserve V1 outputs

The full V2 result became the frozen baseline:

> **552 / 1,334 = 41.3793%**

This is the baseline used by every later paired retrieval evaluation.

Artifact:

- `data/outputs/kazculture_phase1_v2/predictions.jsonl`

Do not regenerate or modify these predictions when comparing S1/S2/S4. They are the frozen S0 comparator.

### 4.4 Interpretation

Our 41.38% is close to the published ~39.51% zero-shot result but is not claimed to be an exact reproduction of the paper's hidden prompt serialization. It is an internally consistent frozen protocol.

The published 64.54% result is used as a **reference target**, not as proof that our retrieval method is directly equivalent to the paper's fine-tuning method.

---

## 5. Initial KG design: V1 pilot on 750 passages

Relevant files:

- `src/construction/kazculture_kg.py`
- `src/construction/kazculture_chunking.py`
- `scripts/extract_kazculture_kg.py`
- `slurm/kazculture_kg_extract.sh`
- `tests/test_kazculture_pipeline.py`

### 5.1 Why start with 750 passages

A full build would require extraction over all 6,749 unique passages and would be expensive. We therefore started with a deterministic 750-passage pilot to answer:

- can Qwen3 produce source-grounded triples?
- can passages be processed within the model context limit?
- how much source coverage do we get?
- are proposed L2 norms and L3 values plausible enough to justify scaling?

### 5.2 First runtime failure and chunking fix

The first pilot extraction failed after **264/750 passages** because one rendered input exceeded Qwen's 4,096-token context.

This was not an OOM.

The pipeline was changed to:

- validate the entire rendered chat prompt with the real tokenizer
- reserve output/safety tokens
- split long passages into overlapping **verbatim source substrings**
- ensure all original source text is represented
- retain chunk/source-span provenance
- validate evidence quotes against the precise substring shown to the model
- resume the existing checkpoint rather than discarding completed work

This is implemented in `src/construction/kazculture_chunking.py`.

### 5.3 V1 pilot result

Final V1 750-passage pilot:

- passages: 750
- passages with accepted candidate triples: **382/750**
- candidate triples: **773**

This demonstrated that the extraction pipeline functioned, but manual/AI review exposed substantial semantic problems.

### 5.4 Main V1 quality problems

Examples of recurring errors included:

- wrong relation direction
- a material of a component being attributed to the whole object
- source/title-like text becoming an entity
- actor/object reversal
- facts converted into social norms without warrant
- cultural values inferred from weak lexical cues
- object construction or game rules treated as normative behavior
- descriptive facts being forced into an L1->L2->L3 pathway

This motivated a redesign rather than scaling V1.

---

## 6. KG V2: separate factual extraction from normative interpretation

Relevant files:

- `docs/KAZCULTURE_KG_V2_PILOT.md`
- `src/construction/kazculture_kg_v2.py`
- `scripts/extract_kazculture_kg_v2.py`
- `scripts/compare_kazculture_kg_v2.py`
- `scripts/review_kazculture_kg_v2.py`
- `slurm/kazculture_kg_v2_pilot.sh`
- `tests/test_kazculture_kg_v2.py`

### 6.1 Why V2

The V1 design effectively encouraged every extracted fact to become part of a cultural norm/value chain. This was scientifically wrong.

A passage may contain a perfectly useful cultural fact without expressing:

- a social obligation
- a taboo
- a social expectation
- a value-level motivation

V2 therefore separated:

1. factual triple correctness
2. whether a norm is warranted at all
3. whether a proposed norm is correct
4. whether a proposed Schwartz value is correct

### 6.2 V2 schema and constraints

V2 introduced an explicit predicate vocabulary such as:

- `IS_A`
- `MADE_FROM`
- `USED_FOR`
- `PART_OF`
- `HAS_PROPERTY`
- `WORN_BY`
- `LOCATED_IN`
- `CREATED_BY`
- `USED_IN`
- `ASSOCIATED_WITH`
- `PERFORMED_DURING`
- `SYMBOLIZES`
- `PRACTICED_BY`
- `REQUIRES`
- `OTHER_SUPPORTED_RELATION`

Every accepted factual candidate required a verbatim quote from the model-visible source.

A norm/value pathway required separate normative evidence rather than being automatically inferred from a material/object fact.

### 6.3 V2 matched 150-passage result

On the first 150 pilot passages:

- covered passages: **109/150 = 72.67%**
- candidate triples: **347**
- proposed full norm/value pathways: **40**
- output-length truncations: **33**
- invalid JSON segments: **33**
- quote mismatches/rejections: **58**

This increased quantity but showed that the output format and generation budget were still fragile.

### 6.4 Important interpretation

Quote matching is only a provenance check. It proves the cited text exists in the input. It does **not** prove that the extracted subject/predicate/object relation is semantically correct.

---

## 7. KG V2.1: improve relation precision and generation reliability

Relevant files:

- `docs/KAZCULTURE_KG_V21_REFINEMENT.md`
- `src/construction/kazculture_kg_v21.py`
- `scripts/extract_kazculture_kg_v21.py`
- `scripts/compare_kazculture_kg_versions.py`
- `scripts/export_kazculture_pathway_audit.py`
- `slurm/kazculture_kg_v21_pilot.sh`
- `tests/test_kazculture_kg_v21.py`

### 7.1 Why V2.1

V2 review showed:

- relation-direction mistakes
- homonym/entity conflation
- gift/material confusion
- output truncation
- invalid JSON
- weak norm/value proposals

V2.1 therefore tightened the prompt/schema and increased output budget.

Key changes included:

- maximum 3 triples rather than 5
- shorter exact evidence quotes
- stricter relation direction
- explicit social-rule kind
- stronger warnings against homonym conflation
- output budget increased from 768 to 1,536 tokens
- full token-budget preflight against the 4,096-token model context

Because several variables changed together, V2.1 is an engineering refinement, not a one-factor scientific ablation.

### 7.2 V2.1 first-150 result

On the first 150 development passages:

- covered passages: **119/150 = 79.33%**
- candidate triples: **260**
- proposed full pathways: **58**
- truncations: **0**
- JSON parse failures: **2**
- invalid predicates: **21**
- quote mismatches: **60**
- orphan pathways: **31**
- duplicates observed: **2**

This was operationally much cleaner than V2 but still contained semantic/pathway problems.

### 7.3 Development-set warning

The first 150 passages influenced prompt revisions, so they cannot be treated as a blind validation set.

That led to creation of a separate held-out-from-prompt-development 150-passage set.

---

## 8. KG V2.2 and the frozen 150-passage held-out quality gate

Relevant files:

- `docs/KAZCULTURE_PHASE2_HELDOUT_QUALITY_GATE.md`
- `src/construction/kazculture_kg_v22.py`
- `scripts/prepare_kazculture_heldout.py`
- `scripts/extract_kazculture_kg_v22.py`
- `scripts/compare_kazculture_kg_holdout.py`
- `scripts/build_kazculture_heldout_audit.py`
- `scripts/score_kazculture_heldout_audit.py`
- `slurm/kazculture_kg_holdout.sh`
- `tests/test_kazculture_kg_v22.py`

### 8.1 Held-out source selection

The held-out set contains 150 train-only source passages selected after:

- excluding the first 150 development passages
- excluding passage IDs touched by earlier audit samples

Artifact:

- `data/kazculture_pipeline/heldout/heldout_train_passages_150.jsonl`

The set is held out from prompt-development decisions. It is **not** an external benchmark and is not a replacement for human validation.

### 8.2 Why V2.2

V2.1 still had a structural problem: pathways referenced factual triples by index. If a factual triple was rejected later, its pathway could become an orphan.

V2.2 nests any norm directly inside its own extracted relation. It also makes the separation between factual-only, norm-only, and full norm/value evidence explicit.

V2.2 principles:

- max 3 triples
- exact contiguous source quotes
- no orphan `triple_index`
- no norms on clearly descriptive predicates such as `MADE_FROM`, `IS_A`, `PART_OF`, `HAS_PROPERTY`, `LOCATED_IN`, `CREATED_BY`
- a norm may exist without a Schwartz value
- game mechanics are not automatically social norms
- object construction is not automatically a social norm
- food recipes are not automatically social norms
- kinship definitions are not automatically social norms
- L3 values remain provisional interpretations

### 8.3 Matched held-out V2.1 vs V2.2 results

**V2.1 held-out**

- passages: 150
- covered: **114**
- coverage: **76.0%**
- candidate triples: **247**
- proposed Schwartz pathways: **62**
- generation truncations: **0**
- invalid JSON: **0**
- invalid quotes: **75**
- invalid predicates: **6**
- orphan pathways: **28**

**V2.2 held-out**

- passages: 150
- covered: **98**
- coverage: **65.33%**
- candidate triples: **162**
- proposed Schwartz pathways: **29**
- norm-only candidates: **3**
- generation truncations: **0**
- invalid JSON: **0**
- invalid quotes: **30**
- invalid predicates: **1**
- orphan pathways: **0**

### 8.4 Interpretation

V2.1 has substantially better coverage.

V2.2 is substantially more conservative and structurally cleaner:

- fewer quote failures
- almost no invalid predicates
- zero orphan pathways

This is a precision/coverage tradeoff. We have **not** established that V2.2 is semantically more accurate until the human audit is completed.

Do not choose the final full-corpus extractor solely because V2.2 has cleaner mechanical diagnostics.

---

## 9. Blinded 100-row audit and AI-assisted screening

The blinded audit builder creates:

- `data/kazculture_pipeline/heldout/blind_review_100.csv`
- `data/kazculture_pipeline/heldout/blind_review_100_key.json`

It samples:

- 50 V2.1 triples
- 50 V2.2 triples
- shuffles them
- hides method identity from the reviewer

Official human fields include:

- `factual_triple_correct_0_or_1`
- `norm_warranted_by_source_0_or_1`
- `proposed_norm_correct_0_or_1_if_present`
- `proposed_values_correct_0_or_1_if_present`
- reviewer notes

### 9.1 AI-assisted screening that has already been done

A separate AI-only review was performed to identify likely error patterns and prioritize human review.

Across the 100 rows, the AI screening categorized factual candidates approximately as:

- 45 appears supported
- 23 needs context review
- 32 likely incorrect

Among 23 proposed norms:

- 11 appears supported
- 9 needs context
- 3 likely incorrect

Among 22 proposed Schwartz-value assignments:

- 14 require conceptual review
- 8 likely unsupported
- 0 were treated as confidently human-certified

Priority labels:

- high: 38
- medium: 22
- routine: 40

These are **not human labels** and must never be copied into the official human-score columns.

### 9.2 Common semantic failure patterns found in screening

Recurring problems included:

- reversed actor/object direction
- `USED_FOR` assigned to a person or recipient instead of an artifact/process
- `PRACTICED_BY` direction reversed
- `CREATED_BY` used for a participant rather than a creator
- component relations treated as `MADE_FROM`
- recipient relations forced into `USED_FOR`
- symbolic relations confused with definitions
- legends/beliefs presented as unqualified facts
- placeholder entity text such as "Kazakh entity or behavior"
- game rules treated as norms
- descriptions containing "traditional" automatically mapped to Schwartz `tradition`
- foreign/regional customs generalized to all Kazakhs
- value labels inferred without separate motivational evidence

This screening materially informed our interpretation of the KG pilots but does not satisfy the human-validation gate.

---

## 10. Retrieval representation definitions

The retrieval implementation is centered in:

- `src/retrieval/kazculture_index.py`

Embedding model:

- `BAAI/bge-m3`

The current representation names are:

### S1 — raw source passage

A direct KazCulture train passage.

This is now the retrieval backbone.

### S2 — factual structured evidence

A source-grounded factual triple plus quote.

S2 is meant to test whether compact explicit factual structure helps relative to raw text.

### S3 — layered triple/path representation

A triple with the proposed norm/value chain where that structure actually exists.

For V2+ extraction, factual-only triples do not automatically become S3 units.

### S4F — source-grounded frame

Frame-level structured evidence local to a source.

### S4R — routed layered representation

Proposed norm/value structured evidence.

Important: V2.1/V2.2 S4R is not exactly the same algorithm as the earlier CultureBank v3 S4R implementation. Avoid treating names as proof of algorithmic identity.

### S4R_combined

Infrastructure also exists for combined KazCulture/CultureBank evidence, but this is not yet the main KazCulture benchmark condition.

---

## 11. Held-out BGE-M3 retrieval diagnostic

Relevant files:

- `docs/KAZCULTURE_PHASE2_RETRIEVAL_DIAGNOSTIC.md`
- `scripts/build_kazculture_holdout_index.py`
- `scripts/probe_kazculture_holdout_retrieval.py`
- `slurm/kazculture_kg_retrieval_diagnostic.sh`
- `tests/test_kazculture_retrieval_diagnostic.py`

Two matched 150-source indexes were built:

- `data/kazculture_pipeline/index/heldout_diagnostic/v21_diagnostic/`
- `data/kazculture_pipeline/index/heldout_diagnostic/v22_diagnostic/`

### 11.1 Index sizes

V2.1:

- S1: 150
- S2: 247
- S3: 62
- S4F: 44
- S4R: 62

V2.2:

- S1: 150
- S2: 162
- S3: 29
- S4F: 20
- S4R: 29

### 11.2 Train-only self-source retrieval results

The diagnostic selected 300 train QA queries across 150 known source passages.

Raw S1 was identical for both methods:

- source availability: 1.0
- Recall@1: **0.9000**
- Recall@3: **0.9567**
- Recall@5: **0.9567**
- Recall@8: **0.9700**
- MRR@8: **0.9294**

This was the first strong evidence that raw passage retrieval should be the backbone.

V2.1 S2:

- source availability: **0.7733**
- Recall@1: **0.6000**
- Recall@3: **0.6633**
- Recall@5: **0.6833**
- Recall@8: **0.7100**
- conditional Recall@5 where S2 exists: **0.8836**

V2.2 S2:

- availability: **0.7000**
- Recall@1: **0.5067**
- Recall@3: **0.6033**
- Recall@5: **0.6200**
- Recall@8: **0.6367**
- conditional Recall@5: **0.8857**

V2.1 S3:

- availability: **0.3300**
- Recall@5: **0.2900**
- conditional Recall@5: **0.8788**

V2.2 S3:

- availability: **0.1567**
- Recall@5: **0.1567**
- conditional Recall@5: **1.0000**

S4F/S4R showed the same general pattern: V2.2 was highly retrievable when it existed, but much too sparse to replace S1.

### 11.3 Main conclusion from this diagnostic

**Do not use the structured KG as the primary retriever.**

The right architectural hypothesis became:

> retrieve with S1, then attach S2/S4 evidence from the retrieved source.

This is the origin of the source-first design.

---

## 12. Source-first enrichment diagnostic

Relevant file:

- `scripts/probe_kazculture_source_enrichment.py`

Output:

- `data/kazculture_pipeline/retrieval_diagnostic/heldout_source_first_enrichment.json`

Protocol:

- freeze S1 rankings
- never allow KG evidence to remove/rerank the S1 source
- ask how often S2/S3/S4F/S4R evidence exists for the source already retrieved by S1

Results on the 300-query held-out train diagnostic:

S1:

- Recall@1: **0.9000**
- Recall@5: **0.9567**

V2.1 S2:

- indexed source count: 114
- gold structured availability: **0.7733**
- enrichment conditional on S1 hit@5: **0.7770**

V2.2 S2:

- indexed source count: 98
- gold structured availability: **0.7000**
- enrichment conditional on S1 hit@5: **0.7108**

V2.1 S4R:

- indexed source count: 44
- gold availability: **0.3300**
- enrichment conditional on S1 hit@5: **0.3380**

V2.2 S4R:

- indexed source count: 20
- gold availability: **0.1567**
- enrichment conditional on S1 hit@5: **0.1638**

Interpretation:

- S2 can enrich a substantial fraction of sources retrieved by S1.
- S4R is currently much sparser.
- V2.1 provides more coverage.
- V2.2 provides more conservative structure.
- this still does not prove downstream QA improvement.

---

## 13. Exploratory 17-question source-first Qwen QA experiment

Relevant files:

- `scripts/eval_kazculture_holdout_enriched_qa.py`
- `slurm/kazculture_holdout_source_first_qa.sh`

This experiment selected test questions whose exact source passage was among the 150 held-out train sources.

It is deliberately labeled:

> `exploratory_overlap_only_not_final_benchmark`

### 13.1 Why run it

The retrieval diagnostics only showed that evidence could be retrieved/attached. They did not show whether Qwen actually answered better.

The small overlap-only QA run compared:

- S1
- V2.1 S1 + S2
- V2.1 S1 + S4R
- V2.1 S1 + S2 + S4R
- V2.2 S1 + S2
- V2.2 S1 + S4R
- V2.2 S1 + S2 + S4R

against frozen S0 predictions.

### 13.2 Preflight

Only **17 test questions / 17 unique sources** overlapped the held-out 150-source set.

Structured evidence availability:

V2.1:

- S2: 17/17
- S4R: 4/17

V2.2:

- S2: 12/17
- S4R: 3/17

### 13.3 Runtime bug and fix

The first enriched condition crashed because same-source attached structured units do not have an independent retrieval cosine score, while the generic prediction serializer assumed every hit had `similarity`.

Fix:

- provenance-attached KG units explicitly get `similarity: null`
- `attachment_mode: same_source_provenance`
- regression test added

This distinction is important scientifically: structured evidence is **attached by source provenance**, not independently retrieved.

### 13.4 QA results

Frozen S0 accuracy on the 17-question subset:

- **29.41% = 5/17**

S1:

- **100% = 17/17**

Every structured condition also scored **17/17**.

Therefore:

- S1 rescued all 12 S0 errors
- structured evidence did not hurt
- structured evidence could not improve because S1 had already hit the ceiling

This experiment demonstrated source retrieval utility but **could not measure incremental KG value**.

That ceiling result is why we moved to a full-scale S1 benchmark before spending GPU time on a full 6,749-passage KG build.

---

## 14. Full-scale S1 passage-only experiment

Relevant files:

- `scripts/build_kazculture_s1_index.py`
- `slurm/kazculture_s1_index.sh`
- `scripts/tune_kazculture_retrieval.py`
- `slurm/kazculture_s1_tune.sh`
- `scripts/eval_kazculture_retrieval.py`
- `slurm/kazculture_s1_eval.sh`
- `src/evaluation/kazculture_rag.py`
- `src/retrieval/kazculture_index.py`

### 14.1 Why isolate S1 first

Before extracting the final KG over 6,749 passages, we need to know how much of the performance gap is solved by ordinary RAG.

If S1 alone reaches or nearly reaches 64.54%, then the scientific burden on S2/S4 is an incremental improvement over a strong passage baseline rather than a large absolute gain.

This prevents wasting a major extraction run before knowing what problem the KG must solve.

### 14.2 Full and strict indexes

Both CPU index jobs completed successfully.

Full index:

- passages/units: **6,749**
- dimensionality: 1,024
- system: S1 only

Strict index:

- passages/units: **5,487**
- exact test-overlapping train passages removed: **1,262**
- system: S1 only

Artifacts:

- `data/kazculture_pipeline/index/s1_full/`
- `data/kazculture_pipeline/index/s1_strict/`

The strict builder independently validates that no excluded passage remains.

### 14.3 Retrieval tuning history

We froze retrieval settings on train QA only.

#### First implementation

The original tuner called BGE-M3 separately for each question. It was extremely slow on CPU.

It was replaced by:

- batched encoding of 400 questions
- one query x passage cosine matrix
- immutable candidate scoring grid

This reduced tuning to roughly one minute on a CPU compute node.

#### Evidence-budget mismatch discovered

The initial tuning result was:

- top_k: 8
- min_similarity: 0.45
- apparent train self-source recall: **77.5%**

However, the evaluator retrieved up to 8 passages and then rendered evidence with a 5,000-character budget. Some retrieved passages counted in recall even though Qwen would never see them.

This was fixed by adding `render_evidence_with_hits()`:

- it returns the text actually rendered
- it returns only the hits that fit in that text
- retrieval metadata now corresponds exactly to what Qwen sees

The tuner was also updated to count recall only over actually rendered evidence.

This is an important methodological fix.

#### Final frozen S1 config

The original 75% feasibility gate became unattainable after correcting the evidence accounting. Maximum prompt-visible train self-source recall was **71.5%**.

Because this decision used only train QA and no test answers, the gate was transparently revised to 70%.

Final config:

- file: `data/kazculture_pipeline/s1_full_dev_thresholds_v3.json`
- source: train-dev only
- dev queries: 400
- `top_k = 8`
- `min_similarity = 0.45`
- evidence budget: 5,000 characters
- prompt-visible train self-source recall: **285/400 = 71.25%**
- target feasibility gate: 70%

The full grid showed:

- top-3 best: about **60.25%**
- top-5 best: **69.0%**
- top-8 best: **71.5%**
- top-8 at 0.45: **71.25%**

The 0.45 threshold was chosen because it was the highest threshold that remained over the frozen 70% gate.

**Do not tune this config using test accuracy.**

---

## 15. Full-scale S1 results: normal/full and strict source-disjoint

Both full-scale S1 conditions have now completed on all **1,334** KazCulture test questions.

### 15.1 Frozen S0 baseline

Closed-book Qwen3-32B V2:

- correct: **552/1,334**
- accuracy: **41.3793%**

This remains the frozen comparator.

### 15.2 Strict exact-passage-disjoint S1

Strict S1 removes all **1,262 unique train passage IDs** that exactly overlap test source passages before retrieval.

Completed run:

- Slurm job: `7667979`
- runtime: **28:25**
- correct: **831/1,334**
- accuracy: **62.2939%**
- change vs S0: **+20.9145 percentage points**
- net additional correct answers vs S0: **+279**

Paired statistics vs S0:

- bootstrap 95% delta interval: **+17.8411 to +24.1379 points**
- exact McNemar p-value: **9.22e-37**
- evidence rescued: **394**
- evidence induced errors: **115**

Other strict metrics:

- no-evidence questions: **15**
- mean actually rendered retrieval units: **6.5127**

Strict category accuracies:

- Clothing: **229/360 = 63.61%**
- Craftsmanship: **5/11 = 45.45%**
- Cuisine: **224/362 = 61.88%**
- Household: **252/403 = 62.53%**
- Livestock: **22/41 = 53.66%**
- Musical instruments: **52/100 = 52.00%**
- National games: **7/10 = 70.00%**
- Superstitions: **4/4 = 100%** (tiny n)
- Traditions and customs: **36/43 = 83.72%**

This is the key generalization-oriented S1 result because the exact source match is unavailable.

### 15.3 Normal/full S1

The first full S1 attempt (`7667978`) stalled on node `ng10706` and hit the three-hour wall before saving a single prediction. The first full and strict prompts were then compared and found to be nearly identical in size/content, making a node/runtime problem more likely than a methodological problem.

The full condition was retried while excluding `ng10706`:

- Slurm job: `7692307`
- node: `ng10803`
- questions: **1,334**
- correct: **1,221/1,334**
- accuracy: **91.5292%**
- change vs frozen S0: **+50.1499 percentage points**
- net additional correct answers vs S0: **+669**
- difference vs strict S1: **+29.2354 percentage points**
- additional correct answers vs strict S1: **+390**

The result files are:

- `data/kazculture_pipeline/evaluation/s1_full/S1.jsonl`
- `data/kazculture_pipeline/evaluation/s1_full/S1_report.json`
- `data/kazculture_pipeline/evaluation/s1_full/comparison.json`

### 15.4 What the full-vs-strict gap means

The contrast is extremely large:

```
S0 closed-book                    41.38%
S1 strict exact-source-disjoint   62.29%
S1 full                           91.53%
published reference target        64.54%
```

Two conclusions must be kept separate.

**First:** passage retrieval is genuinely useful beyond exact source memorization. Strict S1 still gains **+20.91 points** over closed-book Qwen after exact matching sources are removed.

**Second:** the normal/full KazCulture setting is heavily advantaged by exact train/test source overlap. Full S1 is **29.24 points** above strict S1 and retrieves from a corpus containing the exact source for many test items.

Therefore the **91.53% full S1 score must not be presented as unseen-source generalization**. It is a valid benchmark-setting result, but the strict result is the cleaner measure of transfer to related rather than identical source evidence.

The full S1 score exceeds the published 64.54% reference by roughly **26.99 percentage points**, but the methods are not directly comparable: the published number is a two-stage fine-tuned result, while this is retrieval-augmented inference in a benchmark with substantial source overlap.

### 15.5 Consequence for the KG research question

The role of the KG is now clearer.

In the normal/full setting, S1 is already near saturation at **91.53%**, leaving little headroom for S2/S4 and making incremental gains difficult to interpret.

The more scientifically useful target is therefore the **strict S1 baseline of 62.29%**.

Relative to the published 64.54% reference, strict S1 is only about **30 additional net correct answers** short. A source-grounded S2/S4 enrichment method that adds roughly 2-3 percentage points without increasing leakage would be a meaningful result.

The primary next question is:

> **Can source-grounded structured cultural evidence improve the strict 62.29% S1 baseline, especially on questions where raw passage retrieval retrieves related evidence but Qwen still answers incorrectly?**

---

## 16. Immediate interpretation and follow-up analyses

### 16.1 Full S1 is complete

The abnormal first full run is no longer a blocker. The retry completed successfully and wrote the full comparison artifacts.

### 16.2 Detailed full S1 paired statistics should be preserved

The top-line full result is now known. For the branch record, also preserve/report the values in:

`data/kazculture_pipeline/evaluation/s1_full/comparison.json`

especially:

- paired bootstrap interval
- exact McNemar p-value
- evidence-rescued count
- evidence-induced-error count
- exact-source retrieval recall
- no-evidence count
- mean rendered evidence count
- category-level accuracies

These are useful for diagnosing how much of the 91.53% comes from exact source retrieval and where the remaining 113 errors occur.

### 16.3 Analyze full vs strict question-level changes

A useful next analysis is to partition the 1,334 questions into:

1. correct in both full and strict
2. full correct / strict wrong
3. strict correct / full wrong
4. wrong in both

The **full-correct / strict-wrong** set is especially important because it estimates questions whose success depends strongly on access to exact-overlap evidence.

### 16.4 Analyze strict S1 failures before scaling the KG

Strict S1 has **503 errors**.

Before final full KG extraction, inspect this error set to determine:

- whether a relevant related source was retrieved
- whether the retrieved passage actually contained the answer
- whether Qwen failed despite sufficient evidence
- whether retrieval introduced misleading evidence
- whether factual S2 structure could make the needed relation explicit
- whether norm/value S4R structure is plausibly relevant
- which categories have the most recoverable errors

This should guide the next structured-evidence experiment without using test answers to tune the extraction prompt itself.

---

## 17. What we currently know scientifically

### Established

1. The original bare-letter scoring protocol was flawed.
2. The frozen V2 closed-book baseline is **41.38%**.
3. KazCulture has extensive exact source overlap between train and test.
4. BGE-M3 retrieves Kazakh cultural source evidence effectively.
5. Raw S1 passage retrieval is much stronger/more complete than KG-only retrieval.
6. Structured representations are best treated as optional source-local enrichment, not the sole retrieval backbone.
7. V2.2 is mechanically cleaner than V2.1 but substantially sparser.
8. Strict passage-disjoint S1 raises Qwen3 from **41.38% to 62.29%**.
9. The strict S1 gain is large and statistically robust under paired bootstrap/McNemar analysis.
10. The 17-question overlap pilot could not show incremental KG benefit because S1 was already 100%.

### Not established

1. Full S1 test accuracy is now known: **91.53%** in the normal/full setting and **62.29%** in the exact-passage-disjoint strict setting.
2. We do **not** yet have a full 6,749-passage final KG.
3. We do **not** yet have full-test S1+S2 or S1+S4R accuracy.
4. We do **not** yet have independent human semantic validation of the 100-row held-out audit.
5. We cannot claim V2.2 is semantically more accurate based only on mechanical diagnostics.
6. We cannot claim Schwartz values are human-validated.
7. We cannot claim the KG has improved the 62.29% strict S1 result yet.
8. We cannot claim 64.54% has been reached yet.
9. We cannot claim a direct apples-to-apples win over the published fine-tuned method even if raw accuracy exceeds 64.54%; methods and prompting must be reported separately.

---

## 18. Recommended final architecture

Based on the experiments so far, the working architecture should be:

### Retrieval backbone

**S1 raw passage retrieval**

Use the frozen BGE-M3 configuration to retrieve source passages.

### Optional factual enrichment

Attach **S2 factual triples** only when the structured evidence belongs to a source retrieved by S1.

Do not let S2 remove or rerank the S1 source in the primary source-first experiment.

### Optional norm/value enrichment

Attach **S4R** only when a source-grounded normative/value representation exists.

Do not fabricate S4R for factual-only passages merely to improve coverage.

### Planned full comparison

The final experiment should ideally contain:

- S0
- S1
- S2 standalone (ablation)
- S4R standalone (ablation)
- S1 + S2
- S1 + S4R
- S1 + S2 + S4R

and should include both:

- normal/full index condition
- exact-passage-disjoint strict condition

The primary scientific question is no longer "does retrieval help?" That is already clearly yes.

The primary question is:

> **Does source-grounded cultural structure add useful information beyond a strong passage-RAG baseline?**

---

## 19. Next steps, in order

### Step 1 — finish full S1

Retry the full S1 job on a different GPU node and record:

- accuracy
- bootstrap delta vs S0
- McNemar p-value
- rescues/harms
- exact source retrieval recall
- category results
- comparison to strict S1

This quantifies the effect of exact source overlap.

### Step 2 — complete independent human audit

A Kazakh-literate reviewer should complete the 100-row blinded audit.

Do not use the AI screening scores as official labels.

At minimum measure:

- factual triple precision
- proportion of sources/triples where a norm is actually warranted
- proposed norm precision
- proposed value precision

If possible, have a second reviewer label a subset and report inter-rater agreement.

This should inform whether V2.1, V2.2, or a newly frozen refinement is suitable for scaling.

### Step 3 — decide whether a new extraction version is necessary

Do **not** create V2.3 merely because V2.2 has lower coverage.

Use the human audit plus source-first downstream behavior.

The likely engineering goal, if a refinement is needed, is:

- preserve V2.2's conservative norm/value safeguards
- recover some of V2.1's factual S2 coverage
- keep exact quote grounding
- keep relation direction strict
- never force a norm/value onto descriptive facts

If prompt/schema changes are made, freeze a new version and validate it on train-only/held-out development data. Do not tune it repeatedly on test accuracy.

### Step 4 — run a larger train-development source-first QA experiment

Before a full 6,749-passage KG extraction, it may be useful to test S1+S2/S4 on a larger train-development QA sample where S1 is not already at 100%.

This avoids another 17-question ceiling problem and allows an architecture decision without consuming test labels.

### Step 5 — extract the final full KG

Only after the extraction choice is frozen, run over all:

- **6,749 unique train passages**

Build versioned outputs and never overwrite earlier V1/V2/V2.1/V2.2 artifacts.

### Step 6 — build full structured indexes

Generate S2/S3/S4F/S4R units from the final full extraction.

Maintain exact provenance from:

- retrieval unit
- frame/triple/path
- passage ID
- source text
- evidence quote
- culture/category

### Step 7 — freeze any new retrieval settings on train only

No test-answer tuning.

If structured evidence is source-attached rather than independently retrieved, make that explicit in metadata.

### Step 8 — full Qwen evaluation

Run all 1,334 test questions under the frozen Qwen V2 scorer.

Primary comparisons:

- S1 vs S1+S2
- S1 vs S1+S4R
- S1 vs S1+S2+S4R

Report:

- absolute accuracy
- delta in percentage points
- paired bootstrap interval
- exact McNemar
- rescued vs induced-error counts
- evidence coverage
- category breakdown

Repeat in strict exact-passage-disjoint mode.

### Step 9 — analyze the remaining S1 errors

Strict S1 has 503 errors. This error set is now strategically important.

Identify:

- questions where no evidence is retrieved
- questions where useful evidence is retrieved but Qwen still fails
- questions harmed by retrieval
- categories with lower performance, especially craftsmanship, livestock, and musical instruments
- questions where structured facts/norms could plausibly disambiguate the answer

This analysis can tell us where KG enrichment has the best chance to add the required ~30 net correct answers.

### Step 10 — final paper-quality validation

Before publication claims:

- finish human semantic audit
- freeze artifact manifests/checksums
- record all model revisions
- document exact prompt/scoring
- distinguish overlap and strict-disjoint results
- report negative as well as positive ablations
- do not call machine screening "human validation"
- ideally test a second answering backbone if the paper design requires model generality

---

## 20. Important artifact map

### Raw/frozen benchmark

- `data/raw/kazculture_phase1/test.parquet`
- `data/raw/kazculture_phase1/manifest.json`
- `data/outputs/kazculture_phase1/` — original flawed-scoring diagnostic
- `data/outputs/kazculture_phase1_v2/` — frozen S0 baseline

### Corpus

- `data/kazculture_pipeline/corpus/passages.jsonl`
- `data/kazculture_pipeline/corpus/pilot_passages.jsonl`
- `data/kazculture_pipeline/corpus/train_dev_qa.jsonl`
- `data/kazculture_pipeline/corpus/test_overlap.json`
- `data/kazculture_pipeline/corpus/manifest.json`

### KG versions

- `data/kazculture_pipeline/kg/` — V1 pilot
- `data/kazculture_pipeline/kg_v2/`
- `data/kazculture_pipeline/kg_v21/`
- `data/kazculture_pipeline/kg_v21_holdout/`
- `data/kazculture_pipeline/kg_v22_holdout/`

### Held-out audit

- `data/kazculture_pipeline/heldout/heldout_train_passages_150.jsonl`
- `data/kazculture_pipeline/heldout/blind_review_100.csv`
- `data/kazculture_pipeline/heldout/blind_review_100_key.json`
- `data/kazculture_pipeline/heldout/v21_vs_v22_diagnostics.json`

### Retrieval diagnostic indexes

- `data/kazculture_pipeline/index/heldout_diagnostic/v21_diagnostic/`
- `data/kazculture_pipeline/index/heldout_diagnostic/v22_diagnostic/`

### Full S1 indexes

- `data/kazculture_pipeline/index/s1_full/`
- `data/kazculture_pipeline/index/s1_strict/`

### Retrieval reports

- `data/kazculture_pipeline/retrieval_diagnostic/heldout_v21_vs_v22_source_recall.json`
- `data/kazculture_pipeline/retrieval_diagnostic/heldout_source_first_enrichment.json`

### Small source-first QA

- `data/kazculture_pipeline/evaluation/heldout_source_first_qwen/`

### Frozen S1 tuning

Historical configs may exist, but the current prompt-budget-correct config is:

- `data/kazculture_pipeline/s1_full_dev_thresholds_v3.json`

### Full-scale S1 QA

- `data/kazculture_pipeline/evaluation/s1_strict/` — complete
- `data/kazculture_pipeline/evaluation/s1_full/` — complete; **91.53%** full S1

---

## 21. Code map

### Scripts

- `scripts/build_kazculture_heldout_audit.py`
- `scripts/build_kazculture_holdout_index.py`
- `scripts/build_kazculture_index.py`
- `scripts/build_kazculture_s1_index.py`
- `scripts/compare_kazculture_kg_holdout.py`
- `scripts/compare_kazculture_kg_v2.py`
- `scripts/compare_kazculture_kg_versions.py`
- `scripts/compare_kazculture_runs.py`
- `scripts/diagnose_kazculture_phase1.py`
- `scripts/eval_kazculture_holdout_enriched_qa.py`
- `scripts/eval_kazculture_phase1.py`
- `scripts/eval_kazculture_phase1_v2.py`
- `scripts/eval_kazculture_retrieval.py`
- `scripts/export_kazculture_pathway_audit.py`
- `scripts/extract_kazculture_kg.py`
- `scripts/extract_kazculture_kg_v2.py`
- `scripts/extract_kazculture_kg_v21.py`
- `scripts/extract_kazculture_kg_v22.py`
- `scripts/prepare_kazculture_corpus.py`
- `scripts/prepare_kazculture_heldout.py`
- `scripts/probe_kazculture_holdout_retrieval.py`
- `scripts/probe_kazculture_source_enrichment.py`
- `scripts/review_kazculture_kg_v2.py`
- `scripts/score_kazculture_heldout_audit.py`
- `scripts/stage_kazculture_phase1.py`
- `scripts/tune_kazculture_retrieval.py`

### Slurm

- `slurm/kazculture_holdout_source_first_qa.sh`
- `slurm/kazculture_kg_extract.sh`
- `slurm/kazculture_kg_holdout.sh`
- `slurm/kazculture_kg_index.sh`
- `slurm/kazculture_kg_retrieval_diagnostic.sh`
- `slurm/kazculture_kg_v21_pilot.sh`
- `slurm/kazculture_kg_v2_pilot.sh`
- `slurm/kazculture_phase1.sh`
- `slurm/kazculture_phase1_v2.sh`
- `slurm/kazculture_retrieval_eval.sh`
- `slurm/kazculture_s1_eval.sh`
- `slurm/kazculture_s1_index.sh`
- `slurm/kazculture_s1_tune.sh`

### Core modules

Construction:

- `src/construction/kazculture_chunking.py`
- `src/construction/kazculture_kg.py`
- `src/construction/kazculture_kg_v2.py`
- `src/construction/kazculture_kg_v21.py`
- `src/construction/kazculture_kg_v22.py`

Evaluation:

- `src/evaluation/kazculture_hf_scorer.py`
- `src/evaluation/kazculture_hf_scorer_v2.py`
- `src/evaluation/kazculture_phase1.py`
- `src/evaluation/kazculture_rag.py`

Loaders:

- `src/loaders/kazculture_corpus.py`
- `src/loaders/kazculture_loader.py`

Retrieval:

- `src/retrieval/kazculture_index.py`

### Tests

- `tests/test_kazculture_kg_v2.py`
- `tests/test_kazculture_kg_v21.py`
- `tests/test_kazculture_kg_v22.py`
- `tests/test_kazculture_phase1.py`
- `tests/test_kazculture_phase1_v2.py`
- `tests/test_kazculture_pipeline.py`
- `tests/test_kazculture_retrieval_diagnostic.py`

---

## 22. Existing KazCulture docs to read with this handoff

This file is the current high-level source of truth. The following documents contain deeper stage-specific detail:

- `docs/KAZCULTURE_PHASE1.md`
- `docs/KAZCULTURE_PHASES_2_TO_5.md`
- `docs/KAZCULTURE_KG_V2_PILOT.md`
- `docs/KAZCULTURE_KG_V21_REFINEMENT.md`
- `docs/KAZCULTURE_PHASE2_HELDOUT_QUALITY_GATE.md`
- `docs/KAZCULTURE_PHASE2_RETRIEVAL_DIAGNOSTIC.md`

Some older docs describe a planned state before later experiments were run. When an older status line conflicts with this handoff, prefer this handoff plus the actual artifact manifests/results.

---

## 23. Safe workflow for a new partner

Before changing code:

~~~bash
cd ~/projects/def-enaskt/divc/kazculture-restart
source .venv/bin/activate
git status
git branch --show-current
git pull --ff-only

python -m pytest -q   tests/test_kazculture_phase1_v2.py   tests/test_kazculture_pipeline.py   tests/test_kazculture_kg_v2.py   tests/test_kazculture_kg_v21.py   tests/test_kazculture_kg_v22.py   tests/test_kazculture_retrieval_diagnostic.py
~~~

Before a GPU job:

1. inspect the relevant Slurm script
2. run `bash -n` on it
3. make sure the correct output directory is versioned
4. confirm no other job is writing to the same checkpoint
5. check that the model revision is still frozen
6. use `sacct` and inspect both `.out` and `.err`
7. never delete a checkpoint merely because a Slurm wall-time expired

Before changing an extraction prompt:

1. decide whether this is a new version
2. document why the old version is insufficient
3. create a new output path
4. add/update regression tests
5. do not use test-answer accuracy to tune the extraction prompt
6. preserve old artifacts

Before reporting a result:

1. state whether it is full-test, train-only, overlap-only, or held-out-from-prompt-development
2. state whether evidence is human-reviewed
3. state whether exact test source overlap is allowed
4. report the frozen baseline comparator
5. distinguish mechanical quote validity from semantic correctness

---

## 24. Non-negotiable scientific guardrails

- Never call AI-generated audit labels human validation.
- Never fill official human audit fields automatically.
- Never use test answers to tune retrieval thresholds.
- Never use test answers in KG extraction.
- Never silently overwrite old versioned extraction outputs.
- Never interpret exact quote presence as semantic correctness.
- Never force every fact into a norm/value pathway.
- Never improve coverage by inventing unsupported cultural abstractions.
- Always distinguish normal/full retrieval from exact-passage-disjoint retrieval.
- Preserve source provenance for every structured unit.
- Keep Qwen scoring protocol fixed when comparing evidence systems.
- If a method change is motivated by test-answer errors, treat later test results as potentially adapted and create a new independent validation strategy.

---

## 25. Short status summary for collaborators

If you only remember one paragraph:

> We froze a corrected Qwen3-32B closed-book KazCulture baseline at **41.38%**. We built several source-grounded KG extraction versions (V1, V2, V2.1, V2.2) and learned that V2.1 has better coverage while V2.2 is cleaner but sparse. Held-out BGE-M3 experiments showed that raw passages (S1) are much better retrievers than KG-only representations, so the architecture changed to **S1 source retrieval plus optional S2/S4 enrichment**. Full-scale S1 evaluation is now complete: the exact-passage-disjoint strict condition reached **62.29%**, a **+20.91 point** gain over S0 despite removing 1,262 exact-overlap train passages, while the normal/full condition reached **91.53%**. The 29.24-point full-vs-strict gap shows that exact train/test source overlap strongly inflates the normal RAG setting, while the strict gain demonstrates substantial benefit from related cultural evidence even without the exact source. Independent human validation of the held-out KG audit is still pending, and a full 6,749-passage final KG has not yet been built. The next scientific question is whether source-grounded S2/S4 structure can improve the strong **62.29% strict S1 baseline**.

