# Steps 1-7 Summary: Current-Main Meeting Plan

## Purpose

This document explains what Steps 1-7 added to the current `main`-based branch and why each step matters.

The important context is that the current repository already contains a more advanced S4 implementation than the older planning branch. Therefore, these steps do not replace that implementation. They document, stabilize, and prepare it for controlled evaluation.

Branch used:

```text
meeting-steps-1-5-current-main
```

## Big-picture goal

The project is testing whether structured cultural knowledge improves LLM performance on CulturalBench.

The intended comparison is:

```text
S0 no KG
vs.
S1 raw-text RAG
vs.
S2 flat KG
vs.
S3 layered KG
vs.
S4 abstraction-refinement pathway KG
```

The main research question is not simply whether retrieval helps. It is whether **abstraction-refinement pathways** help more than raw text, flat triples, or layered triples alone.

---

## Step 1 — Lock the experimental setup

### File added

```text
docs/MEETING_ACTION_PLAN.md
```

### What it did

Step 1 documented the current experimental setup and the systems being compared:

| System | Meaning |
|---|---|
| `S0_no_kg` | Question/options only; no retrieved evidence |
| `S1_raw_text_rag` | Question/options plus raw retrieved CultureBank passages |
| `S2_flat_kg` | Question/options plus flat KG triples |
| `S3_layered_kg` | Question/options plus layered/refined KG triples |
| `S4_abstraction_refinement_kg` | Question/options plus abstraction-refinement pathway evidence |
| `S4_triple_retrieval` | S4 ablation using refined triples instead of pathways |
| `S4_no_shared_pivot` | S4 ablation excluding shared cross-cultural pivots |

It also locked the comparability rule:

```text
Every system receives the same base CulturalBench question/input.
Only the extra evidence block changes.
```

### Why it matters

This prevents the experiment from becoming unfair. If one system receives a different version of the question, then accuracy differences may come from prompt differences rather than from the retrieval evidence or KG structure.

---

## Step 2 — Fix prompt wording so evidence is augmentation

### File changed

```text
src/prompts.py
```

### What it did

Step 2 changed the prompt wording so retrieved evidence is treated as **additional context**, not as the only source of truth.

The CulturalBench MCQ prompt now tells the model to:

```text
use evidence as additional context
not treat evidence as a replacement for the question/options
not force incomplete, irrelevant, or insufficient evidence
base the choice on the question, options, and most relevant evidence
```

The older raw/refined KG scenario prompts were also updated with the same logic.

### Why it matters

A sparse or imperfect KG can hurt performance if the prompt forces the model to rely only on retrieved evidence. This fix keeps the question/options primary and makes KG/RAG evidence supportive rather than controlling.

---

## Step 3 — Document the current result as a sanity check

### File added/used

```text
docs/MEETING_ACTION_PLAN.md
```

### What it did

Step 3 documented the small current result as a **Phase 1 sanity check**, not as evidence that KG failed.

The existing result should be described cautiously because it used a small retrieval corpus and did not establish statistically decisive differences.

Recommended wording:

```text
We connected the CulturalBench evaluation pipeline and confirmed that S0-S3 can run under the same model, top-k, and scoring protocol. The current result is preliminary and mainly verifies the pipeline.
```

### Why it matters

This keeps the project narrative accurate. Early KG results may look worse than the no-KG baseline because the KG is small, incomplete, noisy, or not yet using the full pathway design.

---

## Step 4 — Add error-analysis workflow

### Files added

```text
scripts/08_error_analysis.py
docs/ERROR_ANALYSIS_GUIDE.md
```

### What it did

Step 4 added a script that reads:

```text
data/outputs/culturalbench_results.json
```

and writes:

```text
data/outputs/culturalbench_error_analysis.csv
data/outputs/culturalbench_error_analysis.md
```

The script selects disagreement cases across S0-S4, including the current S4 system names and ablations.

Examples of useful disagreement patterns:

| Pattern | Possible meaning |
|---|---|
| S0 correct, KG wrong | Retrieval evidence may distract or be irrelevant |
| S0 wrong, KG correct | Retrieved evidence may add useful cultural specificity |
| S4 correct, S3 wrong | Pathway structure may help beyond layered triples |
| S3 correct, S4 wrong | Pathway construction or retrieval may be noisy |
| S4 correct, S4_no_shared_pivot wrong | Shared cross-cultural pivots may help |
| S4_no_shared_pivot correct, S4 wrong | Shared pivots may add noise or overgeneralization |

### Why it matters

Accuracy numbers alone do not explain why a system helps or hurts. Error analysis lets us diagnose retrieval quality, KG coverage, pathway completeness, lost context, prompt behavior, and shared-pivot behavior.

---

## Step 5 — Document the CulturalBench baseline method

### File added

```text
docs/CULTURALBENCH_BASELINE_NOTES.md
```

### What it did

Step 5 clarified that the repo's CulturalBench evaluation is:

```text
zero-shot / retrieval-augmented evaluation of a fixed LLM
```

It is not:

```text
fine-tuning
supervised training
benchmark-specific model adaptation
training on CulturalBench answers
```

It also explains CulturalBench Easy vs. Hard:

| Split | Meaning |
|---|---|
| CulturalBench-Easy | Four-option multiple-choice question |
| CulturalBench-Hard | True/False judgments where all judgments for a group must be correct |

### Why it matters

This answers the meeting question about whether the baseline is trained or zero-shot. It also makes clear that the important internal comparison is within the same model and same benchmark subset.

---

## Step 6 — Add controlled food/hospitality subset workflow

### Files added

```text
scripts/09_select_controlled_subset.py
docs/CONTROLLED_SUBSET_GUIDE.md
```

### What it did

Step 6 added a controlled topic-selection workflow for food/hospitality.

The script scans CultureBank records and selects rows that contain both:

1. food/hospitality terms, and
2. social or cultural interaction terms.

It writes:

```text
data/controlled/food_hospitality_records.jsonl
data/controlled/food_hospitality_audit.csv
data/controlled/food_hospitality_selection_notes.md
```

Optionally, it also selects matching CulturalBench questions:

```text
data/controlled/food_hospitality_culturalbench_questions.jsonl
```

### Why food/hospitality?

Food/hospitality is a useful controlled topic because it often contains all three cultural layers:

| Layer | Example |
|---|---|
| L1 surface practice | offering tea, serving coffee, sharing a meal |
| L2 social norm | guest-host obligation, politeness, welcoming visitors |
| L3 deep value | hospitality, respect, generosity, belonging |

### Why it matters

The project should not immediately scale randomly. A controlled subset helps test whether extracted triples and pathways are meaningful before scaling to many rows.

---

## Step 7 — Document and run the current S4 pathway evaluation

### File added

```text
docs/S4_EVALUATION_GUIDE.md
```

### What it did

Step 7 documents how to run and interpret the current S4 implementation.

The current repo already builds S4 evidence inside:

```text
scripts/06_build_retrieval_corpus.py
```

That script produces:

```text
raw-text passages
raw triples
refined triples
entity map
cross-cultural alignments
resolved graph exports
pathways
persistent pathway index
```

The current S4 evaluation runs through:

```text
scripts/07_eval_culturalbench.py
```

which can evaluate:

```text
S0_no_kg
S1_raw_text_rag
S2_flat_kg
S3_layered_kg
S4_abstraction_refinement_kg
S4_triple_retrieval
S4_no_shared_pivot
```

### Why it matters

Step 7 makes the main research contribution testable. S0-S3 are baseline systems. S4 and its ablations test whether abstraction-refinement pathway structure adds value beyond raw text, flat triples, and layered triples.

---

## Recommended command sequence after Steps 1-7

### 1. Build corpus and S4 pathway index

Local smoke test:

```bash
python scripts/06_build_retrieval_corpus.py \
  --provider ollama \
  --limit 100 \
  --resume
```

Larger vLLM run:

```bash
python scripts/06_build_retrieval_corpus.py \
  --provider vllm \
  --limit 2000 \
  --resume
```

### 2. Evaluate S0-S4

```bash
python scripts/07_eval_culturalbench.py \
  --provider ollama \
  --limit 20 \
  --systems S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg,S4_triple_retrieval,S4_no_shared_pivot
```

### 3. Generate error analysis

```bash
python scripts/08_error_analysis.py \
  --results data/outputs/culturalbench_results.json \
  --limit 10
```

### 4. Run controlled subset selection

```bash
python scripts/09_select_controlled_subset.py \
  --scan_limit 2000 \
  --max_records 100 \
  --include_culturalbench \
  --cb_limit 250
```

Then manually audit:

```text
data/controlled/food_hospitality_audit.csv
```

---

## Current state after Steps 1-7

The branch now supports this workflow:

```text
CultureBank
-> triple extraction
-> critique/refinement
-> entity resolution
-> cross-cultural alignment
-> resolved cultural graph
-> abstraction-refinement pathways
-> pathway retrieval index
-> CulturalBench S0-S4 evaluation
-> manual error analysis
-> controlled food/hospitality subset debugging
```

## What has not been proven yet

These steps prepare the experiment. They do not prove the hypothesis yet.

The hypothesis only becomes meaningful after:

1. the corpus/pathway index is built,
2. S0-S4 are evaluated under the same model/top-k/subset,
3. confidence intervals and McNemar comparisons are inspected,
4. disagreement cases are manually analyzed,
5. pathway relevance and cultural safety are audited.

## How to describe this in a meeting

Use this summary:

> We rebuilt the meeting changes on top of the current main branch. Steps 1-5 document the experimental setup, prompt-fairness rule, baseline interpretation, CulturalBench zero-shot framing, and error-analysis workflow. Step 6 adds a controlled food/hospitality subset workflow for debugging before broad scaling. Step 7 documents how the current S4 abstraction-refinement pathway system is built and evaluated, including S4 ablations for refined triples and shared-pivot removal.
