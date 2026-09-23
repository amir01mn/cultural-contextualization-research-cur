# Step 7 Guide: S4 Abstraction-Refinement Evaluation

## Purpose

This document records Step 7 for the current version of the repository.

Earlier planning treated Step 7 as the first S4 implementation. The current `main` branch already contains a more advanced S4 implementation, so Step 7 now means:

1. preserve the current S4 code,
2. document how it works,
3. define the command sequence for running S0-S4,
4. explain how to interpret S4 and its ablations,
5. connect the S4 run to the error-analysis workflow.

## Current S4 systems

The current repository defines these CulturalBench systems:

| System | Meaning |
|---|---|
| `S0_no_kg` | No retrieved evidence; question/options only |
| `S1_raw_text_rag` | Retrieved raw CultureBank passages |
| `S2_flat_kg` | Retrieved raw KG triples without layer/pathway structure |
| `S3_layered_kg` | Retrieved raw triples with abstraction-layer labels |
| `S4_abstraction_refinement_kg` | Retrieved abstraction-refinement pathways |
| `S4_triple_retrieval` | S4 ablation using refined triple retrieval instead of pathways |
| `S4_no_shared_pivot` | S4 ablation using pathways but excluding shared cross-cultural pivots |

The most important comparison is not just whether S4 beats S0. The main research comparison is whether pathway structure improves over the closest baselines:

```text
S4_abstraction_refinement_kg
vs.
S3_layered_kg
vs.
S4_triple_retrieval
vs.
S4_no_shared_pivot
```

## What S4 adds beyond S0-S3

S0-S3 test whether retrieval or KG structure helps at a basic level:

```text
S0: no external context
S1: raw text context
S2: flat triples
S3: layered triples
```

S4 tests the main project claim:

```text
Do abstraction-refinement pathways help the model answer culturally grounded questions better than raw text, flat triples, or layered triples alone?
```

S4 organizes evidence conceptually across:

```text
L3 deep value / abstract cultural concept
L2 social norm or role expectation
L1 surface practice
```

A FULL pathway requires L1, L2, and L3 evidence within the same connected
source-record component, but the extracted edge directions remain flexible.
S4 preserves and displays the supporting predicates rather than inventing a
strict L3→L2→L1 direction.

The current code also tracks pathway completeness, shared cross-cultural pivots,
relation provenance, and the generated-support ratio of retrieved evidence.

## How the current pipeline builds S4 evidence

The S4 evidence is built by `scripts/06_build_retrieval_corpus.py`.

That script produces:

```text
raw-text passages for S1
raw triples for S2 and S3
refined triples for S4_triple_retrieval and S4 pathway construction
entity map
cross-cultural alignments
resolved graph exports
pathways
persistent pathway retrieval index
```

The key sequence inside the corpus build is:

```text
CultureBank records
-> LLM triple extraction
-> critique/refinement
-> entity resolution
-> cross-cultural alignment
-> resolved graph
-> abstraction-refinement pathway construction
-> pathway index
```

## Step 7 command sequence

### 1. Build the retrieval corpus and pathway index

For a local smoke test:

```bash
python scripts/06_build_retrieval_corpus.py \
  --provider ollama \
  --limit 100 \
  --resume
```

For a larger cluster/vLLM run:

```bash
python scripts/06_build_retrieval_corpus.py \
  --provider vllm \
  --limit 2000 \
  --resume
```

This creates the retrieval files used by S1-S4, including the pathway index required by S4.

### 2. Run S0-S4 on CulturalBench

For a small local evaluation:

```bash
python scripts/07_eval_culturalbench.py \
  --provider ollama \
  --limit 20 \
  --systems S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg,S4_triple_retrieval,S4_no_shared_pivot
```

For a larger vLLM evaluation:

```bash
python scripts/07_eval_culturalbench.py \
  --provider vllm \
  --limit 250 \
  --systems S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg,S4_triple_retrieval,S4_no_shared_pivot
```

### 3. Generate disagreement/error analysis

```bash
python scripts/08_error_analysis.py \
  --results data/outputs/culturalbench_results.json \
  --limit 10
```

Then inspect:

```text
data/outputs/culturalbench_error_analysis.csv
data/outputs/culturalbench_error_analysis.md
```

## What to inspect after running S4

For each run, record:

```text
number of passages
number of raw triples
number of refined triples
number of pathways
retrieval top-k
model/provider
number of Easy and Hard units
S0-S4 accuracy
mean generated-support ratio by system
McNemar comparisons
```

Then manually inspect disagreement cases, especially:

| Pattern | Interpretation |
|---|---|
| `S4_abstraction_refinement_kg` correct, `S3_layered_kg` wrong | Pathway structure may help beyond layered triples |
| `S3_layered_kg` correct, `S4_abstraction_refinement_kg` wrong | Pathway retrieval or pathway construction may be noisy |
| `S4_abstraction_refinement_kg` correct, `S4_triple_retrieval` wrong | Full pathways may help beyond refined triples alone |
| `S4_triple_retrieval` correct, `S4_abstraction_refinement_kg` wrong | The pathway builder/index may be losing useful triple-level evidence |
| `S4_abstraction_refinement_kg` correct, `S4_no_shared_pivot` wrong | Cross-cultural shared pivots may be useful |
| `S4_no_shared_pivot` correct, `S4_abstraction_refinement_kg` wrong | Shared pivots may introduce noise or overgeneralization |

## How to report Step 7

Use cautious wording:

> Step 7 connects the current S4 abstraction-refinement pathway implementation to the CulturalBench evaluation workflow. It compares no-KG, raw text RAG, flat raw triples, layered raw triples, refined-triple retrieval, pathway retrieval, and the no-shared-pivot ablation under the same model and requested top-k. The evaluation also reports generated-support ratios so inferred evidence is visible rather than hidden.

Avoid saying:

> S4 proves the hypothesis.

Better:

> S4 is now testable, but claims depend on accuracy results, statistical comparisons, and manual error analysis.

## Success criteria

Step 7 is successful if:

1. `06_build_retrieval_corpus.py` produces refined triples, entity maps, alignments, pathways, and a pathway index.
2. `07_eval_culturalbench.py` runs all selected systems without missing evidence warnings.
3. `data/outputs/culturalbench_summary.md` reports S0-S4 accuracies and comparisons.
4. `08_error_analysis.py` produces disagreement cases that include S4 and S4 ablations.
5. Manual inspection can explain at least some S4 wins/losses in terms of pathway completeness, retrieval relevance, or shared-pivot behavior.

## Important limitation

The existence of S4 in the pipeline does not mean the pathway system is final. The next research work is to validate whether the retrieved pathways are culturally grounded, relevant, and safe.

Future improvements may include:

- stronger entity resolution,
- better canonical node merging,
- manual pathway validation,
- pathway completeness stratification,
- topic-specific controlled subset testing,
- and safety checks for overgeneralization or stereotyping.
