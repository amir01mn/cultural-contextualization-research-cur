# CulturalBench Baseline Notes

## Why this note exists

During the supervisor meeting, we needed to clarify whether the CulturalBench comparison is zero-shot, trained, fine-tuned, or retrieval-augmented. This note records the baseline method so future results can be explained consistently.

## What CulturalBench is

CulturalBench is a benchmark for measuring cultural knowledge in language models. In this repo, CulturalBench is held out from KG construction and used as the evaluation benchmark.

The benchmark has two evaluation setups:

| Setup | Format | Scoring idea |
|---|---|---|
| CulturalBench-Easy | Four-option multiple choice | Correct if the model selects the gold label |
| CulturalBench-Hard | True/False judgments derived from original questions | A group is correct only if all associated binary judgments are correct |

## Original baseline framing

The original CulturalBench evaluation is best understood as a zero-shot benchmark evaluation of language models.

Important points:

- The benchmark evaluates model answers directly.
- It is not a supervised fine-tuning setup.
- Easy is a four-option multiple-choice setup.
- Hard is stricter because the model must judge candidate answers independently as true or false.

## How this repo uses CulturalBench

This repo does not fine-tune the model on CulturalBench.

Instead, it evaluates a fixed base model under different context conditions:

| System | Context condition |
|---|---|
| `S0_no_kg` | No retrieved context |
| `S1_raw_text_rag` | Retrieved raw CultureBank passages |
| `S2_flat_kg` | Retrieved flat KG triples |
| `S3_layered_kg` | Retrieved layered KG triples |
| `S4_abstraction_refinement_kg` | Retrieved abstraction-refinement pathways |
| `S4_triple_retrieval` | S4 ablation using refined triple retrieval |
| `S4_no_shared_pivot` | S4 ablation without cross-cultural shared pivots |

Therefore, the repo experiment is best described as:

> Zero-shot / retrieval-augmented evaluation of a fixed LLM on CulturalBench.

It is not:

- supervised training,
- fine-tuning,
- benchmark-specific model adaptation,
- or training on CulturalBench answers.

## What score should we compare against?

For internal project tracking, the most important comparison is within-model and within-subset:

```text
S1/S2/S3/S4/S4 ablations vs S0 using the same model, same benchmark subset, same top-k, and same answer format.
```

For the research claim, the strongest comparison is not just whether any KG system beats S0. It is whether the pathway system improves over weaker evidence structures:

```text
S4_abstraction_refinement_kg
  vs S3_layered_kg
  vs S2_flat_kg
  vs S1_raw_text_rag
  vs S0_no_kg
```

The S4 ablations are especially useful for mechanism analysis:

```text
S4_abstraction_refinement_kg vs S4_triple_retrieval
S4_abstraction_refinement_kg vs S4_no_shared_pivot
```

These comparisons ask whether full pathway retrieval and shared abstraction pivots add value beyond refined triples alone.

## Easy vs. Hard

### CulturalBench-Easy

Easy keeps the benchmark in multiple-choice form. Each question has options, and the model receives credit when it selects the correct option.

### CulturalBench-Hard

Hard converts each original question into binary True/False judgments. A hard group is only correct if the required judgments for that group are all correct.

## How to describe current results

Use cautious language:

> The current evaluation compares a fixed model under different evidence conditions. S0 is the no-retrieval baseline. S1-S4 add different forms of evidence to the same base question. This means the experiment tests whether evidence structure improves answer selection without fine-tuning the model.

Avoid saying:

> The KG approach failed.

Better:

> The small early run did not show a KG improvement over S0, but it was a sanity check with a very small corpus. The more relevant test is whether the current S4 pathway system improves after pathway coverage and retrieval quality are sufficient.
