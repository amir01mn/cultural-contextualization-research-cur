# CulturalBench Error Analysis Guide

## Purpose

Before presenting strong claims, manually inspect a small set of CulturalBench disagreement cases. This explains why the current S0-S4 numbers look the way they do.

The goal is to identify whether failures come from:

- retrieval quality,
- prompt wording,
- bad triples,
- missing KG coverage,
- pathway construction,
- shared-pivot ranking,
- or evidence distracting the model.

## Generate the table

After running CulturalBench evaluation, run:

```bash
python scripts/08_error_analysis.py \
  --results data/outputs/culturalbench_results.json \
  --limit 10
```

This writes:

```text
data/outputs/culturalbench_error_analysis.csv
data/outputs/culturalbench_error_analysis.md
```

## Cases to prioritize

| Pattern | What to inspect |
|---|---|
| S0 correct, retrieval/KG wrong | Did retrieved evidence distract the model? Was evidence irrelevant or misleading? |
| S0 wrong, retrieval/KG correct | What evidence helped? Did it add cultural specificity missing from the baseline? |
| S1 correct, KG/pathway wrong | Did raw text preserve context that triples/pathways lost? |
| S4 correct, S3 wrong | Did pathways help beyond layered triples? |
| S4 wrong, S3 correct | Did pathway retrieval over-constrain or mis-rank the evidence? |
| S4 correct, ablation wrong | Did full pathways or shared pivots add value? |
| S4 wrong, ablation correct | Did shared pivots or full-pathway ranking hurt? |

## Manual labels

When filling the generated table, use labels like:

| Label | Meaning |
|---|---|
| `irrelevant_retrieval` | Retrieved evidence does not match the question |
| `missing_coverage` | Relevant cultural information is absent from the retrieval corpus |
| `bad_triple` | Retrieved triple is wrong, unsupported, vague, or misleading |
| `lost_context` | Triple/pathway removed important context from the original passage |
| `pathway_helped` | S4 pathway structure helped answer correctly |
| `pathway_hurt` | S4 pathway structure confused or misled the answer |
| `shared_pivot_helped` | S4 beat no-shared-pivot ablation because shared abstraction helped |
| `shared_pivot_hurt` | No-shared-pivot ablation beat S4, suggesting shared pivot noise |
| `prompt_overweighted_evidence` | Model followed weak evidence instead of the question/options |
| `baseline_prior_knowledge` | S0 was correct from model priors without evidence |
| `needs_better_canonicalization` | Duplicate or inconsistent node labels likely hurt retrieval |

## What to report at the next meeting

Summarize the inspected cases like this:

```text
Manual error analysis of 10 disagreement cases:
- X cases showed irrelevant retrieval.
- Y cases showed missing KG/pathway coverage.
- Z cases showed raw text preserved useful context better than triples/pathways.
- A cases showed S4 pathways helped beyond S3 layered triples.
- B cases showed S4 ablations outperformed full S4, suggesting pathway/shared-pivot noise.
```

This gives a stronger explanation than only reporting accuracy percentages.
