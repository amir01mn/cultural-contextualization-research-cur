# Grounded CultureBank-CulturalBench match discovery

## Why this experiment exists

The exhaustive full-CultureBank coverage audit found candidate benchmark support,
but manual inspection showed that several automated COVERED labels were too
permissive. Related evidence was sometimes treated as direct support.

The grounded-match discovery step therefore removes the PARTIAL category and
constructs a high-precision candidate subset for retrieval evaluation.

## Input

- frozen full CultureBank normalized export: 22,990 records
- SHA-256:
  `60ed452bdba3044d9f7159b6114ab34ee1e606cc449a016799c0c6ed7d31c544`
- controlled CulturalBench development set: 40 units
- model judge: Qwen/Qwen2.5-7B-Instruct

## Exhaustive pairing

For every CulturalBench unit, inspect every CultureBank record whose culture
matches the benchmark target culture under the validated culture matcher.

No semantic top-k candidate filter is used in this discovery step. This avoids
missing a valid support record merely because its wording differs from the
benchmark question.

## Two-stage acceptance rule

A record-question pair is retained only when it passes both verifier stages.

### Stage 1: strict verification

The model must:

1. decompose the complete gold target into essential factual elements;
2. mark every element supported;
3. provide an exact source quote for every element;
4. reject related-topic or partial evidence;
5. require direct contradiction for hard items whose gold judgment is FALSE.

The program independently checks that every claimed support quote occurs in the
record's normalized `source_text`.

### Stage 2: adversarial adjudication

Only Stage-1 matches are reconsidered. The second prompt explicitly attempts to
reject the match and emphasizes common false-positive modes:

- missing causal/reason qualifiers;
- missing common/customary/usually qualifiers;
- incomplete multi-clause answers;
- named practices or events not actually present;
- external definitions;
- treating absence as evidence for a FALSE benchmark item.

A pair becomes a final grounded match only if Stage 2 again returns MATCH and
all exact-quote validations pass.

## Outputs

- `data/outputs/grounded_culturalbench_matches.jsonl`
- `data/outputs/grounded_culturalbench_matches.md`
- resumable Stage-1 and Stage-2 state files under `$SCRATCH`

The Markdown output includes the CultureBank source text and element-level quotes
for every accepted pair so all final candidates can be manually verified.

## Interpretation

This step is intended to discover a high-precision grounded evaluation subset.
It is not itself an accuracy experiment.

After manual review, only verified pairs should be used for claims about
retrieval-grounded CulturalBench performance. Once verified support IDs are
known, the next metric is evidence recall@k: whether each retrieval system places
a known supporting CultureBank record/triple/pathway in the evidence shown to
the answering model.
