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

### Stage 2: adversarial completeness audit

Only Stage-1 matches are reconsidered. Stage 2 audits the **existing Stage-1
element checks** rather than generating a second decomposition or re-deciding
the entire match.

It returns:

- zero-based indices for any Stage-1 elements whose semantic support deserves
  human review;
- any genuinely missing material factual requirements from the gold target.

The unsupported-element indices are diagnostic only. They do not automatically
reject a pair because Stage 1 has already required exact source quotes for every
supported element, and local testing showed that a second LLM pass could
over-reject valid evidence by demanding literal words such as `why` or
`customary`.

The reviewer is explicitly told not to create standalone elements for question
grammar such as `why`, `what`, or `how`, and not to require literal
`common`/`customary` wording when the cultural behavior is already represented
by a Stage-1 element. The program also removes non-material complaints and
duplicate omissions that are already covered by a prior element.

Material omissions remain strict, including:

- missing causal/reason qualifiers such as `beneficial for health`;
- incomplete multi-clause answers;
- named practices or events not actually present;
- external definitions;
- treating absence as evidence for a FALSE benchmark item.

A pair becomes a final grounded match only when Stage 1 is accepted and Stage 2
finds no new material omission after these deterministic filters. Exact
source-quote validation remains enforced in Stage 1, and all final candidates
still require manual review.

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
