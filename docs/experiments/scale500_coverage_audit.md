# Scale-500 CulturalBench coverage audit

## Purpose

The Scale-500 global-retrieval experiment mixes at least three possible failure
modes:

1. the sampled CultureBank corpus does not contain evidence for a benchmark item;
2. relevant evidence exists but retrieval fails to surface it;
3. relevant evidence is retrieved but the answering model still makes an error.

The coverage audit separates the first failure mode before further retrieval
optimization.

## Audit unit

The audit uses the same 40 controlled CulturalBench rows used in the Scale-500
development evaluation:

- 27 Easy items
- 13 Hard true/false items

For Easy items, the audit receives the full question, all options, and the gold
option. For Hard items, it receives the question, the candidate answer, and the
gold true/false judgment.

The gold target is supplied only to define what evidence must be present. The
audit is not an answering condition and must not be compared to S0-S4 as a model
accuracy result.

## Evidence scope

For each benchmark item:

1. include every Scale-500 CultureBank passage whose cultural-group label matches
   the benchmark culture under the validated culture-alias matcher;
2. add a small semantic fallback set from the full corpus to catch plausible
   label variants not represented in the alias table;
3. allow only target-culture passages to count as supporting evidence.

Cross-cultural fallback passages are visible to the audit judge but cannot be
reported as support.

## Labels

- **COVERED**: one or more target-culture passages directly state or clearly
  entail enough information to support the gold answer or judgment.
- **PARTIAL**: related target-culture evidence exists, but it does not establish
  the gold target.
- **ABSENT**: no inspected target-culture passage materially supports the gold
  target.

For a Hard item whose gold judgment is FALSE, absence of the candidate practice
does not establish falsity. COVERED requires affirmative contrary evidence or a
direct refutation.

## Outputs

The audit writes:

- `data/outputs/coverage_audit.jsonl`
- `data/outputs/coverage_audit.md`

The JSONL preserves candidate IDs, target-culture record counts, cited supporting
record IDs, confidence, rationale, and raw judge output for manual review.

## Interpretation

This is an automated evidence audit, not final ground truth. Before publication,
PARTIAL labels, low-confidence labels, and a sample of COVERED/ABSENT labels
should be manually checked against the cited CultureBank records.

The next retrieval experiment should be designed after this audit:

- if many benchmark items are ABSENT, corpus coverage must be treated as a
  separate scaling variable;
- if many are COVERED but S4 fails, retrieval precision/recall is the more direct
  bottleneck;
- category routing should be evaluated primarily where relevant evidence exists.
