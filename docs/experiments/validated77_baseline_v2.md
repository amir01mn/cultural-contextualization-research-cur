# Validated 77-Record Baseline v2

## Status

This document freezes the validated 77-record food/hospitality development experiment after the pathway, provenance, retrieval-fairness, entity-resolution, and cross-cultural-alignment hardening pass.

This is a **controlled development baseline**, not the final large-scale result.

Frozen method commit before this results-document commit:

```text
b4eb022 Fix token matching in alignment actor filter
```

Branch:

```text
validate-flexible-pathways
```

Recommended immutable tag after this document is committed:

```text
validated77-v2
```

---

## Experimental setup

Construction corpus:

```text
data/controlled/food_hospitality_multiculture_records_keep_only.jsonl
```

Evaluation questions:

```text
data/controlled/food_hospitality_culturalbench_questions.jsonl
```

Configuration:

- 77 CultureBank construction records
- 40 CulturalBench scoring units
- 27 Easy units
- 13 Hard units
- model: `Qwen/Qwen2.5-7B-Instruct`
- inference: vLLM on Narval A100
- retrieval top-k: 8
- seed: 42
- alignment verification enabled
- seven evaluation systems run in the same job

Systems:

```text
S0_no_kg
S1_raw_text_rag
S2_flat_kg
S3_layered_kg
S4_abstraction_refinement_kg
S4_triple_retrieval
S4_no_shared_pivot
```

Narval corpus build:

```text
job 3846289
COMPLETED
00:19:32
ExitCode 0:0
```

Narval evaluation:

```text
job 4074408
COMPLETED
00:27:42
ExitCode 0:0
```

---

## Final validated corpus

| Metric | Value |
|---|---:|
| Raw triples | 431 |
| Refined triples | 431 |
| Canonical entities | 496 |
| Entities marked shared, including members and pivots | 34 |
| Verified pairwise alignments | 56 |
| Unique shared pivot IDs | 6 |
| Total pathways | 315 |
| FULL pathways | 39 |
| SOCIAL pathways | 162 |
| SURFACE pathways | 114 |
| Shared-pivot pathways | 67 |
| Unique shared pivots used by pathways | 6 |

The six shared abstractions observed in the final audit were:

- appreciation, L3
- politeness, L2
- hospitality, L3
- reciprocity, L3
- gratitude, L3
- respect, L3

The final audit no longer promoted generic actor-only labels such as `people` or `customers` to shared abstraction pivots.

---

## CulturalBench accuracy

| System | Combined | Easy | Hard |
|---|---:|---:|---:|
| S0 No-KG | 65.0% | 63.0% | 69.2% |
| S1 Raw-text RAG | 65.0% | 66.7% | 61.5% |
| S2 Flat KG | 67.5% | 74.1% | 53.8% |
| S3 Layered KG | 65.0% | 70.4% | 53.8% |
| S4 Pathway KG | 70.0% | 74.1% | 61.5% |
| S4 Triple ablation | 65.0% | 70.4% | 53.8% |
| S4 No-shared ablation | 72.5% | 77.8% | 61.5% |

95% paired-bootstrap intervals were broad because the controlled set contains only 40 scoring units.

---

## Mean generated-support ratio

| System | Generated support |
|---|---:|
| S0 No-KG | 0.0% |
| S1 Raw-text RAG | 0.0% |
| S2 Flat KG | 1.2% |
| S3 Layered KG | 1.2% |
| S4 Pathway KG | 15.4% |
| S4 Triple ablation | 1.2% |
| S4 No-shared ablation | 1.0% |

The higher generated-support ratio in full S4 is largely explained by FULL shared pathways that contain grounded source relations plus a cautious generated L3 bridge such as `may express -> hospitality`, `appreciation`, or `respect`.

---

## Key paired comparisons

The important McNemar comparisons were:

| Comparison | Difference | Discordant units | p-value |
|---|---:|---:|---:|
| S4 Pathway vs S3 Layered | +5.0 percentage points for S4 | 2 gains / 0 losses | 0.5 |
| S4 Pathway vs S4 Triple | +5.0 percentage points for S4 | 2 gains / 0 losses | 0.5 |
| S4 Pathway vs S4 No-shared | -2.5 percentage points for S4 | 0 gains / 1 loss | 1.0 |
| S4 No-shared vs S4 Triple | +7.5 percentage points for No-shared | 3 gains / 0 losses | 0.25 |

None of these differences is statistically significant on the 40-unit development set.

---

## Exact disagreement cases

### Russia, `easy_220`

S3 and the S4 triple ablation were incorrect.

Both S4 pathway systems were correct.

Full S4 retrieved a culture-matched Russian greeting/hospitality pathway containing grounded relations plus a generated hospitality bridge. The no-shared condition also retrieved grounded Russian pathway evidence and remained correct.

Interpretation: this case is consistent with a benefit from pathway-organized evidence, but it does not require a shared cross-cultural pivot.

### South Korea, `hard_5`

S3 and the S4 triple ablation were incorrect.

Both S4 pathway systems were correct.

Full S4 retrieved Korean/South Korean dining pathways, including shared `respect` and `appreciation` pathways with generated L3 bridges. The no-shared pathway condition retained grounded Korean/South Korean relational evidence and was also correct.

Interpretation: again, pathway organization appears useful here, but the shared pivot is not necessary for the gain.

### Bangladesh, `easy_23`

Full S4 was incorrect while the no-shared condition was correct.

Neither condition retrieved Bangladesh-specific evidence. Full S4 retrieved shared pathways from Japanese, Chinese, Turkish, Russian, and Filipino records, with 20% generated support for that question. The no-shared condition retrieved grounded evidence from other cultures and happened to select the correct answer.

Interpretation: this is **not** evidence that removing shared pivots improves Bangladesh reasoning. Neither system was culture-grounded for Bangladesh, so the difference is best treated as evidence sensitivity under missing coverage.

---

## Mechanism analysis

For full S4:

- questions with a retrieved shared pivot: 74.2% accuracy, n=31
- questions without one: 55.6% accuracy, n=9

This is correlational and should not be interpreted causally. Shared-pivot availability is associated with corpus coverage and concept availability.

The direct A4 intervention is more informative:

```text
S4 Pathway KG:       70.0%
S4 No-shared:        72.5%
```

Therefore this controlled experiment does **not** provide evidence that shared cross-cultural pivots improve accuracy.

---

## Comparison with the historical 77-record run

| System | Historical | Validated v2 |
|---|---:|---:|
| S0 No-KG | 65.0% | 65.0% |
| S1 Raw-text RAG | 65.0% | 65.0% |
| S2 Flat KG | 67.5% | 67.5% |
| S3 Layered KG | 65.0% | 65.0% |
| S4 Pathway KG | 75.0% | 70.0% |
| S4 Triple ablation | 65.0% | 65.0% |
| S4 No-shared ablation | 72.5% | 72.5% |

The only headline accuracy that changed after the hardening pass was full S4.

The historical 10-point S4-over-triple difference became a 5-point difference after removing implementation and ablation confounds.

This is why the validation pass was necessary.

---

## Interpretation to carry forward

The defensible conclusion from this controlled development experiment is:

> Pathway-structured retrieval remains promising after methodological hardening. On the 40-unit development set, full pathway retrieval corrected two units missed by both layered and refined-triple retrieval, with no corresponding losses against the triple ablation. However, the difference is not statistically significant. The shared cross-cultural pivot mechanism did not provide an additional accuracy benefit in this experiment, and full S4 retrieved substantially more generated bridge evidence. These results motivate scaling and further mechanism analysis rather than stronger causal claims.

The project should **not** claim from this run that:

- S4 is statistically superior;
- shared pivots improve accuracy;
- the 40-unit result generalizes to CulturalBench overall;
- generated L3 bridges are necessarily beneficial.

---

## Why this is the freeze point

The implementation now has regression coverage for the main methodological invariants discovered during validation, including:

- culture-specific relation provenance;
- connected FULL pathways;
- predicate preservation;
- equal pathway/triple retrieval budgets;
- no duplicate culture penalty;
- no pathway-only post-ranking bonus;
- consistent generated-support treatment;
- no-shared top-k backfill;
- S2/S3 raw-corpus fairness;
- deterministic pathway capping;
- same-layer alignment;
- culture-scoped entity resolution;
- explicit-only shared-pivot semantics;
- same-culture alias exclusion;
- actor-only L2 alignment filtering;
- shared pivot reuse across repeated culture pairs.

Final local regression suite at the frozen method commit:

```text
98 passed
```

---

## Next experiment

Do not further tune the 77-record development corpus against these 40 questions.

The next stage is to scale construction substantially and measure:

- extraction/refinement quality;
- canonical entity count;
- alignment precision and coverage;
- pathway coverage and completeness;
- generated-support ratio;
- per-culture representation;
- cap pressure;
- runtime;
- retrieval culture match;
- downstream accuracy.

Category/index-based retrieval can then be evaluated as a separate change rather than being mixed into this validated baseline.
