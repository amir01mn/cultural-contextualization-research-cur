# Validation and Hardening History

## Why this document exists

This document records the validation pass performed on the cultural contextualization pipeline before scaling it beyond the original 77-record food/hospitality experiment.

The purpose of the validation work was not to redesign the project or chase a higher score. It was to answer a stricter question:

> Does the implementation actually support the scientific claims we want to make about provenance-preserving, connected cultural pathways and fair S4 ablations?

The original 77-record experiment was promising, but a code and research-design audit identified several places where the implementation could make the S4 pathway condition look better than the comparison systems for reasons other than pathway structure. There were also provenance and pathway-definition issues that could become much more serious when scaling from tens of records to thousands.

The validation strategy was therefore:

1. freeze the historical experiment;
2. create a separate validation branch from that exact state;
3. write regression tests that reproduce suspected problems;
4. fix only confirmed problems;
5. keep the research definition, code, tests, and documentation consistent;
6. rerun the exact 77-record experiment under the hardened method;
7. only then scale.

This file summarizes what was found, why each issue mattered, how it was fixed, and what remains before the validated method can be frozen as a new baseline.

---

## Historical baseline that was preserved

The historical branch is:

- branch: `fix-flexible-pathways`
- baseline commit: `d18b2fb`
- immutable tag: `full77-baseline`

The validation branch was created from that exact state:

- branch: `validate-flexible-pathways`

The historical run used:

- 77 manually audited CultureBank food/hospitality records
- 19 cultures represented in pathways
- Qwen/Qwen2.5-7B-Instruct through vLLM
- 40 selected CulturalBench scoring units
- retrieval `top_k = 8`

Historical corpus statistics:

| Metric | Historical 77-record run |
|---|---:|
| Passages | 77 |
| Raw triples | 431 |
| Refined triples | 432 |
| Canonical entities | 443 |
| Cross-cultural alignments | 1 |
| FULL pathways | 52 |
| SOCIAL pathways | 160 |
| SURFACE pathways | 115 |
| Total pathways | 327 |

Historical CulturalBench accuracy:

| System | Combined |
|---|---:|
| S0 No-KG | 65.0% |
| S1 Raw-text RAG | 65.0% |
| S2 Flat KG | 67.5% |
| S3 Layered KG | 65.0% |
| S4 Pathway KG | 75.0% |
| S4 Triple ablation | 65.0% |
| S4 No-shared ablation | 72.5% |

Those numbers were useful evidence that the pathway idea was promising, but they were not treated as final proof. The S4 advantage needed to survive a more controlled implementation.

The historical baseline remains intentionally untouched so that the validated run can be compared against the exact code that produced those numbers.

---

# Validation philosophy

A central rule of this work was:

> Test first. Do not change the algorithm until a regression demonstrates the suspected failure.

This matters because there are two very different categories of problems:

1. **Implementation bugs**, where the code violates the intended method.
2. **Design choices**, where the code is internally consistent but the scientific meaning needs to be clarified.

The validation branch was used to distinguish those cases rather than silently changing the system until the outputs looked better.

---

# 1. Baseline test inconsistency: schema fallback provenance

## Problem

The original test suite contained a test asserting that every schema-fallback triple had:

```text
generated = false
```

However, the extractor already intentionally created cautious inferred value bridges such as:

```text
removes shoes
  -- may express -->
respect
```

Those inferred bridges were correctly marked:

```text
generated = true
confidence = 0.35
```

The test was therefore stale relative to the intended provenance behavior.

## Why it mattered

Changing the extractor merely to satisfy the old test would have made the methodology worse. An inferred cultural interpretation should not be mislabeled as directly grounded evidence.

The distinction we want is:

- directly observed or field-derived relation -> grounded;
- inferred interpretive bridge -> generated, lower confidence.

## Fix

The extractor was left unchanged. The regression test was rewritten to verify the intended distinction.

Commit:

```text
e7ff408 Align schema fallback test with inferred value bridge behavior
```

After this correction, the validation branch had a clean starting checkpoint with all 76 then-existing tests passing.

---

# 2. Cross-cultural relation provenance leakage

## Problem

Entity resolution can canonicalize similar concepts across records. In the original implementation, a canonical node could therefore contain multiple cultures.

For example:

```text
guest
  cultures = {Japan, Netherlands}

tea
  cultures = {Japan, Netherlands}
```

The pathway builder originally decided whether an edge could be used for a culture mainly from the cultures attached to its endpoint nodes.

That creates a dangerous possibility:

```text
Japanese source edge:
guest -> tea
```

could be accepted while building a Dutch pathway simply because both canonical endpoint nodes also occurred in Dutch records.

## Why it mattered

Canonical concepts may be cross-cultural. Source evidence may not silently change culture.

At larger scale, common concepts such as `guest`, `family`, `food`, `respect`, `hospitality`, and `gift` occur across many cultures. A node-level culture test would therefore become increasingly unsafe as the corpus grows.

This could also falsely improve culture-match diagnostics: a pathway could appear to be culture-matched even though its supporting relation came from another culture.

## Regression

A synthetic test deliberately created Japanese and Dutch records using the same endpoint concepts and asserted that:

- Japanese pathways may use the Japanese triple;
- Japanese pathways must not use the Dutch triple;
- Dutch pathways may use the Dutch triple;
- Dutch pathways must not use the Japanese triple.

Regression:

```text
test_pathways_do_not_leak_edges_between_cultures
```

## Fix

Culture provenance was moved to the relation itself.

The resolved graph now carries the source culture from the relation's source record, and pathway construction prefers relation-level `cultural_group` provenance when filtering evidence for a culture.

The endpoint-culture logic remains only as a backward-compatible fallback for older/manually constructed graphs.

Relevant validation commits:

```text
3524cf5 trying to fix the real provenance bug
8b3187e trying to fix the real provenance bug
```

The important design rule established here is:

> Nodes may be shared across cultures. Edges remain tied to their source record and source culture.

---

# 3. FULL pathways could be disconnected same-record bundles

## Problem

The original flexible pathway builder could find one L3 concept, one L2 concept, and one L1 concept in the same source record and label the result `FULL` even when those concepts belonged to separate graph components.

That meant some "FULL pathways" were effectively:

> three useful concepts co-occurring in one CultureBank record

rather than:

> connected relational evidence spanning L3, L2, and L1.

## Why it mattered

The main scientific claim is about structured relational evidence. If FULL does not require connectivity, then an apparent benefit of FULL pathways could really be a benefit of bundling unrelated-but-co-occurring concepts.

We therefore needed the scientific definition and implementation to agree.

## Decision

A FULL pathway was defined as:

> useful L1, L2, and L3 evidence contained in the same weakly connected component of one source record.

Direction is flexible. A valid FULL pathway does **not** need to be a strict directed `L3 -> L2 -> L1` chain, because source statements may express the same relationship in the opposite direction.

What is required is connectivity.

## Regression

```text
test_disconnected_same_record_evidence_does_not_make_full_pathway
```

## Fix

The pathway builder now:

1. groups usable edges by source record and culture;
2. computes weakly connected components inside each record;
3. selects L1/L2/L3 representatives only within a single connected component;
4. creates FULL only if all three layers occur in that component.

Commits:

```text
a56cdb3 added new test
c7617f0 Require connected evidence for FULL pathways
```

---

# 4. S4 dropped the actual pathway predicates

## Problem

The historical S4 prompt showed pathway nodes and layer labels, for example:

```text
LAYER_3_DEEP_VALUE: respect
LAYER_2_SOCIAL: deference norm
LAYER_1_SURFACE: remove shoes
```

but it did not show the model the relations connecting those concepts.

The original `CulturalPathway` mostly retained node references plus `triple_ids`, not fully materialized supporting relation objects.

## Why it mattered

If the model never sees:

```text
respect --mediates--> deference norm
deference norm --expressed as--> remove shoes
```

then the experiment is not really testing relational pathway evidence. It is testing a layer-organized bundle of concepts.

That is a narrower claim.

## Regression

```text
test_s4_prompt_preserves_pathway_predicates
```

## Fix

A provenance-preserving `PathwayEdgeRef` representation was added. Each pathway now retains its exact supporting relations, including:

- triple ID
- source and target entity IDs
- source and target labels
- predicate
- source/predicate/target layers
- source record ID
- cultural group
- generated flag
- evidence text

The S4 formatter now includes those supporting relations in the evidence shown to the LLM.

Commits:

```text
d8b1143 Add S4 predicate preservation regression test
3f67673 Preserve pathway relations in S4 evidence
```

Important design decision:

> We preserve the exact source subgraph. We do not invent a clean representative L3->L2->L1 edge sequence when the source graph does not contain one.

---

# 5. S4 and S4-triple had unequal retrieval candidate budgets

## Problem

Historically, S4 effectively multiplied the requested retrieval budget twice.

For `top_k = 8`:

- triple retrieval could inspect roughly `8 * 5 = 40` semantic candidates;
- S4 could pass `40` into a retriever that multiplied by `5` again, allowing up to approximately `200` candidates.

## Why it mattered

The S4 vs S4-triple comparison is intended to test the **retrieval unit**:

- isolated refined triples;
- whole pathways.

If S4 also receives a much larger search pool, then any S4 advantage cannot cleanly be attributed to pathway structure.

## Regression

```text
test_s4_pathway_and_triple_ablation_use_same_requested_retrieval_budget
```

## Fix

Both systems now enter their retrievers with the same requested `top_k`. Each retriever applies the same internal candidate multiplier.

Commits:

```text
651d120 Add S4 retrieval budget fairness test
b111007 Equalize S4 pathway and triple retrieval budgets
```

---

# 6. S4 culture mismatch was penalized twice

## Problem

`PathwayRetriever.retrieve()` already applied culture-aware reranking.

After that, the S4 post-ranking stage applied culture matching again.

The triple ablation did not receive the same second penalty.

## Why it mattered

This made S4's ranking behavior different for a reason unrelated to pathway structure and made score interpretation harder.

## Regression

```text
test_s4_postranking_does_not_apply_culture_penalty_twice
```

## Fix

Culture-aware ranking is now performed once inside the shared retrieval reranking logic. The S4 post-ranking stage preserves the retriever's score.

Commits:

```text
8ab2857 Add S4 double culture penalty regression test
37a0830 Avoid double culture penalty in S4 reranking
```

---

# 7. S4 received pathway-only completeness and shared-pivot bonuses

## Problem

After retrieval, S4 multiplied pathway scores by extra factors based on:

- completeness;
- shared-pivot availability.

The triple ablation did not receive analogous bonuses.

## Why it mattered

The A3 retrieval-unit ablation is supposed to ask whether whole pathways help relative to isolated triples.

If S4 receives extra pathway-specific score boosts, the comparison changes more than one variable at a time.

Shared-pivot usefulness is also already tested separately in the A4 no-shared-pivot ablation.

## Regression

```text
test_s4_postranking_preserves_retriever_scores_without_pathway_bonuses
```

## Fix

The pathway-only post-retrieval bonuses were removed. S4 now preserves the semantic/culture-aware score returned by the retriever.

Commits:

```text
5ad7358 Add S4 pathway bonus fairness regression test
1670eda Remove pathway-only post-retrieval ranking bonuses
```

---

# 8. Generated/inferred evidence was penalized in triple retrieval but not pathway retrieval

## Problem

Triple retrieval already downweighted `generated=true` evidence.

Pathway retrieval initially treated a pathway containing generated critique bridges as if the entire pathway were grounded.

That created a provenance asymmetry:

```text
refined triple retrieval -> inferred evidence penalized
pathway retrieval       -> same inferred evidence could avoid penalty
```

## Why it mattered

Some apparent pathway benefit could actually come from easier access to inferred bridge information rather than pathway structure.

This is particularly important because generated critique suggestions are intentionally lower-confidence interpretations.

## Regression and first fix

A regression compared two otherwise identical pathways:

- fully grounded;
- fully generated.

Before the fix, both received the same score.

Regression:

```text
test_pathway_retrieval_penalizes_fully_generated_support
```

Initial commits:

```text
87e3526 Add pathway generated-support penalty regression test
2331928 Penalize fully generated pathway support
```

## Final fix for mixed support

A pathway can be partly grounded and partly generated, so a binary rule was still too coarse.

The shared reranker now accepts a continuous generated-support ratio.

For an otherwise identical set of pathways, the intended ordering is:

```text
fully grounded > mixed support > fully generated
```

Regression:

```text
test_pathway_retrieval_scales_mixed_generated_support
```

The penalty is scaled from the fraction of known supporting pathway edges that have `generated=true`.

The evaluation outputs now also record and summarize generated-support ratio by system.

Relevant commits:

```text
135bf4e Harden pathway retrieval and shared pivots
92642b2 Equalize evaluation evidence and track generated support
738e46e Add remaining pathway validation regressions
52a299d Add evaluation fairness regressions
```

---

# 9. S4-no-shared filtered after top-k and could receive less evidence

## Problem

The no-shared-pivot ablation originally worked approximately like this:

1. retrieve final `top_k` pathways;
2. remove pathways containing shared pivots.

If several of the retrieved pathways had shared pivots, the ablation could end with fewer than `top_k` evidence units.

## Why it mattered

That confounds the A4 alignment experiment.

The intended question is:

> What happens when shared-pivot pathways are unavailable?

It should not also mean:

> What happens when the model receives less evidence overall?

## Regression

```text
test_pathway_retrieval_no_shared_backfills_to_top_k
```

## Fix

Shared pathways are excluded from the eligible candidate set **before** final top-k selection. The retriever then fills the requested budget from the remaining eligible pathways when enough are available.

This keeps the evidence-volume comparison much cleaner.

---

# 10. S2 vs S3 used different underlying triple corpora

## Problem

The historical system definitions did not cleanly isolate layering:

- S2 used raw triples;
- S3 preferentially used refined triples.

Therefore S2 vs S3 changed two things:

1. layer labels;
2. the source triple corpus.

## Why it mattered

The S2/S3 comparison is supposed to isolate the value of adding abstraction-layer structure to the **same information**.

If S3 also receives critique/refinement changes, it is not a pure layering ablation.

## Regression

```text
test_s2_and_s3_use_same_raw_triple_retriever
```

## Fix

S2 and S3 now both retrieve from the raw triple corpus.

The difference is:

- S2 formats them as flat triples without layer labels;
- S3 formats the same raw triple evidence with layer labels.

Refined triples are reserved for the S4 triple-retrieval ablation and for constructing the refined pathway corpus.

Commit group:

```text
92642b2 Equalize evaluation evidence and track generated support
52a299d Add evaluation fairness regressions
```

---

# 11. Pathway deduplication could erase distinct source evidence

## Problem

Pathways that produced the same canonical node sequence could collapse to the same deduplication key even when they were supported by different source relations.

## Why it mattered

Two records can express similar concepts but provide different predicates, provenance, or grounded/generated support.

Collapsing them based only on canonical node IDs loses source evidence.

## Fix

The pathway identity now includes the supporting `triple_ids` in addition to culture and node IDs.

This preserves distinct provenance-bearing evidence bundles even when their canonical node sequence is the same.

---

# 12. MAX_PATHWAYS_PER_CULTURE was order-dependent

## Problem

The original fixed cap could make pathway survival depend on which record happened to be encountered first.

At small scale this may be subtle. At larger scale it becomes a serious reproducibility and quality problem.

## Why it mattered

A cap is an engineering constraint. It should not silently turn source-record ordering into a ranking function.

## Regression

```text
test_pathway_cap_prefers_grounded_full_evidence_over_record_order
```

## Fix

The builder now generates candidate pathways before truncation and applies a deterministic priority based on:

1. pathway completeness;
2. generated-support ratio, preferring more grounded support;
3. stable pathway ID as a deterministic tie-breaker.

Only then is the per-culture cap applied.

This does not claim that the ranking is a final scientific definition of pathway quality. It removes accidental record-order dependence.

---

# 13. Shared-pivot pathways were not reliably connected to explicit alignments

## Problem

The cross-cultural aligner creates a pivot object representing an aligned abstraction.

However, the graph/pathway representation did not naturally contain that newly created pivot as an extracted triple endpoint.

A pathway containing an aligned member entity could therefore fail to be marked as using the corresponding shared pivot.

## Why it mattered

If `has_shared_pivot` is rarely or incorrectly set, the S4 vs S4-no-shared comparison cannot be interpreted as an alignment ablation.

## Fix

The builder now constructs a mapping:

```text
aligned member entity -> explicit pivot ID
```

A pathway is considered shared when its representative/support nodes include an entity belonging to an explicit alignment.

Regression:

```text
test_aligned_member_marks_pathway_as_shared
```

An additional bug was found when both the member entity itself and the explicit pivot were eligible pivot IDs. The entity's own ID could win lexicographically.

Fix:

```text
21f100b Prefer explicit alignment pivots in pathways
```

The explicit alignment pivot now takes precedence.

---

# 14. Alignment needed same-layer and provenance safeguards

## Problem

Cross-cultural candidate generation could consider highly similar abstract concepts without explicitly enforcing same-layer matching.

The synthetic pivot entity also needed to retain record-to-culture provenance from its member entities.

## Why it mattered

An L2 social norm should not be aligned to an L3 deep value merely because the labels are similar.

Likewise, creating a cross-cultural pivot must not erase the source-record cultural provenance needed later by the resolved graph and pathway builder.

## Fix

The aligner now:

- requires aligned entities to belong to the same layer;
- merges `source_instance_cultures` into the pivot;
- marks the returned entity copies, rather than mutating only stale/original objects;
- preserves explicit alignment membership for downstream pathway construction.

Regressions include:

```text
test_alignment_preserves_provenance_and_marks_returned_members_shared
test_cross_cultural_alignment_does_not_merge_different_layers
```

---

# 15. The JSON entity map was persisted before alignment

## Problem discovered during the first Narval validation rerun

The first hardened 77-record corpus build produced:

| Metric | Intermediate validation run |
|---|---:|
| Raw triples | 431 |
| Refined triples | 431 |
| Canonical entities | 443 |
| Alignments | 1 |
| FULL pathways | 39 |
| SOCIAL pathways | 163 |
| SURFACE pathways | 114 |
| Total pathways | 316 |

The validation report also said:

```text
Shared abstraction nodes: 0
Alignments: 1
```

Those two lines were inconsistent.

Inspection showed that:

- `entity_map.jsonl` was written **before** cross-cultural alignment;
- the resolved graph and `entity_map.parquet` used the post-alignment entity objects.

The validation report reads the JSONL file, so it was reporting stale pre-alignment flags.

## Fix

`entity_map.jsonl` is now persisted after alignment, so JSONL and Parquet representations describe the same post-alignment entity state.

Commit:

```text
3e8d074 Persist aligned entity map after cross-cultural alignment
```

This issue did **not** require changing extraction, critique, or pathway logic by itself. It was an artifact/reporting consistency problem.

---

# 16. Major late finding: entity resolution itself was creating implicit cross-cultural pivots

## Problem discovered from the Narval diagnostic

The intermediate hardened corpus contained:

```text
316 total pathways
100 pathways marked as shared-pivot
1 explicit cross-cultural alignment
```

That ratio was implausibly large if A4 was meant to test the explicit alignment mechanism.

The cause was deeper than the stale JSON entity map.

Entity resolution clustered candidates globally by label/layer. Therefore, the same abstract concept appearing in multiple cultures could already become one multicultural canonical entity **before** the cross-cultural alignment stage.

The aligner then marked such pre-merged multicultural abstract entities as shared abstractions.

That meant the no-shared-pivot ablation was removing evidence created by two mechanisms:

1. explicit cross-cultural alignment;
2. implicit cross-cultural merging during ordinary entity resolution.

## Why it mattered

A4 is intended to isolate one cause:

> Does the explicit shared-abstraction alignment mechanism help?

If entity resolution already performs cross-cultural merging, then the ablation cannot answer that question cleanly.

It also makes "shared pivot" ambiguous:

- was this node intentionally aligned across cultures?
- or did entity resolution simply cluster similar labels globally?

## Design decision

Entity resolution and cross-cultural alignment now have separate responsibilities.

### Entity resolution

Entity resolution canonicalizes lexical/semantic variants **within a culture**.

### Cross-cultural alignment

The later explicit alignment stage is the only mechanism allowed to connect abstractions across cultures for the shared-pivot experiment.

This creates a much cleaner causal architecture:

```text
within-culture canonicalization
        ↓
culture-specific entities
        ↓
explicit cross-cultural alignment
        ↓
shared abstraction pivots
```

## Regression

```text
test_entity_resolution_keeps_cultures_separate_before_alignment
```

The entity mapping key now includes culture:

```text
(normalized_label, layer, culture)
```

rather than only:

```text
(normalized_label, layer)
```

Legacy two-field mappings remain supported where needed for manually constructed tests and backward compatibility.

The pathway builder now derives shared-pivot metadata only from explicit alignment objects.

Commit:

```text
b433994 Isolate cross-cultural alignment from entity resolution
```

This is a method-level change, so it requires rebuilding the 77-record corpus before the final validated evaluation.

---

# 17. Documentation was brought back into agreement with the implementation

## Problem

Parts of the repository documentation still described:

- strict directed `L3 -> L2 -> L1` pathways;
- old S2/S3 evidence-source behavior;
- older MVP system descriptions.

## Why it mattered

A method is not reproducible if the paper/design document says one thing and the implementation does another.

## Fix

The research documentation was updated to state that:

- FULL requires connected L1/L2/L3 evidence within one source-record component;
- edge direction remains faithful to the extracted source relation;
- S2 and S3 share the raw triple corpus;
- S4 preserves supporting predicates and provenance;
- generated-support ratio is reported;
- the old README system description is legacy.

Commit:

```text
9c28788 Align research docs with validated pathway method
```

---

# 18. Test-suite growth

The branch started with the historical baseline and an outdated provenance test.

Major checkpoints during validation included:

```text
76 passing
...
83 passing
...
93 passing
...
94 passing
```

At the implementation state summarized through commit `b433994`, the local Windows environment reports:

```text
94 passed
```

The additional tests are not just unit coverage. They encode scientific assumptions that the experiment now depends on:

- relation provenance cannot cross cultures;
- FULL must be connected;
- predicates must reach the S4 prompt;
- retrieval-unit ablations must use equal requested budgets;
- S4 cannot receive a second culture penalty;
- S4 cannot receive hidden completeness/shared bonuses in A3;
- generated support must be treated consistently;
- mixed generated support must be distinguishable from fully grounded support;
- the no-shared ablation must backfill its evidence budget;
- S2 and S3 must use the same underlying raw triples;
- alignment must preserve provenance and stay within a layer;
- per-culture pathway caps must not be decided by record order;
- entity resolution must remain culture-scoped before explicit alignment.

---

# Consolidated commit history

The validation branch was created from `d18b2fb`. The main hardening commits through the current method state are:

| Commit | Purpose |
|---|---|
| `e7ff408` | Align schema-fallback test with inferred bridge provenance |
| `3524cf5` | Initial provenance bug correction work |
| `8b3187e` | Finalize relation-level culture provenance behavior |
| `a56cdb3` | Add disconnected-FULL regression |
| `c7617f0` | Require connected evidence for FULL pathways |
| `d8b1143` | Add S4 predicate-preservation regression |
| `3f67673` | Preserve pathway relations in S4 evidence |
| `651d120` | Add S4 retrieval-budget fairness test |
| `b111007` | Equalize S4 pathway/triple requested budgets |
| `8ab2857` | Add S4 double-culture-penalty regression |
| `37a0830` | Remove double culture penalty |
| `5ad7358` | Add pathway-only bonus fairness regression |
| `1670eda` | Remove pathway-only post-retrieval bonuses |
| `87e3526` | Add generated-support penalty regression |
| `2331928` | Penalize fully generated pathway support |
| `135bf4e` | Harden pathway retrieval and shared pivots |
| `92642b2` | Equalize evaluation evidence and track generated support |
| `738e46e` | Add remaining pathway validation regressions |
| `52a299d` | Add evaluation fairness regressions |
| `9c28788` | Align research docs with validated method |
| `21f100b` | Prefer explicit alignment pivots |
| `3e8d074` | Persist post-alignment entity map |
| `b433994` | Isolate explicit cross-cultural alignment from entity resolution |

This history is intentionally granular. Several suspected issues were committed as a failing/targeted regression before the corresponding implementation fix so the repository retains evidence of why the change was made.

---

# What changed conceptually

The biggest conceptual changes are not individual lines of code. They are clearer separations of responsibility.

## Provenance

Before:

```text
node culture could indirectly determine edge culture
```

Now:

```text
edge/source-record provenance determines relation culture
```

## FULL pathway definition

Before:

```text
same record + L1/L2/L3 could be enough
```

Now:

```text
same source record + same connected component + L1/L2/L3
```

## Pathway evidence shown to the model

Before:

```text
layer-labelled concepts
```

Now:

```text
layer-labelled concepts + exact supporting predicates/relations
```

## S4 vs S4-triple

Before:

```text
retrieval unit + candidate budget + ranking bonuses + provenance handling differed
```

Now:

```text
comparison is much closer to:
whole pathways vs isolated refined triples
```

## S2 vs S3

Before:

```text
flat raw triples vs layered refined triples
```

Now:

```text
same raw triples:
flat formatting vs layer-aware formatting
```

## Shared-pivot ablation

Before:

```text
implicit global entity merging + explicit alignment could both create "shared" nodes
```

Now:

```text
within-culture entity resolution
        vs
explicit cross-cultural alignment
```

That makes the no-shared-pivot ablation substantially more interpretable.

---

# Current Narval rerun status

The first hardened Narval rerun, before the final entity-resolution/alignment separation, produced:

```text
Raw triples:       431
Refined triples:   431
Entities:          443
Alignments:          1
FULL:               39
SOCIAL:            163
SURFACE:           114
Total pathways:    316
```

The decrease in FULL pathways from 52 to 39 was expected after enforcing connected evidence.

However, a diagnostic then found approximately 100 pathways marked as shared despite only one explicit alignment. That exposed the implicit cross-cultural entity-resolution confound described above.

Therefore:

> The 316-pathway intermediate run should not be treated as the final validated 77-record baseline.

A new Narval corpus build is required from the culture-scoped entity-resolution implementation at `b433994`, with alignment verification enabled.

At the time this document was written, that rebuild was being submitted/run.

---

# What we should inspect after the new 77-record build

Before running CulturalBench, record:

- raw triple count;
- refined triple count;
- canonical entity count;
- explicit alignment count;
- shared abstraction node count;
- FULL/SOCIAL/SURFACE pathway counts;
- total pathways;
- number of pathways with `has_shared_pivot=true`;
- number of unique shared pivot IDs;
- generated-support ratio;
- culture coverage.

For shared pivots specifically, verify that the number and examples are explainable from explicit alignment objects rather than implicit entity merging.

Then manually inspect a small stratified sample of:

- FULL pathways;
- SOCIAL pathways;
- pathways containing generated support;
- pathways using shared pivots;
- high-scoring retrieved pathways for several cultures.

The automated tests protect invariants, but they do not replace qualitative validation of whether the cultural relations themselves are sensible.

---

# Exact evaluation to rerun after the corpus is accepted

Use the same controlled comparison as the historical baseline:

- 77 CultureBank records;
- same 40 selected CulturalBench scoring units;
- Qwen/Qwen2.5-7B-Instruct;
- vLLM;
- `top_k = 8`;
- same question subset;
- all seven systems:

```text
S0_no_kg
S1_raw_text_rag
S2_flat_kg
S3_layered_kg
S4_abstraction_refinement_kg
S4_triple_retrieval
S4_no_shared_pivot
```

The purpose is not to preserve the historical 75% S4 score.

The purpose is to answer:

> Does the pathway result remain promising when the implementation is methodologically cleaner?

The validated run should be compared against the frozen historical baseline on:

- pathway counts and completeness;
- culture-match behavior;
- generated-support ratio;
- S0-S4 accuracy;
- S4 vs S3 disagreements;
- S4 vs S4-triple disagreements;
- S4 vs S4-no-shared disagreements;
- qualitative evidence quality.

Only after that comparison should the validated implementation be frozen as Baseline v2.

---

# Known limitations that are not "fixed" by this validation pass

Some limitations are experimental-design constraints rather than code bugs.

## 1. The 77-record controlled corpus is benchmark-aware

The controlled corpus was selected with CulturalBench target cultures in mind and manually audited for useful food/hospitality evidence.

Gold answers were not used, so this is not answer leakage, but it is test-set-informed corpus construction.

Therefore the 77-record experiment should remain a **development/controlled proof of concept**, not the final held-out generalization result.

## 2. The original run used TF-IDF fallback retrieval

The historical environment did not consistently install `sentence-transformers`.

Changing the retrieval backend while simultaneously validating pathway logic would have added another independent variable, so it was not silently changed during this hardening pass.

Retrieval backend choice should be explicitly locked for the next formal scaled experiment.

## 3. One or a few shared alignments are not enough to establish H2

Even if the implementation is now correct, cross-cultural alignment quality and coverage still need to be demonstrated empirically at larger scale.

## 4. The generated-support penalty is a modeling choice

The implementation now treats generated support transparently and consistently, but the exact penalty magnitude remains a hyperparameter. The explicit generated-triples ablation remains important.

## 5. Unit tests cannot validate cultural truth

The tests verify structural and methodological invariants. They do not establish that every extracted cultural interpretation or cross-cultural alignment is correct.

Human/manual quality auditing remains necessary.

---

# Why this validation work was worth doing before scaling

The original plan was to move quickly from the 77-record proof of concept to a much larger CultureBank run.

The audit showed why that would have been risky.

Several issues become more dangerous with scale:

- global concepts occur in more cultures;
- provenance leakage becomes easier;
- record-order caps affect more candidates;
- implicit cross-cultural merging becomes more common;
- generated bridges can occupy more pathways;
- unfair candidate budgets can create larger retrieval advantages.

Running thousands of records before fixing those issues could have produced a large, expensive experiment whose result was difficult to interpret.

The validation branch therefore serves as a methodological checkpoint between:

```text
promising proof of concept
        ↓
validated implementation
        ↓
larger-scale experiment
        ↓
category/index retrieval work
```

---

# Next steps after the validated 77-record rerun

If the rebuilt corpus and 40-unit evaluation look healthy:

1. freeze the exact validated implementation and outputs as Baseline v2;
2. archive the historical vs validated comparison;
3. scale the same construction method to a substantially larger CultureBank corpus;
4. measure pathway coverage, alignment quality, generated-support ratio, per-culture representation, build time, and failure rates;
5. only after construction is stable, add the category/index retrieval architecture discussed for scaling;
6. compare global similarity retrieval against category-filtered pathway retrieval;
7. later add recursive/LLM-directed filtering and additional domains/models.

The important sequencing principle is:

> Do not change construction, retrieval architecture, model, and dataset scale all at once.

Each new experiment should isolate one meaningful change.

---

# Summary

The validation pass changed the project from a promising but partially confounded proof of concept into a substantially more controlled experimental pipeline.

The main improvements are:

- source relations retain culture-specific provenance;
- FULL pathways require connected evidence;
- pathway predicates reach the LLM;
- pathway/triple candidate budgets are equalized;
- duplicate culture penalties and pathway-only bonuses are removed;
- generated support is penalized and measured consistently;
- no-shared retrieval keeps a fair top-k budget;
- S2/S3 now isolate layer-label structure;
- pathway capping is deterministic rather than record-order driven;
- explicit alignments are wired into pathway shared-pivot metadata;
- alignment preserves provenance and same-layer semantics;
- the post-alignment entity map is persisted correctly;
- entity resolution is culture-scoped so explicit alignment is the only shared-pivot mechanism used by A4;
- the research documentation now matches the implementation;
- the regression suite grew from the original baseline to 94 passing tests at the method state summarized through `b433994`.

The final question is now empirical rather than architectural:

> After rebuilding and rerunning the exact 77-record experiment with these controls, does the S4 pathway condition still show useful behavior?

That is the result that should determine whether this implementation is frozen and scaled.


---

# 19. Alignment audit after the culture-scoped rebuild

## What the Narval rebuild showed

After culture-scoped entity resolution was introduced, the 77-record corpus was rebuilt with LLM alignment verification enabled.

The structural output was:

```text
Raw triples:                         431
Refined triples:                     431
Canonical entities:                  555
Verified pairwise alignments:         66
FULL pathways:                        39
SOCIAL pathways:                     162
SURFACE pathways:                    114
Total pathways:                      315
Shared-pivot pathways:                87
Unique pivot IDs used by pathways:    28
```

The increase from 443 to 555 canonical entities was expected because concepts are no longer merged across cultures during ordinary entity resolution.

The stable reduction from 52 historical FULL pathways to 39 also remained consistent with the connected-FULL definition.

However, manual inspection of the 66 verified alignments exposed three additional alignment-quality problems.

## Problem A: same-culture aliases were treated as cross-cultural

Examples included:

```text
Japanese        vs Japanese people
Dutch           vs Dutch people
Turkish         vs Turkish people
```

These are lexical variants of the cultural-group label, not distinct cultures.

The candidate generator previously compared culture strings literally, so `Japanese` and `Japanese people` counted as different cultures.

### Fix

Culture labels are now normalized before candidate-pair generation. Obvious `... people` / `... peoples` aliases collapse to the same culture key and therefore cannot form a cross-cultural candidate pair.

Regression:

```text
test_same_culture_aliases_are_not_cross_cultural
```

## Problem B: generic actors were being promoted to shared abstractions

The L2 layer contains both:

- genuine social norms/expectations; and
- social participants/roles.

The previous aligner treated all L2 nodes as potentially alignable abstractions. The verifier therefore accepted examples such as:

```text
people
customers
foreigners and family members
family members and guests
service providers/customers
```

Those labels may participate in cultural relations, but they are not themselves the mid-level shared abstractions that A4 is meant to test.

### Fix

Alignment eligibility is now conservative for L2:

- obvious actor/group-only labels are excluded before embedding candidate generation;
- norm/value-like L2 labels such as `guest etiquette`, obligations, expectations, hierarchies, politeness systems, and relations remain eligible;
- the LLM verifier prompt now explicitly rejects labels that merely name participants, populations, national groups, customers, guests, tourists, family members, authors, service providers, etc.

Regressions:

```text
test_actor_only_l2_labels_are_not_shared_abstraction_candidates
test_norm_like_l2_labels_remain_alignment_candidates
```

## Problem C: one concept produced many pair-specific pivot IDs

Repeated concepts such as `hospitality`, `appreciation`, `reciprocity`, and `politeness` appeared across several cultures.

The previous pivot ID included the culture pair. Therefore:

```text
Brazilian hospitality <-> Pakistani hospitality
Brazilian hospitality <-> Russian hospitality
Chinese hospitality   <-> Russian hospitality
...
```

could each create a different pivot even though they represented the same normalized abstraction.

That caused a single entity to participate in many equivalent pivot IDs and made shared-node counts difficult to interpret.

### Fix

The pivot ID is now based on:

```text
(layer, normalized abstraction label)
```

rather than the culture pair.

Pairwise alignment records are still preserved as verification evidence, but repeated verified matches of the same abstraction reuse one shared pivot entity whose cultures, aliases, source IDs, and provenance are accumulated.

Regression:

```text
test_same_concept_across_three_cultures_reuses_one_pivot
```

## Validation-report improvement

The Phase 2 report now distinguishes:

- entities marked shared, including alignment members and pivots;
- verified pairwise alignments;
- unique shared pivot IDs;
- pathways using shared pivots;
- unique shared pivots actually used by pathways.

This avoids treating pairwise alignment count, pivot count, and marked-member count as if they were the same quantity.

## Commit

```text
60b70d5 Tighten shared abstraction alignment
```

Because this changes alignment candidate selection and pivot identity, the 77-record corpus must be rebuilt once more before the final CulturalBench rerun.
