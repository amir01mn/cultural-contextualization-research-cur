# Cultural Knowledge Graph Research — Project Overview

This is the current end-to-end reference for the repository. It describes the
implemented system, the checked-in full experiment, the present research stage,
and the remaining work. For formal hypotheses and the preregistered-style
evaluation protocol, see [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md). For cluster
execution, see [`NARVAL_SETUP.md`](NARVAL_SETUP.md).

## 1. Current stage

The project is a **full research pipeline under experimental validation**, not an
MVP. The construction, pathway-retrieval, evaluation, ablation, and mechanism-
analysis code is implemented and has been run on the full checked-in corpus.

The main full-benchmark result is negative: every retrieval condition performs
worse than the closed-book baseline, and S4 pathway retrieval performs worse
than refined-triple retrieval. The current stage is therefore retrieval-quality
diagnosis, controlled multi-culture validation, and provenance-safe rebuilding.
The repository is not yet ready for a final paper claim.

## 2. Research objective

The project tests whether cultural knowledge organized as
**abstraction→refinement pathways** improves LLM reasoning:

```text
L3 deep value → L2 social norm or role → L1 observable practice
```

Examples include an abstract value such as hospitality, a host–guest norm, and
a concrete practice such as offering food or drink.

The design separates construction from evaluation:

- **CultureBank** supplies source records for KG construction.
- **CulturalBench Easy and Hard** provide held-out multiple-choice evaluation.
- The same base LLM and question scaffold are used for all systems.
- Only the retrieved evidence representation changes between systems.

## 3. Hypotheses

- **H1 — Structure helps:** pathway KG > flat KG > raw-text RAG > no-KG.
- **H2 — Pathways, not fact count:** any S4 gains should concentrate on
  questions with complete L3→L2→L1 pathways or shared cross-cultural pivots.
- **H3 — Safe refinement:** critique should improve pathway completeness without
  excessive generated support or increased stereotyping.

The checked-in full run does not support H1 or H2. H3 cannot be resolved until
the human quality and stereotype audits are complete.

## 4. Systems and ablations

| ID | Name | Evidence shown to the LLM |
|---|---|---|
| S0 | `S0_no_kg` | no retrieved evidence |
| S1 | `S1_raw_text_rag` | retrieved CultureBank passages |
| S2 | `S2_flat_kg` | raw triples with layer structure hidden |
| S3 | `S3_layered_kg` | refined triples with layer labels |
| S4 | `S4_abstraction_refinement_kg` | ranked abstraction→refinement pathways |
| A3 | `S4_triple_retrieval` | refined-triple retrieval instead of pathways |
| A4 | `S4_no_shared_pivot` | pathway retrieval without shared pivots |

The current evaluator also supports relevance floors and minimum-score sweeps so
evidence can be withheld when retrieval confidence is too low.

## 5. Implemented architecture

```text
CultureBank
  → load and normalize records
  → batched LLM triple extraction
  → critique and cautious refinement
  → raw/refined triple stores and inspection graphs
  → entity resolution
  → canonical culture normalization
  → cross-cultural abstraction alignment
  → resolved graph
  → L3/L2/L1 pathway construction
  → persistent pathway embedding index

CulturalBench
  → S0–S4 answering with a shared base LLM
  → Easy item and Hard group exact-match scoring
  → paired bootstrap confidence intervals
  → paired McNemar tests
  → completeness/shared-pivot mechanism analysis
  → per-culture results and run reports
```

### LLM backends

All model calls use the interface in `src/llm_backend.py`:

| Backend | Intended use |
|---|---|
| Ollama | local development and small smoke tests |
| vLLM | batched GPU inference on Narval |

The research pipeline uses local/open models and does not require paid APIs.

## 6. Cultural representation

Each triple element has one of four layer labels:

| Layer | Meaning |
|---|---|
| `LAYER_3_DEEP_VALUE` | values, beliefs, and moral principles |
| `LAYER_2_SOCIAL` | roles, norms, expectations, and etiquette |
| `LAYER_1_SURFACE` | observable practices, gestures, rituals, and objects |
| `UNKNOWN` | genuinely unresolved cases; excluded from pathway analysis |

Every triple retains its source record, evidence, extraction method, confidence,
and `generated` flag. Critique-suggested bridges remain marked as generated and
must not be presented as ground truth.

## 7. Construction pipeline

### Loading and normalization

`src/data_loader.py` maps CultureBank records into a stable internal schema.
`src/culture_normalizer.py` canonicalizes culture names so variants such as
adjectival and country forms can match during graph construction and retrieval.

### Triple extraction

`src/triple_extractor.py` prompts the selected LLM for layered cultural triples.
Extraction supports batched generation, content-addressed caching, checkpoint
resume, JSON repair, and an explicitly enabled schema fallback for development.
Research runs should not use fallback output as if it were LLM-derived evidence.

### Critique and refinement

`src/critique_agent.py` identifies unsupported statements, missing layers, and
possible bridge triples. Original evidence is retained; suggestions are added
with uncertainty metadata and generated status.

### Entity resolution and alignment

`src/entity_resolver.py` merges lexical variants into canonical entities.
`src/cross_cultural_aligner.py` proposes and optionally LLM-verifies shared
abstraction nodes across cultures. `src/graph_builder.py` exports raw and resolved
graphs in GraphML, JSON, CSV, and HTML formats.

### Pathway construction

`src/pathway_builder.py` constructs culture-coherent paths by pivoting on L2
social nodes. The source graph commonly stores both `L2→L1` and `L2→L3`; pathway
discovery therefore treats relevant edges as traversable for discovery and emits
the semantic order `L3→L2→L1`.

Pathways are classified as:

- `FULL`: all three layers
- `SOCIAL`: a two-layer path containing L2
- `SURFACE`: partial surface-bearing evidence
- `NONE`: no usable evidence

`src/pathway_retriever.py` persists and loads the pathway embedding index used by
S4. `src/retrieval_ranking.py` applies relevance and structural ranking signals.

## 8. Evaluation

`scripts/07_eval_culturalbench.py` evaluates S0–S4 on held-out CulturalBench.

- Easy: one multiple-choice question is one scoring unit.
- Hard: a group is correct only when every True/False item is correct.
- Combined: Easy questions and Hard groups are pooled.
- Statistics: paired bootstrap confidence intervals and paired McNemar tests.

The open-ended heuristic evaluator in `src/evaluator.py` and the scripts numbered
01–05 are retained for debugging and historical comparison. They are not the
primary research metric.

## 9. Checked-in full-run artifacts

The latest checked-in full evaluation reports:

| Artifact | Count |
|---|---:|
| CultureBank passages | 22,990 |
| raw triples | 153,882 |
| refined triples | 186,691 |
| canonical entities | 114,472 |
| cross-cultural alignments | 998 |
| pathways | 29,578 |
| CulturalBench scoring units | 2,454 |

Pathway composition from the corresponding comparison report:

| Completeness | Count |
|---|---:|
| FULL | 22,064 |
| SOCIAL | 7,514 |
| SURFACE | 0 |
| with shared pivot | 212 |

### Accuracy

Model: Qwen/Qwen2.5-7B-Instruct; retrieval top-k: 8.

| System | Combined | Easy | Hard |
|---|---:|---:|---:|
| S0 No-KG | **52.2%** | 71.6% | 32.8% |
| S1 Raw-text RAG | 41.4% | 67.3% | 15.5% |
| S2 Flat KG | 42.1% | 67.0% | 17.3% |
| S3 Layered KG | 42.5% | 67.3% | 17.7% |
| S4 Pathway KG | **38.4%** | 66.0% | 10.8% |
| S4 Triple ablation | 42.3% | 67.2% | 17.5% |
| S4 No-shared ablation | 38.2% | 66.3% | 10.1% |

S0 significantly outperforms every retrieval condition in this run. S4 is also
significantly worse than S3 and the S4 triple-retrieval ablation. The evidence
points to retrieval mismatch and evidence distraction rather than insufficient
pathway count.

### Controlled follow-up evidence

A later manually audited food/hospitality experiment used 77 construction
records across 19 pathway cultures and 40 CulturalBench units. S4 reached 75.0%
versus 65.0% for S0 and gained four items over S3 without a loss. The sample was
too small for statistical significance and had incomplete culture coverage, so
it is promising diagnostic evidence rather than a final result.

## 10. Artifact lineage warning

The checked-in artifacts were not all generated by one frozen run:

- `data/outputs/phase2_validation.md` describes an older 2,000-record pilot with
  502 pathways and two pathway cultures.
- the later corpus contains 29,578 pathways and has a separate full checkpoint.
- some older diagnostic denominators do not match the official 2,454-unit
  evaluation count.

Before the next formal experiment, rebuild the corpus and regenerate validation,
evaluation, mechanism, and comparison reports from the same manifest and
checkpoint. Historical reports should remain clearly labeled rather than being
treated as current validation.

## 11. Source modules

| Module | Responsibility |
|---|---|
| `config.py` | paths, environment settings, datasets, retrieval defaults |
| `schemas.py` | records, triples, entities, alignments, pathways, MCQs |
| `data_loader.py` | CultureBank loading and normalization |
| `culturalbench_loader.py` | CulturalBench loading and grouping |
| `culture_normalizer.py` | canonical culture matching |
| `triple_extractor.py` | LLM extraction and resumable batching |
| `critique_agent.py` | critique, bridge suggestions, refined triples |
| `entity_resolver.py` | canonical entity clustering |
| `cross_cultural_aligner.py` | shared abstraction alignment |
| `graph_builder.py` | raw/resolved graphs and exports |
| `pathway_builder.py` | complete and partial cultural pathways |
| `pathway_retriever.py` | persistent pathway index and retrieval |
| `retriever.py` | passage and triple retrieval |
| `retrieval_ranking.py` | culture/relevance ranking signals |
| `mcq_systems.py` | S0–S4 prompt and answer execution |
| `mcq_evaluator.py` | accuracy, CIs, McNemar, reports |
| `mechanism_analysis.py` | completeness and shared-pivot analysis |
| `storage.py` | JSONL/Parquet, checkpoints, and LLM cache |

## 12. Primary commands

### Local smoke run

```bash
python scripts/06_build_retrieval_corpus.py --provider ollama --limit 30
python scripts/09_validate_phase2.py --strict
python scripts/07_eval_culturalbench.py --provider ollama --limit 6 \
  --systems S0_no_kg,S3_layered_kg,S4_abstraction_refinement_kg
```

### Narval

```bash
sbatch slurm/build_corpus.sh
sbatch slurm/eval_full.sh
```

### Tests

```bash
python -m pytest -q
```

The current repository-local virtual environment is incomplete: `scipy` and `pandas`
are declared dependencies but are not installed there. The observed result is
117 passing tests and 13 dependency-related failures out of 130 collected tests.

## 13. Current priorities

1. Consolidate the active branch, later controlled-experiment work, and the
   provenance-safe rebuild work into one canonical branch.
2. Freeze a complete environment and pass the full test suite.
3. Regenerate every report from one manifest-backed corpus and checkpoint.
4. Complete independent human audits of calibration triples, holdout triples,
   layer labels, entity merges, pathways, and retrieval.
5. Improve culture-matched retrieval, relevance gating, and evidence abstention.
6. Repeat controlled multi-culture experiments at adequate scale.
7. Run at least two base models, at least three seeds, top-k and generated-
   evidence ablations, and multiple-comparison correction.
8. Complete stereotype, representation, and per-culture fairness analysis.

## 14. Limitations and responsible use

- Retrieved evidence can reduce accuracy even when it is structurally complete.
- Cultural coverage is uneven; broad aggregate accuracy can hide this.
- Automatic completeness does not establish human-valid pathway semantics.
- Generated bridge triples may encode unsupported or stereotyped inferences.
- The graph represents source records, not universal facts about cultural groups.
- No publishable claim should be made until the frozen run and human audits are
  complete.

Last updated: 2026-09-22.
