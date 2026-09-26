# Abstraction→Refinement Cultural Knowledge Graph Research

A research system for testing whether structured cultural knowledge improves
LLM performance on culturally grounded questions. The system builds a layered
knowledge graph from CultureBank and evaluates it on the held-out CulturalBench
benchmark.

This repository is no longer a small MVP. It contains the full S0–S4 research
pipeline, scalable vLLM/SLURM execution, full-corpus artifacts, ablations, and
mechanism analysis. The project is currently in **experimental validation and
retrieval-quality diagnosis**: the engineering pipeline is implemented, but the
main full-benchmark result does not support the central hypothesis yet.

## Research question

Does an abstraction→refinement cultural KG improve culturally grounded
reasoning compared with closed-book answering, raw-text RAG, and flat or layered
triple retrieval?

The graph organizes cultural evidence into:

- `LAYER_3_DEEP_VALUE`: values and beliefs such as respect or hospitality
- `LAYER_2_SOCIAL`: roles, norms, expectations, and etiquette
- `LAYER_1_SURFACE`: observable practices, objects, rituals, and gestures

The target retrieval unit is a provenance-preserving pathway from an abstract
value through a social norm to a culture-specific practice.

## Systems evaluated

| ID | System | Evidence supplied to the same base LLM |
|---|---|---|
| S0 | No-KG | Question and options only |
| S1 | Raw-text RAG | Retrieved CultureBank passages |
| S2 | Flat KG | Retrieved triples without layer structure |
| S3 | Layered KG | Retrieved refined triples with layer labels |
| S4 | Pathway KG | Retrieved abstraction→refinement pathways |
| A3 | S4 triple ablation | Refined triples instead of pathways |
| A4 | S4 no-shared ablation | Pathways without shared cross-cultural pivots |

CultureBank is used only for construction; CulturalBench is held out for
evaluation. The same model, question, options, and decoding setup are used
across systems. Only the evidence block changes.

## Current project status

Implemented:

- LLM-based extraction and critique/refinement with provenance
- entity resolution and canonical culture normalization
- cross-cultural abstraction alignment
- L3→L2→L1 pathway construction and persistent pathway retrieval
- S0–S4 CulturalBench evaluation and targeted ablations
- paired-bootstrap confidence intervals, McNemar tests, and mechanism analysis
- checkpoint/resume, caching, Parquet mirrors, Ollama, vLLM, and SLURM support
- fail-loud corpus quality gates

Still required before publication-quality claims:

- consolidate the later experimental branches and establish a canonical run
- complete human audits of triples, layers, merges, pathways, and retrieval
- improve culture matching, retrieval precision, and evidence gating
- run at least two base models and at least three seeds
- complete generated-evidence, top-k, stereotype, and fairness analyses
- reconcile and regenerate reports from one frozen corpus/checkpoint

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | end-to-end overview of the method, results, and priorities |
| [`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md) | formal definitions, hypotheses H1–H3, ablations, statistics |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | package map, data flow, artifact files, configuration |
| [`docs/RUN_HISTORY.md`](docs/RUN_HISTORY.md) | every run so far, artifact lineage, branches, open issues |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | coding and research-integrity rules |
| [`docs/NARVAL_SETUP.md`](docs/NARVAL_SETUP.md) | cluster setup, offline staging, SLURM jobs |
| [`AGENTS.md`](AGENTS.md) | briefing for AI agents (loaded by Claude Code via `CLAUDE.md`) |

## Current full-run result

The checked-in full evaluation uses Qwen2.5-7B-Instruct, top-k 8, 22,990
CultureBank passages, 153,882 raw triples, 186,691 refined triples, 29,578
pathways, and 2,454 CulturalBench units.

| System | Combined accuracy |
|---|---:|
| S0 No-KG | **52.2%** |
| S1 Raw-text RAG | 41.4% |
| S2 Flat KG | 42.1% |
| S3 Layered KG | 42.5% |
| S4 Pathway KG | **38.4%** |
| S4 Triple ablation | 42.3% |
| S4 No-shared ablation | 38.2% |

This result is negative: every retrieval condition underperforms the closed-book
baseline, and pathway retrieval underperforms refined-triple retrieval. It
indicates that retrieved evidence is often mismatched or distracting. The result
is a diagnosis of the current system, not evidence for the target claim.

Caveat: S3 and the S4 triple ablation currently receive identical evidence
(refined triples) and differ only in a one-line header, yet they disagree on 70
units. Treat differences of about 3 points as within prompt-wording noise. See
open issue 1 in [`docs/RUN_HISTORY.md`](docs/RUN_HISTORY.md).

Later controlled food/hospitality experiments on a manually audited,
multi-culture subset produced promising but small-sample pathway gains. Those
results remain preliminary and do not supersede the full-run result.

Primary checked-in reports:

- `data/outputs/culturalbench_summary.md`
- `data/outputs/mechanism_analysis.md`
- `data/outputs/before_after_comparison.md`
- `data/outputs/culturalbench_results.json`

The existing `phase2_validation.md` was generated from an older pilot artifact
set and must not be interpreted as validation of the later 29,578-pathway corpus.

## Repository layout

```text
.
├── src/
│   ├── core/            config, schemas, IO, checkpoints, LLM cache, culture names, embeddings
│   ├── llm/             Ollama / vLLM backends and all prompt templates
│   ├── loaders/         CultureBank (construction) and CulturalBench (evaluation) loaders
│   ├── construction/    extraction, critique, entity resolution, alignment, graph, pathways
│   ├── retrieval/       passage, triple, and pathway retrieval with reranking
│   ├── evaluation/      S0–S4 systems, scoring, statistics, mechanism analysis
│   └── legacy/          original MVP evaluator (not a research metric)
├── scripts/
│   ├── prep_offline_data.py     stage datasets/model for offline GPU jobs
│   ├── build_corpus.py          1. build passages, triples, entities, pathways, index
│   ├── validate_corpus.py       2. quality gates (fail loud)
│   ├── eval_culturalbench.py    3. evaluate S0–S4 and ablations
│   ├── compare_runs.py          4. before/after report against an archived run
│   └── legacy/                  MVP steps 01–05 and run_mvp.py (debugging only)
├── slurm/                       Narval jobs; watchers/ holds overnight babysitters
├── tests/                       offline unit, pipeline-integrity, and architecture tests
├── data/processed/              triples, entities, pathways, indexes (git-ignored)
├── data/outputs/                graphs, benchmark results, and reports
└── docs/                        design, architecture, conventions, run history, cluster guide
```

Subpackages depend on each other in one direction only (`core` ← `llm` ←
`construction`; `core` ← `retrieval` ← `evaluation`), and
`tests/test_architecture.py` enforces it. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

Requirements:

- Python 3.11+
- Ollama for local development, or vLLM on a GPU node

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Narval, install the pinned environment instead with
`.venv/bin/pip install -r requirements-lock.txt` after loading the modules. The
datasets and model live in the sibling `hf_cache/` folder; see
[`docs/NARVAL_SETUP.md`](docs/NARVAL_SETUP.md) §0–2.

`sentence-transformers` is optional but recommended for dense retrieval. Without
it, supported retrieval components use TF-IDF where applicable.

## Local smoke test

```bash
ollama serve
ollama pull llama3.1:8b

python scripts/build_corpus.py \
  --provider ollama --limit 30

# Report only: the gates are sized for the full corpus (>= 5,000 pathways)
# and will FAIL on 30 records. Lower GATE_MIN_* to gate a pilot.
python scripts/validate_corpus.py

python scripts/eval_culturalbench.py \
  --provider ollama --limit 6 \
  --systems S0_no_kg,S3_layered_kg,S4_abstraction_refinement_kg
```

Do not use `--allow_fallback` for a research run. It is intended only for
development and tests.

## Full cluster workflow

Stage datasets and model weights on the Narval login node, then run:

```bash
sbatch slurm/build_corpus.sh
sbatch slurm/eval_full.sh
```

See [`docs/NARVAL_SETUP.md`](docs/NARVAL_SETUP.md) for environment variables,
offline staging, outputs, and validation requirements.

## Testing

The suite is offline and deterministic: 133 tests, about 1–2 minutes. On Narval
the venv takes scipy, pandas, and pyarrow from the cluster modules, so load them
first, in the current shell:

```bash
source slurm/load_modules.sh && load_login_modules
source slurm/env_hf.sh && setup_hf_offline
export EMBEDDING_BACKEND=tfidf
.venv/bin/python -m pytest -q -p no:cacheprovider
```

`bash slurm/run_tests.sh` wraps these steps, and `sbatch slurm/run_tests.sh`
runs them on a compute node. On a machine with every requirement pip-installed,
`python -m pytest -q` is enough.

## Research and ethical constraints

- Graph claims apply to source records, not entire cultures.
- Generated or critique-suggested evidence must remain visibly uncertain.
- Benchmark gains are insufficient without human pathway and stereotype audits.
- Culture coverage and per-culture accuracy must be reported.
- CultureBank and CulturalBench remain subject to their original licenses and
  terms.
