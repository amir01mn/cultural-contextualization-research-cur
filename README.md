# Abstraction→Refinement Cultural Knowledge Graph Research

A research system for testing whether structured cultural knowledge improves
LLM performance on culturally grounded questions. The system builds a layered
knowledge graph (KG) from CultureBank and compares it against closed-book
answering, raw-passage retrieval, and flat or layered triple retrieval.

**Current state (October 2026, KG v3).** The rebuilt layered KG beats raw
passages on every in-corpus test, with two answering models. Its abstraction
route helps the weaker model a lot on an external benchmark (NormAd). It does not
yet generalize to records missing from the graph. Results:
[`docs/results/eval_v3/RESULTS.md`](docs/results/eval_v3/RESULTS.md).

## Research question

Does an abstraction→refinement cultural KG improve culturally grounded
reasoning compared with closed-book answering, raw-text RAG, and flat or layered
triple retrieval?

The graph organizes cultural evidence into three layers:

- **L3 deep value**: values and beliefs such as respect or hospitality
- **L2 social norm**: roles, norms, expectations, and etiquette
- **L1 surface practice**: observable practices, objects, rituals, and gestures

The retrieval unit is a provenance-preserving chain from an abstract value
through a social norm to a culture-specific practice. Cultures that share a value
or a norm type are linked through those abstract nodes. This lets the system
reason about a culture it has little data on.

## How the project got here

| When | Version | What happened |
|---|---|---|
| Jul 2026 | KG v1, CulturalBench | Every KG system lost to the baseline. Cause: a global 512-token limit truncated extraction, and **96%** of records were silently replaced by template triples. |
| Aug 2026 | KG v1 rebuilt | A clean build (153,882 raw triples, 29,578 pathways) still lost on CulturalBench: S0 52.2%, S4 38.4%. CulturalBench mostly asks about facts CultureBank does not contain (oracle retrieval: 27.9% vs 25% chance). |
| Sep 2026 | Eval v2 | New benchmarks matched to the KG: CultureBank grounded questions, NormAd, and an intrinsic graph audit. Passages and triples now help (S1 +12.8 points on multiple choice), but the **pathway builder** was the bottleneck: it found the gold record only 41–47% of the time, and only 22.5% of pathways were judged coherent. |
| Oct 2026 | **KG v3, eval v3** | Rebuilt the layered graph as one **frame** per CultureBank descriptor, with **routed** retrieval. Results below. |

Full history, artifact lineage, and job IDs: [`docs/RUN_HISTORY.md`](docs/RUN_HISTORY.md).

## KG v3: frames and routing

Each of the 22,990 CultureBank descriptors becomes a frame:

```text
value (Schwartz's 19 values) → norm type (clustered, culture-neutral) → culture's norm → practice (the descriptor)
```

- 3,077 norm types, of which 1,358 are shared by two or more cultures, across
  2,159 cultural groups.
- The LLM (Qwen2.5-7B) writes only the norm, the norm type, and the value labels.
  Descriptor fields are mapped directly. CultureBank's `eval_*` columns are never
  read.

At answer time, **S4R routes** each question to one of three kinds of evidence:

1. **refine**: the culture's own value → norm → practice chains, when they match
2. **abstract**: the shared value or norm type, plus how similar cultures handle it
3. **none**: no evidence, when nothing is relevant

## Systems evaluated

All systems use the same answering model, prompt scaffold, and decoding settings.
Only the evidence block changes.

| ID | Evidence |
|---|---|
| S0 | none (closed-book) |
| S1 / S1g | raw CultureBank passages / with a culture filter |
| S2 | flat triples |
| S3 / S3g | raw layered triples / with a culture filter |
| A3 | refined layered triples |
| S4 | the August abstraction→refinement pathways (KG v1) |
| S4L | record-local pathways over the August triples |
| S4F | KG v3 frames, always injected |
| **S4R** | **KG v3 with routing (refine / abstract / none): the main system** |
| S4R_noabs | S4R without the abstraction route (ablation) |

## Headline results (eval v3)

Answering models: Qwen2.5-7B-Instruct (primary) and OLMo-2-1124-7B-Instruct
(replication). Statistics: McNemar exact tests and paired bootstrap 95% CIs, with
Holm–Bonferroni correction over pre-registered comparisons. Protocol, fixed before
any test run: [`docs/EVAL_V3_PLAN.md`](docs/EVAL_V3_PLAN.md).

**Graph quality (no answering model)**

| | August pathways (S4) | **KG v3 frames** |
|---|---:|---:|
| Records reachable | 38% | **100%** |
| Gold record in top 8 (test) | 41.2% | **94.7%** (passages 87.5%) |
| Chains judged coherent (phi-4, n=200) | 22.5% | **88.5%** |

**CultureBank grounded questions (Qwen2.5-7B)**

| System | Multiple choice | Hard multiple choice | Open-ended (judged) |
|---|---:|---:|---:|
| S0 | 78.6% | 48.5% | 79.2% |
| S1 passages | 91.4% | 76.5% | 88.8% |
| S3g | 85.5% | 64.0% | 88.8% |
| S4 (August) | 78.6% | 49.0% | 82.6% |
| **S4R** | **92.9%** | **79.6%** | **92.6%** |
| S4R vs S1 | +1.5 (Holm p=0.015) | +3.2 (p<0.001) | +3.8 (Holm p=0.009) |

With OLMo-2-7B, S4R scores 93.4% vs 90.0% for passages on multiple choice
(+3.4, p<0.001).

**NormAd (external; 2,315 stories, 5.8% covered by CultureBank)**

| | Qwen2.5-7B | OLMo-2-7B |
|---|---:|---:|
| Country only | 57.2% | 44.1% |
| + NormAd's gold value | 61.5% | 49.4% |
| S4R_noabs | 57.5% | 47.5% |
| **S4R** | **57.8%** | **54.4%** |

For OLMo, S4R beats country-only by +10.3 points and its own no-abstraction
ablation by +6.9 (both Holm p<0.001). It also scores above the gold-value
condition. For Qwen the gains are not significant, although S4R beats passages by
+2.5 (Holm p=0.004).

**Held-out records (still negative).** When a question's own record is removed
from the graph, S4R scores 75.0% vs 80.0% for S0, and so does every other
retrieval system. The routing thresholds were tuned in-corpus, so held-out
questions are sent to the refine route with other, misleading records of the
same culture (233 of 300).

### Caveats

- **The in-corpus tests are open-book by design.** The question's situation comes
  from the record's own context, and the gold answer paraphrases its behaviour.
  The gold record is in S4R's evidence for 1,421 of 1,500 questions.
- **The August triples leaked evaluation text.** Their extraction prompt included
  `eval_whole_desc`, and 16% of triples carry eval wording. Eval v3 drops those
  triples from every triple system. Read eval v2's triple numbers as upper bounds.
- **The judge is not yet human-validated.** The phi-4 judge scores the open-ended
  answers and the chain audit. Blind annotation sheets are in
  [`docs/annotation/v3/`](docs/annotation/v3/) but are not yet labeled.
- **The same model labels and answers.** KG v3 labels come from Qwen2.5-7B, the
  primary answering model.
- **NormAd was run once per condition.** Multiple choice uses three option orders.

### Next step

Tune the refinement gate on a held-out *dev* split, so S4R falls back to
abstraction when only other records of the culture match. This needs a new
pre-registered run.

## Documentation

| Document | Contents |
|---|---|
| [`docs/results/eval_v3/RESULTS.md`](docs/results/eval_v3/RESULTS.md) | latest results: KG v3 and eval v3 |
| [`docs/results/eval_v2/RESULTS.md`](docs/results/eval_v2/RESULTS.md) | eval v2 results on the August KG |
| [`docs/EVAL_V3_PLAN.md`](docs/EVAL_V3_PLAN.md), [`docs/EVAL_V2_PLAN.md`](docs/EVAL_V2_PLAN.md) | pre-registered evaluation protocols |
| [`docs/RUN_HISTORY.md`](docs/RUN_HISTORY.md) | every run, artifact lineage, branches, open issues |
| [`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md) | formal definitions, hypotheses H1–H3, ablations, statistics |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | package map, data flow, artifact files, configuration |
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | end-to-end overview of the v1 pipeline (Sep 2026) |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | coding and research-integrity rules |
| [`docs/NARVAL_SETUP.md`](docs/NARVAL_SETUP.md) | cluster setup, offline staging, SLURM jobs |
| [`AGENTS.md`](AGENTS.md) | briefing for AI agents (loaded by Claude Code via `CLAUDE.md`) |

## Repository layout

```text
.
├── src/
│   ├── core/            config, schemas, IO, checkpoints, LLM cache, culture names, embeddings, leakage check
│   ├── llm/             Ollama / vLLM backends and all prompt templates
│   ├── loaders/         CultureBank, CultureBank eval columns, CulturalBench, NormAd
│   ├── construction/    extraction, critique, entity resolution, alignment, graph, pathways, frames (v3)
│   ├── retrieval/       passage, triple, pathway, and frame retrieval with routing
│   ├── evaluation/      CulturalBench, CultureBank grounded, NormAd, intrinsic audit, statistics
│   └── legacy/          original MVP evaluator (not a research metric)
├── scripts/
│   ├── build_corpus.py, validate_corpus.py     KG v1: triples, entities, pathways, quality gates
│   ├── eval_culturalbench.py, compare_runs.py  KG v1 on CulturalBench
│   ├── prep_*_v2.py, run_*_v2.py               eval v2 data prep and evaluation
│   ├── build_kg_v3.py                          KG v3: frames, labels, norm types, indexes, checks
│   ├── prep_culturebank_hard_v3.py             hard-distractor multiple choice
│   ├── make_annotation_sheets.py               blind human-label sheets
│   └── legacy/                                 MVP steps (debugging only)
├── slurm/                       Narval jobs; watchers/ holds overnight babysitters
├── tests/                       offline unit, pipeline-integrity, and architecture tests
├── data/                        built KGs and outputs (mostly git-ignored; full copies on the cluster)
└── docs/                        design, protocols, results, run history, cluster guide
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
datasets and models live in the sibling `hf_cache/` folder; see
[`docs/NARVAL_SETUP.md`](docs/NARVAL_SETUP.md) §0–2.

`sentence-transformers` is optional but recommended for dense retrieval. Without
it, supported retrieval components use TF-IDF where applicable.

## Running on Narval

GPU work runs only through SLURM. vLLM model loading alone takes about 16–18
minutes per job.

**KG v3 and eval v3** (current):

```bash
sbatch slurm/build_kg_v3.sh          # frames, LLM labels, norm types, indexes (GPU)
sbatch slurm/build_kg_v3_cpu.sh      # re-run the non-LLM stages from cached labels
sbatch --export=ALL,EVAL=culturebank slurm/eval_v3.sh
# EVAL = culturebank | culturebank_hard | normad | judge | olmo
```

Eval jobs wait for the build's `READY` flag. They write only under
`data/kg_v3/`, `data/eval_v3/`, and `data/outputs/eval_v3/`.

**KG v1 and CulturalBench** (historical):

```bash
RUN_TAG=<new-tag> sbatch slurm/build_corpus.sh
sbatch slurm/eval_full.sh
```

A CPU-only smoke build of KG v3 with fake labels:
`python scripts/build_kg_v3.py --stages frames,kg,local,check --smoke`.

Do not use `--allow_fallback` for a research run. It is intended only for
development and tests.

## Testing

The suite is offline and deterministic: 199 tests, about 1–2 minutes. On Narval
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
- CultureBank, CulturalBench, and NormAd remain subject to their original
  licenses and terms.
