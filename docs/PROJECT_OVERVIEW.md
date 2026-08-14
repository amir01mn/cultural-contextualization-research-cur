# Cultural KG MVP — Full Project Overview

This document describes **what the project does**, **every pipeline and process**, **how data flows from input to output**, **what is implemented today vs. planned for the publishable version**, and **how evaluation works**. It is the single reference for understanding the codebase end-to-end.

For the formal research claim, hypotheses, and ablation matrix, see [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md).  
For cluster/GPU setup, see [`NARVAL_SETUP.md`](NARVAL_SETUP.md).

---

## Table of contents

1. [Project purpose](#1-project-purpose)
2. [Research contribution (target paper)](#2-research-contribution-target-paper)
3. [High-level architecture](#3-high-level-architecture)
4. [Datasets](#4-datasets)
5. [Cultural layer model](#5-cultural-layer-model)
6. [End-to-end data flow](#6-end-to-end-data-flow)
7. [Pipeline A — MVP experiment (`run_mvp.py`)](#7-pipeline-a--mvp-experiment-run_mvppy)
8. [Pipeline B — Publishable CulturalBench evaluation (scripts 06–07)](#8-pipeline-b--publishable-culturalbench-evaluation-scripts-0607)
9. [Pipeline C — Step-by-step scripts (01–05)](#9-pipeline-c--step-by-step-scripts-0105)
10. [How triples are extracted](#10-how-triples-are-extracted)
11. [How the knowledge graph is built](#11-how-the-knowledge-graph-is-built)
12. [How critique and refinement work](#12-how-critique-and-refinement-work)
13. [How retrieval works (current)](#13-how-retrieval-works-current)
14. [How pathways will work (planned — S4 system)](#14-how-pathways-will-work-planned--s4-system)
15. [How evaluation works](#15-how-evaluation-works)
16. [LLM backends](#16-llm-backends)
17. [Source modules reference](#17-source-modules-reference)
18. [Directory layout and artifacts](#18-directory-layout-and-artifacts)
19. [Configuration and environment](#19-configuration-and-environment)
20. [Commands cheat sheet](#20-commands-cheat-sheet)
21. [Roadmap — what comes next](#21-roadmap--what-comes-next)
22. [Limitations and design principles](#22-limitations-and-design-principles)

---

## 1. Project purpose

**cultural-kg-mvp** is a Python research pipeline that tests whether a **layered cultural knowledge graph (KG)**, organized as **abstraction→refinement pathways**, improves an LLM's ability to answer **culturally grounded questions** — compared to baseline prompting, raw-text retrieval, and flat (non-layered) KGs.

Everything runs on **local / open-source LLMs** (Ollama for development, vLLM on GPU clusters for scale). **No paid API** (OpenAI, Claude, Anthropic) is required or used.

The project has two operational modes:

| Mode | Purpose | Entry point |
|---|---|---|
| **MVP pipeline** | Quick end-to-end prototype on a small CultureBank sample; generates scenarios and heuristic scores | `scripts/run_mvp.py` |
| **Publishable pipeline** | Held-out evaluation on **CulturalBench** with objective MCQ accuracy, bootstrap CIs, and McNemar tests | `scripts/06_build_retrieval_corpus.py` + `scripts/07_eval_culturalbench.py` |

Both share the same core building blocks (extraction, graph, retrieval, LLM adapter) but differ in evaluation design.

---

## 2. Research contribution (target paper)

### One-line pitch

We build a theory-grounded cultural KG organized as **abstraction→refinement pathways** — from shared human/universal concepts refined downward into culture-specific practices — and show that this **pathway structure (not raw fact count)** improves culturally grounded reasoning while controlling stereotyping.

### Core idea

Cultural reasoning fails when a model knows a surface practice ("remove shoes") but cannot connect it to the **social norm** ("guest-host expectation") and **underlying value** ("respect / cleanliness"). A flat list of facts does not expose that chain or enable **cross-cultural transfer** (e.g., "respect for elders" → touching feet in India vs. bowing in Japan).

The KG is therefore built so that knowledge flows:

```
Shared abstraction (e.g., respect for elders)
  → social norm (deference to elders)
    → refined practice (touch elder's feet / bow deeply)
```

### Hypotheses (from `RESEARCH_DESIGN.md`)

- **H1 — Structure helps:** Abstraction→refinement KG > flat KG > raw-text RAG > no-KG on CulturalBench accuracy.
- **H2 — Pathways, not facts:** Gains concentrate where a **complete L3→L2→L1 pathway** is retrievable and/or a **shared abstraction node** connects cultures.
- **H3 — Refinement completes pathways safely:** Critique increases pathway completeness without excessive generated/hallucinated triples or stereotyping.

### What is implemented vs. planned

| Component | Status |
|---|---|
| Layered LLM triple extraction | ✅ Implemented |
| Graph construction (networkx) | ✅ Implemented |
| Critique-refinement | ✅ Implemented |
| Triple-level retrieval (TF-IDF / sentence-transformers) | ✅ Implemented |
| CulturalBench MCQ evaluation (S0–S3) | ✅ Implemented |
| Pluggable LLM backend (Ollama + vLLM) | ✅ Implemented |
| Entity resolution / node canonicalization | ✅ Implemented |
| Cross-cultural abstraction alignment | ✅ Implemented |
| Abstraction→refinement **pathway** construction | ✅ Implemented |
| Pathway retrieval (S4 system) | ✅ Implemented |
| Mechanism analysis (completeness buckets, shared-node pivots) | ✅ Implemented |
| Canonical culture normalization | ✅ Implemented |
| Fail-loud corpus quality gates (`09_validate_phase2.py --strict`) | ✅ Implemented |

---

## 3. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                    │
│  CultureBank (HF: SALT-NLP/CultureBank)  →  KG construction             │
│  CulturalBench (HF: kellycyy/CulturalBench) →  held-out evaluation      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NORMALIZATION & LOADING                              │
│  data_loader.py          culturalbench_loader.py                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ TRIPLE EXTRACTION│  │ CRITIQUE/REFINE  │  │  GRAPH BUILD     │
│ triple_extractor │  │ critique_agent   │  │  graph_builder   │
│ (local LLM main) │  │ (local LLM main) │  │  (networkx)      │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LAYER                                 │
│  retriever.py — TripleRetriever + TextRetriever                         │
│  (TF-IDF or sentence-transformers; pathway retrieval planned)           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌──────────────────────────┐            ┌──────────────────────────┐
│ MVP EVALUATION           │            │ CULTURALBENCH EVALUATION │
│ evaluator.py             │            │ mcq_systems.py           │
│ (scenario gen + scoring) │            │ mcq_evaluator.py         │
└──────────────────────────┘            └──────────────────────────┘
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │   LLM BACKEND        │
                    │ llm_backend.py       │
                    │ Ollama / vLLM        │
                    └──────────────────────┘
```

---

## 4. Datasets

### CultureBank — KG construction only

| Property | Value |
|---|---|
| Hugging Face ID | `SALT-NLP/CultureBank` |
| Splits | `tiktok`, `reddit` (no `train` split) |
| Loader | `src/data_loader.py` |
| Usage | Build triples, graph, retrieval corpus |

If `--split train` is passed, the loader **auto-detects available splits** and reads from `tiktok` + `reddit` until `--limit` is reached. On HF failure, a **built-in fallback** (7 examples) keeps the pipeline runnable.

Each record is normalized to:

```json
{
  "dataset_id": "culturebank",
  "instance_id": "...",
  "cultural_group": "...",
  "context": "...",
  "goal": "...",
  "actor": "...",
  "recipient": "...",
  "relation": "...",
  "actor_behavior": "...",
  "recipient_behavior": "...",
  "topic": "...",
  "agreement": "...",
  "source_text": "...",
  "raw": { "...original fields..." }
}
```

Missing fields become `null`.

### CulturalBench — evaluation only (held out)

| Property | Value |
|---|---|
| Hugging Face ID | `kellycyy/CulturalBench` |
| Variants | `CulturalBench-Easy`, `CulturalBench-Hard` |
| Loader | `src/culturalbench_loader.py` |
| Usage | MCQ accuracy evaluation only |

**Held-out principle:** the KG is built **only** from CultureBank; evaluation is **only** on CulturalBench. No overlap → avoids circularity.

- **Easy:** 4-way multiple choice; one correct letter per question.
- **Hard:** each question is split into several True/False items; a **group** is correct only if **all** items in the group are correct.

---

## 5. Cultural layer model

Every triple element (subject, predicate, object) is assigned one of four layers:

| Layer | Enum | Abstraction | Examples |
|---|---|---|---|
| L3 | `LAYER_3_DEEP_VALUE` | Most abstract | respect, harmony, hospitality, honor, dignity |
| L2 | `LAYER_2_SOCIAL` | Intermediate | host-guest roles, politeness norms, hierarchy, obligation |
| L1 | `LAYER_1_SURFACE` | Most refined | remove shoes, bow, touch feet, offer coffee |
| — | `UNKNOWN` | Unclear only | used when genuinely ambiguous |

**Abstraction→refinement direction:** L3 (abstract) → L2 → L1 (concrete). A complete **pathway** spans all three layers for a coherent culture.

Graph nodes carry a soft **layer distribution** (e.g., an L1 node gets `[0.70, 0.15, 0.15]` for [L1, L2, L3]) used for visualization and analysis.

Theoretical grounding (for the paper): Schein's three levels, Schwartz's basic values, Brown's human universals — see `RESEARCH_DESIGN.md`.

---

## 6. End-to-end data flow

### Publishable path (recommended for research)

```
CultureBank records
    → normalize (data_loader)
    → LLM triple extraction (triple_extractor, batched)
    → save corpus_triples.jsonl + corpus_passages.jsonl
    → build networkx graph + export (graph_builder)
    → [future: entity resolution, pathway construction]
    → index for retrieval (retriever)

CulturalBench questions
    → load & normalize (culturalbench_loader)
    → for each system S0–S3:
        retrieve evidence (passages or triples)
        → build MCQ prompt (mcq_systems)
        → LLM answer (llm_backend)
        → parse letter / True-False (mcq_systems)
    → score accuracy + bootstrap CI + McNemar (mcq_evaluator)
    → save JSON/CSV/MD/plot → data/outputs/
```

### MVP path (quick prototype)

```
CultureBank (N records)
    → normalize
    → LLM extract triples
    → build raw KG
    → LLM critique → refined triples → refined KG
    → LLM generate scenario questions
    → 3 systems answer (baseline / raw KG / refined KG)
    → heuristic scoring (0–100) + PASS/PARTIAL/FAIL
    → save artifacts → data/outputs/
```

---

## 7. Pipeline A — MVP experiment (`run_mvp.py`)

**Purpose:** End-to-end smoke test and prototype on a small sample (default 100 records, configurable).

**Command:**
```bash
python scripts/run_mvp.py \
  --dataset culturebank --split train --limit 100 \
  --extraction_mode llm_based --critic llm_based \
  --llm_provider ollama --ollama_model llama3.1:8b
```

### Steps (16 total)

| Step | Action | Module | Output |
|---|---|---|---|
| 0 | Preflight: check Ollama, warmup model | `local_llm` | — |
| 1 | Load CultureBank records | `data_loader` | `data/processed/culturebank_sample_{N}.jsonl` |
| 2 | Save normalized records | `utils.write_jsonl` | same |
| 3 | LLM extract triples (main method) | `triple_extractor` | per-record triple lists |
| 4 | Save raw triples | `utils` | `data/processed/raw_triples_{N}.jsonl` |
| 5 | Build raw KG | `graph_builder` | `data/outputs/raw_cultural_kg.*` |
| 6 | LLM critique each record | `critique_agent` | critique results |
| 7 | Save critiques | `utils` | `data/processed/critiques_{N}.jsonl` |
| 8 | Merge into refined triples | `critique_agent.build_refined_triples` | `data/processed/refined_triples_{N}.jsonl` |
| 9 | Build refined KG | `graph_builder` | `data/outputs/refined_cultural_kg.*` |
| 10 | Generate scenario questions (~25) | `evaluator.generate_scenarios` | in-memory examples |
| 11 | Baseline answers (no KG) | `evaluator` | — |
| 12 | Raw KG answers | `evaluator` + `retriever` | — |
| 13 | Refined KG answers | `evaluator` + `retriever` | — |
| 14 | Score all answers (heuristic 0–100) | `evaluator.score_answer` | — |
| 15 | Save evaluation outputs | `evaluator.save_evaluation_outputs` | `data/outputs/evaluation_*` |
| 16 | Print terminal summary | `run_mvp.print_summary` | PASS / PARTIAL / FAIL |

### Three MVP systems compared

| System | ID | What the LLM sees |
|---|---|---|
| Baseline | `baseline_prompt_only` | Scenario question only |
| Raw KG | `raw_kg_augmented` | Question + top-k retrieved **raw** triples |
| Refined KG | `critique_refined_kg_augmented` | Question + top-k retrieved **refined** triples |

### Fallback behavior

- Default: **`--allow_fallback` is OFF** — if Ollama is unavailable, the run **stops** with setup instructions (never fakes LLM extraction).
- With `--allow_fallback`: schema/rule-based extraction and critique are used; results are marked as not a true LLM experiment.

---

## 8. Pipeline B — Publishable CulturalBench evaluation (scripts 06–07)

**Purpose:** Objective, held-out evaluation on the established **CulturalBench** benchmark — the sole metric for the publishable paper.

### Step 6 — Build retrieval corpus

```bash
python scripts/06_build_retrieval_corpus.py --provider ollama --limit 2000
# or on GPU cluster:
python scripts/06_build_retrieval_corpus.py --provider vllm --limit 2000
```

| Sub-step | What happens | Output |
|---|---|---|
| Load CultureBank | Up to `--limit` records | in-memory |
| Build raw-text passages | One searchable passage per record (for S1) | `data/processed/corpus_passages.jsonl` |
| Batched LLM triple extraction | All records via `extract_triples_llm_batched` | `data/processed/corpus_triples.jsonl` |
| Build & export graph | networkx graph for inspection | `data/outputs/corpus_cultural_kg.*` |

**Out of scope in step 6 (later phases):** entity resolution, cross-cultural alignment, pathway construction, S4 system.

### Step 7 — Evaluate on CulturalBench

```bash
python scripts/07_eval_culturalbench.py --provider ollama --limit 250
```

| Sub-step | What happens | Output |
|---|---|---|
| Load CulturalBench (Easy + Hard) | Normalized MCQ questions | in-memory |
| Load corpus (passages + triples) | Retrieval indexes built once | `RetrievalContext` |
| Run each system S0–S3 | Batch LLM MCQ answering | per-system answers |
| Score | Accuracy, 95% bootstrap CI, McNemar | summary dict |
| Save | JSON, CSV, MD, plot | `data/outputs/culturalbench_*` |

### Four publishable systems (S0–S3)

| ID | Name | Evidence injected into prompt |
|---|---|---|
| **S0** | `S0_no_kg` | None — closed-book MCQ |
| **S1** | `S1_raw_text_rag` | Top-k retrieved **CultureBank text passages** |
| **S2** | `S2_flat_kg` | Top-k retrieved triples as a **flat bag** (layer labels **stripped**) |
| **S3** | `S3_layered_kg` | Top-k retrieved triples **with layer labels** shown |

All systems share the same LLM, prompt scaffold, and retrieval budget (`top_k`). Only the evidence differs — so accuracy differences are attributable to structure.

**S4 (planned):** `S4_abstraction_refinement_kg` — retrieve complete **pathways** with shared abstraction nodes and critique-completed bridges.

### Scoring (CulturalBench)

- **Easy unit:** one question = one unit; correct iff predicted letter matches gold.
- **Hard unit:** one question group = one unit; correct iff **all** True/False items in the group are correct.
- **Combined:** Easy questions + Hard groups pooled.
- **Statistics:** paired bootstrap 95% CIs; McNemar tests between system pairs.
- **No custom keyword scoring** — exact-match MCQ accuracy only.

---

## 9. Pipeline C — Step-by-step scripts (01–05)

These mirror the MVP stages individually for debugging:

| Script | Purpose |
|---|---|
| `01_load_dataset.py` | Load & normalize CultureBank → JSONL |
| `02_extract_triples.py` | LLM (or fallback) triple extraction |
| `03_build_graph.py` | Build networkx graph from triples JSONL |
| `04_run_critique.py` | Critique + build refined triples + refined graph |
| `05_evaluate.py` | MVP scenario evaluation (heuristic scoring) |

Additional utilities:

| Script | Purpose |
|---|---|
| `prep_offline_data.py` | Pre-download CultureBank + CulturalBench to HF cache for offline cluster runs |
| `run_mvp.py` | Full MVP orchestration (steps 01–05 combined) |
| `06_build_retrieval_corpus.py` | Corpus for CulturalBench systems |
| `07_eval_culturalbench.py` | CulturalBench MCQ evaluation |

SLURM job scripts (`slurm/build_corpus.sh`, `slurm/eval_culturalbench.sh`) wrap steps 06–07 for cluster execution.

---

## 10. How triples are extracted

**Module:** `src/triple_extractor.py`  
**Main method:** local LLM (Ollama or vLLM via backend).  
**Fallback:** schema/rule-based field mapping (`schema_fallback`) — **only** when LLM fails and `--allow_fallback` is set.

### Modes

| Mode | Behavior |
|---|---|
| `llm_based` (default) | LLM extraction only; optional fallback on failure |
| `schema_fallback` | Deterministic rules from record fields |
| `both` | LLM + schema, deduplicated |

### LLM extraction process (per record)

1. Format the normalized record into the **exact user prompt** (`src/prompts.py` → `TRIPLE_EXTRACTION_USER`).
2. Send to LLM with **system prompt** (`TRIPLE_EXTRACTION_SYSTEM`) requesting JSON only.
3. Parse JSON (`local_llm.generate_json` with repair/retry).
4. Validate layer labels via `layer_classifier.validate_triple_layers`.
5. Build `CulturalTriple` objects with full provenance.

### CulturalTriple schema (every triple preserves provenance)

```json
{
  "triple_id": "t_abc123",
  "subject": "...",
  "predicate": "...",
  "object": "...",
  "subject_layer": "LAYER_2_SOCIAL",
  "predicate_layer": "LAYER_1_SURFACE",
  "object_layer": "LAYER_1_SURFACE",
  "confidence": 0.8,
  "source_text": "...",
  "dataset_id": "culturebank",
  "instance_id": "...",
  "raw_fields": {},
  "extraction_method": "llm_based",
  "generated": false,
  "evidence": "actor_behavior",
  "notes": null
}
```

- **`generated=false`:** directly grounded in the source record.
- **`generated=true`:** inferred bridge triple; lower confidence; treated as uncertain in refined-KG answering.

### Batched extraction (scalable)

`extract_triples_llm_batched(records, backend)` formats all prompts upfront and calls `backend.generate(prompts)` in one batch — enabling efficient vLLM inference on GPU.

### Layer classifier (validation only)

`src/layer_classifier.py` is **not** the main extraction method. It:
- converts invalid LLM layer names → `UNKNOWN`
- provides keyword-based guessing as a last-resort fallback

---

## 11. How the knowledge graph is built

**Module:** `src/graph_builder.py`  
**Library:** `networkx.MultiDiGraph`

### From triples to graph

For each `CulturalTriple`:

1. **Subject node** — created/merged by normalized label; layer distribution accumulated from `subject_layer`.
2. **Object node** — same for `object_layer`.
3. **Directed edge** — subject → object, keyed by `triple_id`, carrying:
   - `predicate`, `predicate_layer`
   - `dataset_id`, `instance_id`, `source_text`
   - `confidence`, `generated`, `evidence`, `notes`, `extraction_method`

### Node attributes

| Attribute | Meaning |
|---|---|
| `label` | Canonical text label |
| `layer_distribution` | Soft [L1, L2, L3] distribution (accumulated from triples) |
| `dominant_layer` | Argmax of distribution |
| `sources` | Set of `instance_id`s contributing to this node |
| `frequency` | How many triples reference this node |
| `generated` | True only if all contributing triples are generated |

### Layer distribution priors (from `config.py`)

```
LAYER_1_SURFACE:    [0.70, 0.15, 0.15]
LAYER_2_SOCIAL:     [0.15, 0.70, 0.15]
LAYER_3_DEEP_VALUE: [0.15, 0.15, 0.70]
UNKNOWN:            [0.33, 0.33, 0.33]
```

### Exports (per graph prefix, e.g. `raw_cultural_kg`, `refined_cultural_kg`, `corpus_cultural_kg`)

| Format | File | Use |
|---|---|---|
| GraphML | `{prefix}.graphml` | Gephi, Cytoscape |
| JSON (node-link) | `{prefix}.json` | Programmatic analysis |
| Node CSV | `{prefix}_nodes.csv` | Spreadsheet / stats |
| Edge CSV | `{prefix}_edges.csv` | Spreadsheet / stats |
| Interactive HTML | `{prefix}.html` | pyvis browser visualization (layer-colored nodes) |

### What the current graph does NOT yet do

- **Entity resolution:** "remove shoes" and "take off shoes" remain separate nodes.
- **Cross-cultural alignment:** no shared abstraction nodes linking cultures.
- **Explicit pathway objects:** paths are implicit in the edge structure, not precomputed or indexed.

These are the main gaps for the publishable S4 system.

---

## 12. How critique and refinement work

**Module:** `src/critique_agent.py`

### Purpose

Inspect extracted triples per record and:
- identify missing Layer-2 social / Layer-3 value mediation
- flag unsupported or stereotyping-prone triples
- suggest cautious bridge triples (`generated=true`, confidence < 0.5)

### Main method: LLM critique

Uses exact prompts in `src/prompts.py` (`CRITIQUE_SYSTEM`, `CRITIQUE_USER`). Returns JSON with:
- `issues[]` — typed problems with severity
- `suggested_triples[]` — new bridge triples
- `triples_to_flag[]` — existing triples to annotate
- `summary`

### Fallback: rule-based critique

Heuristic checks only when LLM unavailable + `--allow_fallback`. Flags missing L2/L3 layers, surface-without-bridge, missing cultural group.

### Refined triple builder (`build_refined_triples`)

1. Keep **all original triples** (annotate flagged ones in `notes`).
2. Add critique **suggested triples** (deduplicated by subject+predicate+object per instance).
3. Grounded triples take priority over generated duplicates.

Output: `refined_triples_{N}.jsonl` → fed into `graph_builder` for the refined KG.

---

## 13. How retrieval works (current)

**Module:** `src/retriever.py`

### Two retriever classes

| Class | Indexes | Used by |
|---|---|---|
| `TripleRetriever` | `CulturalTriple` objects | S2, S3, MVP raw/refined KG |
| `TextRetriever` | Raw passage dicts (`text` field) | S1 raw-text RAG |

### Indexing

Each triple/passage is converted to a **rich text document** (subject, predicate, object, evidence, notes, source_text, cultural_group, topic) and embedded or vectorized.

### Backend selection (automatic)

1. **`sentence-transformers`** (`all-MiniLM-L6-v2`) if installed → dense cosine similarity.
2. **`TF-IDF`** (scikit-learn) otherwise → sparse cosine similarity.

### Query-time flow

1. Build query string from question (+ cultural group + options for MCQ).
2. Compute similarity against all indexed documents.
3. Return top-k `(item, score)` pairs.

### MVP retrieval (`evaluator.py`)

Same `TripleRetriever`; separate indexes for raw vs. refined triple sets. Retrieved triple IDs and scores logged to `retrieval_debug_{N}.json`.

### Limitations (current)

- Retrieves **isolated triples**, not connected pathways.
- No persistent index (rebuilt in memory each run).
- No hybrid BM25+dense or FAISS persistence yet.
- No culture-aware filtering or pathway-completeness ranking.

---

## 14. How pathways will work (planned — S4 system)

This is the **target retrieval unit** for the publishable paper. Not yet implemented.

> **Implementation note (important).** Pathways are assembled by pivoting on the
> **L2 (social norm) hub**, not by walking downward from an L3 node.
>
> The corpus stores cultural structure as `L2 → L1` (norm to practice) and
> `L2 → L3` (norm to value), because the social norm is the natural *subject* of
> both statements. Measured on the 22,990-record corpus: 57,966 `L2→L1` edges and
> 22,521 `L2→L3` edges, but only **19** `L3→L2` edges. A traversal that starts at
> L3 and follows outgoing edges therefore finds almost nothing — it produced 150
> pathways with **zero** complete ones.
>
> An L2 hub with at least one L3 neighbour and one L1 neighbour *is* a complete
> value→norm→practice chain, just stored with the norm as subject. Edges are
> treated as undirected for discovery and nodes are ordered L3→L2→L1 for output.

### Pathway definition

A path `p = (v3 → v2 → v1)` where:
- `layer(v3) = L3`, `layer(v2) = L2`, `layer(v1) = L1`
- Nodes share at least one **canonical** culture (see `src/culture_normalizer.py`)
- Every hop carries provenance (at least one contributing `triple_id`)
- Optionally passes through a **shared abstraction node** linking ≥2 cultures

Example:
```
respect for elders (L3)
  → deference norm (L2)
    → touch elder's feet (L1, Indian)
    → bow deeply (L1, Japanese)   ← cross-cultural pivot
```

### Planned construction pipeline

```
corpus triples
  → entity resolution (merge lexical variants)
  → cross-cultural abstraction alignment (shared mid-level nodes)
  → pathway assembly (L3→L2→L1 chains per culture)
  → mark completeness (FULL / SOCIAL / SURFACE / NONE)
  → index pathways (FAISS/Qdrant + BM25)
```

### Planned retrieval flow

1. **Candidate lookup** — hybrid dense+sparse over abstraction nodes, refinement nodes, and precomputed pathways.
2. **Graph expansion** — from candidates, traverse upward (practice→value) and downward (value→practice); include cross-cultural analogies via shared nodes.
3. **Pathway ranking** — relevance × completeness × provenance strength × culture match; penalize generated dominance.
4. **Prompt injection** — structured pathway bundle (not flat triple list) into MCQ prompt.

### S4 system (planned)

`S4_abstraction_refinement_kg` — same as S3 but retrieves **ranked pathways** from the critique-completed, entity-resolved graph. Compared against S0–S3 on CulturalBench; H2 tested by bucketing accuracy on pathway completeness.

---

## 15. How evaluation works

The project has **two evaluation tracks**:

### Track 1 — MVP heuristic evaluation (`src/evaluator.py`)

Used by `run_mvp.py` and `05_evaluate.py`. **Not** the publishable metric.

**Scenario generation:** LLM (or fallback) creates open-ended cultural scenario questions from CultureBank records.

**Three systems:** baseline / raw KG / refined KG — free-text answers.

**Scoring formula (out of 100):**

| Component | Points |
|---|---|
| Keyword coverage | 30 |
| Cultural group match | 15 |
| Provenance use | 20 |
| Layer coverage (L2/L3 terms in answer) | 20 |
| Answer specificity | 10 |
| Avoids bad keywords | 5 |
| Generated dominance penalty | up to −15 |

**Verdict:** PASS / PARTIAL PASS / FAIL based on improvement thresholds and generated-support ratio.

**Outputs:** `evaluation_examples_100.json`, `evaluation_results_100.json/.csv`, `evaluation_summary_100.md`, `evaluation_average_scores_100.png`, `retrieval_debug_100.json`.

### Track 2 — CulturalBench MCQ evaluation (`src/mcq_evaluator.py`) — **primary for publication**

Used by `07_eval_culturalbench.py`.

**Metric:** exact-match MCQ accuracy (Easy items + Hard groups).

**Statistics:**
- Per-system accuracy with **95% paired bootstrap CIs**
- **McNemar's test** for paired significance between systems
- Reported for Easy, Hard, and Combined

**Outputs:** `culturalbench_results.json`, `culturalbench_results.csv`, `culturalbench_summary.md`, `culturalbench_accuracy.png`, per-item results.

**Systems:** S0 (no KG) → S1 (raw text) → S2 (flat KG) → S3 (layered KG) → *[future S4 pathways]*.

---

## 16. LLM backends

**Module:** `src/llm_backend.py`

All LLM work goes through a pluggable backend implementing `generate(prompts) -> list[str]`.

| Backend | When | Batching | Module |
|---|---|---|---|
| **OllamaBackend** | Local dev | Thread pool (`OLLAMA_MAX_WORKERS=4`) | wraps `local_llm.py` |
| **VLLMBackend** | GPU cluster (Narval) | True batched inference | lazy vLLM import |

**Selection:** `--provider ollama` or `--provider vllm` (env: `LLM_PROVIDER`).

**Default models:**
- Ollama: `llama3.1:8b`
- vLLM: `Qwen/Qwen2.5-7B-Instruct`

**JSON generation:** `local_llm.generate_json` — used for extraction and critique; includes JSON repair, retry, and Ollama `format=json` mode.

**Warmup:** `local_llm.warmup_model()` / `backend.warmup()` — preloads model to avoid first-call timeout on cold start.

**No paid APIs.** Ever.

---

## 17. Source modules reference

| Module | Responsibility |
|---|---|
| `config.py` | Paths, env vars, dataset IDs, layer distributions, model defaults |
| `schemas.py` | Pydantic models: `CulturalRecord`, `CulturalTriple`, `CritiqueResult`, `EvaluationExample`, `MCQQuestion`, etc. |
| `prompts.py` | Verbatim LLM prompts (extraction, critique, scenario, answering, MCQ) |
| `data_loader.py` | Load & normalize CultureBank |
| `culturalbench_loader.py` | Load & normalize CulturalBench MCQ |
| `local_llm.py` | Ollama adapter: generate, JSON parse, warmup, availability check |
| `llm_backend.py` | Pluggable Ollama / vLLM backend |
| `triple_extractor.py` | LLM + schema triple extraction (single + batched) |
| `layer_classifier.py` | Layer validation / keyword fallback (not main extraction) |
| `graph_builder.py` | networkx graph build + multi-format export |
| `critique_agent.py` | LLM + rule critique; refined triple merger |
| `retriever.py` | `TripleRetriever`, `TextRetriever` (TF-IDF / sentence-transformers) |
| `evaluator.py` | MVP scenario gen, 3-system free-text eval, heuristic scoring |
| `mcq_systems.py` | CulturalBench S0–S3 prompt building, parsing, batch answering |
| `mcq_evaluator.py` | CulturalBench accuracy, bootstrap CI, McNemar, output saving |
| `utils.py` | Logging, JSONL I/O, IDs, text normalization |

---

## 18. Directory layout and artifacts

```
cultural-kg-mvp/
├── README.md                    # Quick start
├── requirements.txt
├── .env.example
├── docs/
│   ├── PROJECT_OVERVIEW.md      # ← this file
│   ├── RESEARCH_DESIGN.md       # Formal claim, hypotheses, ablations
│   └── NARVAL_SETUP.md          # Cluster/GPU instructions
├── data/
│   ├── raw/                     # Optional raw downloads
│   ├── processed/               # Normalized records, triples, corpus
│   └── outputs/                 # Graphs, evaluation results, plots
├── src/                         # Core library (see §17)
├── scripts/                     # Runnable pipelines (see §9)
├── slurm/                       # Cluster job scripts
└── tests/                       # pytest suite (offline, LLM mocked)
```

### Key processed artifacts

| File | Produced by | Contents |
|---|---|---|
| `culturebank_sample_{N}.jsonl` | 01 / run_mvp | Normalized CultureBank records |
| `raw_triples_{N}.jsonl` | 02 / run_mvp | LLM-extracted triples |
| `refined_triples_{N}.jsonl` | 04 / run_mvp | Original + critique-suggested triples |
| `critiques_{N}.jsonl` | 04 / run_mvp | Per-record critique JSON |
| `corpus_passages.jsonl` | 06 | Raw-text retrieval passages (S1) |
| `corpus_triples.jsonl` | 06 | Full-corpus triple store (S2/S3) |

### Key output artifacts

| File | Contents |
|---|---|
| `raw_cultural_kg.*` / `refined_cultural_kg.*` / `corpus_cultural_kg.*` | Graph exports (graphml, json, csv, html) |
| `evaluation_*` | MVP heuristic eval results |
| `culturalbench_*` | CulturalBench MCQ eval results |
| `run_meta.json` | Run configuration snapshot |

---

## 19. Configuration and environment

Copy `.env.example` → `.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `vllm` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_TIMEOUT` | `600` | Seconds (generous for cold start) |
| `VLLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Cluster model |
| `CULTUREBANK_HF_ID` | `SALT-NLP/CultureBank` | KG source |
| `CULTURALBENCH_HF_ID` | `kellycyy/CulturalBench` | Eval benchmark |
| `CORPUS_LIMIT` | `2000` | Max CultureBank records for corpus |
| `RETRIEVAL_TOP_K` | `8` | Triples/passages retrieved per question |
| `USE_LLM_EXTRACTION` | `true` | Feature flag |
| `USE_LLM_CRITIQUE` | `true` | Feature flag |

---

## 20. Commands cheat sheet

### Local development (Ollama)

```bash
# Setup
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama serve

# MVP quick test (3 records)
python scripts/run_mvp.py --limit 3 --num_scenarios 3 \
  --extraction_mode llm_based --critic llm_based --ollama_model llama3.1:8b

# MVP full sample
python scripts/run_mvp.py --limit 100 --extraction_mode llm_based --critic llm_based

# Build retrieval corpus (small)
python scripts/06_build_retrieval_corpus.py --provider ollama --limit 50

# CulturalBench smoke test
python scripts/07_eval_culturalbench.py --provider ollama --limit 6 \
  --systems S0_no_kg,S3_layered_kg

# Run tests
pytest -q
```

### Cluster (vLLM / GPU)

```bash
# On login node: stage datasets offline
python scripts/prep_offline_data.py

# On compute node
python scripts/06_build_retrieval_corpus.py --provider vllm --limit 2000
python scripts/07_eval_culturalbench.py --provider vllm --limit 250
```

---

## 21. Roadmap — what comes next

Aligned with `RESEARCH_DESIGN.md` phases:

### Phase 1 — Design lock & benchmark wiring ✅ (mostly done)

- [x] CulturalBench loader
- [x] S0–S3 MCQ systems
- [x] Accuracy + bootstrap CI + McNemar
- [x] Pluggable LLM backend (Ollama + vLLM)
- [ ] First full CulturalBench numbers on meaningful corpus size

### Phase 2 — Scalable pipeline ✅ (implemented)

- [x] Batched inference with caching & checkpoint/resume
- [x] Parquet storage for triples / entities / pathways
- [x] **Entity resolution / node canonicalization**
- [x] **Cross-cultural abstraction alignment** (shared nodes; embedding + LLM verify)
- [x] **Pathway construction** (L3→L2→L1 chains)
- [x] Persistent pathway embedding index (`pathway_retriever.py`)
- [ ] FAISS/Qdrant hybrid index (optional upgrade; TF-IDF/MiniLM baseline in place)

### Phase 3 — Full construction + S4 + ablations

- [ ] Implement **S4 abstraction→refinement pathway system**
- [ ] Full CultureBank corpus (both splits)
- [ ] Ablations: S2 vs S3 vs S4; triple vs pathway retrieval; with/without shared nodes
- [ ] ≥2 base LLM models

### Phase 4 — Evaluation & mechanism analysis

- [ ] Full CulturalBench Easy + Hard evaluation
- [ ] **H2 money figures:** accuracy by pathway-completeness bucket; shared-node availability
- [ ] Human validation of extraction quality, layer labels, pathway edges
- [ ] Stereotype audit

### Phase 5 — Paper

- [ ] Error analysis, qualitative pathway examples
- [ ] Ethics / representation / fairness analysis
- [ ] Reproducibility artifacts (released KG, prompts, configs)
- [ ] Draft for target venue (EMNLP / ACL / NAACL or FAccT / CSCW)

---

## 22. Limitations and design principles

### Design principles

1. **Local LLM is the main method** — rule/schema extraction is fallback only, never default.
2. **Provenance on every triple** — dataset_id, instance_id, source_text, extraction_method, confidence, generated flag.
3. **Held-out evaluation** — build on CultureBank, evaluate on CulturalBench.
4. **No stereotyping by design** — prompts explicitly forbid generalizing beyond the source record; generated triples marked uncertain.
5. **Reproducible & open** — no paid API; pinned models; released artifacts.

### Current limitations

- Graph nodes are **not entity-resolved** — duplicate labels fragment the graph.
- Retrieval is **triple-level**, not pathway-level — S4 not yet built.
- MVP evaluation uses **heuristic scoring**, not human judgment — use CulturalBench for research claims.
- CultureBank field mapping may miss dataset-specific columns (preserved under `raw`).
- Local 8B models can produce imperfect JSON; repair/retry mitigates but some records may yield fewer triples.
- The KG reflects **source records only** — must not be read as generalizations about entire cultures.

---

*Last updated: reflects Phase 2 pipeline (entity resolution, alignment, pathways, checkpoint/resume). S4 MCQ system remains Phase 3.*
