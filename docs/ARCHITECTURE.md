# Architecture

How the code is organized, how data flows through it, and which files each
stage reads and writes. For the research question and hypotheses see
[`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md); for coding rules see
[`CONVENTIONS.md`](CONVENTIONS.md).

## 1. Repository map

```text
.
├── AGENTS.md                 context and ground rules for AI agents (CLAUDE.md imports it)
├── src/                      the library (all logic lives here)
│   ├── core/                 config, schemas, IO, checkpoints, cache, culture names, embeddings
│   ├── llm/                  LLM backends (Ollama / vLLM) and every prompt template
│   ├── loaders/              CultureBank (construction) and CulturalBench (evaluation)
│   ├── construction/         extraction → critique → resolution → alignment → graph → pathways
│   ├── retrieval/            passage / triple / pathway retrieval and reranking
│   ├── evaluation/           S0–S4 systems, scoring, statistics, mechanism analysis
│   └── legacy/               original MVP open-ended evaluator (not a research metric)
├── scripts/                  thin CLI entry points; one per pipeline step
│   ├── prep_offline_data.py  stage datasets/model on the login node
│   ├── build_corpus.py       step 1: build every corpus artifact
│   ├── validate_corpus.py    step 2: quality gates (fail loud)
│   ├── eval_culturalbench.py step 3: S0–S4 + ablations on CulturalBench
│   ├── compare_runs.py       step 4: before/after report vs an archived run
│   └── legacy/               MVP steps 01–05 and run_mvp.py (debugging only)
├── slurm/                    Narval job scripts and shared shell helpers
│   ├── load_modules.sh       module setup (source it; never pipe it)
│   ├── env_hf.sh             offline Hugging Face cache env
│   ├── build_corpus.sh       GPU job → build_corpus.py + validate_corpus.py
│   ├── eval_full.sh          GPU job → eval (all systems + floor sweep) + compare_runs.py
│   ├── eval_culturalbench.sh GPU job → eval with a custom system set
│   ├── run_tests.sh          CPU job (or plain `bash`) → pytest
│   ├── prep_offline.sh       login node → prep_offline_data.py
│   └── watchers/             overnight babysitters that chain/resubmit jobs
├── tests/                    offline unit + pipeline-integrity + architecture tests
├── data/                     raw/ processed/ outputs/ (mostly git-ignored, see §5)
└── docs/                     research design, architecture, conventions, run history
```

## 2. Package layering

Each subpackage may import only from itself and the packages in its "May
import" column. Dependencies run one way, from `evaluation` down to `core`.
`tests/test_architecture.py` enforces the table, so a violation fails the test
suite instead of eroding silently. To add an edge, update this table and
`ALLOWED` in that test together.

| Package | May import | Contains |
|---|---|---|
| `core` | nothing in `src` | `config`, `schemas`, `utils`, `storage`, `culture_normalizer`, `embedding_utils` |
| `llm` | `core` | `llm_backend` (interface + Ollama/vLLM), `local_llm` (Ollama client, JSON repair), `prompts` |
| `loaders` | `core` | `culturebank_loader`, `culturalbench_loader` |
| `construction` | `core`, `llm` | `triple_extractor`, `critique_agent`, `layer_classifier`, `entity_resolver`, `cross_cultural_aligner`, `graph_builder`, `pathway_builder` |
| `retrieval` | `core` | `retriever` (passages, triples), `pathway_retriever` (persistent index), `retrieval_ranking` |
| `evaluation` | `core`, `llm`, `retrieval` | `mcq_systems`, `mcq_evaluator`, `mechanism_analysis` |
| `legacy` | `core`, `llm`, `retrieval` | `evaluator` (MVP heuristic scorer) |

Scripts orchestrate across packages; the library modules never import scripts.

## 3. Pipeline and data flow

```text
                      ┌─────────────── scripts/build_corpus.py ───────────────┐
CultureBank (HF) ──►  loaders.culturebank_loader
                        │  records ──► corpus_passages.jsonl ───────────────────────────► S1
                        ▼
                      construction.triple_extractor   (LLM, batched, cached, resumable)
                        │  ──► corpus_triples.jsonl (+ .parquet) ───────────────────────► S2
                        ▼
                      construction.critique_agent     (+ layer_classifier)
                        │  ──► corpus_critiques.jsonl
                        │  ──► corpus_triples_refined.jsonl (+ .parquet) ─────────────► S3, A3
                        ▼   (schema-fallback triples, llm_failed=True, are dropped here)
                      construction.entity_resolver    ──► canonical entities
                        ▼
                      construction.cross_cultural_aligner ──► cross_cultural_alignments.jsonl
                        │                                ──► entity_map.jsonl (+ .parquet)
                        ▼
                      construction.graph_builder      ──► outputs/resolved_cultural_kg.*
                        ▼
                      construction.pathway_builder    ──► corpus_pathways.jsonl (+ .parquet)
                        ▼
                      retrieval.pathway_retriever     ──► pathway_index/ ─────────────────► S4, A4
                      └──────────────────────────────────────────────────────────────────┘

scripts/validate_corpus.py  reads the above + the checkpoint → outputs/phase2_validation.md

CulturalBench (HF) ──► loaders.culturalbench_loader
                        ▼
scripts/eval_culturalbench.py
   evaluation.mcq_systems      build one prompt per (system, question); one batched generate()
   evaluation.mcq_evaluator    Easy item / Hard group units, paired bootstrap, McNemar
   evaluation.mechanism_analysis   accuracy by completeness bucket and shared pivot
     ──► outputs/culturalbench_{results.json,results.csv,summary.md,accuracy.png}
     ──► outputs/mechanism_analysis.md, mechanism_h2_completeness.png

scripts/compare_runs.py ──► outputs/before_after_comparison.md
```

### Stage notes

- **Extraction** asks the LLM for layered triples per record. Responses are
  cached by `core.storage.generate_with_cache`, keyed on prompt, model, and
  token budget. Progress is checkpointed in `corpus_build_state*.json`. If the LLM
  returns empty or unparseable output, `CORPUS_SCHEMA_FALLBACK_ON_EMPTY` lets the
  schema mapper emit field-based triples. Those are tagged `llm_failed=True`:
  they stay in the flat stores so S1/S2 remain comparable, but they are
  **excluded from entity resolution and pathways**.
- **Critique** flags unsupported or stereotyping triples and suggests at most
  `MAX_CRITIQUE_SUGGESTIONS` bridge triples. Suggestions are always
  `generated=True` with low confidence. Only *content* issues on explicitly
  flagged triples drop a triple. *Structural* issues such as
  `surface_without_bridge` never erase a record.
- **Entity resolution** merges surface variants within a layer, using
  embeddings with a bounded neighbour search (`ER_MAX_NEIGHBORS`). Culture
  names are canonicalized by `core.culture_normalizer`, so "American",
  "Americans" and "United States" are the same culture.
- **Alignment** proposes cross-culture pairs of L2/L3 entities by embedding
  similarity and optionally verifies them with the LLM. Verified pairs become
  *shared abstraction pivots*.
- **Pathways** pivot on L2 hubs. The corpus stores `L2→L1` and `L2→L3`, since
  the norm is the natural subject of both. An L2 node that has an L3 neighbour
  and an L1 neighbour in the same culture therefore forms a FULL `L3→L2→L1`
  pathway. Every hop must share a culture.
- **Retrieval** ranks by similarity, then applies `retrieval_ranking.rerank_hits`:
  a culture-match multiplier, a generated-triple penalty, a per-culture
  diversity cap, and an absolute relevance floor (`RETRIEVAL_MIN_SCORE`). If
  nothing clears the floor, the system answers closed-book with an empty
  evidence block.

## 4. Systems → evidence source

| ID | Name in code | Evidence store | Formatter |
|---|---|---|---|
| S0 | `S0_no_kg` | none | — |
| S1 | `S1_raw_text_rag` | `corpus_passages.jsonl` | passages |
| S2 | `S2_flat_kg` | `corpus_triples` (raw) | triples, layer labels stripped |
| S3 | `S3_layered_kg` | `corpus_triples_refined` (falls back to raw) | triples with layer labels |
| S4 | `S4_abstraction_refinement_kg` | `pathway_index/` | pathways |
| A3 | `S4_triple_retrieval` | `corpus_triples_refined` | **same as S3** |
| A4 | `S4_no_shared_pivot` | `pathway_index/`, shared-pivot pathways removed | pathways |

> **Known design/code mismatch.** `RESEARCH_DESIGN.md` §7 defines S3 as the
> *raw* layered KG, so that S3 vs S4 isolates the effect of refinement (ablation
> A2). Since commit `033425a` the code feeds S3 the *refined* triples. As a result:
> (1) S3 and A3 receive identical evidence and differ only in the one-line
> evidence header; (2) S2 vs S3 now changes both layering *and* refinement.
> In the full run S3 and A3 still disagreed on 70 of 2,454 units, a useful
> estimate of prompt-wording noise. Whether to restore raw triples for S3 is a
> research decision; it is listed in `RUN_HISTORY.md` under open issues.

The answering prompt, model, decoding, and `top_k` are shared by all systems.
Only the evidence block and its header (`llm.prompts.MCQ_EVIDENCE_HEADERS`)
change.

## 5. Artifacts on disk

`config.PROJECT_ROOT/data/`:

| Path | Written by | Tracked in git |
|---|---|---|
| `processed/corpus_passages.jsonl` | build_corpus | no |
| `processed/corpus_triples.jsonl` / `.parquet` | build_corpus | no |
| `processed/corpus_critiques.jsonl` | build_corpus | no |
| `processed/corpus_triples_refined.jsonl` / `.parquet` | build_corpus | no |
| `processed/entity_map.jsonl` / `.parquet` | build_corpus | no |
| `processed/cross_cultural_alignments.jsonl` | build_corpus | no |
| `processed/corpus_pathways.jsonl` / `.parquet` | build_corpus | no |
| `processed/pathway_index/` (`config.json`, `embeddings.npy`, `pathways.jsonl`) | build_corpus | no |
| `processed/corpus_build_state[_<RUN_TAG>].json` | build_corpus (checkpoint) | no |
| `processed/llm_cache[_<RUN_TAG>]/` | LLM response cache | no |
| `outputs/corpus_cultural_kg.*` | raw inspection graph | yes (`.json/.graphml/.html` via Git LFS; CSVs plain) |
| `outputs/resolved_cultural_kg.*` | resolved graph | no |
| `outputs/culturalbench_*` | eval_culturalbench | yes (plain git; results.json is ~26 MB) |
| `outputs/phase2_validation.md` | validate_corpus | no |
| `outputs/mechanism_analysis.md`, `before_after_comparison.md` | eval / compare_runs | no |
| `outputs/broken_run_baseline/` | archived July run, input to compare_runs | no |

Eval reads the Parquet mirror when one sits next to a JSONL store. On Lustre
this is roughly 10× faster than parsing the 430–525 MB JSONL files.

The `RUN_TAG` environment variable in `slurm/build_corpus.sh` isolates a
rebuild's cache and checkpoint. It does **not** isolate the triple, pathway, or
output files, which always use the default names above. Two runs in one
checkout therefore overwrite each other's artifacts. See
[`RUN_HISTORY.md`](RUN_HISTORY.md).

## 6. Configuration

All settings live in `src/core/config.py`, which reads environment variables
(and `.env` if present). Scripts may override a value for one run
(`eval_culturalbench.py --min_score` sets `config.RETRIEVAL_MIN_SCORE`). The
knobs that matter most:

| Variable | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `vllm` on GPU nodes (SLURM scripts set it) |
| `VLLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | model for every stage and for answering |
| `EXTRACTION_MAX_TOKENS` / `CRITIQUE_MAX_TOKENS` | 2048 / 1536 | per-stage budgets; part of the cache key |
| `CORPUS_SCHEMA_FALLBACK_ON_EMPTY` | true | tags fallback triples `llm_failed` (kept out of pathways) |
| `EMBEDDING_BACKEND` | `auto` | `tfidf` for tests; MiniLM loads slowly from Lustre |
| `ER_SIMILARITY_THRESHOLD` / `ALIGNMENT_SIMILARITY_THRESHOLD` | 0.88 / 0.82 | merge / pivot candidate thresholds |
| `MAX_PATHWAYS_PER_CULTURE` / `MAX_TOTAL_PATHWAYS` | 500 / 150000 | pathway index caps |
| `RETRIEVAL_TOP_K` | 8 | evidence budget for every system |
| `RETRIEVAL_MIN_SCORE` | 0.15 | relevance floor; below it, answer closed-book |
| `RETRIEVAL_MAX_PER_CULTURE` | 3 | diversity cap per result set |
| `GATE_MIN_*` | see `validate_corpus.py` | quality-gate thresholds (lower them for pilots) |

`USE_LLM_*` and `NUM_SCENARIOS` are used only by the legacy MVP path.

## 7. Guards against known failure modes

Each guard exists because a past run failed that way (see `RUN_HISTORY.md`).
Tests in `tests/test_pipeline_integrity.py` pin most of them.

| Failure seen | Guard |
|---|---|
| One global 512-token budget truncated 96% of extraction JSON | per-stage budgets and a retry at `JSON_RETRY_TOKEN_MULTIPLIER` |
| A raised budget replayed ~44k truncated cached completions | cache key includes model and `max_tokens` |
| Resume skipped extraction because an old checkpoint marked all records done | `RUN_TAG` → fresh state and cache paths |
| Failed records were marked complete | failures go to `*_failed`, not `*_completed` |
| Schema-fallback triples (hardcoded layers) starved pathways | `llm_failed` tag; excluded from ER and pathways |
| Structural critique issues erased whole records | only flagged content issues drop a triple |
| 150 pathways, 0 FULL, from walking down from L3 | L2-hub pivoting |
| 2,410 culture strings; no overlap with benchmark cultures | `core.culture_normalizer` |
| Irrelevant evidence injected unconditionally | relevance floor and closed-book fallback |
| Hard scored per True/False row, inflating accuracy | Hard units grouped by `question_idx`; all rows must be correct |
| A job reported success on a broken corpus | `validate_corpus.py --strict` gates in `build_corpus.sh` |
