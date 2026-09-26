# AGENTS.md

Guidance for AI agents (Claude Code, its subagents, and other coding agents)
working in this repository. Read this before changing code or interpreting
results.

## What this project is

A research pipeline that tests whether a **layered cultural knowledge graph**
helps an LLM answer culturally grounded questions. The layers are:

- **L3**: deep value
- **L2**: social norm
- **L1**: observable practice

The KG is built from **CultureBank** and evaluated on the held-out
**CulturalBench** benchmark (Easy MCQ plus Hard True/False groups). There are
seven systems:

| ID | Evidence |
|---|---|
| S0 | closed-book, no evidence |
| S1 | raw passages |
| S2 | flat triples |
| S3 | layered triples |
| S4 | abstraction→refinement pathways |
| A3 | ablation: refined triples instead of pathways |
| A4 | ablation: pathways without shared pivots |

All systems share one model and prompt scaffold; only the evidence changes.

**Current state (Sep 2026).** The checked-in full run is a *negative* result:
S0 gets 52.2%, and every retrieval system is worse (S4 38.4%). Work is in the
diagnosis and consolidation stage, not tuning for a paper. Read
`docs/RUN_HISTORY.md` before drawing conclusions from anything in `data/`.

## Where to look

| Need | Read |
|---|---|
| Package map, data flow, artifact files, config knobs | `docs/ARCHITECTURE.md` |
| Coding and research-integrity rules | `docs/CONVENTIONS.md` |
| Past runs, artifact lineage, branches, stash, open issues | `docs/RUN_HISTORY.md` |
| Hypotheses H1–H3, ablations, statistics protocol | `docs/RESEARCH_DESIGN.md` |
| Narval setup, SLURM jobs, offline staging, **where models/caches live** | `docs/NARVAL_SETUP.md` (§0 storage layout) |
| Plain-language overview and results | `README.md`, `docs/PROJECT_OVERVIEW.md` |

Code layout: `src/{core,llm,loaders,construction,retrieval,evaluation,legacy}`.
The pipeline scripts are `scripts/{build_corpus,validate_corpus,eval_culturalbench,compare_runs}.py`.
`scripts/legacy/` and `src/legacy/` are the frozen MVP; do not build on them.

## Running things on Narval

The venv does not include scipy, pandas, or pyarrow; they come from the cluster
modules. **Load the modules first, in the current shell.** Never pipe
`load_modules.sh` or run it in a subshell, because the module environment would
be lost.

```bash
cd <repo root>
source slurm/load_modules.sh && load_login_modules
source slurm/env_hf.sh && setup_hf_offline
export EMBEDDING_BACKEND=tfidf          # else MiniLM loads from Lustre for ~10 min
.venv/bin/python -m pytest -q -p no:cacheprovider      # 133 tests, ~1-2 min
```

`bash slurm/run_tests.sh` does the same thing. Use `sbatch slurm/run_tests.sh`
if the login node's Lustre client stalls.

- A single test file: `.venv/bin/python -m pytest -q tests/test_x.py`.
- A bare `pytest` from the repo root is safe, because `pytest.ini` limits
  collection to `tests/`.
- GPU work (corpus build, eval) runs only through SLURM (`slurm/*.sh`). vLLM
  model load alone takes about 16–18 minutes per job; jobs cap at 3 hours on
  `gpubase_bygpu_b1`.
- The repo sits inside a workspace folder with sibling `models/` (backup weights)
  and `hf_cache/` (empty). Jobs use `$SCRATCH/hf_cache`. Never copy weights or
  caches into the repo. See NARVAL_SETUP §0, including the `.venv` path issue.
- Never start a GPU job, or anything that overwrites `data/processed/` or
  `data/outputs/`, without the user's explicit go-ahead. Those files are the
  only copy of the August full run.

## Rules that are easy to break

- **No `--allow_fallback` in research runs.** Fallback output must stay tagged
  (`llm_failed`, `generated`) and excluded from pathways.
- **Do not introduce a single global LLM token budget.** Use the per-stage
  `STAGE_MAX_TOKENS`. A 512-token budget once truncated 96% of the corpus.
- **The LLM cache key must include the model and `max_tokens`.** Otherwise
  stale truncated completions replay silently.
- **Rebuilds need a fresh `RUN_TAG`.** An old checkpoint marks every record
  complete and skips extraction. `RUN_TAG` isolates only the cache and
  checkpoint, *not* the output files; see `ARCHITECTURE.md` §5.
- **Pass `--state_path` to `validate_corpus.py`** so it matches the build.
  `slurm/build_corpus.sh` does this.
- **Pilot samples must span cultures.** The first N CultureBank records are
  about 99.8% United States, so a sequential pilot cannot test cross-cultural
  alignment.
- **Treat S3 and A3 as identical evidence for now.** Both use refined triples,
  which contradicts the design. This is open issue #1 in `RUN_HISTORY.md`.
  Do not "fix" it without asking; it is a research decision.
- **Respect subpackage layering.** `tests/test_architecture.py` enforces it.
  New modules go in a subpackage, never at the top level of `src/`.
- **Every prompt lives in `src/llm/prompts.py`.** Changing a prompt changes the
  experiment, so say so in the commit.

## Working agreements

- Keep numbers in docs traceable to a file in `data/outputs/` or a log in
  `slurm/logs/`. Never invent or round results into a stronger claim.
- After a run finishes or a branch changes state, update `docs/RUN_HISTORY.md`.
- After changing structure or config, update `docs/ARCHITECTURE.md`.
- Branch before committing. Unmerged work lives on `origin/fix-flexible-pathways`
  and in `stash@{0}`. Both predate the subpackage split; see RUN_HISTORY §3
  before merging them.
- When delegating to a subagent, point it at this file and at the specific doc
  section it needs. Tell it not to launch SLURM jobs or overwrite `data/`.
