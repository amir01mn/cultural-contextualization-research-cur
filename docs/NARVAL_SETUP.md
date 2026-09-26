# Running on Narval (Digital Research Alliance of Canada)

This guide covers building the full cultural KG and evaluating systems S0–S4 on
Narval via SLURM. It assumes you have an Alliance account and an allocation (the
`def-<pi>` account string).

Two hard facts shape the workflow:

1. Compute nodes have NO internet. Datasets and model weights must be downloaded
   on the LOGIN node first, then read offline by the GPU job.
2. We use vLLM for batched GPU inference (not Ollama). Ollama remains the local
   dev backend.

```mermaid
flowchart TD
  login["Login node: create venv and stage data/model"] --> corpus["GPU job: build_corpus.sh"]
  corpus --> validate["Strict corpus and pathway validation"]
  validate --> eval["GPU job: eval_full.sh (S0-S4 + ablations)"]
  eval --> out["Evaluation, mechanism, and comparison reports"]
```

## 0. Storage layout (where things live)

The git repo is one folder inside a larger workspace. Model weights and
dataset caches deliberately stay **outside** the repo: they are tens of GB,
must never be committed, and would slow every Lustre directory walk (pytest,
`git status`).

```text
/project/def-enaskt/amir01mn/cultural-contextualization-research-cur/   workspace (backed up; group quota 1 TB)
├── cultural-contextualization-research-cur/   this git repo (the active checkout)
├── models/
│   ├── Qwen2.5-7B-Instruct/   13 GB, byte-identical to HF revision a09a354 that every run loaded
│   └── Qwen2.5-3B-Instruct/   5 GB, not referenced by any code (carried over from culfit)
└── hf_cache/                  empty; nothing points here

/scratch/amir01mn/   ($SCRATCH: fast, NOT backed up, unused files purged after ~60 days)
├── hf_cache/        THE LIVE CACHE. HF_HOME for every job (slurm/env_hf.sh):
│                    CultureBank, CulturalBench, Qwen2.5-7B-Instruct@a09a354,
│                    and an NLI model used only by the abandoned rebuild v2
├── models/          another copy of the same two Qwen models
├── cultural-contextualization-research-cur/   OLD clone (Aug 3); do not edit
└── venvs/           culfit project environments (unrelated)
```

Jobs read datasets and weights from `$SCRATCH/hf_cache` only. The
`/project` copy of the 7B weights is the purge-proof backup. If scratch is
purged, point vLLM at it directly (a local path is accepted):

```bash
VLLM_MODEL=/project/def-enaskt/amir01mn/cultural-contextualization-research-cur/models/Qwen2.5-7B-Instruct \
  sbatch slurm/eval_full.sh
```

Datasets have no backup copy; re-stage them with `slurm/prep_offline.sh` on a
login node.

> **Known issue: the venv points at the old scratch clone.** This checkout's
> `.venv` was copied from the scratch clone, and 62 of its `bin/` scripts
> (including `activate` and `pip`) still hardcode
> `/scratch/amir01mn/cultural-contextualization-research-cur/.venv`. So
> `source .venv/bin/activate`, which every SLURM job runs, actually switches
> to the scratch venv, and `pip install` installs there. The package sets are
> currently identical, so results are unaffected. But if scratch purges the
> old clone, jobs will lose their packages. Calling `.venv/bin/python`
> directly (as the test commands do) is unaffected.

## 1. One-time setup (login node)

```bash
# Clone / copy the project into your project or scratch space.
cd $SCRATCH
git clone <your-repo-url> cultural-contextualization-research
cd cultural-contextualization-research

# Load modules exactly as the jobs do (StdEnv/2023, python/3.11, arrow/21.0.0,
# opencv/4.13.0). Source it in the current shell; never pipe it.
source slurm/load_modules.sh && load_login_modules

# Create an isolated virtualenv (Alliance recommends venv + prebuilt wheels).
python -m venv .venv
source .venv/bin/activate
pip install --no-index --upgrade pip

# Install dependencies from the Alliance wheelhouse where possible.
# Check availability first, e.g.:  avail_wheels vllm
pip install --no-index -r requirements.txt

# vLLM (cluster only). If a wheel is available:
pip install --no-index vllm
# If not available via --no-index, install on the login node with internet:
#   pip install vllm
```

Notes:
- `pyvis` is only used for the optional HTML graph view; if it is not in the
  wheelhouse you can omit it (the corpus build still writes graphml/json/csv).
- `sentence-transformers` is optional; without it, retrieval uses TF-IDF.

## 2. Stage data (and model) for offline use (login node)

```bash
export HF_HOME=$SCRATCH/hf_cache
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct   # non-gated default
bash slurm/prep_offline.sh
```

This caches CultureBank + CulturalBench and (if `VLLM_MODEL` is set) the model
weights into `$HF_HOME`. Gated models (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
require `huggingface-cli login` and accepting the license first.

## 3. Verify the allocation in the SLURM scripts

The GPU scripts currently use `--account=def-enaskt_gpu`; `run_tests.sh` uses
`--account=def-enaskt`. Confirm these are still the intended allocations and
change them to the appropriate `def-<pi>` account if the deployment changes.

## 4. Build and validate the retrieval corpus

```bash
export HF_HOME=$SCRATCH/hf_cache
# Full CultureBank run (checkpoint/resume enabled by default):
RUN_TAG=<new-tag> CORPUS_LIMIT=100000 BATCH_SIZE=32 VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch slurm/build_corpus.sh

# Smoke test. Still overwrites data/processed and data/outputs; use a separate
# checkout if those hold artifacts you need. STRICT=0: full-size gates fail at 50.
RUN_TAG=smoke CORPUS_LIMIT=50 BATCH_SIZE=16 STRICT=0 sbatch slurm/build_corpus.sh
```

Primary construction outputs:

| Artifact | Path |
|---|---|
| Checkpoint state | `data/processed/corpus_build_state.json` |
| Raw triples | `data/processed/corpus_triples.jsonl` |
| Refined triples | `data/processed/corpus_triples_refined.jsonl` |
| Entity map | `data/processed/entity_map.jsonl` |
| Alignments | `data/processed/cross_cultural_alignments.jsonl` |
| Pathways | `data/processed/corpus_pathways.jsonl` |
| Pathway index | `data/processed/pathway_index/` |
| Parquet mirrors | `data/processed/*.parquet` |
| Resolved graph | `data/outputs/resolved_cultural_kg.*` |
| Validation report | `data/outputs/phase2_validation.md` |

The SLURM script runs `scripts/validate_corpus.py --strict --state_path "$STATE_PATH"`
after the build, so the gates check the checkpoint this build wrote.
Do not evaluate a corpus that fails these gates. Also verify that the validation
report, checkpoint, pathway index, and evaluation all belong to the same run;
older checked-in reports span more than one artifact generation.

Optional env vars for `build_corpus.sh`:

- `RUN_TAG=<tag>`: use `llm_cache_<tag>/` and `corpus_build_state_<tag>.json`.
  **Always set a new tag for a rebuild.** An old checkpoint marks every record
  complete, so a resumed run would skip extraction entirely. The tag does *not*
  rename the triple, pathway, or report files, so copy `data/` aside before
  rebuilding in a checkout whose artifacts you need (see
  [`RUN_HISTORY.md`](RUN_HISTORY.md) §2).
- `RESUME=0`: disable checkpoint resume and start fresh.
- `SKIP_ALIGNMENT_VERIFY=1`: embedding-only alignment (faster smoke).
- `STRICT=0`: report gate failures without failing the job.
- `GATE_MIN_PATHWAYS`, `GATE_MIN_EXTRACTION_SUCCESS`, `GATE_MIN_CRITIQUE_SUCCESS`,
  `GATE_MIN_L3_SHARE`, `GATE_MIN_FULL_PATHWAYS`, `GATE_MIN_SHARED_PIVOT`: gate
  thresholds. The defaults are sized for the full corpus; lower them for pilots.

Install optional embedding model for better entity resolution / pathway retrieval:

```bash
pip install sentence-transformers
```

## 5. Evaluate S0–S4 and ablations

```bash
export HF_HOME=$SCRATCH/hf_cache
LIMIT=100000 TOP_K=8 VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch slurm/eval_full.sh
```

Outputs to `data/outputs/`:
- `culturalbench_summary.md` (accuracy table + McNemar tests)
- `culturalbench_results.json` / `.csv`
- `culturalbench_accuracy.png`
- `mechanism_analysis.md` / `mechanism_h2_completeness.png`
- `before_after_comparison.md`

`eval_full.sh` runs S0–S4 plus the triple-retrieval and no-shared-pivot
ablations. Use `eval_culturalbench.sh` only when you intentionally need a smaller
or custom system set.

## Offline environment variables (set by the GPU job scripts)

```bash
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

If a dataset/model is not found offline, re-run `slurm/prep_offline.sh` on the
login node to cache it.

## Useful SLURM tips

- Interactive GPU debugging: `salloc --account=def-<pi> --gres=gpu:a100:1 --cpus-per-task=6 --mem=48G --time=1:00:00`
- Monitor: `squeue -u $USER`; logs stream to `slurm/logs/`.
- Use `$SLURM_TMPDIR` for fast node-local scratch if you read/write large files.
- Right-size `--time` and `--mem`; A100 (40/80GB) easily holds a 7-8B model at
  `gpu_memory_utilization=0.90`.

## Running the tests

```bash
source slurm/load_modules.sh && load_login_modules   # venv needs scipy/pandas/pyarrow from modules
source slurm/env_hf.sh && setup_hf_offline
export EMBEDDING_BACKEND=tfidf                        # MiniLM from Lustre can take ~10 minutes
.venv/bin/python -m pytest -q -p no:cacheprovider
```

`bash slurm/run_tests.sh` wraps these steps. If the login node's Lustre client
stalls, `sbatch slurm/run_tests.sh` runs the same thing on a CPU compute node.

## Overnight watchers

`slurm/watchers/watch_*_run.sh` are nohup babysitters from the July runs. They
resubmit `build_corpus.sh` after timeouts until the checkpoint is complete, then
submit evaluation. Their limits (`CORPUS_LIMIT`, `EVAL_LIMIT`) are hardcoded for
those runs, so review them before reuse. Their logs go to `slurm/watch/`, which
is git-ignored.

## Local development (no cluster)

You can develop and smoke-test everything locally with Ollama:

```bash
ollama serve & ollama pull llama3.1:8b
python scripts/build_corpus.py --provider ollama --limit 30
python scripts/validate_corpus.py      # report only; full-corpus gates fail at 30 records
python scripts/eval_culturalbench.py --provider ollama --limit 6 \
  --systems S0_no_kg,S3_layered_kg,S4_abstraction_refinement_kg
```

The harness is identical; only the backend (`--provider`) changes.

## Current-result warning

The checked-in full Qwen2.5-7B-Instruct run is a negative result: S0 reaches
52.2% combined accuracy, S3 42.5%, and S4 38.4%. Treat a successful job as an
engineering outcome, not evidence that the research hypothesis passed. Formal
claims also require the human KG-quality, retrieval, stereotype, and fairness
audits defined in `RESEARCH_DESIGN.md`.

Last updated: 2026-09-25.
