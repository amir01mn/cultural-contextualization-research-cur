# Phase 2 next step: exploratory Kazakh source-retrieval diagnostic

**Scope:** Candidate KG V2.1 vs V2.2 on the same 150 train-only,
previously screened-source-excluded holdout passages. This step does **not**
replace human factual/norm/value accuracy review, and must NOT be called
a Qwen3-32B answer-quality result.

## Why now

- V2.1 held-out: 114/150 passages covered, 247 candidates, 62 Schwartz
  pathways, 75 quote rejects, 28 orphan pathways.
- V2.2 held-out: 98/150 covered, 162 candidates, 29 Schwartz pathways,
  30 quote rejects, no orphan pathways.
- Both: 0 JSON failures and 0 generation truncations on these sources.
- The 100-item blinded review is awaiting independent Kazakh-literate human
  decisions. AI-only screening of examples cannot be fed into the human score
  columns and cannot satisfy the full-KG quality gate.
- BGE-M3 multilingual retrieval has not yet been empirically tested on these
  Kazakh source questions.

This is a *self-source retrieval sanity check*: for a TRAIN QA question
associated with passage P, we ask whether passage P is retrieved from
an index that contains P. This does **not** establish generalization to
unseen passages, human-verified semantic precision or final QA accuracy.
The QA question itself is never added to the KG or embedding index,
nor are gold answers or test labels. The same sampled train questions and
BGE-M3 model generate a matched paired diagnostic for both methods.

## 1. Pull and test

```bash
cd ~/projects/def-enaskt/divc/kazculture-restart
git pull --ff-only
source slurm/load_modules.sh
load_login_modules
source .venv/bin/activate
python -m pytest -q tests/test_kazculture_retrieval_diagnostic.py \
  tests/test_kazculture_kg_v22.py tests/test_kazculture_kg_v21.py \
  tests/test_kazculture_pipeline.py
bash -n slurm/kazculture_kg_retrieval_diagnostic.sh
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers import OK')"
```

If the import fails, repair only the missing dependencies shown in the
traceback before attempting to build indexes. Do not reinstall the entire
requirements file or overwrite the working Qwen3 virtual environment.

## 2. Cache the multilingual embedding model

The compute-node Slurm script sets HF offline mode. BGE-M3 must already be
in the same project HF cache used by `slurm/env_hf.sh`. Stage on an
internet-enabled login node *before* submission:

```bash
source slurm/env_hf.sh
setup_hf_cache
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("BAAI/bge-m3")
print("BGE-M3 cached:", path)
PY
```

This downloads a frozen copy *for the diagnostic* into the project cache.
For publication-quality runs, record/pin the exact embedding model
revision and store it with the index manifest.

## 3. Two exploratory CPU indexing jobs

```bash
mkdir -p slurm/logs
sbatch --export=ALL,KG_DIAG_STAGE=v21_index \
  slurm/kazculture_kg_retrieval_diagnostic.sh
sbatch --export=ALL,KG_DIAG_STAGE=v22_index \
  slurm/kazculture_kg_retrieval_diagnostic.sh
```

These jobs each use 12 CPU cores, 64 GB memory and a maximum three-hour
allocation. No Qwen3 GPU or test answers are involved. They write:

- `data/kazculture_pipeline/index/heldout_diagnostic/v21_diagnostic/`
- `data/kazculture_pipeline/index/heldout_diagnostic/v22_diagnostic/`

Each index has BGE-M3 embeddings, source-bearing retrieval units,
resumable embedding shards and a diagnostic-only provenance manifest.
They **do not** generate the standard `build_manifest.json` needed by
the final test evaluation script, and cannot be mistaken for an accepted
full graph index.

Check the actual job IDs (replace JOB_ID):

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 35 slurm/logs/kzkg_ret_diag_JOB_ID.out
```

If a job finishes with failure due to an unset/missing model weight,
stage BGE-M3 then resubmit the same method's job. Embedding shard
checkpointing can resume safely.

## 4. Retrieve matched train questions

**Only once both index jobs are COMPLETED with exit code 0:0:**

```bash
sbatch --export=ALL,KG_DIAG_STAGE=probe \
  slurm/kazculture_kg_retrieval_diagnostic.sh
```

The probe selects up to 300 train questions across the same 150 source
passages, with at most three questions per source. No answer text is used.
It reports absolute source Recall@1/3/5/8, gold-source availability and
conditional Recall@k per representation, plus MRR@8.
It uses the exact same query vectors for the two indexes.

Print the report:

```bash
cat data/kazculture_pipeline/retrieval_diagnostic/heldout_v21_vs_v22_source_recall.json
```

Interpretation:
- **S1:** raw-source retrieval baseline
- **S2:** flat triple retrieval
- **S3:** triples with proposed norm/value chains, only where actually present
- **S4F:** source-local frame evidence; V2.1/V2.2 implementations only
- **S4R:** proposed L3→L2→L1 pathways; not equivalent to CultureBank v3 routed
  abstraction algorithm
- **source_recall_at_5_all:** proportion of ALL selected questions whose correct
  source appears in top 5; missing KG representations count as misses
- **source_recall_at_5_conditional:** same quantity only where the model
  generated a relevant representation for the source; useful diagnostically,
  but can exaggerate effectiveness when many passages have no graph/path.

If S3/S4R coverage is poor, that is an important finding. Do not
retroactively turn unrelated factual triples into fake norms to raise recall.

## 5. What is still necessary before Phases 3 and 4

1. Independent human review of the 100 source-matched, version-blinded
   factual triples and all relevant norm/value proposals. AI-only
   screening must remain separate; blanks cannot be auto-filled.
2. Source-level/cultural plausibility review, not only automated exact
   quote matching.
3. Retrieval evaluation on appropriate held-out QA **after** choosing
   a config, with source-overlap control and near-duplicate assessment.
   Existing benchmark test overlaps the training descriptions extensively;
   interpret naive RAG test accuracy accordingly.
4. Only then build the 6,749-source KG and run frozen Qwen3-32B QA
   evaluation. No current data justify a Phase 4 accuracy improvement claim.

## No destructive operations

All results live under dedicated `index/heldout_diagnostic/` and
`retrieval_diagnostic/` directories. The original Phase 1 S0 baseline,
all V1/V2/V2.1/V2.2 extraction checkpoints, and blinded human review
remain unchanged.
