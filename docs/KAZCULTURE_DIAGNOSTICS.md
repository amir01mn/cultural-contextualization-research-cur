# KazCulture offline diagnostics, 2026-11-07

Source reviewed: `kazculture-full-eval` at
`91c995c2e3410251e22ea434d7c3e659c3e91155`. Source files were retrieved through
the GitHub connection; direct Git clone credentials were unavailable in the local
executor. These are offline code checks in Python 3.12, not reproduction of the
pinned Narval model environment or of the benchmark accuracy figures.

## Confirmed problems

### Interrupted cache writes can misassign vectors

A first cache writer can leave vectors without a complete key file. Previously,
opening that cache returned early without trimming those vectors. A later append
then paired a new key with the earlier vector. A live process also appended after
another process's incomplete write without checking the shared files again.

Four synthetic regressions failed on the original source: missing first key file,
empty first key file, a partial first key, and an interrupted append after another
process had already opened the cache. For example, a new key intended for `[0, 1]`
reloaded as `[1, 0]`. Repair now runs under the shared lock before reads and appends;
dimension metadata creation is also protected by that lock.

This prevents further misalignment. It cannot detect or reconstruct historical
key/vector pairs that already look complete but point to the wrong embeddings.

### Corrected runs can silently reuse old checkpoints

The evaluator and train-dev gate checked the pinned embedding-model revision but
did not record the active embedding backend. A synthetic completed S4R run could
be resumed after changing the backend, reusing its earlier predictions. The same
problem reproduced for the train-dev gate. The recorded revision was a constant,
so it did not establish which encoder actually ran.

Both paths now record the requested backend and the embedding implementation hash.
They fail closed on a backend/code change or legacy settings instead of treating
old output as corrected output. The gate also records adapter kwargs and context
length. Use a new versioned `--out` path; preserve the old directory.

## Validation

- Original offline suite: **302 passed, 1 skipped**.
- Fixed offline suite: **307 passed, 1 skipped**.
- Added regressions failed before the fixes: four cache cases and two backend-
  change scenarios (one extends the existing train-dev test).
- Shell syntax: **42 scripts passed** `bash -n`.
- Narval and Nibi submission dry runs produced the intended A100/H100 GPU flags
  for the frame build, train-dev gate, and strict evaluation jobs.
- Four concurrent cache writers preserved all **100** independently checked
  key/vector pairs. All seven KazCulture build/tune/evaluation/audit CLIs passed
  their `--help` startup checks.
- One scorer test requires Torch, which is absent locally. No real encoder or
  answering model was loaded. Local dependencies are not the pinned cluster lock.

These findings do not establish which error Amir saw or that S4R improves accuracy.
His SLURM logs and the actual cluster artifacts are still needed for that diagnosis.

## Next checks on Narval

From the user's existing checkout (`~/projects/def-enaskt/divc/kazculture-restart`),
inspect `git status --short`, fetch origin, and switch to the diagnostics branch.
Keep local work if the switch reports a conflict; do not reset or clean the tree.

Load the modules in the current shell, as the job scripts do:

```bash
source slurm/load_modules.sh && load_login_modules
source slurm/env_hf.sh && setup_hf_offline
EMBEDDING_BACKEND=tfidf .venv/bin/python -m pytest -q -p no:cacheprovider
```

Before a GPU submission, establish whether these inputs exist in this checkout:

- `data/raw/kazculture_phase1/test.parquet`
- `data/kazculture_pipeline/corpus/train_dev_qa.jsonl`
- `data/kazculture_pipeline/s1_full_dev_thresholds_v3.json`
- `data/kazculture_pipeline/index/{s1_strict,s1_full}/build_manifest.json`
- `data/kazculture_pipeline/frames_v3k/{strict,full}/kg_meta.json`
- `data/kazculture_pipeline/frames_v3k/routing_thresholds.json`
- `data/kazculture_pipeline/frames_v3k/routing_tuning_train_row_ids.json`

The frame artifacts are not tracked in Git. Missing files require access to a
verified artifact set or rebuilding the corresponding stage; a branch switch
does not bring those files across from Amir's cluster workspace.

Start with the train-dev gate, using BGE-M3, a **fresh** embedding-cache directory,
a **fresh** `KZ_GATE_OUT`, and `KZ_GATE_EXCLUDE_IDS` pointing to the routing-tuning
IDs. The exclusion is optional in the shell script and must be supplied explicitly.
Also verify that existing frame indexes were not built from a misaligned cache.
Preserve frozen routing thresholds; do not tune against test results.

Inspect the gate's `.out`/`.err` logs and `gate_decision.json` before scheduling
the preregistered benchmark conditions. No GPU jobs were submitted by these checks.
