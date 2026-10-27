# KazCulture Phases 2–5: one codebase, staged Narval experiments

> IMPLEMENTATION STATUS: Code for the stages below has been committed. The
> 750-record audit, full KG generation, retrieval calibration, full retrieval
> comparisons, and combined CultureBank evaluation have NOT been performed yet.
> CPU smoke tests do not validate LLM extraction quality, multilingual retrieval
> quality, vLLM runtime or scientific accuracy.

## Goals and frozen benchmark

- Qwen3-32B model snapshot:
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Baseline: independent V2 protocol, **552/1334 = 41.3793%**, non-thinking,
  assistant-prefixed `Answer:` and context-valid spaced A/B/C/D probabilities.
- Paper: 39.51% published zero-shot and 64.54% published two-stage fine-tuned.
- For all S1–S4R comparisons, only the *evidence* differs. Question/options,
  model snapshot, scoring and tokenization are fixed.

This is an end-to-end implementation, **not** permission to skip quality gates.

## Safety and source rules

1. Graph extraction receives ONLY KazCulture `train.item_description`,
   `category_english`, and `source`; never `answer`, `question`,
   `answer_label`, or the test descriptions.
2. The train QA file exists for *train-only* retrieval diagnostics, physically
   separate from KG construction.
3. Dedup by normalized exact passage text; retain original source, row IDs,
   culture and category. This does NOT identify paraphrased/near-duplicate
   overlaps.
4. Every extracted triple has a verbatim source quote. Quote validation is
   mechanical and **does not prove its interpretation is true**.
5. L2 norm and L3 value classifications are provisional until reviewed.
6. Audit at least 60 randomly sampled triples AND associated pathways,
   entering 0/1 decisions for both. The gate requires >=80% agreement with
   the evidence by this rubric; failures block full extraction and index.
   (Audit acceptance measures reviewer approval, not inter-rater agreement.)
7. A later full-corpus scientific audit is still needed. Pilot approval is not
   evidence that all generated triples are correct.
8. Similar values across cultures provide *only a provisional L3 bridge*.
   CultureBank descriptions are labeled INDIRECT, preserving the original
   cultural group and frame ID. No direct shared custom is inferred.
9. The strict passage-disjoint condition removes exact test-matching
   source passages from train. Near duplicates require a separate analysis.

## One-time commands: in kazculture-restart, on the login node

```bash
git pull --ff-only
source slurm/load_modules.sh && load_login_modules
source .venv/bin/activate
python -m pytest -q tests/test_kazculture_pipeline.py
bash -n slurm/kazculture_kg_extract.sh
bash -n slurm/kazculture_kg_index.sh
bash -n slurm/kazculture_retrieval_eval.sh

# Prepare shared offline HF model cache, without moving personal HF auth token.
unset HF_HOME HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_CACHE="$(dirname "$PWD")/hf_cache/hub"
export HF_DATASETS_CACHE="$(dirname "$PWD")/hf_cache/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# Stage multilingual BGE-M3 before compute nodes switch to HF_OFFLINE.
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('BAAI/bge-m3', device='cpu'); print('BGE-M3 staged')"

mkdir -p slurm/logs
```

BGE-M3 is recommended as a multilingual candidate, not yet proven best for
Kazakh cultural texts. A valid Phase 2 includes empirical Kazakh retrieval
recall/precision diagnostics.

## 2026-10-26 Narval pilot failure and safe resume (Job 7474841)

The first GPU pilot (`7474841`) failed after **264 of 750** passages
were saved, because one input exceeded vLLM's **4,096-token** context.
This was not an OOM: Qwen3-32B and tensor-parallel workers initialized.

The extraction runner now validates the *entire rendered chat prompt* with
the actual tokenizer, reserving **512 output tokens plus 64 safety tokens**.
Passages that exceed the limit are split into overlapping **verbatim source
substrings**; every substring is seen by the LLM and each quote is checked
against the precise substring shown to it. Each triple retains its segment
source span and its own provisional norm/value labels; source passages and
question IDs are not changed. A separate `.chunking.json` provenance
sidecar records the strategy. The original pilot `.manifest.json` is
intentionally compatible because the model, sampling, extraction instructions,
and dataset did not change; the original 264 checkpoints remain valid.

**First check the GPU job is no longer running. Do not delete or move the
checkpoint or its manifest.** On the login node:

```bash
git pull --ff-only
python -m pytest -q tests/test_kazculture_pipeline.py
wc -l data/kazculture_pipeline/kg/pilot_frames.jsonl

# CPU-only check using cached Qwen3 tokenizer: proves no oversized inputs
# will be sent to the next GPU pilot.
python scripts/extract_kazculture_kg.py \
  --passages data/kazculture_pipeline/corpus/pilot_passages.jsonl \
  --out data/kazculture_pipeline/kg/pilot_frames.jsonl \
  --preflight_only

# When validation passes and the old job is terminated:
sbatch --export=ALL,KG_STAGE=pilot slurm/kazculture_kg_extract.sh
```

The follow-up job skips all checkpointed passages; it will write an audit CSV
only after all 750 source passages finish. Other infrastructure errors must
be examined separately rather than interpreted as passage-length problems.

## Phase 2 — prepare and extract a 750-unique-passage pilot

On the login node:

```bash
python scripts/prepare_kazculture_corpus.py \
  --test_file data/raw/kazculture_phase1/test.parquet \
  --pilot_size 750
cat data/kazculture_pipeline/corpus/manifest.json
cat data/kazculture_pipeline/corpus/test_overlap.json | head -n 15
```

This stages training passages and separate train-QA development queries.
Despite 14,803 training QA rows, the number of **unique source passages** may
be lower due to deduplication.

Submit GPU extraction using Qwen3-32B non-thinking mode and vLLM:

```bash
export KAZCULTURE_MODEL_REVISION=9216db5781bf21249d130ec9da846c4624c16137
sbatch --export=ALL,KG_STAGE=pilot slurm/kazculture_kg_extract.sh
```

If 2x A100 40GB vLLM fails for memory, inspect the actual log. If your
allocation permits, use `sbatch --gres=gpu:a100:4
--export=ALL,KG_STAGE=pilot,VLLM_TP=4 slurm/kazculture_kg_extract.sh`.
Do not submit a concurrent extractor against the same checkpoint. Slurm B1
has a three-hour maximum; subsequent same-stage jobs resume completed records.

After completion, audit this file manually (UTF-8 CSV):

```text
data/kazculture_pipeline/kg/audit_pilot.csv
```

Review `passage`, `subject`, `predicate`, `object`, `verbatim_quote`,
and the provisional norm/value pathway. Fill both
`triple_correct_0_or_1` and `pathway_correct_0_or_1` with 0 or 1 for every
row. A source substring's presence is NOT proof that the relation is true.
Keep the annotated file in project storage, not Git. The extraction script
will not overwrite an existing audit file.

Gate:

```bash
python - <<'PY'
from pathlib import Path
from src.construction.kazculture_kg import validate_audit
print(validate_audit(Path('data/kazculture_pipeline/kg/audit_pilot.csv')))
PY
```

Then create the pilot index as a CPU Slurm job:

```bash
sbatch --export=ALL,KG_INDEX_MODE=pilot slurm/kazculture_kg_index.sh
```

After it completes, tune and inspect retrieval on **train QA only**:

```bash
python scripts/tune_kazculture_retrieval.py \
  --index data/kazculture_pipeline/index/pilot \
  --dev data/kazculture_pipeline/corpus/train_dev_qa.jsonl \
  --max_queries 14803 \
  --out data/kazculture_pipeline/pilot_dev_thresholds.json
```

Retrieval quality audit should inspect question/passages in Kazakh,
source IDs, false-positive matches and the provisional L1/L2/L3 paths.
The train-QA self-passage recall is a *retrieval sanity test*, NOT an estimate
of performance on novel unseen passages. Do not tune on test MCQ answers.

## Phase 3 — full source-indexed graph

Only after the pilot audit passes:

```bash
sbatch --export=ALL,KG_STAGE=full slurm/kazculture_kg_extract.sh
```

Extraction produces checkpointed
`data/kazculture_pipeline/kg/full_frames.jsonl`.
If interrupted by the B1 walltime, check logs and submit the same stage again
to resume. One active job per checkpoint.

Then build the full and exact-passage-disjoint graph indexes:

```bash
sbatch --export=ALL,KG_INDEX_MODE=full slurm/kazculture_kg_index.sh
sbatch --export=ALL,KG_INDEX_MODE=strict slurm/kazculture_kg_index.sh
```

These are independent *CPU* index builds; wait for each to finish before
evaluating. Each index contains provenance-tagged `units.jsonl`,
`embeddings.npy`, `graph_frames.jsonl`, `graph_triples.jsonl`,
`graph_pathways.jsonl` and build metadata.

Freeze training-only thresholds using the full graph:

```bash
python scripts/tune_kazculture_retrieval.py \
  --index data/kazculture_pipeline/index/full \
  --dev data/kazculture_pipeline/corpus/train_dev_qa.jsonl \
  --max_queries 400 \
  --out data/kazculture_pipeline/full_dev_thresholds.json
export KAZCULTURE_DEV_CONFIG="$PWD/data/kazculture_pipeline/full_dev_thresholds.json"
```

If dev recall fails the gate, diagnose multilingual retrieval and fix the
embedding/index **without consulting test labels**, then create a newly
versioned index and threshold file. Do not silently lower the gate.

## Phase 4 — full 1,334-test MCQ comparison

First run CPU-only retrieval previews (requires full index, no 32B weights):

```bash
python scripts/eval_kazculture_retrieval.py \
  --test_file data/raw/kazculture_phase1/test.parquet \
  --baseline_predictions data/outputs/kazculture_phase1_v2/predictions.jsonl \
  --index data/kazculture_pipeline/index/full \
  --out data/kazculture_pipeline/retrieval_preview \
  --config data/kazculture_pipeline/full_dev_thresholds.json \
  --retrieval_only
```

Then submit the GPU experiments, keeping the dev config exported:

```bash
export KAZCULTURE_DEV_CONFIG="$PWD/data/kazculture_pipeline/full_dev_thresholds.json"
sbatch --export=ALL,KG_EVAL_MODE=full slurm/kazculture_retrieval_eval.sh
sbatch --export=ALL,KG_EVAL_MODE=strict slurm/kazculture_retrieval_eval.sh
```

Both jobs use the same published **V2** 41.3793% saved S0 predictions and
write separate per-system JSONL checkpoints with `--resume`. For
S1/S2/S3/S4F/S4R inspect
`data/kazculture_pipeline/evaluation/{full,strict}/comparison.json`:
micro-accuracy, category accuracy, paired bootstrap delta confidence
interval, exact McNemar p-value, evidence-induced errors, and overlap-
conditional retrieval recall. Under exact source-disjoint evaluation,
source-overlap recall is undefined, not 0.

## Phase 5 — combined KazCulture + CultureBank graph

This stage additionally requires an **accessible, already constructed**
CultureBank KG v3 source frame file. It is not included in the KazCulture
model cache and must not be invented.

Example: find a copy in YOUR older project workspaces:

```bash
find "$HOME" -path '*/data/kg_v3/full/frames.jsonl' -print 2>/dev/null
```

If it exists, specify its real path (not a guessed path):

```bash
export KAZCULTURE_CB_FRAMES="/actual/path/to/data/kg_v3/full/frames.jsonl"
test -f "$KAZCULTURE_CB_FRAMES" || echo "Missing source frames"
sbatch --export=ALL,KG_INDEX_MODE=combined slurm/kazculture_kg_index.sh
sbatch --export=ALL,KG_INDEX_MODE=combined_strict slurm/kazculture_kg_index.sh
```

The combined index does not erase source group identity. Only shared
Schwartz L3 value labels enable *provisional* alignment. At retrieval time
a Kazakh source anchor must exist, and CultureBank analogs are marked
INDIRECT, never asserted to be Kazakhstan-specific.

After the index builds, evaluate with the same frozen dev config:

```bash
sbatch --export=ALL,KG_EVAL_MODE=combined slurm/kazculture_retrieval_eval.sh
sbatch --export=ALL,KG_EVAL_MODE=combined_strict slurm/kazculture_retrieval_eval.sh
```

Compare the exact paired predictions directly:

```bash
python scripts/compare_kazculture_runs.py \
  --kazculture_only data/kazculture_pipeline/evaluation/full/S4R.jsonl \
  --combined data/kazculture_pipeline/evaluation/combined/S4R_combined.jsonl \
  --out data/kazculture_pipeline/evaluation/kaz_only_vs_combined.json

python scripts/compare_kazculture_runs.py \
  --kazculture_only data/kazculture_pipeline/evaluation/strict/S4R.jsonl \
  --combined data/kazculture_pipeline/evaluation/combined_strict/S4R_combined.jsonl \
  --out data/kazculture_pipeline/evaluation/strict_kaz_vs_combined.json
```

## Limits of the current implementation

- The extraction prompt creates grounded candidate triples and provisional
  norm/value chains, **not automatically verified cultural facts**.
- Exact normalized passage dedup and exact test-source overlap exclusion do
  **not** address paraphrase/semantic leakage; near-duplicate audits should
  precede publication.
- BGE-M3 has broad multilingual support but must be evaluated empirically on
  these Kazakh questions; its model revision should be pinned for final reports.
- `S4R` currently implements same-source provisional L3→L2→L1 evidence
  retrieval and threshold gating. It is a first integrated adaptation, not a
  complete reproduction of the CultureBank v3 route/abstraction algorithm.
  Routing complexity and refinement/abstraction ablations remain necessary.
- Combined CultureBank alignment only bridges exact common value taxonomy
  nodes; additional cross-source norm equivalence needs human review.
- No quality, accuracy or speed results for Phases 2–5 may be claimed until
  these jobs complete and their artifacts are inspected.
