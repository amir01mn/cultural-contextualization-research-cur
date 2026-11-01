# Phase 2: frozen holdout and independent quality gate (V2.1 vs V2.2)

**Status:** Scripts, parser, extraction protocol and tests committed. The GPU
runs, human audit and Kazakh semantic retrieval validation are NOT yet run.

## What was established on the matched first 150 KazCulture pilot sources

| Metric | V1 | V2 | V2.1 |
|---|---:|---:|---:|
| Passages with candidate triples | 75 | 109 | 119 |
| Candidate triples | 147 | 347 | 260 |
| Proposed fully-layered pathways | not equivalent | 40 | 58 |
| JSON failures | not logged | 33 | 2 |
| Output truncations | not logged | 33 | 0 |

These are **candidate counts**, not correctness measures. AI-assisted
screening flagged serious L2/L3 over-inference. No independent human
validation means Phase 2 is not yet ready to scale.

## Frozen V2.2 refinement (no more prompt editing after held-out runs)

V2.2 changes the output to `triples: [{subject, predicate, object, evidence,
norm: null|{behavior,norm,norm_type,rule_kind,evidence,values: []}}]`.
A norm belongs to its own practice relation rather than referencing a
`triple_index`. This removes mechanically orphaned pathways when source-
unverified triples are dropped. Only exact quotes from the model-visible
chunk are accepted. Distinct output statuses: fact-only, norm-only (no
L3 value), and a provisional full pathway if the model explicitly proposes
a Schwartz value. **L3 is never mechanically proven by quoting a passage.**
Source descriptions of tool materials or game mechanics should not have
forced social norms. Retrieval construction now suppresses S3/S4R units
for V2, V2.1 and V2.2 factual-only triples.

This differs from V2.1 in multiple ways, so treat it as a new candidate
method, not a single-variable ablation. Validation on the first 150 already
inspected sources would be biased.

## 1. Pull code and run tests on Narval

```bash
cd ~/projects/def-enaskt/divc/kazculture-restart
source .venv/bin/activate
git pull --ff-only
python -m pytest -q tests/test_kazculture_kg_v22.py \
  tests/test_kazculture_kg_v21.py tests/test_kazculture_kg_v2.py \
  tests/test_kazculture_pipeline.py
bash -n slurm/kazculture_kg_holdout.sh
```

## 2. Freeze a train-only, audit-excluded set of 150 source passages

Crucial nuance: the original V1 audit sampled from all 750 source passages,
not just the first 150. Selecting positions 150–299 without exclusions
could reuse passages that influenced V2/V2.1 prompt revisions. The selector
excludes the first 150 AND any passage IDs in the earlier three audits.

```bash
python scripts/prepare_kazculture_heldout.py \
  --passages data/kazculture_pipeline/corpus/pilot_passages.jsonl \
  --audit data/kazculture_pipeline/kg/audit_pilot.csv \
  --audit data/kazculture_pipeline/kg_v2/audit_v2_pilot_150.csv \
  --audit data/kazculture_pipeline/kg_v21/audit_v21_pilot_150.csv \
  --out data/kazculture_pipeline/heldout/heldout_train_passages_150.jsonl \
  --dev_prefix 150 --count 150
```

This creates **immutable** train-only source inputs, and
`heldout_train_passages_150.selection.json` documenting source checksums,
audit exclusions and the selection policy. Input is intentionally not
test questions or gold answers.

The set is held out from *prompt development*, not an independent
external test corpus. The model's prior V1 extraction pass did touch the
original 750, and we do not claim an independent generalization benchmark.

## 3. CPU-only prompt/token preflights for both methods

```bash
python scripts/extract_kazculture_kg_v21.py \
  --passages data/kazculture_pipeline/heldout/heldout_train_passages_150.jsonl \
  --start 0 --limit 150 \
  --out data/kazculture_pipeline/kg_v21_holdout/heldout_150_frames.jsonl \
  --audit_csv data/kazculture_pipeline/kg_v21_holdout/audit_heldout_150.csv \
  --max_tokens 1536 --preflight_only

python scripts/extract_kazculture_kg_v22.py \
  --passages data/kazculture_pipeline/heldout/heldout_train_passages_150.jsonl \
  --start 0 --limit 150 \
  --out data/kazculture_pipeline/kg_v22_holdout/heldout_150_frames.jsonl \
  --audit_csv data/kazculture_pipeline/kg_v22_holdout/audit_heldout_150.csv \
  --max_tokens 1536 --preflight_only
```

Both must print `preflight_passed`. The 4,096 context window reserves
1,536 tokens for output and 64 for safety, and long sources are losslessly
split with overlap. If any prompt is too long, the preflight reveals it
before requesting GPUs.

## 4. GPU jobs on Narval

```bash
mkdir -p slurm/logs

sbatch --export=ALL,KG_HOLDOUT_VERSION=v21 slurm/kazculture_kg_holdout.sh
sbatch --export=ALL,KG_HOLDOUT_VERSION=v22 slurm/kazculture_kg_holdout.sh
```

Each job uses 2×A100, TP=2, Qwen3-32B revision
`9216db5781bf21249d130ec9da846c4624c16137`. Jobs write to
**separate resumable checkpoints**:

- `data/kazculture_pipeline/kg_v21_holdout/heldout_150_frames.jsonl`
- `data/kazculture_pipeline/kg_v22_holdout/heldout_150_frames.jsonl`

Check job status with `sacct -j ACTUAL_JOB_ID
--format=JobID,State,ExitCode,Elapsed`. Log names are
`slurm/logs/kzkg_hout_JOBID.out` and `.err`.

## 5. Matched-source extraction diagnostics (after both finish)

```bash
python scripts/compare_kazculture_kg_holdout.py \
  --v21 data/kazculture_pipeline/kg_v21_holdout/heldout_150_frames.jsonl \
  --v22 data/kazculture_pipeline/kg_v22_holdout/heldout_150_frames.jsonl \
  --out data/kazculture_pipeline/heldout/v21_vs_v22_diagnostics.json
```

Coverage, quoted candidate yield, invalid predicates/quotes and truncation
are descriptive. None establish KG semantic correctness.

## 6. Blind independent human review

```bash
python scripts/build_kazculture_heldout_audit.py \
  --v21 data/kazculture_pipeline/kg_v21_holdout/heldout_150_frames.jsonl \
  --v22 data/kazculture_pipeline/kg_v22_holdout/heldout_150_frames.jsonl \
  --audit data/kazculture_pipeline/heldout/blind_review_100.csv \
  --mapping data/kazculture_pipeline/heldout/blind_review_100_key.json \
  --sample_per_version 50
```

The audit is 100 shuffled triples, 50 per method, with review IDs not
revealing method/version. Keep `blind_review_100_key.json` away from
reviewers until labels are complete. Each record includes source text,
triple evidence and any proposed norm/value. A qualified reviewer must
fill the official `0_or_1` fields (no auto-labeling):

- factual_triple_correct_0_or_1: correct actor/direction/predicate/object
- norm_warranted_by_source_0_or_1: norm genuinely applicable
- proposed_norm_correct_0_or_1_if_present: valid source-grounded norm,
  only when a norm was proposed
- proposed_values_correct_0_or_1_if_present: supported Schwartz value,
  only when values were proposed
- reviewer_notes: disambiguation, cultural context, or corrections

If possible, have a second Kazakh-literate reviewer label a subset and
resolve differences. Note that a sampled triple-level audit does not
measure source-level recall of facts never proposed.

After human review is complete and the corrected CSV is back on Narval:

```bash
python scripts/score_kazculture_heldout_audit.py \
  --audit data/kazculture_pipeline/heldout/blind_review_100.csv \
  --mapping data/kazculture_pipeline/heldout/blind_review_100_key.json \
  --out data/kazculture_pipeline/heldout/blind_quality_report.json
```

Reports factual precision, norm quality and complete-pathway precision;
refuses unanswered review fields. A preliminary **80% precision gate**
requires enough reviewed pathways as well as correct facts. This is a
research screening threshold, not a population-level guarantee.

Separately export ALL V2.2 fully-layered proposed pathways for a targeted
precision audit:

```bash
python scripts/export_kazculture_pathway_audit.py \
  --frames data/kazculture_pipeline/kg_v22_holdout/heldout_150_frames.jsonl \
  --out data/kazculture_pipeline/kg_v22_holdout/all_proposed_pathways_for_review.csv
```

## 7. Before Phase 3

Use the human audit to freeze a method. Then evaluate Kazakh query to
source-passage retrieval using train-only dev QA and passage-disjoint controls.
Only after factual quality, pathway precision and retrieval utility are
acceptable should extraction of all **6,749** unique train passages proceed.
Keep test answers out of extraction, retrieval tuning and embedding input.
