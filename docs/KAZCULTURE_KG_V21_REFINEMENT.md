# KazCulture KG V2.1: relation precision and all-pathways review

**Status:** Code and regression tests prepared. No V2.1 GPU extraction,
manual validation or downstream QA accuracy improvement is claimed.

## What we learned

The original 750-source KG V1 pilot recovered 773 quote-matched candidate
triples from 382/750 sources. KG V2, run on 150 of those exact source passages,
recovered 347 candidate triples from 109/150 sources, against V1's
147 triples from 75/150 on the same subset. V2 generated 40 *proposed*
pathways. Its run logged **33 output-length truncations, 33 invalid JSON
segments** and 58 source-quote rejections. These are extraction mechanics,
not knowledge correctness. Preliminary AI-assisted screening on the 60-row
sample: 31 appeared supported, 14 needed context review and 15 appeared
incorrect; only 9/60 samples contained proposed pathways, and most of
those were screened as questionable. Human review was **not** completed.

## Scientific next step

Perform two tasks in parallel:

A. **All-40 pathway audit:** inspect every V2 proposed pathway, rather than
relying on the 9 incidental cases in the 60-triple sample. Independently
label triple correctness, whether a social norm is warranted, and pathway
correctness. Keep the official human-label fields blank until a human
reviewer completes them; never copy previous machine screening scores.
No full-corpus extraction gate can pass on unreviewed outputs.

B. **Matched 150-source V2.1 engineering pilot:** a new prompt requires
short exact evidence quotes, max 3 triples instead of 5, strict relation
direction, no homonym conflation or gift-as-material confusion, and a
declared social rule kind for pathways. Output tokens raised **768 -> 1536**
with an actual token-budget preflight to prevent the earlier 4096-token
input overflow. Extraction is *versioned* and writes to `kg_v21/`, leaving
V1 and V2 unchanged. Changing the output budget AND prompt/schema means this
is a combined engineering comparison, NOT an isolated one-factor ablation.
The first 150 sources are a *development set* because their previous
screening influenced this prompt. Validate final accuracy later on disjoint
source passages (e.g., positions 150:300 of the 750 source pilot).

## A. Export every original V2 proposed pathway (no GPU)

```bash
cd ~/projects/def-enaskt/divc/kazculture-restart
source .venv/bin/activate
git pull --ff-only
python -m pytest -q tests/test_kazculture_kg_v21.py tests/test_kazculture_kg_v2.py

python scripts/export_kazculture_pathway_audit.py \
  --frames data/kazculture_pipeline/kg_v2/pilot_150_frames.jsonl \
  --out data/kazculture_pipeline/kg_v2/all_v2_pathways_for_review.csv
```

Output should enumerate all 40 **proposed** pathways if the original
checkpoint is intact. The output file uses exclusive-create mode and
refuses to overwrite an existing human audit.

From Windows PowerShell, outside the SSH session:

```powershell
scp divc@narval.alliancecan.ca:/lustre06/project/6113454/divc/kazculture-restart/data/kazculture_pipeline/kg_v2/all_v2_pathways_for_review.csv "$HOME\Downloads\all_v2_pathways_for_review.csv"
```

The sampling audit and complete proposed-pathway audit assess different
quantities. Reviewing only proposed pathways estimates precision and helps
diagnose false positives; a random triple audit is still needed to identify
missed legitimate pathways and measure recall/coverage.

## B. V2.1 150-passage preflight (no GPU)

```bash
python scripts/extract_kazculture_kg_v21.py \
  --passages data/kazculture_pipeline/corpus/pilot_passages.jsonl \
  --start 0 --limit 150 \
  --out data/kazculture_pipeline/kg_v21/pilot_150_frames.jsonl \
  --audit_csv data/kazculture_pipeline/kg_v21/audit_v21_pilot_150.csv \
  --max_tokens 1536 --preflight_only
```

Expected: `"status": "preflight_passed"`. This may generate more than
151 model-visible segments because the higher output budget reduces
allowed prompt input; it does not discard characters.

## C. Submit V2.1 GPU extraction

```bash
mkdir -p slurm/logs
sbatch --export=ALL slurm/kazculture_kg_v21_pilot.sh
```

Inspect `sacct -j ACTUAL_JOB_ID --format=JobID,State,ExitCode,Elapsed`
and `tail -n 50 slurm/logs/kazkg_v21_ACTUAL_JOB_ID.out`.
The job resumes its own v2.1 checkpoint; it cannot overwrite prior
v1/v2 extraction or human audit. When finished:

```bash
cat data/kazculture_pipeline/kg_v21/pilot_150_frames.summary.json

python scripts/compare_kazculture_kg_versions.py \
  --v1 data/kazculture_pipeline/kg/pilot_frames.jsonl \
  --v2 data/kazculture_pipeline/kg_v2/pilot_150_frames.jsonl \
  --v21 data/kazculture_pipeline/kg_v21/pilot_150_frames.jsonl \
  --out data/kazculture_pipeline/kg_v21/v1_v2_v21_matched_comparison.json
```

The comparison enforces shared passage IDs. It reports coverage, candidate
counts, and truncation/JSON diagnostics. It DOES NOT measure semantic
accuracy, reliable cultural value inference, or final QA performance.

## D. Manual quality gate / next held-out test

- Review 60 V2.1 sampled triples plus all V2 proposed pathways.
- If the norm count is low, audit every V2.1 proposed pathway separately via
  `export_kazculture_pathway_audit.py --frames ... --out ...`.
- Triple precision must be measured using human labels. Proposed-pathway
  precision is *only* among generated pathways. Independently assess
  normative eligibility on random source/triple samples for missed paths.
- Examine new invalid JSON, output length and quote mismatch diagnostics.
- Only then apply the promising configuration to untouched source indices
  `--start 150 --limit 150` with a *new* output path. Do not treat
  these initial 150 revised-prompt passages as a blind test.
- Do not proceed to the 6,749-passage extraction or publish accuracy gains
  without an adequate manual validation and retrieval test.
