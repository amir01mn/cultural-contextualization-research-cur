# KazCulture KG v2: factual-versus-normative extraction pilot

**Status:** Code complete for an *independent* smaller extraction and audit
experiment. **Do not report improved extraction accuracy yet:** no new
Qwen3-32B GPU results or human judgments have been collected.

## Motivation

Original 750-passage pilot: 750 source passages, 382 frames with at least one
validated-quote candidate, 773 candidate triples, 50.93% passage coverage.
A preliminary 60-row screen found source-quoted factual triples with
relationship-direction errors (e.g., bowstring material wrongly attributed to
the entire bow), spurious source-title entities, and unsupported or missing
social norms. 33/60 had no proposed L2/L3 annotation; 25/60 were screened
as needing triple revision. These are **screening flags**, not certified
human accuracy labels.

A source object description can yield a useful *factual* triple without any
social-norm or Schwartz-value pathway. The previous audit gate conflated
absence of a pathway with an incorrect pathway, making its 80% target
unrealistic. V2 separates the two denominators and avoids forcing norms
onto factual descriptions.

## V2 controlled changes

- Immutable sample: the **first 150 unique source passages of the same
  750-passage pilot**. For a fair matched-source comparison, v1 results are
  restricted to these exact 150 IDs.
- Qwen3-32B checkpoint identical to Phase 1 V2,
  `9216db5781bf21249d130ec9da846c4624c16137`. Non-thinking mode,
  temperature 0, deterministic seed 42, tensor parallelism 2.
- New explicit predicate vocabulary: `IS_A`, `MADE_FROM`,
  `USED_FOR`, `PART_OF`, `HAS_PROPERTY`, `WORN_BY`,
  `LOCATED_IN`, `CREATED_BY`, `USED_IN`, `ASSOCIATED_WITH`,
  `PERFORMED_DURING`, `SYMBOLIZES`, `PRACTICED_BY`,
  `REQUIRES`, and `OTHER_SUPPORTED_RELATION`. Every valid
  triple has a verbatim quote from the model-visible source.
- A model-proposed **normative pathway** requires an explicit, separate
  quote for a social rule/custom/obligation/taboo, a norm, a norm type, and
  a recognized Schwartz value. No automatic L2/L3 pathway is inferred from
  an object's materials or construction. **All generated interpretations
  remain provisional, even when their quotes are valid.**
- Passages are segmented without losing any original source bytes if they
  exceed Qwen's 4,096-token context. New output reserves 768 generation
  tokens and 64 safety tokens; the preflight validates the chat template.
- The model sees the cultural passage and the allowed taxonomy labels, but
  NOT source book title, topic, category, QA answer or test data.
- Rejected candidates are counted by reason. JSON failures and output-length
  truncations are recorded, rather than silently aggregated as weak KG.
- V2 v1 outputs are **never overwritten**, nor is the Phase 1 41.38%
  Qwen3 closed-book baseline modified.

### What is and is not mechanically verified

A substring quote proves a cited piece of text exists in the actual input.
It does NOT prove that the candidate subject/predicate/object relation or
the cultural interpretation is correct. `subject_in_quote` and
`object_in_quote` are preliminary string-match indicators, which can
be false for legitimate Kazakh inflection. Humans still audit semantics.

## Run on Narval

From the existing repo checkout, in the project directory:

```bash
cd ~/projects/def-enaskt/divc/kazculture-restart
source .venv/bin/activate
git pull --ff-only

# Offline tests; no GPU or LLM weights.
python -m pytest -q tests/test_kazculture_kg_v2.py tests/test_kazculture_pipeline.py
bash -n slurm/kazculture_kg_v2_pilot.sh

# CPU-only actual cached Qwen tokenizer check, no extraction started.
python scripts/extract_kazculture_kg_v2.py \
  --passages data/kazculture_pipeline/corpus/pilot_passages.jsonl \
  --limit 150 \
  --out data/kazculture_pipeline/kg_v2/pilot_150_frames.jsonl \
  --audit_csv data/kazculture_pipeline/kg_v2/audit_v2_pilot_150.csv \
  --preflight_only

mkdir -p slurm/logs
sbatch --time=03:00:00 --export=ALL slurm/kazculture_kg_v2_pilot.sh
```

When the job finishes, check `sacct -j JOB_ID` and the
`slurm/logs/kazkg_v2_JOB_ID.out` / `.err` files.

Outputs (new, independent directory):
- `data/kazculture_pipeline/kg_v2/pilot_150_frames.jsonl`
- `data/kazculture_pipeline/kg_v2/pilot_150_frames.manifest.json`
- `data/kazculture_pipeline/kg_v2/pilot_150_frames.summary.json`
- `data/kazculture_pipeline/kg_v2/audit_v2_pilot_150.csv`

The runner resumes completed passages, validates its manifest and
does not overwrite annotated audit decisions.

## Compare against v1 on matching 150 source passages

```bash
python scripts/compare_kazculture_kg_v2.py \
  --old_frames data/kazculture_pipeline/kg/pilot_frames.jsonl \
  --new_frames data/kazculture_pipeline/kg_v2/pilot_150_frames.jsonl \
  --out data/kazculture_pipeline/kg_v2/matched_150_comparison.json
```

This compares coverage, candidate triple counts and provisional pathway
counts only. **The counts do not measure factual accuracy or retrieval
performance.** The v1 norm/value presence metric is an imperfect proxy,
not directly comparable with the v2 valid-pathway count.

## Independent manual audit of new candidate triples

V2 creates an **unannotated** 60-row random audit CSV:
`data/kazculture_pipeline/kg_v2/audit_v2_pilot_150.csv`.

Review EACH triple against its Kazakh source passage:
- `triple_correct_0_or_1`: whether subject, direction, predicate and object
  accurately express the source relationship.
- `normative_eligible_0_or_1`: **whether the source/triple warrants a
  social-norm pathway at all** (regardless of what the model proposed).
- `pathway_correct_0_or_1_if_proposed`: fill 0 or 1 ONLY when
  `model_proposed_pathway_0_or_1=1`. Leave it blank if no pathway was
  proposed. Mark 0 for an unsupported or misassigned value/norm.
- `reviewer_notes`: note language ambiguity, wrong relation direction,
  metadata contamination, weak cultural inference, or useful corrections.

Reviewers must make the actual decisions; previous machine screening must
not be copied into the official 0/1 judgment fields.

After the audit is manually completed and uploaded back to Narval:

```bash
python scripts/review_kazculture_kg_v2.py \
  --audit_csv data/kazculture_pipeline/kg_v2/audit_v2_pilot_150.csv \
  --out data/kazculture_pipeline/kg_v2/reviewed_quality_report.json
```

Reports **triple precision**, **precision among *proposed* pathways**,
**human normative eligibility**, and **accepted pathway coverage**.
If too few normative pathways are sampled for meaningful precision,
the layered-scale gate fails as *insufficient evidence*, not as evidence
that every factual triple should receive an invented norm.

The 80% thresholds are pilot research gates, not universally validated
accuracy guarantees. Only after a satisfactory review should a new
versioned full-corpus build be considered. Independently evaluate Kazakh
question-to-passage retrieval using train-only development questions
before test-set evaluations.
