# KazCulture Phase 1: Qwen3-32B closed-book reproduction

Branch: `experiment/kazculture-qwen3-phase1` (based on `kg-v3`).

## Research objective

Reproduce the KazCulture paper's Qwen3-32B **39.51%** zero-shot
non-thinking MCQ accuracy on all **1,334 test questions**. The **64.54%**
two-stage fine-tuned result is a later KG retrieval target, not Phase 1.

This is a **documented independent reproduction protocol**. The paper's exact
closed-book prompt template may differ from our prompt. Any resulting difference
must be documented; never change the prompt after looking at test labels.

## Integrity

- Input to model: `question`, four choices A/B/C/D.
- **Never**: `item_description`, `answer`, or `answer_label` in the model prompt.
- Gold `answer_label` used only by evaluation reducer.
- Non-thinking mode: `apply_chat_template(..., enable_thinking=False)`.
- Score next-token **log probability** of A/B/C/D and take argmax.
- Fixed Kazakh prompt `kazculture-closedbook-kz-v1`; one order of options.
- No retrieval; no finetuning; no temperature/sample generation.
- Report full test set (1,334) separately from pilots/subsets.
- Tokenization is checked; scoring fails loudly if any letter spans multiple tokens.
- Outputs include metadata, checkpointed prediction JSONL, accuracy and 95% Wilson CI.
- No model-generated answers in dataset or graph construction during Phase 1.

## On Narval

1. Accept gated terms at https://huggingface.co/datasets/issai/KazCulture
   and configure HF credentials on a network-accessible login node.
2. Stage a fixed KazCulture **test** split (JSONL/JSON/Parquet) in project space,
   and stage `Qwen/Qwen3-32B` model weights in the shared HF cache.
   **Never commit either to Git.** Record their immutable revisions.
3. Install dependencies in your existing `.venv` (torch, transformers,
   accelerate, datasets, huggingface_hub, pyarrow). Check Qwen3 requires
   a sufficiently recent transformers version. Full BF16 weights need
   approximately 64 GB plus significant overhead; the default SLURM script
   requests two A100 80GB GPUs. Use `device_map=auto`.
4. Dry run before requesting the GPU:

   ```bash
   python scripts/eval_kazculture_phase1.py \
     --test_file /project/path/kazculture_test.parquet --dry_run
   ```

5. Submit:

   ```bash
   export KAZCULTURE_TEST_FILE=/project/path/kazculture_test.parquet
   sbatch --export=ALL,KAZCULTURE_TEST_FILE="$KAZCULTURE_TEST_FILE" \
     slurm/kazculture_phase1.sh
   ```

6. Read `data/outputs/kazculture_phase1/summary.json`. If interrupted,
   rerun with `--resume`. Results are unmeasured until a real model run finishes.

## Notes about reproducibility and limitations

- The script **fails** if dataset length is not 1,334; local smoke files may
  specify `--expected_size 1`.
- Pin exact dataset and model revisions when possible.
- HF trust/model download is not performed in tests. This project environment
  may lack access to the gated dataset and GPUs.
- The current prompt adds the assistant continuation `Жауап: `, and scores
  A/B/C/D as the *next token*. It must be calibrated against the paper's own
  prompting implementation if made available. Do not claim an exact reproduction
  of 39.51% if this protocol differs.
- The full repo's `kg-v3` artifacts and historical evaluation results are not
  modified.
- Phase 2 will add multilingual evidence retrieval and KG construction **only
  after Phase 1's frozen baseline is measured**.
