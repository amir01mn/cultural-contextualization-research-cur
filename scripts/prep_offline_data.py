"""Pre-stage datasets (and optionally model weights) for offline cluster runs.

Run this on the Narval LOGIN node (which has internet). It downloads CultureBank
and CulturalBench into the Hugging Face cache (HF_HOME) so that compute nodes,
which have NO internet, can load them with HF_DATASETS_OFFLINE=1.

Usage (login node):
    export HF_HOME=$SCRATCH/hf_cache
    python scripts/prep_offline_data.py
    # optionally also pre-download the model weights:
    python scripts/prep_offline_data.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import config  # noqa: E402
from src.core.utils import get_logger  # noqa: E402

LOGGER = get_logger("prep_offline_data")


def _stage_culturebank() -> None:
    from datasets import get_dataset_split_names, load_dataset

    LOGGER.info("Staging CultureBank (%s)...", config.CULTUREBANK_HF_ID)
    try:
        splits = get_dataset_split_names(config.CULTUREBANK_HF_ID)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not list CultureBank splits: %s", exc)
        splits = ["tiktok", "reddit"]
    for sp in splits:
        try:
            ds = load_dataset(config.CULTUREBANK_HF_ID, split=sp)
            LOGGER.info("  cached CultureBank split '%s' (%d rows).", sp, len(ds))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("  failed to cache split '%s': %s", sp, exc)


def _stage_culturalbench() -> None:
    from datasets import load_dataset

    LOGGER.info("Staging CulturalBench (%s)...", config.CULTURALBENCH_HF_ID)
    for cfg in ("CulturalBench-Easy", "CulturalBench-Hard"):
        cached = False
        for sp in ("test", "train", "validation"):
            try:
                ds = load_dataset(config.CULTURALBENCH_HF_ID, cfg, split=sp)
                LOGGER.info("  cached %s split '%s' (%d rows).", cfg, sp, len(ds))
                cached = True
                break
            except Exception:  # noqa: BLE001
                continue
        if not cached:
            LOGGER.warning("  could not cache config '%s'.", cfg)


def _stage_model(model: str) -> None:
    LOGGER.info("Pre-downloading model weights: %s", model)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model)
        LOGGER.info("  model '%s' cached.", model)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("  failed to download model '%s': %s", model, exc)
        LOGGER.warning("  (gated models like Llama need `huggingface-cli login`.)")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stage data/models for offline runs.")
    p.add_argument("--skip_culturebank", action="store_true", default=False)
    p.add_argument("--skip_culturalbench", action="store_true", default=False)
    p.add_argument("--model", default=None,
                   help="Optional HF model id to pre-download (e.g. the vLLM model).")
    args = p.parse_args(argv)

    import os
    LOGGER.info("HF_HOME=%s", os.getenv("HF_HOME", "(default ~/.cache/huggingface)"))

    if not args.skip_culturebank:
        _stage_culturebank()
    if not args.skip_culturalbench:
        _stage_culturalbench()
    if args.model:
        _stage_model(args.model)
    LOGGER.info("Offline staging complete. Compute nodes can now run with "
                "HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1.")


if __name__ == "__main__":
    main()
