#!/usr/bin/env python3
"""Stage gated KazCulture test Parquet and model weights for offline Narval use.

Requires prior Hugging Face access approval and 'hf auth login'. Run on an
internet-enabled login node, not from an offline SLURM GPU job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out_dir", default="data/raw/kazculture_phase1")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--skip_model", action="store_true")
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("issai/KazCulture", split="test",
                          revision=args.dataset_revision, token=True)
    except Exception as exc:
        raise RuntimeError("Accept KazCulture dataset terms on Hugging Face and authenticate with 'hf auth login'") from exc
    if len(ds) != 1334:
        raise ValueError(f"Expected exactly 1,334 test rows; found {len(ds)}")
    path = out / "test.parquet"
    ds.to_parquet(str(path))
    model_path = None
    if not args.skip_model:
        model_path = snapshot_download(repo_id=args.model, revision=args.model_revision)
    manifest = {
        "dataset": "issai/KazCulture",
        "dataset_revision_requested": args.dataset_revision,
        "dataset_fingerprint": ds._fingerprint,
        "test_rows": len(ds),
        "test_file": str(path.resolve()),
        "test_sha256": sha256(path),
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "model_cache_path": model_path,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
