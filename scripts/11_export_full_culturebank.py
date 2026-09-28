"""Export the complete cached CultureBank dataset as normalized JSONL.

This script does no LLM work. It loads the requested Hugging Face splits,
normalizes each CultureBank record with the same normalize_record() function used
by the corpus pipeline, preserves the source split in raw["_split"], writes a
single deterministic JSONL file, and reports its SHA-256 checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import normalize_record  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402

LOGGER = get_logger("11_export_full_culturebank")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Export all requested CultureBank splits as normalized JSONL."
    )
    p.add_argument("--dataset", default=config.CULTUREBANK_HF_ID)
    p.add_argument(
        "--splits",
        default="tiktok,reddit",
        help="Comma-separated Hugging Face splits in deterministic output order.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination JSONL path, preferably under $SCRATCH on Narval.",
    )
    p.add_argument(
        "--metadata_output",
        default=None,
        help="Optional metadata JSON path. Defaults to <output>.meta.json.",
    )
    args = p.parse_args(argv)

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if not splits:
        raise ValueError("At least one split is required.")

    from datasets import load_dataset

    output = Path(args.output)
    ensure_dir(output.parent)
    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output
        else Path(str(output) + ".meta.json")
    )
    ensure_dir(metadata_output.parent)

    split_counts: Counter[str] = Counter()
    culture_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    seen_instance_ids: set[str] = set()
    total = 0

    with output.open("w", encoding="utf-8") as f:
        for split in splits:
            LOGGER.info("Loading CultureBank split '%s' from %s", split, args.dataset)
            ds = load_dataset(args.dataset, split=split)
            LOGGER.info("Split '%s' contains %d rows.", split, len(ds))

            for row_idx, item in enumerate(ds):
                raw = dict(item)
                raw.setdefault("_split", split)
                rec = normalize_record(raw, total, dataset_id="culturebank")
                instance_id = str(rec["instance_id"])

                if instance_id in seen_instance_ids:
                    raise RuntimeError(
                        f"Duplicate normalized instance_id {instance_id!r} "
                        f"encountered in split {split!r}, row {row_idx}."
                    )
                seen_instance_ids.add(instance_id)

                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                total += 1
                split_counts[split] += 1
                culture_counts[str(rec.get("cultural_group") or "unknown")] += 1
                topic_counts[str(rec.get("topic") or "unknown")] += 1

    checksum = sha256_file(output)
    metadata = {
        "dataset": args.dataset,
        "splits": splits,
        "total_records": total,
        "split_counts": dict(split_counts),
        "unique_raw_cultural_group_labels": len(culture_counts),
        "unique_topics": len(topic_counts),
        "top_cultural_groups": culture_counts.most_common(30),
        "top_topics": topic_counts.most_common(50),
        "sha256": checksum,
        "output": str(output),
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== FULL CULTUREBANK EXPORT =====\n")
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Records: {total}")
    for split in splits:
        print(f"  {split}: {split_counts[split]}")
    print(f"Unique raw culture labels: {len(culture_counts)}")
    print(f"Unique topics: {len(topic_counts)}")
    print(f"Output: {output}")
    print(f"SHA-256: {checksum}")
    print(f"Metadata: {metadata_output}")


if __name__ == "__main__":
    main()
