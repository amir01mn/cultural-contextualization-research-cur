"""KazCulture test-split adapter for Phase 1 (closed-book Qwen3-32B).

Only question, options, and gold label reach this adapter's public evaluation
records. The source passage and free-text answer are intentionally not exposed.
Dataset access requires accepting issai/KazCulture terms on Hugging Face.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

DATASET = "issai/KazCulture"
EXPECTED_TEST_SIZE = 1334
CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class KazCultureQuestion:
    question_id: str
    question: str
    options: dict[str, str]
    answer_key: str
    category: str


def convert_test_rows(rows: Iterable[Mapping]) -> list[KazCultureQuestion]:
    """Validate dataset fields, unique IDs, and letters; exclude source passages.

    Row ordinal becomes part of the ID so duplicate question text is not lost.
    """
    questions: list[KazCultureQuestion] = []
    for index, row in enumerate(rows):
        if row.get("set") not in (None, "test"):
            raise ValueError(f"Row {index}: expected test split")
        question = str(row.get("question") or "").strip()
        options = {letter: str(row.get(letter.lower()) or "").strip() for letter in CHOICES}
        answer = str(row.get("answer_label") or "").strip().upper()
        if not question or any(not opt for opt in options.values()):
            raise ValueError(f"Row {index}: missing question or A/B/C/D option")
        if answer not in CHOICES:
            raise ValueError(f"Row {index}: invalid gold answer label {answer!r}")
        # Stable for a fixed dataset revision and row ordering.
        suffix = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        questions.append(KazCultureQuestion(
            question_id=f"kazculture_test_{index:04d}_{suffix}",
            question=question, options=options, answer_key=answer,
            category=str(row.get("category_english") or "Unknown").strip(),
        ))
    return questions


def load_test(*, local_file: str | None = None,
              revision: str | None = None, limit: int | None = None,
              expected_size: int = EXPECTED_TEST_SIZE) -> tuple[list[KazCultureQuestion], dict]:
    """Load authorized Hugging Face data or offline JSONL/JSON fixture.

    No fallback/synthetic dataset is permitted in research runs.
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if local_file:
        path = Path(local_file)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        else:
            raise ValueError("Local test file must be .jsonl, .json, or .parquet")
        dataset_revision = f"local:{path.name}"
        source = str(path.resolve())
    else:
        from datasets import load_dataset
        try:
            dataset = load_dataset(DATASET, split="test", revision=revision, token=True)
        except Exception as exc:
            raise RuntimeError(
                "KazCulture is gated. Accept the terms at "
                "https://huggingface.co/datasets/issai/KazCulture and log in "
                "with 'hf auth login' on the compute environment before staging."
            ) from exc
        rows = list(dataset)
        dataset_revision = str(getattr(dataset, "_fingerprint", None) or revision or "unknown")
        source = DATASET

    if expected_size and len(rows) != expected_size:
        raise ValueError(
            f"Expected exactly {expected_size} KazCulture test rows; received {len(rows)}. "
            "Stop before scoring to avoid incomparable accuracy."
        )
    questions = convert_test_rows(rows)
    if limit is not None:
        questions = questions[:limit]
    return questions, {
        "source": source, "dataset_revision_or_fingerprint": dataset_revision,
        "full_test_count": len(rows), "scored_subset_size": len(questions),
        "expected_test_size": expected_size,
    }
