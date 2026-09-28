"""Shared helpers: logging, JSONL IO, id generation, small text utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def get_logger(name: str = "cultural_kg") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOGGER = get_logger()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_id(*parts: Any, prefix: str = "", length: int = 12) -> str:
    """Deterministic short id derived from the given parts."""
    payload = "||".join(str(p) for p in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_node_label(label: Any) -> str:
    """Canonical label used to merge equivalent graph nodes."""
    return normalize_text(label).lower()


def truncate(text: Any, max_chars: int = 600) -> str:
    s = normalize_text(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def coalesce(*values: Any) -> Any:
    """Return the first non-empty value (None / "" / "nan" treated as empty)."""
    for v in values:
        if v is None:
            continue
        sv = str(v).strip()
        if sv and sv.lower() != "nan" and sv.lower() != "none":
            return v
    return None
