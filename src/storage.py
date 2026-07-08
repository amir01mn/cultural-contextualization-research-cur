"""Checkpoint, Parquet, and LLM response cache helpers for Phase 2."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .schemas import CorpusBuildState
from .utils import ensure_dir, get_logger, read_json, write_json

LOGGER = get_logger("storage")


def append_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Append records to a JSONL file (creates parent dirs as needed)."""
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def read_jsonl_list(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus_state(path: str | Path) -> CorpusBuildState:
    path = Path(path)
    if not path.exists():
        return CorpusBuildState()
    try:
        return CorpusBuildState(**read_json(path))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load corpus state from %s (%s); starting fresh.", path, exc)
        return CorpusBuildState()


def prepare_stage_state(path: str | Path, stage: str, *, resume: bool) -> CorpusBuildState:
    """Load checkpoint state; when ``resume=False``, clear only this stage."""
    state = load_corpus_state(path)
    if not resume:
        if stage == "extraction":
            state.extraction_completed = []
        elif stage == "critique":
            state.critique_completed = []
        else:
            raise ValueError(f"Unknown stage: {stage}")
    return state


def save_corpus_state(path: str | Path, state: CorpusBuildState) -> None:
    state.updated_at = datetime.now(timezone.utc).isoformat()
    write_json(path, state.model_dump())


def read_completed_ids(state: CorpusBuildState, stage: str) -> set[str]:
    if stage == "extraction":
        return set(state.extraction_completed)
    if stage == "critique":
        return set(state.critique_completed)
    raise ValueError(f"Unknown stage: {stage}")


def mark_completed(state: CorpusBuildState, stage: str, instance_ids: Iterable[str]) -> None:
    ids = list(instance_ids)
    if stage == "extraction":
        done = set(state.extraction_completed)
        done.update(ids)
        state.extraction_completed = sorted(done)
    elif stage == "critique":
        done = set(state.critique_completed)
        done.update(ids)
        state.critique_completed = sorted(done)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def write_parquet(path: str | Path, records: list[dict]) -> None:
    import pandas as pd

    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(records).to_parquet(path, index=False)


def read_parquet(path: str | Path) -> list[dict]:
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict(orient="records")


def _cache_key(stage: str, system_prompt: Optional[str], user_prompt: str) -> str:
    payload = f"{stage}\n{system_prompt or ''}\n{user_prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_llm_cache_path(cache_dir: str | Path, stage: str,
                       system_prompt: Optional[str], user_prompt: str) -> Path:
    key = _cache_key(stage, system_prompt, user_prompt)
    return Path(cache_dir) / stage / f"{key}.txt"


def read_llm_cache(cache_dir: str | Path, stage: str,
                   system_prompt: Optional[str], user_prompt: str) -> Optional[str]:
    path = get_llm_cache_path(cache_dir, stage, system_prompt, user_prompt)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_llm_cache(cache_dir: str | Path, stage: str,
                    system_prompt: Optional[str], user_prompt: str, response: str) -> None:
    path = get_llm_cache_path(cache_dir, stage, system_prompt, user_prompt)
    ensure_dir(path.parent)
    path.write_text(response, encoding="utf-8")


def generate_with_cache(
    backend,
    stage: str,
    user_prompts: list[str],
    system_prompt: Optional[str],
    cache_dir: str | Path,
) -> list[str]:
    """Generate completions, reading/writing per-prompt cache entries."""
    cache_dir = Path(cache_dir)
    outputs: list[Optional[str]] = [None] * len(user_prompts)
    pending_idx: list[int] = []
    pending_prompts: list[str] = []

    for i, prompt in enumerate(user_prompts):
        cached = read_llm_cache(cache_dir, stage, system_prompt, prompt)
        if cached is not None:
            outputs[i] = cached
        else:
            pending_idx.append(i)
            pending_prompts.append(prompt)

    if pending_prompts:
        fresh = backend.generate(pending_prompts, system_prompt=system_prompt)
        for idx, prompt, text in zip(pending_idx, pending_prompts, fresh):
            write_llm_cache(cache_dir, stage, system_prompt, prompt, text)
            outputs[idx] = text

    return [o or "" for o in outputs]
