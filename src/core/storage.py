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
            state.extraction_failed = []
        elif stage == "critique":
            state.critique_completed = []
            state.critique_failed = []
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
    """Mark records as successfully completed (and clear any prior failure)."""
    ids = set(instance_ids)
    if stage == "extraction":
        state.extraction_completed = sorted(set(state.extraction_completed) | ids)
        state.extraction_failed = sorted(set(state.extraction_failed) - ids)
    elif stage == "critique":
        state.critique_completed = sorted(set(state.critique_completed) | ids)
        state.critique_failed = sorted(set(state.critique_failed) - ids)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def mark_failed(state: CorpusBuildState, stage: str, instance_ids: Iterable[str]) -> None:
    """Record records whose LLM output could not be parsed.

    Deliberately does NOT mark them complete, so a resumed run retries them.
    Previously every record was marked complete regardless of outcome, which is
    why the checkpoint claimed 22,990 critiques while only 20,990 rows existed
    and 20,644 of those were parse failures.
    """
    ids = set(instance_ids)
    if stage == "extraction":
        state.extraction_failed = sorted(set(state.extraction_failed) | ids)
    elif stage == "critique":
        state.critique_failed = sorted(set(state.critique_failed) | ids)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def read_failed_ids(state: CorpusBuildState, stage: str) -> set[str]:
    if stage == "extraction":
        return set(state.extraction_failed)
    if stage == "critique":
        return set(state.critique_failed)
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


def _cache_key(stage: str, system_prompt: Optional[str], user_prompt: str,
               *, model: Optional[str] = None,
               max_tokens: Optional[int] = None) -> str:
    """Hash the full generation context, not just the prompt text.

    The model id and token budget MUST be part of the key. They are what
    determine whether a response was truncated, so a cache keyed only on prompt
    text would replay previously truncated completions verbatim after the budget
    is raised - silently reproducing the broken corpus.
    """
    payload = (
        f"{stage}\n{model or ''}\n{max_tokens or ''}\n"
        f"{system_prompt or ''}\n{user_prompt}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_llm_cache_path(cache_dir: str | Path, stage: str,
                       system_prompt: Optional[str], user_prompt: str,
                       *, model: Optional[str] = None,
                       max_tokens: Optional[int] = None) -> Path:
    key = _cache_key(stage, system_prompt, user_prompt,
                     model=model, max_tokens=max_tokens)
    return Path(cache_dir) / stage / f"{key}.txt"


def read_llm_cache(cache_dir: str | Path, stage: str,
                   system_prompt: Optional[str], user_prompt: str,
                   *, model: Optional[str] = None,
                   max_tokens: Optional[int] = None) -> Optional[str]:
    path = get_llm_cache_path(cache_dir, stage, system_prompt, user_prompt,
                              model=model, max_tokens=max_tokens)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_llm_cache(cache_dir: str | Path, stage: str,
                    system_prompt: Optional[str], user_prompt: str, response: str,
                    *, model: Optional[str] = None,
                    max_tokens: Optional[int] = None) -> None:
    path = get_llm_cache_path(cache_dir, stage, system_prompt, user_prompt,
                              model=model, max_tokens=max_tokens)
    ensure_dir(path.parent)
    path.write_text(response, encoding="utf-8")


def generate_with_cache(
    backend,
    stage: str,
    user_prompts: list[str],
    system_prompt: Optional[str],
    cache_dir: str | Path,
    *,
    max_tokens: Optional[int] = None,
) -> list[str]:
    """Generate completions, reading/writing per-prompt cache entries.

    ``max_tokens`` defaults to the configured budget for ``stage`` and is part of
    the cache key, so changing a stage budget invalidates its cached responses.
    """
    from . import config

    cache_dir = Path(cache_dir)
    if max_tokens is None:
        max_tokens = config.stage_max_tokens(stage)
    model = getattr(backend, "model", None)

    outputs: list[Optional[str]] = [None] * len(user_prompts)
    pending_idx: list[int] = []
    pending_prompts: list[str] = []

    for i, prompt in enumerate(user_prompts):
        cached = read_llm_cache(cache_dir, stage, system_prompt, prompt,
                                model=model, max_tokens=max_tokens)
        if cached is not None:
            outputs[i] = cached
        else:
            pending_idx.append(i)
            pending_prompts.append(prompt)

    if pending_prompts:
        fresh = backend.generate(pending_prompts, system_prompt=system_prompt,
                                 max_tokens=max_tokens)
        for idx, prompt, text in zip(pending_idx, pending_prompts, fresh):
            write_llm_cache(cache_dir, stage, system_prompt, prompt, text,
                            model=model, max_tokens=max_tokens)
            outputs[idx] = text

    return [o or "" for o in outputs]
