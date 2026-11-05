"""KazCulture frames v3k: the CultureBank KG v3 frame method ported to KazCulture.

One frame per KazCulture train passage (docs/KAZCULTURE_FRAMES_V3K.md):

    L1  practice    the full Kazakh passage text (never truncated in the frame)
    L2c norm        one English sentence, labelled by Qwen3-32B
    L2  norm type   culture-neutral label, clustered at cosine 0.75 (frame_builder)
    L3  values      1-2 Schwartz values (frame_builder.SCHWARTZ_VALUES)

The Kazakh frames are added to the CultureBank v3 frames (``data/kg_v3/full``)
and the union goes through ``frame_builder.build_frame_kg`` unchanged, so norm
types are shared across the two sources and culture profiles exist for Kazakh
and for the CultureBank groups that serve as abstract-route analogs.

Only the labelling prompt differs from v3 (``prompts.KAZCULTURE_FRAME_LABEL_USER``);
its reply is parsed by ``frame_builder.parse_label``. Only the passage text, its
category and the fixed culture "Kazakh" reach the prompt: train QA answers and
test fields never do.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..core.culture_normalizer import canonical_culture
from ..core.schemas import CulturalFrame
from ..core.utils import get_logger
from ..llm import prompts
from . import frame_builder as fb

LOGGER = get_logger("kazculture_frames_v3k")

SCHEMA_VERSION = "kazculture-frames-v3k"
LABEL_MODEL = "Qwen/Qwen3-32B"
LABEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
LABEL_MAX_TOKENS = fb.LABEL_MAX_TOKENS          # same cap as v3 (200)
PRACTICE_MAX_TOKENS = 4096                       # prompt-only cap on the passage
CHARS_PER_TOKEN_FALLBACK = 2                     # conservative for Kazakh Cyrillic
CULTURE_LABEL = "Kazakh"
KAZAKH = canonical_culture(CULTURE_LABEL) or "kazakh"
TRUNCATION_MARK = " [...]"
STATUS_OK, STATUS_NONE, STATUS_FAILED = "ok", "none", "failed"
_NONE_WORDS = {"none", "null", "n/a", "na", ""}


def is_kazakh_culture(culture: str) -> bool:
    """CultureBank groups that would collide with KazCulture as "direct" Kazakh."""
    return str(culture or "").lower().startswith("kazakh")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def frame_from_passage(p: dict) -> CulturalFrame:
    """L1 practice is the whole passage; the category is the frame topic."""
    cats = p.get("categories") or []
    return CulturalFrame(
        frame_id=str(p["passage_id"]), culture=KAZAKH, culture_label=CULTURE_LABEL,
        topic=str(cats[0]) if cats else "", text=str(p["text"]))


def frames_from_passages(passages: Iterable[dict]) -> list[CulturalFrame]:
    return [frame_from_passage(p) for p in passages]


# ---------------------------------------------------------------------------
# Labelling prompt
# ---------------------------------------------------------------------------

def value_list() -> str:
    return "\n".join(f"  - {name}: {d}" for name, d in fb.SCHWARTZ_VALUES.items())


def prompt_sha256(max_tokens: int = LABEL_MAX_TOKENS,
                  practice_max_tokens: int = PRACTICE_MAX_TOKENS) -> str:
    """Hash of everything that shapes a label besides the passage itself."""
    blob = json.dumps({
        "schema": SCHEMA_VERSION, "system": prompts.FRAME_LABEL_SYSTEM,
        "user": prompts.KAZCULTURE_FRAME_LABEL_USER,
        "retry": prompts.FRAME_LABEL_RETRY_SUFFIX, "values": value_list(),
        "max_tokens": max_tokens, "practice_max_tokens": practice_max_tokens,
        "truncation_mark": TRUNCATION_MARK,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_label_prompt(f: CulturalFrame, practice: str) -> str:
    return prompts.KAZCULTURE_FRAME_LABEL_USER.format(
        topic=f.topic or "unspecified", passage=practice, culture=CULTURE_LABEL,
        value_list=value_list())


def truncate_practice(text: str, max_tokens: int, tokenizer=None) -> tuple[str, bool, int]:
    """Cut the passage to ``max_tokens`` for the labelling prompt only.

    Returns ``(text, truncated, n_tokens_before)``. Without a tokenizer the cap
    is ``max_tokens * CHARS_PER_TOKEN_FALLBACK`` characters.
    """
    if tokenizer is None:
        limit = max_tokens * CHARS_PER_TOKEN_FALLBACK
        n = -(-len(text) // CHARS_PER_TOKEN_FALLBACK)
        if len(text) <= limit:
            return text, False, n
        return text[:limit].rstrip() + TRUNCATION_MARK, True, n
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, False, len(ids)
    return (tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).rstrip()
            + TRUNCATION_MARK, True, len(ids))


def practice_budget(tokenizer, *, max_model_len: int, max_tokens: int = LABEL_MAX_TOKENS,
                    practice_max_tokens: int = PRACTICE_MAX_TOKENS, margin: int = 64) -> int:
    """Largest passage token budget that keeps prompt + reply inside the context."""
    if tokenizer is None:
        return practice_max_tokens
    probe = CulturalFrame(frame_id="probe", culture=KAZAKH, culture_label=CULTURE_LABEL,
                          topic="x" * 40)
    msgs = [{"role": "system", "content": prompts.FRAME_LABEL_SYSTEM},
            {"role": "user", "content": render_label_prompt(probe, "")
             + prompts.FRAME_LABEL_RETRY_SUFFIX}]
    try:
        overhead = len(tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    except Exception:  # noqa: BLE001 - tokenizer without a chat template
        overhead = len(tokenizer.encode(msgs[0]["content"] + msgs[1]["content"]))
    room = max_model_len - overhead - max_tokens - margin
    if room < 256:
        raise ValueError(f"max_model_len={max_model_len} leaves {room} tokens for the passage")
    return min(practice_max_tokens, room)


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def is_explicit_none(text: str) -> bool:
    """True when the reply is the prompt's "no norm" object (not a parse failure)."""
    m = _JSON.search(text or "")
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    norm = str(obj.get("norm") or "").strip().lower().strip(".")
    return norm in _NONE_WORDS


# ---------------------------------------------------------------------------
# Sharding and the per-passage checkpoint
# ---------------------------------------------------------------------------

def shard(passages: list[dict], index: int, n_shards: int) -> list[dict]:
    """Deterministic round-robin shard over passage-id order."""
    if n_shards < 1 or not 0 <= index < n_shards:
        raise ValueError(f"bad shard {index}/{n_shards}")
    ordered = sorted(passages, key=lambda p: p["passage_id"])
    return ordered[index::n_shards]


def shard_path(out_dir: Path, index: int, n_shards: int) -> Path:
    return Path(out_dir) / "labels" / f"shard_{index:03d}_of_{n_shards:03d}.jsonl"


def read_checkpoint(path: Path, *, model: str, revision: str, prompt_hash: str,
                    max_tokens: int, repair: bool = True) -> dict[str, dict]:
    """Rows of a label checkpoint that match this exact labelling setup.

    A truncated last line (job killed mid-write) is dropped and, with
    ``repair``, cut from the file so the next append starts on a clean line.
    A malformed line anywhere else is an error, not something to skip.
    """
    path = Path(path)
    if not path.exists():
        return {}
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    body, tail = raw[:cut], raw[cut:]
    if tail.strip():
        try:
            json.loads(tail.decode("utf-8"))
            body, tail = raw + b"\n", b""          # complete row, missing its newline
        except (json.JSONDecodeError, UnicodeDecodeError):
            LOGGER.warning("%s: dropping truncated last line (%d bytes)", path, len(tail))
    if repair and body != raw:
        with path.open("r+b") as fh:
            fh.truncate(min(len(body), len(raw)))
            if len(body) > len(raw):
                fh.seek(0, 2)
                fh.write(b"\n")
    done: dict[str, dict] = {}
    for i, line in enumerate(body.split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path}: malformed line {i} inside the checkpoint") from exc
        if (r.get("model"), r.get("revision"), r.get("prompt_sha256"),
                r.get("max_tokens")) == (model, revision, prompt_hash, max_tokens):
            done[r["passage_id"]] = r
    return done


def label_passages(backend, passages: list[dict], checkpoint: Path, *, model: str,
                   revision: str, tokenizer=None, max_model_len: Optional[int] = None,
                   max_tokens: int = LABEL_MAX_TOKENS,
                   practice_max_tokens: int = PRACTICE_MAX_TOKENS,
                   batch_size: int = 256) -> dict[str, dict]:
    """Label passages with one retry, appending one JSONL row per passage.

    Each row records model, revision, prompt hash and ``max_tokens``; resume
    keeps only rows that match all four (AGENTS.md cache-key rule). An explicit
    "none" reply is final (status ``none``, no retry): retrying would push the
    model to invent a norm for a descriptive passage.
    """
    phash = prompt_sha256(max_tokens, practice_max_tokens)
    done = read_checkpoint(checkpoint, model=model, revision=revision,
                           prompt_hash=phash, max_tokens=max_tokens)
    budget = (practice_budget(tokenizer, max_model_len=max_model_len, max_tokens=max_tokens,
                              practice_max_tokens=practice_max_tokens)
              if tokenizer is not None and max_model_len else practice_max_tokens)
    todo = [p for p in passages if p["passage_id"] not in done]
    LOGGER.info("KazCulture labels: %d done, %d to do (budget %d passage tokens)",
                len(done), len(todo), budget)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            frames = [frame_from_passage(p) for p in batch]
            cut = [truncate_practice(f.text, budget, tokenizer) for f in frames]
            base = [render_label_prompt(f, c[0]) for f, c in zip(frames, cut)]
            outs = backend.generate(base, system_prompt=prompts.FRAME_LABEL_SYSTEM,
                                    temperature=0.0, max_tokens=max_tokens)
            parsed = [fb.parse_label(o) for o in outs]
            nones = [p is None and is_explicit_none(o) for p, o in zip(parsed, outs)]
            retry = [i for i, p in enumerate(parsed) if p is None and not nones[i]]
            if retry:
                outs2 = backend.generate(
                    [base[i] + prompts.FRAME_LABEL_RETRY_SUFFIX for i in retry],
                    system_prompt=prompts.FRAME_LABEL_SYSTEM, temperature=0.0,
                    max_tokens=max_tokens)
                for i, o in zip(retry, outs2):
                    parsed[i], outs[i] = fb.parse_label(o), o
                    nones[i] = parsed[i] is None and is_explicit_none(o)
            for f, (_t, truncated, ntok), p, o, none in zip(frames, cut, parsed, outs, nones):
                status = STATUS_OK if p else STATUS_NONE if none else STATUS_FAILED
                r = {"passage_id": f.frame_id, "schema": SCHEMA_VERSION, "model": model,
                     "revision": revision, "prompt_sha256": phash, "max_tokens": max_tokens,
                     "status": status, "label": p,
                     "raw": None if p else (o or "")[-500:],
                     "practice_truncated": truncated, "practice_tokens": ntok}
                done[f.frame_id] = r
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            LOGGER.info("Labelled %d/%d (%s)", min(start + batch_size, len(todo)), len(todo),
                        dict(Counter(done[f.frame_id]["status"] for f in frames)))
    return done


def merge_label_shards(paths: list[Path], passages: list[dict], *, model: str, revision: str,
                       max_tokens: int = LABEL_MAX_TOKENS,
                       practice_max_tokens: int = PRACTICE_MAX_TOKENS) -> tuple[list[dict], dict]:
    """One row per passage, in passage-id order; refuses gaps, strays and conflicts."""
    phash = prompt_sha256(max_tokens, practice_max_tokens)
    rows: dict[str, dict] = {}
    for path in paths:
        part = read_checkpoint(path, model=model, revision=revision, prompt_hash=phash,
                               max_tokens=max_tokens, repair=False)
        for pid, r in part.items():
            if pid in rows and rows[pid]["label"] != r["label"]:
                raise ValueError(f"Conflicting labels for {pid} across shards")
            rows[pid] = r
    want = {p["passage_id"] for p in passages}
    missing, stray = sorted(want - set(rows)), sorted(set(rows) - want)
    if stray:
        raise ValueError(f"{len(stray)} labelled ids are not in the corpus, e.g. {stray[:3]}")
    if missing:
        raise ValueError(f"{len(missing)} passages have no label row, e.g. {missing[:3]}")
    merged = [rows[pid] for pid in sorted(want)]
    stats = {"n": len(merged), "status": dict(Counter(r["status"] for r in merged)),
             "practice_truncated": sum(bool(r["practice_truncated"]) for r in merged),
             "prompt_sha256": phash}
    return merged, stats


def apply_label_rows(frames: list[CulturalFrame], rows: Iterable[dict]) -> dict:
    """Copy labels onto frames. ``none``/``failed`` frames keep no norm (refine-only)."""
    by_id = {r["passage_id"]: r for r in rows}
    stats: Counter = Counter()
    for f in frames:
        r = by_id.get(f.frame_id)
        lab = r.get("label") if r else None
        if lab is None and r and r.get("status") == STATUS_FAILED and r.get("raw"):
            lab = fb.parse_label(r["raw"])     # a parser fix needs no new LLM calls
            if lab:
                stats["reparsed"] += 1
        if lab:
            f.norm, f.norm_type_label, f.values = lab["norm"], lab["norm_type"], list(lab["values"])
            f.label_status = STATUS_OK
        else:
            f.label_status = r["status"] if r else "unlabeled"
        stats[f.label_status] += 1
    return dict(stats)


# ---------------------------------------------------------------------------
# KG assembly: KazCulture + CultureBank v3 frames
# ---------------------------------------------------------------------------

def load_culturebank_frames(kg_dir: Path) -> tuple[list[CulturalFrame], dict]:
    """CultureBank v3 frames, minus Kazakh groups (KazCulture is the only Kazakh source)."""
    frames, dropped = [], Counter()
    with (Path(kg_dir) / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            f = CulturalFrame(**json.loads(line))
            if is_kazakh_culture(f.culture):
                dropped[f.culture_label or f.culture] += 1
                continue
            # Re-clustered in the union; the old cluster id must not leak through.
            f.norm_type_id = None
            frames.append(f)
    return frames, {"n_culturebank_frames": len(frames),
                    "dropped_kazakh_culturebank_frames": dict(dropped)}


def build_kazculture_kg(kaz_frames: list[CulturalFrame], cb_frames: list[CulturalFrame],
                        embed_fn: Callable[[list[str]], np.ndarray], *,
                        exclude_ids: Optional[Iterable[str]] = None,
                        threshold: float = fb.NORM_TYPE_SIM_THRESHOLD) -> dict:
    """``frame_builder.build_frame_kg`` over the union (excluded passages shape nothing)."""
    ids = [f.frame_id for f in kaz_frames] + [f.frame_id for f in cb_frames]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate frame ids between KazCulture and CultureBank frames")
    excluded = set(exclude_ids or ())
    unknown = excluded - {f.frame_id for f in kaz_frames}
    if unknown:
        raise ValueError(f"{len(unknown)} excluded ids are not KazCulture frames")
    kg = fb.build_frame_kg(list(kaz_frames) + list(cb_frames), embed_fn,
                           exclude_ids=excluded, threshold=threshold)
    kept_kaz = [f for f in kg["frames"] if f.culture == KAZAKH]
    kg["meta"].update({
        "schema": SCHEMA_VERSION,
        "n_kazculture_frames": len(kept_kaz),
        "n_culturebank_frames": len(kg["frames"]) - len(kept_kaz),
        "kazculture_label_status": dict(Counter(f.label_status for f in kept_kaz)),
        "n_norm_types_with_kazakh": sum(1 for n in kg["norm_types"] if KAZAKH in n.cultures),
        "n_norm_types_kazakh_and_other": sum(
            1 for n in kg["norm_types"] if KAZAKH in n.cultures and len(n.cultures) >= 2),
    })
    return kg
