"""Load and normalize the CulturalBench benchmark (sole evaluation metric).

CulturalBench has two formats:
  - CulturalBench-Easy: 4-way single-correct multiple choice.
  - CulturalBench-Hard: each underlying question is split into several
    (question, option) pairs, each judged True/False. A question (group) is
    counted correct only if every one of its True/False judgments is right.

Design notes:
  - This loader is robust to field-name variation (mirrors the CultureBank
    loader): it tries several candidate column names and config/split names.
  - It supports offline operation on cluster compute nodes: if HF_DATASETS_OFFLINE
    is set, ``datasets`` reads from the pre-staged cache under HF_DATASETS_CACHE.
    By default load failures raise; pass ``allow_fallback=True`` for a tiny
    built-in sample (local dev / unit tests only).
  - CulturalBench is held out from KG construction (built only from CultureBank).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from ..core import config
from ..core.schemas import MCQQuestion, MCQVariant
from ..core.utils import coalesce, get_logger, normalize_text, stable_id, write_jsonl

LOGGER = get_logger("culturalbench_loader")

# Config names as staged by scripts/prep_offline_data.py (test split only).
_EASY_CONFIGS = ["CulturalBench-Easy"]
_HARD_CONFIGS = ["CulturalBench-Hard"]
_CULTURALBENCH_SPLITS = ["test"]

_QUESTION_KEYS = ["prompt_question", "question", "prompt", "query"]
_COUNTRY_KEYS = ["country", "Country", "cultural_group", "culture", "nation"]
_REGION_KEYS = ["region", "Region", "continent", "area"]
# Order matters: this picks the key that GROUPS the True/False items belonging to
# one CulturalBench-Hard question. "question_idx" is that key; "data_idx" is a
# per-row identifier and must come after it. With data_idx first, every item
# became its own group, so Hard was scored as per-item True/False accuracy
# instead of the official metric (a question counts only when all of its items
# are correct) - which reported Hard accuracy far higher than it truly was.
_GROUP_KEYS = ["question_idx", "q_idx", "question_id", "data_idx", "idx", "id"]
_ANSWER_KEYS = ["answer", "correct_answer", "label", "gold", "gold_answer", "is_correct"]
_HARD_OPTION_KEYS = ["prompt_option", "option", "answer_option", "candidate", "choice"]

_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _pick(raw: dict, keys: list[str]):
    for k in keys:
        if k in raw and raw[k] not in (None, "", []):
            return raw[k]
    return None


def _extract_easy_options(raw: dict) -> dict[str, str]:
    """Find 4-way options under several possible key conventions."""
    options: dict[str, str] = {}
    # Convention 1: prompt_option_a / option_a / a
    for letter in _LETTERS:
        low = letter.lower()
        for key in (f"prompt_option_{low}", f"option_{low}", f"option{low}",
                    f"choice_{low}", low, letter):
            if key in raw and raw[key] not in (None, "", []):
                options[letter] = normalize_text(raw[key])
                break
    if options:
        return options
    # Convention 2: a list/dict field of options.
    for key in ("options", "choices", "prompt_options"):
        val = raw.get(key)
        if isinstance(val, dict):
            for i, (_k, v) in enumerate(val.items()):
                if i < len(_LETTERS):
                    options[_LETTERS[i]] = normalize_text(v)
            return options
        if isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                if i < len(_LETTERS):
                    options[_LETTERS[i]] = normalize_text(v)
            return options
    return options


def _normalize_answer_letter(value, options: dict[str, str]) -> Optional[str]:
    """Map a gold answer (letter or option text) to a canonical letter."""
    if value is None:
        return None
    s = normalize_text(value)
    if not s:
        return None
    # Single letter (a / A / "a)").
    first = s.strip().strip(").:").upper()
    if first in options:
        return first
    if len(first) >= 1 and first[0] in options and (len(first) == 1 or not first[1].isalnum()):
        return first[0]
    # Numeric index (1-based or 0-based).
    if s.isdigit():
        idx = int(s)
        for cand in (idx - 1, idx):
            if 0 <= cand < len(options):
                return _LETTERS[cand]
    # Match against option text.
    s_low = s.lower()
    for letter, text in options.items():
        if text and text.lower() == s_low:
            return letter
    return None


def _normalize_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = normalize_text(value).lower()
    if s in {"true", "yes", "y", "1", "correct", "right"}:
        return True
    if s in {"false", "no", "n", "0", "incorrect", "wrong"}:
        return False
    return None


def _load_hf_split(config_names: list[str]):
    """Load a CulturalBench variant from the staged HF datasets cache."""
    from datasets import load_dataset

    last_exc = None
    for cfg in config_names:
        for sp in _CULTURALBENCH_SPLITS:
            try:
                return load_dataset(config.CULTURALBENCH_HF_ID, cfg, split=sp), cfg, sp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError("No CulturalBench config could be loaded.")


def _normalize_easy(rows, limit: int) -> list[MCQQuestion]:
    questions: list[MCQQuestion] = []
    for i, raw in enumerate(rows):
        if len(questions) >= limit:
            break
        raw = {str(k): v for k, v in dict(raw).items()}
        options = _extract_easy_options(raw)
        if len(options) < 2:
            continue
        answer_key = _normalize_answer_letter(_pick(raw, _ANSWER_KEYS), options)
        group = coalesce(_pick(raw, _GROUP_KEYS), i)
        qid = f"easy_{group}"
        questions.append(MCQQuestion(
            question_id=qid,
            variant=MCQVariant.EASY,
            group_id=qid,
            question=normalize_text(_pick(raw, _QUESTION_KEYS)),
            cultural_group=normalize_text(_pick(raw, _COUNTRY_KEYS)) or None,
            region=normalize_text(_pick(raw, _REGION_KEYS)) or None,
            options=options,
            answer_key=answer_key,
            raw=raw,
        ))
    return questions


def _normalize_hard(rows, limit_groups: int) -> list[MCQQuestion]:
    questions: list[MCQQuestion] = []
    seen_groups: dict[str, int] = {}
    for i, raw in enumerate(rows):
        raw = {str(k): v for k, v in dict(raw).items()}
        group = str(coalesce(_pick(raw, _GROUP_KEYS), i))
        if group not in seen_groups and len(seen_groups) >= limit_groups:
            continue
        item_idx = seen_groups.get(group, 0)
        seen_groups[group] = item_idx + 1
        option_text = normalize_text(_pick(raw, _HARD_OPTION_KEYS))
        answer_bool = _normalize_bool(_pick(raw, _ANSWER_KEYS))
        gid = f"hard_{group}"
        qid = f"{gid}_{item_idx}"
        questions.append(MCQQuestion(
            question_id=qid,
            variant=MCQVariant.HARD,
            group_id=gid,
            question=normalize_text(_pick(raw, _QUESTION_KEYS)),
            cultural_group=normalize_text(_pick(raw, _COUNTRY_KEYS)) or None,
            region=normalize_text(_pick(raw, _REGION_KEYS)) or None,
            option_text=option_text,
            answer_bool=answer_bool,
            raw=raw,
        ))
    return questions


def _fallback_dataset() -> list[MCQQuestion]:
    """Small built-in CulturalBench-style sample for offline/failure runs."""
    easy = [
        MCQQuestion(
            question_id="easy_fb1", variant=MCQVariant.EASY, group_id="easy_fb1",
            question="In Japan, what is the customary thing to do with your shoes "
                     "when entering someone's home?",
            cultural_group="Japan",
            options={
                "A": "Keep them on throughout the visit",
                "B": "Remove them at the entrance (genkan)",
                "C": "Place them on the dining table",
                "D": "Hand them to the host as a gift",
            },
            answer_key="B",
        ),
        MCQQuestion(
            question_id="easy_fb2", variant=MCQVariant.EASY, group_id="easy_fb2",
            question="In Saudi Arabian hospitality, how does a guest typically "
                     "signal they have had enough Arabic coffee (gahwa)?",
            cultural_group="Saudi Arabia",
            options={
                "A": "Turn the cup upside down on the floor",
                "B": "Loudly say they are finished",
                "C": "Gently shake or wiggle the cup",
                "D": "Leave the gathering immediately",
            },
            answer_key="C",
        ),
    ]
    hard_specs = [
        ("In India, how might a younger person greet a respected elder?",
         "India", "By touching the elder's feet as a sign of respect", True),
        ("In India, how might a younger person greet a respected elder?",
         "India", "By refusing to acknowledge the elder", False),
        ("In India, how might a younger person greet a respected elder?",
         "India", "By seeking the elder's blessing", True),
        ("In India, how might a younger person greet a respected elder?",
         "India", "By demanding the elder bow first", False),
    ]
    hard = []
    for j, (q, country, opt, gold) in enumerate(hard_specs):
        hard.append(MCQQuestion(
            question_id=f"hard_fb1_{j}", variant=MCQVariant.HARD,
            group_id="hard_fb1", question=q, cultural_group=country,
            option_text=opt, answer_bool=gold,
        ))
    return easy + hard


def load_culturalbench(variants: tuple[str, ...] = ("easy", "hard"),
                       limit: int = 250,
                       allow_fallback: bool = False) -> list[MCQQuestion]:
    """Load and normalize CulturalBench questions.

    ``limit`` caps the number of Easy questions and the number of Hard groups
    (each Hard group expands into several True/False items). Raises on failure
    unless ``allow_fallback=True`` (tiny built-in sample for tests only).
    """
    questions: list[MCQQuestion] = []
    try:
        if "easy" in variants:
            rows, cfg, sp = _load_hf_split(_EASY_CONFIGS)
            LOGGER.info("Loaded CulturalBench-Easy from config '%s' split '%s' (%d rows).",
                        cfg, sp, len(rows))
            questions.extend(_normalize_easy(rows, limit))
        if "hard" in variants:
            rows, cfg, sp = _load_hf_split(_HARD_CONFIGS)
            LOGGER.info("Loaded CulturalBench-Hard from config '%s' split '%s' (%d rows).",
                        cfg, sp, len(rows))
            questions.extend(_normalize_hard(rows, limit))
        if not questions:
            raise RuntimeError("CulturalBench returned no usable questions.")
        LOGGER.info("Normalized %d CulturalBench items.", len(questions))
    except Exception as exc:  # noqa: BLE001
        cache = os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HOME", "(default)")
        LOGGER.error("Could not load CulturalBench from Hugging Face: %s", exc)
        LOGGER.error("HF_DATASETS_CACHE=%s  HF_DATASETS_OFFLINE=%s",
                     cache, os.getenv("HF_DATASETS_OFFLINE", "0"))
        if not allow_fallback:
            raise RuntimeError(
                "CulturalBench load failed and allow_fallback=False. "
                "Run slurm/prep_offline.sh on the login node, then ensure GPU jobs "
                "set HF_HOME/HF_DATASETS_CACHE to $SCRATCH/hf_cache."
            ) from exc
        LOGGER.warning("Using built-in fallback CulturalBench sample.")
        questions = _fallback_dataset()
        if "easy" not in variants:
            questions = [q for q in questions if q.variant == MCQVariant.HARD]
        if "hard" not in variants:
            questions = [q for q in questions if q.variant == MCQVariant.EASY]
    return questions


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Load and normalize CulturalBench.")
    parser.add_argument("--variants", default="easy,hard",
                        help="Comma-separated subset of {easy,hard}.")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--allow_fallback", action="store_true", default=False)
    parser.add_argument("--output",
                        default=str(config.PROCESSED_DIR / "culturalbench_normalized.jsonl"))
    args = parser.parse_args(argv)

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    questions = load_culturalbench(variants=variants, limit=args.limit,
                                   allow_fallback=args.allow_fallback)
    count = write_jsonl(args.output, [q.model_dump() for q in questions])
    LOGGER.info("Wrote %d CulturalBench items -> %s", count, args.output)


if __name__ == "__main__":
    main()
