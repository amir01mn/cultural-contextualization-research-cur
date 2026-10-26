"""Keep CultureBank's evaluation columns out of KG construction and evidence.

Every CultureBank row ships ``eval_whole_desc``, ``eval_question``,
``eval_persona`` and ``eval_scenario``, which eval v2 uses as the test set. The
August extraction prompt serialized the whole raw row, so some triples were
written from ``eval_whole_desc`` rather than from the descriptor. The verbatim
leak guard in eval v2 cannot see paraphrased content; ``triple_uses_eval_text``
is a lexical proxy for it.
"""

from __future__ import annotations

import re
from typing import Optional

EVAL_FIELDS = ("eval_whole_desc", "eval_question", "eval_persona", "eval_scenario")

_WORD = re.compile(r"[a-z]+")
_STOP = frozenset(
    "a an the of and or to in for with on as is are be by their its it that this "
    "at from".split())


def descriptor_fields(raw: Optional[dict]) -> dict:
    """``raw`` without the evaluation columns."""
    return {k: v for k, v in (raw or {}).items() if k not in EVAL_FIELDS}


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def eval_only_words(text: str, raw: Optional[dict]) -> set[str]:
    """Content words of ``text`` found in the eval columns but in no descriptor field."""
    raw = raw or {}
    src = _words(" ".join(str(v) for v in descriptor_fields(raw).values() if v))
    ev = _words(" ".join(str(raw.get(k) or "") for k in EVAL_FIELDS))
    return {w for w in _words(text) - _STOP if w in ev and w not in src}


def triple_uses_eval_text(subject: str, obj: str, raw: Optional[dict],
                          min_words: int = 2) -> bool:
    """True when a triple's subject/object carry >= ``min_words`` eval-only words."""
    return len(eval_only_words(f"{subject} {obj}", raw)) >= min_words
