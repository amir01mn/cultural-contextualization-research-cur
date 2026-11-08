"""Option-grounded retrieval over the KazCulture passages / v3k frames (v2 dev).

Diagnosis behind it (v2 dev, `docs/KAZCULTURE_V2_OPTION_GROUNDED.md`): when an
answer-bearing passage is in the evidence, Qwen3-32B answers ~93% of dev MCQ
correctly; the question-only dense query (S1) puts one in the top 8 for only
~27% of questions, while one exists in the corpus for ~73%. KazCulture
questions are item/term lookups in Kazakh; the answer options name the
candidate items, so they are the strongest available retrieval key.

Scores (no tuned weights): ``z(dense(question)) + z(BM25(question + text))``,
z-scored over KazCulture passages, where ``text`` is all options (pooled) or
one option (per-option grounding). BM25 uses 5-character stem prefixes of
words with >= 4 letters, a crude but language-agnostic stemmer for Kazakh
agglutination.

Per option the score adds ``z(BM25(distinctive stems))``: the option's stems
that are in neither the question nor another option, so options do not all
retrieve the question's passages. Grounding of an option in a passage = share
of those distinctive stems that occur in it. Nothing looks at the answer key.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def stems(text: str) -> list[str]:
    return [w[:5] for w in _WORD.findall((text or "").lower()) if len(w) >= 4]


def distinctive_stems(question: str, options: dict[str, str]) -> dict[str, list[str]]:
    """Per option, its stems that are neither in the question nor in another option.

    Falls back to the option's stems minus the question's, then to all its stems,
    so no option is left without a key.
    """
    q = set(stems(question))
    per = {k: list(dict.fromkeys(stems(v))) for k, v in options.items()}
    count = Counter(w for ws in per.values() for w in set(ws))
    out = {}
    for k, ws in per.items():
        d = [w for w in ws if w not in q and count[w] == 1]
        out[k] = d or [w for w in ws if w not in q] or ws
    return out


def grounding(option: str | list[str], passage_stems: set[str]) -> float:
    """Share of the option's (distinctive) stems that occur in the passage."""
    o = stems(option) if isinstance(option, str) else option
    return sum(w in passage_stems for w in o) / len(o) if o else 0.0


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        tfs = [Counter(stems(d)) for d in docs]
        self.n = len(docs)
        self.lens = np.array([sum(c.values()) for c in tfs], dtype=float)
        self.avg = float(self.lens.mean()) if self.n else 1.0
        df = Counter(w for c in tfs for w in c)
        self.idf = {w: math.log(1 + (self.n - k + 0.5) / (k + 0.5)) for w, k in df.items()}
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(tfs):
            for w, f in c.items():
                self.post[w].append((i, f))

    def score(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for w in set(stems(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.post[w]:
                out[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return out


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


class OptionGroundedRetriever:
    """Hybrid dense+BM25 ranking of KazCulture passages, pooled or per option.

    ``units``: S1 index units (KazCulture, direct). ``dense_fn(question)`` returns
    cosine scores aligned with ``units`` (BGE-M3 in the harness; injectable for
    tests).
    """

    def __init__(self, units: list[dict], dense_fn: Callable[[str], np.ndarray]):
        self.units = units
        self.dense_fn = dense_fn
        self.bm25 = BM25([u["text"] for u in units])
        self.stemsets = [set(stems(u["text"])) for u in units]
        self.pid_row = {u["passage_id"]: i for i, u in enumerate(units)}

    def _mask(self, exclude: Optional[Iterable[str]]) -> np.ndarray:
        m = np.zeros(len(self.units), dtype=bool)
        for p in exclude or ():
            if p in self.pid_row:
                m[self.pid_row[p]] = True
        return m

    def _hits(self, score: np.ndarray, k: int) -> list[dict]:
        order = np.argsort(-score, kind="stable")[:k]
        return [{**self.units[i], "similarity": round(float(score[i]), 6)} for i in order]

    def pooled(self, question: str, options: dict[str, str], *, k: int = 40,
               exclude: Optional[Iterable[str]] = None, dense: Optional[np.ndarray] = None
               ) -> list[dict]:
        d = self.dense_fn(question) if dense is None else dense
        s = _z(np.asarray(d, dtype=float)) + _z(
            self.bm25.score(question + " " + " ".join(options.values())))
        s[self._mask(exclude)] = -np.inf
        return self._hits(s, k)

    def per_option(self, question: str, options: dict[str, str], *, k: int = 5,
                   exclude: Optional[Iterable[str]] = None, dense: Optional[np.ndarray] = None
                   ) -> dict[str, list[dict]]:
        d = _z(np.asarray(self.dense_fn(question) if dense is None else dense, dtype=float))
        mask = self._mask(exclude)
        keys = distinctive_stems(question, options)
        out = {}
        for letter, text in options.items():
            s = d + _z(self.bm25.score(question + " " + text)) \
                + _z(self.bm25.score(" ".join(keys[letter])))
            s[mask] = -np.inf
            hits = self._hits(s, k)
            for h in hits:
                h["grounding"] = round(grounding(
                    keys[letter], self.stemsets[self.pid_row[h["passage_id"]]]), 4)
            out[letter] = hits
        return out

    @staticmethod
    def interleave(per: dict[str, list[dict]], k: int) -> list[dict]:
        """Round-robin over options, first hit of each option first, no repeats."""
        seen, out = set(), []
        depth = max((len(v) for v in per.values()), default=0)
        for r in range(depth):
            for letter in sorted(per):
                if r < len(per[letter]) and per[letter][r]["passage_id"] not in seen:
                    seen.add(per[letter][r]["passage_id"])
                    out.append(per[letter][r])
                    if len(out) >= k:
                        return out
        return out


# ---------------------------------------------------------------------------
# Rendering (S4R-OG / S4R-OGR): the layered KG view, grouped by option
# ---------------------------------------------------------------------------

OG_HEADER = ("Evidence per answer option from Kazakh training sources "
             "(each practice with its model-labelled norm and values):")


def _layer_line(frame) -> str:
    parts = []
    if frame is not None and getattr(frame, "values", None):
        parts.append("values: " + ", ".join(frame.values))
    if frame is not None and getattr(frame, "norm_type_label", None):
        parts.append("norm type: " + frame.norm_type_label)
    if frame is not None and getattr(frame, "norm", None):
        parts.append("norm: " + frame.norm)
    return " | ".join(parts)


def render_option_grounded(
    options: dict[str, str], per: dict[str, list[dict]], frames: dict, *,
    budget_chars: int, per_option: int = 2, tau_ground: Optional[float] = None,
    abstraction: Optional[Callable[[str, list[dict]], list[str]]] = None,
) -> tuple[list[str], list[dict]]:
    """Blocks grouped by option; a passage is printed once and referenced after.

    With ``tau_ground`` set, an option whose best grounding is below it is
    marked as not directly grounded, and (if ``abstraction`` is given) also gets
    norm-level lines (the abstract route) before its best-ranked practices.
    Returns (blocks, meta); the joined text (``"\n"``) fits ``budget_chars``.
    """
    blocks: list[str] = [OG_HEADER]
    meta: list[dict] = [{"id": None, "layer": "header"}]
    printed: dict[str, int] = {}
    n_used = len(OG_HEADER)

    def fits(text: str) -> bool:
        return n_used + 1 + len(text) <= budget_chars

    rounds = [(r, letter) for r in range(per_option) for letter in sorted(options)]
    # first pass: option headers + first practice each; later rounds add depth
    header_done: set[str] = set()
    for r, letter in rounds:
        hits = per.get(letter, [])
        best = max((h.get("grounding", 0.0) for h in hits), default=0.0)
        grounded = tau_ground is None or best >= tau_ground
        if letter not in header_done:
            tag = (" (no retrieved source mentions this option's key words)"
                   if tau_ground is not None and best == 0.0 else "")
            head = f"Option {letter}: {options[letter]}{tag}"
            if not fits(head):
                break
            blocks.append(head)
            meta.append({"id": None, "layer": "option", "option": letter,
                         "grounded": grounded, "best_grounding": best})
            n_used += 1 + len(head)
            header_done.add(letter)
            if not grounded and abstraction is not None:
                for line in abstraction(letter, hits):
                    if fits(line):
                        blocks.append(line)
                        meta.append({"id": None, "layer": "abstract", "option": letter})
                        n_used += 1 + len(line)
        if r >= len(hits):
            continue
        h = hits[r]
        pid = h["passage_id"]
        if pid in printed:
            line = f"  - see [{printed[pid]}]"
            if fits(line):
                blocks.append(line)
                meta.append({"id": pid, "layer": "ref", "option": letter})
                n_used += 1 + len(line)
            continue
        num = len(printed) + 1
        lay = _layer_line(frames.get(pid))
        text = f"  [{num}] practice: {h['text']}" + (f"\n      {lay}" if lay else "")
        if not fits(text):
            continue
        printed[pid] = num
        blocks.append(text)
        meta.append({"id": pid, "unit_id": h.get("unit_id"), "layer": "frame", "option": letter,
                     "similarity": h.get("similarity"), "grounding": h.get("grounding")})
        n_used += 1 + len(text)
    return blocks, meta


def norm_abstraction(frames: dict, nt_members: dict[str, list[str]], *,
                     n_related: int = 2, exclude: Optional[set] = None
                     ) -> Callable[[str, list[dict]], list[str]]:
    """Abstract route for an ungrounded option: the norm type of its best-ranked
    frame and up to ``n_related`` sibling norms (English labels, L2/L3 only)."""
    exclude = exclude or set()

    def fn(letter: str, hits: list[dict]) -> list[str]:
        for h in hits:
            f = frames.get(h["passage_id"])
            if f is None or not f.norm_type_id:
                continue
            sibs = [frames[p].norm for p in nt_members.get(f.norm_type_id, [])
                    if p != h["passage_id"] and p not in exclude and p in frames
                    and frames[p].norm][:n_related]
            lines = [f"  related norm type: {f.norm_type_label}"
                     + (f" (values: {', '.join(f.values)})" if f.values else "")]
            lines += [f"    - {s}" for s in [f.norm, *sibs] if s]
            return lines
        return []
    return fn


# ---------------------------------------------------------------------------
# Flat layered rendering (S4R-OGF / S4R-OGFR)
# ---------------------------------------------------------------------------
# Dev a2 showed that grouping practices under "Option X" headers misattributes
# them (a passage retrieved for several options is printed under the first, so
# the layout asserts support it does not have; Qwen over-picked A). The flat
# layout keeps S1-OGpo's passages and order exactly and only adds the KG layers.

FLAT_HEADER = ("Kazakh training sources, each practice with its model-labelled "
               "values, norm type and norm:")
ABSTRACT_HEADER = ("Broader Kazakh norms (abstraction layer; no source above names "
                   "some options' key words):")


def render_flat_layered(hits: list[dict], frames: dict, *, budget_chars: int,
                        abstract_lines: Optional[list[str]] = None
                        ) -> tuple[list[str], list[dict]]:
    """Numbered practices (same order as ``hits``) with their layer line, then the
    optional abstraction section, all within ``budget_chars`` (joined by "\\n").

    The abstraction section is reserved first, so it is never squeezed out by
    the practices; practices are packed in order and stop at the first that
    does not fit (the frozen S1 packing rule).
    """
    abstract = [ABSTRACT_HEADER, *abstract_lines] if abstract_lines else []
    reserve = len("\n".join(abstract)) + 1 if abstract else 0
    if reserve > budget_chars // 3:          # never let abstraction crowd out practices
        abstract, reserve = [], 0
    blocks, meta = [FLAT_HEADER], [{"id": None, "layer": "header"}]
    used = len(FLAT_HEADER)
    for h in hits:
        lay = _layer_line(frames.get(h["passage_id"]))
        num = sum(1 for m in meta if m["layer"] == "frame") + 1
        text = f"[{num}] practice: {h['text']}" + (f"\n    {lay}" if lay else "")
        if used + 1 + len(text) + reserve > budget_chars:
            break
        blocks.append(text)
        meta.append({"id": h["passage_id"], "unit_id": h.get("unit_id"), "layer": "frame",
                     "similarity": h.get("similarity"), "grounding": h.get("grounding")})
        used += 1 + len(text)
    for line in abstract:
        blocks.append(line)
        meta.append({"id": None, "layer": "abstract"})
    return blocks, meta


def abstraction_for_ungrounded(per: dict[str, list[dict]], frames: dict,
                               nt_members: dict[str, list[str]], *, tau_ground: float,
                               exclude: Optional[set] = None, n_related: int = 2
                               ) -> list[str]:
    """Abstract route: for each option whose best grounding is below ``tau_ground``,
    the norm type of its top frame and up to ``n_related`` sibling norms; norm
    types are listed once, options are not named (no attribution claims)."""
    fn = norm_abstraction(frames, nt_members, n_related=n_related, exclude=exclude)
    lines, seen = [], set()
    for letter in sorted(per):
        hits = per[letter]
        if max((h.get("grounding", 0.0) for h in hits), default=0.0) >= tau_ground:
            continue
        got = fn(letter, hits)
        if got and got[0] not in seen:
            seen.add(got[0])
            lines += got
    return lines
