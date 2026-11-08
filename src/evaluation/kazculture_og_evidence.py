"""Option-grounded evidence (KazCulture v2), shared by the dev and the test harness.

One builder per (index, frames) pair, so dev (`scripts/eval_kazculture_v2_dev.py`)
and test (`scripts/eval_kazculture_final.py`) render byte-identical evidence for a
given question, exclusion set and budget.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.core.schemas import CulturalFrame
from src.evaluation.kazculture_final import Evidence, format_labels, s1_evidence
from src.retrieval.kazculture_option_grounded import (
    OptionGroundedRetriever, abstraction_for_ungrounded, norm_abstraction,
    render_flat_layered, render_option_grounded)

OG_CONDITIONS = ("S1-OG", "S1-OG+L", "S1-OGpo", "S4R-OG", "S4R-OGR", "S4R-OGF", "S4R-OGFR")
TAU_GROUND = 1.0      # fixed before any dev result: "all distinctive option stems found"
PER_OPTION_K = 4      # candidates retrieved per option
PER_OPTION_SHOWN = 2  # practices shown per option in S4R-OG(R)


def load_kazakh_frames(frames_jsonl: Path) -> dict[str, CulturalFrame]:
    out = {}
    with Path(frames_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("culture") == "kazakh":
                    out[r["frame_id"]] = CulturalFrame(**r)
    return out


def frame_labels(f: Optional[CulturalFrame]) -> Optional[dict]:
    """The S1+L label dict for one frame (its own raw labels)."""
    if f is None or not (f.norm or f.norm_type_label or f.values):
        return None
    return {"norm": f.norm, "norm_type": f.norm_type_label, "values": list(f.values or [])}


class OGEvidenceBuilder:
    """``index``: a MultilingualIndex; ``frames``: Kazakh frames of the same setting
    (``KazCultureFrameRetriever.kazakh_frames``); ``labels_fn``: the S1+L label
    source (``labels_for_passage``), so S1-OG+L annotates exactly like S1+L."""

    def __init__(self, index, frames: dict[str, CulturalFrame], *,
                 labels_fn: Optional[Callable[[str], Optional[dict]]] = None,
                 dense_fn: Optional[Callable[[str], np.ndarray]] = None):
        rows = [k for k, u in enumerate(index.units)
                if u["system"] == "S1" and u["dataset"] == "KazCulture" and not u["indirect"]]
        self.rows = np.asarray(rows)
        self.index = index
        self._dense: dict[str, np.ndarray] = {}
        self.og = OptionGroundedRetriever([index.units[k] for k in rows],
                                          dense_fn or self._dense_scores)
        self.frames = frames
        self.labels_fn = labels_fn or (lambda pid: frame_labels(frames.get(pid)))
        nt: dict[str, list[str]] = {}
        for pid, f in frames.items():
            if f.norm_type_id:
                nt.setdefault(f.norm_type_id, []).append(pid)
        self.nt_members = nt

    def _dense_scores(self, question: str) -> np.ndarray:
        if question not in self._dense:
            vec = np.asarray(self.index._load_model().encode(
                [question], normalize_embeddings=True, convert_to_numpy=True)[0],
                dtype=np.float32)
            self._dense[question] = np.asarray(self.index.emb[self.rows] @ vec, dtype=float)
        return self._dense[question]

    def evidence(self, condition: str, question: str, options: dict[str, str],
                 exclude: set[str], *, top_k: int, budget: int) -> Evidence:
        if condition in ("S1-OG", "S1-OG+L"):
            annotate = None
            if condition == "S1-OG+L":
                annotate = lambda u: format_labels(self.labels_fn(u["passage_id"]))
            return s1_evidence(self.og.pooled(question, options, exclude=exclude),
                               top_k=top_k, max_chars=budget, annotate=annotate)
        per = self.og.per_option(question, options, exclude=exclude, k=PER_OPTION_K)
        if condition == "S1-OGpo":
            return s1_evidence(self.og.interleave(per, top_k), top_k=top_k, max_chars=budget)
        if condition in ("S4R-OGF", "S4R-OGFR"):
            lines = None
            if condition == "S4R-OGFR":
                lines = abstraction_for_ungrounded(per, self.frames, self.nt_members,
                                                   tau_ground=TAU_GROUND, exclude=exclude)
            blocks, meta = render_flat_layered(self.og.interleave(per, top_k), self.frames,
                                               budget_chars=budget, abstract_lines=lines)
            for m in meta:
                m.setdefault("unit_id", None)
                m["route"] = "abstract" if m["layer"] == "abstract" else "refine"
                m.setdefault("indirect", False)
            return Evidence(blocks, meta, sep="\n",
                            n_abstract_lines=sum(m["layer"] == "abstract" for m in meta))
        if condition not in ("S4R-OG", "S4R-OGR"):
            raise ValueError(condition)
        tau = abstraction = None
        if condition == "S4R-OGR":
            tau = TAU_GROUND
            abstraction = norm_abstraction(self.frames, self.nt_members, exclude=exclude)
        blocks, meta = render_option_grounded(
            options, per, self.frames, budget_chars=budget, per_option=PER_OPTION_SHOWN,
            tau_ground=tau, abstraction=abstraction)
        for m in meta:
            m.setdefault("unit_id", None)
            m["route"] = "abstract" if m["layer"] == "abstract" else "refine"
            m.setdefault("indirect", False)
        return Evidence(blocks, meta, sep="\n",
                        n_abstract_options=sum(1 for m in meta if m["layer"] == "option"
                                               and not m.get("grounded", True)))
