"""KG v3 retrieval: frames, norm types, and abstraction->refinement routing.

``FrameRetriever.route`` decides *what kind* of evidence a question gets, using
the layers rather than a single relevance score:

    refine    the question's culture has a frame that matches the situation
              (score >= tau_ref): give that culture's frames as
              value -> norm type -> norm -> practice chains, with full text.
    abstract  no matching frame, but the culture is in the graph and a norm type
              matches (score >= tau_abs): give the norm type and its values,
              the culture's own related norms, and how culturally similar groups
              handle that norm type (labelled indirect).
    none      otherwise: no evidence, the model answers closed-book.

Cultures are similar when their value profiles and norm-type usage are
similar (``culture_similarity``), i.e. similarity is read off the abstract
layers, not off geography or surface entities.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import config
from ..core.culture_normalizer import canonical_culture
from ..core.embedding_utils import backend_name, embed_texts
from ..core.schemas import CulturalFrame, NormType
from ..core.utils import get_logger, normalize_text
from .retrieval_ranking import rerank_hits

LOGGER = get_logger("frame_retriever")

ROUTE_REFINE, ROUTE_ABSTRACT, ROUTE_NONE = "refine", "abstract", "none"


def frame_document(f: CulturalFrame) -> str:
    parts = [f"{f.culture_label}.", f"Topic: {f.topic}." if f.topic else "", f.text]
    if f.norm:
        parts.append(f"Norm: {f.norm}")
    if f.norm_type_label:
        parts.append(f"Norm type: {f.norm_type_label}.")
    return " ".join(p for p in parts if p)


def norm_type_document(nt: NormType) -> str:
    others = [m for m in nt.member_labels if m != nt.label][:12]
    return "; ".join([nt.label, *others])


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_frame_index(kg_dir: str | Path) -> None:
    """Embed frame and norm-type documents into ``kg_dir`` (one fit, one space)."""
    kg_dir = Path(kg_dir)
    frames = [CulturalFrame(**r) for r in _read_jsonl(kg_dir / "frames.jsonl")]
    nts = [NormType(**r) for r in _read_jsonl(kg_dir / "norm_types.jsonl")]
    docs = [frame_document(f) for f in frames] + [norm_type_document(n) for n in nts]
    emb = embed_texts(docs, fit=True)
    np.save(kg_dir / "frame_embeddings.npy", emb[:len(frames)])
    np.save(kg_dir / "norm_type_embeddings.npy", emb[len(frames):])
    (kg_dir / "index_config.json").write_text(json.dumps(
        {"backend": backend_name(), "frames": len(frames), "norm_types": len(nts)}, indent=2))
    LOGGER.info("Frame index: %d frames + %d norm types (%s) -> %s",
                len(frames), len(nts), backend_name(), kg_dir)


class FrameRetriever:
    """Load a KG v3 directory (``save_frame_kg`` + ``build_frame_index``)."""

    def __init__(self, kg_dir: str | Path):
        self.kg_dir = Path(kg_dir)
        self.frames = [CulturalFrame(**r) for r in _read_jsonl(self.kg_dir / "frames.jsonl")]
        self.norm_types = [NormType(**r) for r in _read_jsonl(self.kg_dir / "norm_types.jsonl")]
        self.profiles: dict = json.loads((self.kg_dir / "culture_profiles.json").read_text())
        self.nt_by_id = {n.norm_type_id: n for n in self.norm_types}
        self.frame_index = {f.frame_id: i for i, f in enumerate(self.frames)}
        self.by_culture: dict[str, list[int]] = defaultdict(list)
        self.by_norm_type: dict[str, list[int]] = defaultdict(list)
        for i, f in enumerate(self.frames):
            self.by_culture[f.culture].append(i)
            if f.norm_type_id:
                self.by_norm_type[f.norm_type_id].append(i)
        cfg_path = self.kg_dir / "index_config.json"
        self.backend = json.loads(cfg_path.read_text()).get("backend") if cfg_path.exists() else None
        fe, ne = self.kg_dir / "frame_embeddings.npy", self.kg_dir / "norm_type_embeddings.npy"
        self.frame_emb = np.load(fe) if fe.exists() else None
        self.nt_emb = np.load(ne) if ne.exists() else None
        if self.backend != backend_name() or self.backend == "tfidf" or self.frame_emb is None:
            # Re-embed in the active space (TF-IDF must be fitted in-process).
            docs = ([frame_document(f) for f in self.frames]
                    + [norm_type_document(n) for n in self.norm_types])
            emb = embed_texts(docs, fit=True)
            self.frame_emb, self.nt_emb = emb[:len(self.frames)], emb[len(self.frames):]
            self.backend = backend_name()
        self._sim_cache: dict[str, list[tuple[str, float]]] = {}
        LOGGER.info("FrameRetriever: %d frames, %d norm types, %d cultures from %s",
                    len(self.frames), len(self.norm_types), len(self.profiles), self.kg_dir)

    # -- shared helpers -----------------------------------------------------

    @property
    def documents(self) -> list[str]:
        return [frame_document(f) for f in self.frames]

    def mask_for_instance_ids(self, instance_ids) -> np.ndarray:
        ids = set(instance_ids or ())
        return np.array([f.frame_id in ids for f in self.frames], dtype=bool)

    def _scores(self, query: str, exclude_mask=None) -> np.ndarray:
        q = embed_texts([normalize_text(query)], fit=False)
        scores = (self.frame_emb @ q[0]).astype(float)
        if exclude_mask is not None:
            scores[np.asarray(exclude_mask, dtype=bool)] = -np.inf
        return scores

    def retrieve(self, query: str, top_k: int = 8, *,
                 question_culture: Optional[str] = None,
                 exclude_mask=None) -> list[tuple[CulturalFrame, float]]:
        """Plain frame retrieval with the shared culture-aware reranking (S4F)."""
        if not self.frames or not normalize_text(query):
            return []
        scores = self._scores(query, exclude_mask)
        cand_k = min(len(self.frames), max(top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, top_k))
        order = np.argsort(-scores)[:cand_k]
        hits = [(self.frames[i], float(scores[i])) for i in order if np.isfinite(scores[i])]
        return rerank_hits(hits, top_k=top_k, question_culture=question_culture,
                           get_culture=lambda f: f.culture_label)

    def culture_similarity(self, culture: str, k: int = 10) -> list[tuple[str, float]]:
        """Cultures ranked by similarity of value profile and norm-type usage."""
        if culture in self._sim_cache:
            return self._sim_cache[culture]
        p = self.profiles.get(culture)
        if not p:
            return []

        def vec(d: dict, keys: list[str]) -> np.ndarray:
            v = np.array([d.get(x, 0.0) for x in keys], dtype=float)
            n = np.linalg.norm(v)
            return v / n if n else v

        vkeys = sorted({v for q in self.profiles.values() for v in q["values"]})
        mine_v = vec(p["values"], vkeys)
        mine_nt = p["norm_types"]
        out = []
        for c, q in self.profiles.items():
            if c == culture or q["n_frames"] < 5:
                continue
            cos_v = float(mine_v @ vec(q["values"], vkeys))
            shared = set(mine_nt) & set(q["norm_types"])
            num = sum(mine_nt[t] * q["norm_types"][t] for t in shared)
            den = (np.sqrt(sum(x * x for x in mine_nt.values()))
                   * np.sqrt(sum(x * x for x in q["norm_types"].values())))
            cos_nt = num / den if den else 0.0
            out.append((c, 0.5 * cos_v + 0.5 * cos_nt))
        out.sort(key=lambda x: -x[1])
        self._sim_cache[culture] = out[:k]
        return self._sim_cache[culture]

    # -- routing ------------------------------------------------------------

    def route(self, query: str, question_culture: Optional[str], *, top_k: int = 8,
              tau_ref: float, tau_abs: float, allow_abstraction: bool = True,
              exclude_mask=None, n_norm_types: int = 2, n_related: int = 3,
              n_analogs: int = 2) -> dict:
        qc = canonical_culture(question_culture) or normalize_text(question_culture or "").lower()
        empty = {"mode": ROUTE_NONE, "culture": qc, "frames": [], "norm_types": [],
                 "related": [], "analogs": [], "best_same_culture": None, "best_norm_type": None}
        if not self.frames or not normalize_text(query):
            return empty
        scores = self._scores(query, exclude_mask)
        same = [i for i in self.by_culture.get(qc, []) if np.isfinite(scores[i])]
        best_same = max((scores[i] for i in same), default=None)
        empty["best_same_culture"] = None if best_same is None else float(best_same)
        if best_same is not None and best_same >= tau_ref:
            ranked = sorted(same, key=lambda i: -scores[i])
            chosen = [i for i in ranked if scores[i] >= tau_ref][:top_k]
            return {**empty, "mode": ROUTE_REFINE,
                    "frames": [(self.frames[i], float(scores[i])) for i in chosen]}
        if not allow_abstraction or qc not in self.profiles or self.nt_emb is None \
                or not len(self.norm_types):
            return empty
        q = embed_texts([normalize_text(query)], fit=False)[0]
        nt_scores = self.nt_emb @ q
        nt_order = [int(j) for j in np.argsort(-nt_scores)[:n_norm_types]
                    if nt_scores[j] >= tau_abs]
        empty["best_norm_type"] = float(nt_scores.max()) if len(nt_scores) else None
        if not nt_order:
            return empty
        nts = [(self.norm_types[j], float(nt_scores[j])) for j in nt_order]
        nt_ids = {n.norm_type_id for n, _ in nts}
        nt_values = {v for n, _ in nts for v in n.values}
        # The culture's own norms in these norm types, else sharing their values.
        own = sorted(same, key=lambda i: (self.frames[i].norm_type_id not in nt_ids,
                                          -scores[i]))
        related = [self.frames[i] for i in own
                   if self.frames[i].norm and (self.frames[i].norm_type_id in nt_ids
                                               or set(self.frames[i].values) & nt_values)
                   ][:n_related]
        # Similar cultures' frames in the same norm types.
        sim_rank = {c: r for r, (c, _s) in enumerate(self.culture_similarity(qc, k=25))}
        cands = []
        for nid in nt_ids:
            for i in self.by_norm_type.get(nid, []):
                f = self.frames[i]
                if f.culture != qc and f.culture in sim_rank and np.isfinite(scores[i]):
                    cands.append((sim_rank[f.culture], -scores[i], i))
        cands.sort()
        analogs, seen = [], set()
        for _r, _s, i in cands:
            f = self.frames[i]
            if f.culture in seen:
                continue
            seen.add(f.culture)
            analogs.append(f)
            if len(analogs) >= n_analogs:
                break
        return {**empty, "mode": ROUTE_ABSTRACT, "norm_types": nts,
                "related": related, "analogs": analogs}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def format_frame(f: CulturalFrame, index: int, nt_label: Optional[str] = None) -> str:
    values = ", ".join(f.values) if f.values else "unlabeled"
    nt = nt_label or f.norm_type_label or "unlabeled"
    lines = [f"[{index}] {f.culture_label}",
             f"  value: {values}",
             f"  norm type: {nt}"]
    if f.norm:
        lines.append(f"  norm: {f.norm}")
    agree = f" (agreement {f.agreement:.2f})" if f.agreement is not None else ""
    lines.append(f"  practice: {f.text}{agree}")
    return "\n".join(lines)


def render_frames(frames: list[CulturalFrame], nt_by_id: dict) -> str:
    return "\n".join(
        format_frame(f, i, nt_by_id[f.norm_type_id].label if f.norm_type_id in nt_by_id else None)
        for i, f in enumerate(frames, 1))


def render_route(r: dict, profiles: dict, nt_by_id: dict, culture_label: str) -> str:
    if r["mode"] == ROUTE_REFINE:
        return (f"Specific knowledge about {culture_label} "
                "(value → norm type → norm → practice):\n"
                + render_frames([f for f, _ in r["frames"]], nt_by_id))
    if r["mode"] != ROUTE_ABSTRACT:
        return ""
    lines = [f"No record matches this exact situation for {culture_label}. "
             "General knowledge from the graph (use with judgement):"]
    for nt, _s in r["norm_types"]:
        vals = ", ".join(nt.values) if nt.values else "unlabeled"
        lines.append(f"- Relevant norm type: {nt.label} (underlying values: {vals})")
    prof = profiles.get(r["culture"]) or {}
    top_vals = sorted(prof.get("values", {}).items(), key=lambda x: -x[1])[:3]
    if top_vals:
        lines.append(f"- Values most often expressed in {culture_label}'s norms: "
                     + ", ".join(v for v, _ in top_vals))
    if r["related"]:
        lines.append(f"- Related norms in {culture_label}:")
        lines += [f"    * {f.norm}" for f in r["related"]]
    if r["analogs"]:
        lines.append("- How culturally similar groups handle this norm type "
                     f"(indirect; {culture_label} may differ):")
        lines += [f"    * {f.culture_label}: {f.norm or f.actor_behavior}" for f in r["analogs"]]
    return "\n".join(lines)


def route_ids(r: dict) -> list[str]:
    if r["mode"] == ROUTE_REFINE:
        return [f.frame_id for f, _ in r["frames"]]
    if r["mode"] == ROUTE_ABSTRACT:
        return [f.frame_id for f in r["related"]] + [f.frame_id for f in r["analogs"]]
    return []
