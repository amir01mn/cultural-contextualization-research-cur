"""KazCulture S4R: a thin adapter over the KG v3 ``FrameRetriever``.

Nothing here re-implements routing. ``FrameRetriever.route`` picks refine /
abstract / none exactly as for CultureBank, and ``render_route`` produces the
text. This module only

* loads the strict and full v3k KG directories (``scripts/build_kazculture_frames.py``),
* splits the rendered route into evidence blocks, one per frame / norm type /
  profile / norm / analog, whose texts joined by ``"\\n"`` equal ``render_route``,
* keeps the longest prefix of blocks that fits a character budget, with the
  semantics of ``kazculture_index.render_evidence_with_hits`` (blocks in order,
  separators counted, stop at the first block that does not fit, report only
  what was rendered),
* exposes per-passage labels for S1+L, and
* builds the CTRL-SHUF variant (labels / norm types / analogs swapped for those
  of a deterministic non-matching donor).

CultureBank frames reach a KazCulture question only as abstract-route analogs
(or, rarely, as members of a matched norm type); they are flagged ``indirect``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..core.schemas import CulturalFrame
from .frame_retriever import (ROUTE_ABSTRACT, ROUTE_NONE, ROUTE_REFINE, FrameRetriever,
                              format_frame, render_route)

CULTURE_LABEL = "Kazakh"
KAZAKH = "kazakh"
SETTINGS = ("strict", "full")
# route_mode -> (allow_abstraction, shuffled). "routed"/"shuffled" are the names
# src/evaluation/kazculture_final.py passes for S4R / CTRL-SHUF.
ROUTE_MODES = {"full": (True, False), "routed": (True, False), "noabs": (False, False),
               "shuffled": (True, True)}
SEPARATOR = "\n"            # "\n".join(block texts) == render_route(...)
SHUFFLE_SEED = 0
THRESHOLDS_FILE = "routing_thresholds.json"


def check_embedding_backend(kg_dir: Path) -> None:
    """Refuse to query a KG in a different embedding space than it was built in.

    ``FrameRetriever`` re-embeds every frame when the active backend differs
    from ``index_config.json`` (by design for TF-IDF/CultureBank). For KazCulture
    that silently swapped BGE-M3 for English-only MiniLM in the first v1 runs.
    """
    from ..core.embedding_utils import backend_name
    cfg = kg_dir / "index_config.json"
    if not cfg.exists():
        return
    stored = json.loads(cfg.read_text(encoding="utf-8")).get("backend")
    active = backend_name()
    if stored and stored != active:
        raise RuntimeError(f"{kg_dir} was embedded with {stored} but the active backend is "
                           f"{active}; set EMBEDDING_BACKEND=bge-m3")


def render_s4r_evidence(blocks: list[dict]) -> str:
    """The evidence string for the prompt (what the budget was checked against)."""
    return SEPARATOR.join(b["text"] for b in blocks)


def fit_blocks(blocks: list[dict], budget_chars: int) -> list[dict]:
    """Greedy in-order prefix; the joined text never exceeds ``budget_chars``."""
    texts, used = [], []
    for b in blocks:
        if len(SEPARATOR.join(texts + [b["text"]])) > budget_chars:
            break
        texts.append(b["text"])
        used.append(b)
    return used


def _block(text: str, source_id: Optional[str], layer: str, route: str, indirect: bool,
           similarity: Optional[float]) -> dict:
    return {"text": text, "source_id": source_id, "layer": layer, "route": route,
            "indirect": bool(indirect),
            "similarity": None if similarity is None else round(float(similarity), 6)}


def route_blocks(r: dict, fr: FrameRetriever, scores: Optional[np.ndarray],
                 culture_label: str = CULTURE_LABEL) -> list[dict]:
    """Split ``render_route(r, ...)`` into blocks without changing a character."""
    full = render_route(r, fr.profiles, fr.nt_by_id, culture_label)
    if not full:
        return []
    mode = r["mode"]
    header, rest = full.split("\n", 1)
    lines = rest.split("\n")
    pieces: list[tuple[list[str], Optional[str], str, bool, Optional[float]]] = []

    def sim(fid: str) -> Optional[float]:
        if scores is None or fid not in fr.frame_index:
            return None
        v = float(scores[fr.frame_index[fid]])
        return v if np.isfinite(v) else None

    if mode == ROUTE_REFINE:
        pos = 0
        for i, (f, s) in enumerate(r["frames"], 1):
            nt = fr.nt_by_id.get(f.norm_type_id)
            n = format_frame(f, i, nt.label if nt else None).count("\n") + 1
            pieces.append((lines[pos:pos + n], f.frame_id, "frame",
                           f.culture != KAZAKH, s))
            pos += n
    elif mode == ROUTE_ABSTRACT:
        pos = 0
        for nt, s in r["norm_types"]:
            pieces.append(([lines[pos]], nt.norm_type_id, "norm_type",
                           KAZAKH not in nt.cultures, s))
            pos += 1
        if pos < len(lines) and lines[pos].startswith("- Values most often expressed"):
            pieces.append(([lines[pos]], f"profile:{r['culture']}", "culture_profile",
                           False, None))
            pos += 1
        for group, layer, indirect in ((r["related"], "norm", False),
                                       (r["analogs"], "analog", True)):
            if not group:
                continue
            sub = lines[pos]
            pos += 1
            for j, f in enumerate(group):
                item = [sub, lines[pos]] if j == 0 else [lines[pos]]
                pieces.append((item, f.frame_id, layer,
                               indirect or f.culture != KAZAKH, sim(f.frame_id)))
                pos += 1
    blocks = []
    for k, (ls, sid, layer, indirect, s) in enumerate(pieces):
        text = "\n".join(([header] if k == 0 else []) + ls)
        blocks.append(_block(text, sid, layer, mode, indirect, s))
    assert SEPARATOR.join(b["text"] for b in blocks) == full, "block split changed the text"
    return blocks


# ---------------------------------------------------------------------------
# Deterministic donors for CTRL-SHUF
# ---------------------------------------------------------------------------

def _rng_for(key: str, seed: int = SHUFFLE_SEED) -> np.random.Generator:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)
    return np.random.default_rng([seed, h])


def donor_map(ids: Iterable[str], group: Callable[[str], object],
              seed: int = SHUFFLE_SEED) -> dict[str, str]:
    """id -> a different id, preferring a different ``group``; deterministic per id."""
    pool = sorted(set(ids))
    out = {}
    for i in pool:
        g = group(i)
        cands = [x for x in pool if x != i and group(x) != g] or [x for x in pool if x != i]
        if cands:
            out[i] = cands[int(_rng_for(i, seed).integers(len(cands)))]
    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class KazCultureFrameRetriever:
    """S4R retrieval over ``<frames_dir>/{strict,full}`` (v3k KG directories)."""

    def __init__(self, frames_dir: str | Path, *, settings: Iterable[str] = SETTINGS,
                 tau_ref: Optional[float] = None, tau_abs: Optional[float] = None,
                 retrievers: Optional[dict[str, FrameRetriever]] = None):
        self.frames_dir = Path(frames_dir)
        self.settings = tuple(settings)
        for s in self.settings:
            if s not in SETTINGS:
                raise ValueError(f"unknown setting {s!r}")
        # Loaded on first use, so a strict-only run never reads full/.
        self.retrievers: dict[str, FrameRetriever] = dict(retrievers or {})
        if tau_ref is None or tau_abs is None:
            path = self.frames_dir / THRESHOLDS_FILE
            if path.exists():
                t = json.loads(path.read_text(encoding="utf-8"))
                tau_ref = t["tau_ref"] if tau_ref is None else tau_ref
                tau_abs = t["tau_abs"] if tau_abs is None else tau_abs
        self.tau_ref, self.tau_abs = tau_ref, tau_abs
        self._donors: dict[str, dict] = {}

    # -- helpers --------------------------------------------------------------

    def _fr(self, setting: str) -> FrameRetriever:
        if setting not in self.retrievers:
            if setting not in self.settings:
                raise ValueError(f"setting {setting!r} not enabled (have {self.settings})")
            check_embedding_backend(self.frames_dir / setting)
            self.retrievers[setting] = FrameRetriever(self.frames_dir / setting)
        return self.retrievers[setting]

    def kazakh_frames(self, setting: str) -> dict[str, CulturalFrame]:
        fr = self._fr(setting)
        return {fr.frames[i].frame_id: fr.frames[i] for i in fr.by_culture.get(KAZAKH, [])}

    def exclude_mask(self, setting: str, passage_ids: Optional[Iterable[str]]):
        ids = set(passage_ids or ())
        return self._fr(setting).mask_for_instance_ids(ids) if ids else None

    def route(self, question_text: str, *, setting: str = "strict", top_k: int = 8,
              route_mode: str = "full", tau_ref: Optional[float] = None,
              tau_abs: Optional[float] = None,
              exclude_passage_ids: Optional[Iterable[str]] = None) -> dict:
        """The raw ``FrameRetriever.route`` result for a KazCulture question."""
        if route_mode not in ROUTE_MODES:
            raise ValueError(f"route_mode must be one of {sorted(ROUTE_MODES)}")
        tau_ref = self.tau_ref if tau_ref is None else tau_ref
        tau_abs = self.tau_abs if tau_abs is None else tau_abs
        if tau_ref is None or tau_abs is None:
            raise ValueError("Routing thresholds unset: pass tau_ref/tau_abs or write "
                             f"{self.frames_dir / THRESHOLDS_FILE} (tune on train only)")
        return self._fr(setting).route(
            question_text, CULTURE_LABEL, top_k=top_k, tau_ref=tau_ref, tau_abs=tau_abs,
            allow_abstraction=ROUTE_MODES[route_mode][0],
            exclude_mask=self.exclude_mask(setting, exclude_passage_ids))

    # -- public API -------------------------------------------------------------

    def retrieve_s4r(self, question_text: str, options: Optional[dict] = None,
                     setting: str = "strict", top_k: int = 8, budget_chars: int = 5000,
                     route_mode: str = "full", *, shuffled: bool = False,
                     exclude_passage_ids: Optional[Iterable[str]] = None,
                     tau_ref: Optional[float] = None,
                     tau_abs: Optional[float] = None) -> list[dict]:
        """Evidence blocks that fit ``budget_chars``; ``[]`` on the none route.

        ``options`` is accepted for API symmetry but not used in the query,
        matching S1 (``kazculture_rag.selected_evidence`` queries the question
        only; ``config.RETRIEVAL_QUERY_INCLUDES_OPTIONS`` is off for the same
        reason). ``shuffled=True`` (or ``route_mode="shuffled"``) gives CTRL-SHUF.
        """
        del options
        shuffled = shuffled or ROUTE_MODES.get(route_mode, (True, False))[1]
        r = self.route(question_text, setting=setting, top_k=top_k, route_mode=route_mode,
                       tau_ref=tau_ref, tau_abs=tau_abs,
                       exclude_passage_ids=exclude_passage_ids)
        if r["mode"] == ROUTE_NONE:
            return []
        fr = self._fr(setting)
        scores = fr._scores(question_text,
                            self.exclude_mask(setting, exclude_passage_ids))
        if shuffled:
            r = self.shuffle_route(r, setting)
        return fit_blocks(route_blocks(r, fr, scores), budget_chars)

    def labels_for_passage(self, passage_id: str, *, setting: str = "full",
                           shuffled: bool = False) -> Optional[dict]:
        """The passage's own ``{norm, norm_type, values}`` (S1+L); None if unlabelled.

        ``norm_type`` is the labeller's own label, not the cluster name, so the
        result is the same in every setting and no other passage (in particular
        no test-overlap passage) influences it. ``setting`` only picks which KG
        file to read; "full" holds every passage. With ``shuffled`` the labels
        of the passage's deterministic donor are returned.
        """
        frames = self.kazakh_frames(setting)
        f = frames.get(passage_id)
        if f is None or f.label_status != "ok":
            return None
        if shuffled:
            f = frames[self.donors(setting)["frame"][passage_id]]
        return {"norm": f.norm, "norm_type": f.norm_type_label, "values": list(f.values)}

    # -- CTRL-SHUF ---------------------------------------------------------------

    def donors(self, setting: str) -> dict:
        """Deterministic (seed 0) donor maps, built from this setting's KG only."""
        if setting in self._donors:
            return self._donors[setting]
        fr = self._fr(setting)
        kaz = {fid: f for fid, f in self.kazakh_frames(setting).items()
               if f.label_status == "ok"}
        cb = {f.frame_id: f for f in fr.frames
              if f.culture != KAZAKH and f.label_status == "ok" and f.norm_type_id}
        d = {
            "frame": donor_map(kaz, lambda i: kaz[i].topic),
            "norm_type": donor_map(fr.nt_by_id, lambda i: i),
            "analog": donor_map(cb, lambda i: cb[i].norm_type_id),
        }
        self._donors[setting] = d
        return d

    def shuffle_route(self, r: dict, setting: str) -> dict:
        """Same layout and practices; labels, norm types and analogs from donors."""
        fr, d = self._fr(setting), self.donors(setting)
        kaz = self.kazakh_frames(setting)
        cb = {f.frame_id: f for f in fr.frames}

        def relabel(f: CulturalFrame) -> CulturalFrame:
            if f.label_status != "ok" or f.frame_id not in d["frame"]:
                return f
            g = kaz[d["frame"][f.frame_id]]
            return f.model_copy(update={"norm": g.norm, "norm_type_label": g.norm_type_label,
                                        "norm_type_id": g.norm_type_id,
                                        "values": list(g.values)})

        out = dict(r)
        if r["mode"] == ROUTE_REFINE:
            out["frames"] = [(relabel(f), s) for f, s in r["frames"]]
        elif r["mode"] == ROUTE_ABSTRACT:
            out["norm_types"] = [(fr.nt_by_id[d["norm_type"].get(n.norm_type_id,
                                                                 n.norm_type_id)], s)
                                 for n, s in r["norm_types"]]
            out["related"] = [kaz[d["frame"][f.frame_id]] if f.frame_id in d["frame"] else f
                              for f in r["related"]]
            out["analogs"] = [cb[d["analog"][f.frame_id]] if f.frame_id in d["analog"] else f
                              for f in r["analogs"]]
        return out


# ---------------------------------------------------------------------------
# Module-level API (src/evaluation/kazculture_final.py imports this module by
# name and calls these; configure() first to change the defaults)
# ---------------------------------------------------------------------------

_DEFAULT: Optional[KazCultureFrameRetriever] = None
_DEFAULT_KW: dict = {}


def default_frames_dir() -> Path:
    from ..core import config
    return config.DATA_DIR / "kazculture_pipeline" / "frames_v3k"


def configure(**kwargs) -> KazCultureFrameRetriever:
    """(Re)build the shared retriever: frames_dir, settings, tau_ref, tau_abs."""
    global _DEFAULT, _DEFAULT_KW
    _DEFAULT_KW = dict(kwargs)
    _DEFAULT = None
    return _get()


def _get() -> KazCultureFrameRetriever:
    global _DEFAULT
    if _DEFAULT is None:
        kw = dict(_DEFAULT_KW)
        frames_dir = kw.pop("frames_dir", None) or default_frames_dir()
        _DEFAULT = KazCultureFrameRetriever(frames_dir, **kw)
    return _DEFAULT


def retrieve_s4r(question_text: str, options: Optional[dict] = None, setting: str = "strict",
                 top_k: int = 8, budget_chars: int = 5000, route_mode: str = "full", *,
                 exclude_passage_ids: Optional[Iterable[str]] = None,
                 shuffled: bool = False) -> list[dict]:
    """``KazCultureFrameRetriever.retrieve_s4r`` on the shared instance."""
    return _get().retrieve_s4r(question_text, options, setting, top_k, budget_chars,
                               route_mode, shuffled=shuffled,
                               exclude_passage_ids=exclude_passage_ids)


def labels_for_passage(passage_id: str, *, shuffled: bool = False) -> Optional[dict]:
    """``KazCultureFrameRetriever.labels_for_passage`` on the shared instance."""
    return _get().labels_for_passage(passage_id, shuffled=shuffled)
