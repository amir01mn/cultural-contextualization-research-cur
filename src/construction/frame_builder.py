"""KG v3: a frame per CultureBank descriptor, linked through abstract nodes.

Why (eval v2, docs/results/eval_v2/RESULTS.md): the August pathways reached only
38% of records, 65% of them stitched triples from different records, and they
rendered as bare labels. Here the unit is the descriptor itself:

    L3  value          fixed taxonomy (Schwartz et al. 2012, 19 refined values)
    L2  norm type      culture-neutral, clustered from per-frame labels
    L2c culture norm   this group's norm (one sentence, per frame)
    L1  practice       the descriptor text (context, actor, behaviour, ...)

Every record is a frame, so every record is reachable. Cultures connect only
through norm types and values, never through surface entities like "people".
The LLM only *names* the norm and picks values; the descriptor fields are mapped
directly, and CultureBank's ``eval_*`` columns are never read.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..core.culture_normalizer import canonical_culture
from ..core.leakage import EVAL_FIELDS
from ..core.schemas import CulturalFrame, NormType
from ..core.utils import get_logger, normalize_text, stable_id
from ..llm import prompts

LOGGER = get_logger("frame_builder")

# Schwartz, Cieciuch, Vecchione et al. (2012), "Refining the theory of basic
# individual values", JPSP 103(4). Name -> short definition shown to the labeler.
SCHWARTZ_VALUES: dict[str, str] = {
    "self-direction (thought)": "freedom to cultivate one's own ideas and abilities",
    "self-direction (action)": "freedom to determine one's own actions",
    "stimulation": "excitement, novelty and change",
    "hedonism": "pleasure and sensuous gratification",
    "achievement": "success according to social standards",
    "power (dominance)": "power through exercising control over people",
    "power (resources)": "power through control of material and social resources",
    "face": "maintaining one's public image and avoiding humiliation",
    "security (personal)": "safety in one's immediate environment",
    "security (societal)": "safety and stability in the wider society",
    "tradition": "maintaining cultural, family or religious traditions",
    "conformity (rules)": "compliance with rules, laws and formal obligations",
    "conformity (interpersonal)": "avoiding upsetting or harming other people",
    "humility": "recognizing one's insignificance in the larger scheme of things",
    "benevolence (dependability)": "being a reliable, trustworthy member of the in-group",
    "benevolence (caring)": "devotion to the welfare of in-group members",
    "universalism (concern)": "commitment to equality, justice and protection for all",
    "universalism (nature)": "preservation of the natural environment",
    "universalism (tolerance)": "acceptance and understanding of those who are different",
}
VALUE_NAMES = tuple(SCHWARTZ_VALUES)

# Descriptor columns, in rendering order. Anything else in the CSV row (in
# particular EVAL_FIELDS) is ignored.
DESCRIPTOR_KEYS = ("context", "actor", "actor_behavior", "recipient",
                   "recipient_behavior", "relation", "goal", "other_descriptions")

NORM_TYPE_SIM_THRESHOLD = 0.75   # leader clustering of norm-type labels (cosine)
LABEL_MAX_TOKENS = 200


def _clean(value) -> Optional[str]:
    text = normalize_text(value) if value is not None else ""
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return None
    return text


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def descriptor_text(f: CulturalFrame) -> str:
    """The practice (L1) as one readable sentence group, descriptor fields only."""
    parts = []
    if f.context:
        parts.append(f"Context: {f.context}.")
    actor = f.actor or "people"
    if f.actor_behavior:
        parts.append(f"{actor[0].upper() + actor[1:]}: {f.actor_behavior}.")
    if f.recipient and f.recipient_behavior:
        parts.append(f"{f.recipient[0].upper() + f.recipient[1:]}: {f.recipient_behavior}.")
    elif f.recipient_behavior:
        parts.append(f"Response: {f.recipient_behavior}.")
    if f.relation:
        parts.append(f"Relation: {f.relation}.")
    if f.goal:
        parts.append(f"Goal: {f.goal}.")
    if f.other_descriptions:
        parts.append(f"Also: {f.other_descriptions}.")
    return " ".join(parts)


def frame_from_row(row: dict) -> Optional[CulturalFrame]:
    """A frame from one CultureBank CSV row (``rows_with_instance_ids``)."""
    group = _clean(row.get("cultural group") or row.get("cultural_group"))
    behaviour = _clean(row.get("actor_behavior"))
    if not group or not behaviour:
        return None
    fields = {k: _clean(row.get(k)) for k in DESCRIPTOR_KEYS}
    frame = CulturalFrame(
        frame_id=str(row["instance_id"]),
        culture=canonical_culture(group) or group.lower(),
        culture_label=group,
        topic=_clean(row.get("topic")) or "",
        agreement=_float(row.get("agreement_f", row.get("agreement"))),
        **fields,
    )
    frame.text = descriptor_text(frame)
    return frame


def frames_from_rows(rows: Iterable[dict]) -> list[CulturalFrame]:
    frames = [f for f in (frame_from_row(r) for r in rows) if f is not None]
    LOGGER.info("Built %d frames.", len(frames))
    return frames


# ---------------------------------------------------------------------------
# LLM labeling (norm, norm type, values)
# ---------------------------------------------------------------------------

def label_prompt(f: CulturalFrame) -> str:
    descriptor = f"Group: {f.culture_label}. Topic: {f.topic}. {f.text}"
    for key in EVAL_FIELDS:  # defensive: the descriptor is built from fields only
        assert key not in descriptor
    value_list = "\n".join(f"  - {name}: {d}" for name, d in SCHWARTZ_VALUES.items())
    return prompts.FRAME_LABEL_USER.format(
        descriptor=descriptor, culture=f.culture_label, value_list=value_list)


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _match_value(name: str) -> Optional[str]:
    key = normalize_text(name).lower().replace("–", "-").replace("—", "-")
    key = re.sub(r"\s*-\s*", " ", key)
    for v in VALUE_NAMES:
        flat = v.replace("(", "").replace(")", "")
        if key in (v, flat) or key.replace(" ", "") == flat.replace(" ", ""):
            return v
    # A bare family name ("benevolence") is ambiguous for split values; only
    # accept it when it names exactly one value.
    hits = [v for v in VALUE_NAMES if v.split(" (")[0] == key]
    return hits[0] if len(hits) == 1 else None


def parse_label(text: str) -> Optional[dict]:
    """``{"norm", "norm_type", "values"}`` or None when unusable."""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    norm = _clean(obj.get("norm"))
    norm_type = _clean(obj.get("norm_type"))
    raw_values = obj.get("values") or []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    values = []
    for v in raw_values:
        mv = _match_value(str(v))
        if mv and mv not in values:
            values.append(mv)
    if not norm or not norm_type or not values:
        return None
    return {"norm": norm, "norm_type": norm_type.lower().strip(" ."), "values": values[:2]}


def label_frames(backend, frames: list[CulturalFrame], cache_path: Path, *,
                 max_tokens: int = LABEL_MAX_TOKENS, batch_size: int = 2048) -> dict:
    """Label every frame (resumable via ``cache_path`` JSONL), one retry on failure.

    The cache records the model and ``max_tokens`` so a changed budget never
    replays old completions (AGENTS.md). Frames that still fail stay in the KG
    as ``label_status="failed"``: reachable for refinement, absent from the
    abstraction layers. Nothing is substituted.
    """
    model = getattr(backend, "model", "unknown")
    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model and r.get("max_tokens") == max_tokens:
                    done[r["frame_id"]] = r
    todo = [f for f in frames if f.frame_id not in done]
    LOGGER.info("Frame labels: %d cached, %d to do (model=%s)", len(done), len(todo), model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            base = [label_prompt(f) for f in batch]
            outs = backend.generate(base, system_prompt=prompts.FRAME_LABEL_SYSTEM,
                                    temperature=0.0, max_tokens=max_tokens)
            parsed = [parse_label(o) for o in outs]
            retry = [i for i, p in enumerate(parsed) if p is None]
            if retry:
                outs2 = backend.generate(
                    [base[i] + prompts.FRAME_LABEL_RETRY_SUFFIX for i in retry],
                    system_prompt=prompts.FRAME_LABEL_SYSTEM, temperature=0.0,
                    max_tokens=max_tokens)
                for i, o in zip(retry, outs2):
                    parsed[i] = parse_label(o)
                    outs[i] = o
            for f, p, o in zip(batch, parsed, outs):
                r = {"frame_id": f.frame_id, "model": model, "max_tokens": max_tokens,
                     "label": p, "raw": None if p else (o or "")[-500:]}
                done[f.frame_id] = r
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            LOGGER.info("Labeled %d/%d (batch failures after retry: %d)",
                        min(start + batch_size, len(todo)), len(todo),
                        sum(p is None for p in parsed))
    return done


def apply_labels(frames: list[CulturalFrame], labels: dict[str, dict]) -> dict:
    stats = Counter()
    for f in frames:
        lab = (labels.get(f.frame_id) or {}).get("label")
        if lab:
            f.norm, f.norm_type_label, f.values = lab["norm"], lab["norm_type"], list(lab["values"])
            f.label_status = "ok"
        else:
            f.label_status = "failed" if f.frame_id in labels else "unlabeled"
        stats[f.label_status] += 1
    return dict(stats)


# ---------------------------------------------------------------------------
# Norm types (L2): leader clustering of culture-neutral labels
# ---------------------------------------------------------------------------

def cluster_norm_types(frames: list[CulturalFrame],
                       embed_fn: Callable[[list[str]], np.ndarray], *,
                       threshold: float = NORM_TYPE_SIM_THRESHOLD) -> list[NormType]:
    """Assign ``norm_type_id`` to each labeled frame and return the norm types.

    Deterministic leader clustering: labels are visited from most to least
    frequent; each joins the most similar existing leader at cosine >=
    ``threshold`` or becomes a new leader. The leader (the most frequent label)
    names the cluster, so no extra LLM call is needed.
    """
    labeled = [f for f in frames if f.label_status == "ok" and f.norm_type_label]
    counts = Counter(f.norm_type_label for f in labeled)
    uniq = sorted(counts, key=lambda s: (-counts[s], s))
    if not uniq:
        return []
    emb = np.asarray(embed_fn(uniq), dtype=np.float32)
    leader_of: dict[str, str] = {}
    leaders: list[int] = []
    leader_mat = np.zeros((len(uniq), emb.shape[1]), dtype=np.float32)
    for i, lab in enumerate(uniq):
        if leaders:
            sims = leader_mat[:len(leaders)] @ emb[i]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                leader_of[lab] = uniq[leaders[j]]
                continue
        leader_mat[len(leaders)] = emb[i]
        leaders.append(i)
        leader_of[lab] = lab

    members: dict[str, list[CulturalFrame]] = defaultdict(list)
    for f in labeled:
        lead = leader_of[f.norm_type_label]
        f.norm_type_id = stable_id("norm_type", lead, prefix="nt_")
        members[lead].append(f)

    norm_types = []
    for lead, fs in members.items():
        vc = Counter(v for f in fs for v in f.values)
        norm_types.append(NormType(
            norm_type_id=stable_id("norm_type", lead, prefix="nt_"),
            label=lead,
            member_labels=sorted({f.norm_type_label for f in fs}),
            frame_ids=[f.frame_id for f in fs],
            cultures=sorted({f.culture for f in fs}),
            values=[v for v, _ in vc.most_common(2)],
        ))
    norm_types.sort(key=lambda n: (-len(n.frame_ids), n.label))
    multi = sum(1 for n in norm_types if len(n.cultures) >= 2)
    LOGGER.info("Norm types: %d from %d labels (%d shared by >=2 cultures).",
                len(norm_types), len(uniq), multi)
    return norm_types


def culture_profiles(frames: list[CulturalFrame]) -> dict[str, dict]:
    """Per culture: agreement-weighted value distribution and norm-type counts."""
    out: dict[str, dict] = {}
    by_c: dict[str, list[CulturalFrame]] = defaultdict(list)
    for f in frames:
        by_c[f.culture].append(f)
    for c, fs in by_c.items():
        vw: Counter = Counter()
        for f in fs:
            w = f.agreement if f.agreement is not None else 0.5
            for v in f.values:
                vw[v] += w / len(f.values)
        total = sum(vw.values()) or 1.0
        out[c] = {
            "n_frames": len(fs),
            "label": Counter(f.culture_label for f in fs).most_common(1)[0][0],
            "values": {v: round(vw[v] / total, 4) for v in VALUE_NAMES if vw[v]},
            "norm_types": dict(Counter(f.norm_type_id for f in fs if f.norm_type_id)),
        }
    return out


# ---------------------------------------------------------------------------
# Assembly and persistence
# ---------------------------------------------------------------------------

def build_frame_kg(frames: list[CulturalFrame],
                   embed_fn: Callable[[list[str]], np.ndarray], *,
                   exclude_ids: Optional[Iterable[str]] = None,
                   threshold: float = NORM_TYPE_SIM_THRESHOLD) -> dict:
    """Frames (minus ``exclude_ids``) -> norm types -> culture profiles.

    ``exclude_ids`` builds a held-out KG: excluded records shape neither the
    norm-type clusters nor the culture profiles, unlike masking at query time.
    """
    excluded = set(exclude_ids or ())
    kept = [f.model_copy(deep=True) for f in frames if f.frame_id not in excluded]
    norm_types = cluster_norm_types(kept, embed_fn, threshold=threshold)
    profiles = culture_profiles(kept)
    meta = {
        "n_frames": len(kept), "n_excluded": len(excluded),
        "label_status": dict(Counter(f.label_status for f in kept)),
        "n_norm_types": len(norm_types),
        "n_norm_types_multi_culture": sum(1 for n in norm_types if len(n.cultures) >= 2),
        "n_cultures": len(profiles),
        "norm_type_threshold": threshold,
        "value_taxonomy": "Schwartz et al. 2012 (19 refined values)",
    }
    return {"frames": kept, "norm_types": norm_types, "profiles": profiles, "meta": meta}


def save_frame_kg(kg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as f:
        for fr in kg["frames"]:
            f.write(json.dumps(fr.model_dump(), ensure_ascii=False) + "\n")
    with (out_dir / "norm_types.jsonl").open("w", encoding="utf-8") as f:
        for nt in kg["norm_types"]:
            f.write(json.dumps(nt.model_dump(), ensure_ascii=False) + "\n")
    (out_dir / "culture_profiles.json").write_text(
        json.dumps(kg["profiles"], ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "kg_meta.json").write_text(json.dumps(kg["meta"], indent=2), encoding="utf-8")
    LOGGER.info("Saved frame KG (%d frames, %d norm types) -> %s",
                len(kg["frames"]), len(kg["norm_types"]), out_dir)
