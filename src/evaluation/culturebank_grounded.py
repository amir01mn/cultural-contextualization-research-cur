"""Eval v2, Evaluation 1: CultureBank grounded QA (docs/EVAL_V2_PLAN.md).

Stage functions ``run_<stage>(backend, ctx, data, out_dir, **cfg) -> dict``.
``data`` is the dict loaded by the driver: ``{"dev": [...], "test": [...],
"masks": {"near": [...], "far": [...]}}`` with items as written by
``scripts/prep_culturebank_eval_v2.py``. Evaluation modules never read the KG
themselves; the driver passes a ``RetrievalContext``.

Systems use short labels (``S0`` ... ``S4g``); gated ``*g`` labels are the same
system with ``RETRIEVAL_STRICT_CULTURE=True`` and ``RETRIEVAL_MIN_SCORE=tau``,
tau tuned on dev only and frozen in ``tau.json``.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import re
import time
import zlib
from pathlib import Path
from typing import Iterable, Optional

from ..core import config
from ..core.culture_normalizer import canonical_culture
from ..core.schemas import MCQQuestion, MCQVariant
from ..core.utils import get_logger, normalize_text
from ..llm import prompts
from . import mcq_systems as ms
from .mcq_evaluator import mcnemar, paired_bootstrap

LOGGER = get_logger("culturebank_grounded")

SYSTEM_IDS = {
    "S0": ms.SYSTEM_S0,
    "S1": ms.SYSTEM_S1,
    "S2": ms.SYSTEM_S2,
    "S3": ms.SYSTEM_S3,
    "S4": ms.SYSTEM_S4,
    "S4_triple_retrieval": ms.SYSTEM_S4_TRIPLE,
    "S4_no_shared_pivot": ms.SYSTEM_S4_NO_SHARED,
    "S1g": ms.SYSTEM_S1,
    "S3g": ms.SYSTEM_S3,
    "S4g": ms.SYSTEM_S4,
    # KG v3 (docs/EVAL_V3_PLAN.md)
    "S4L": ms.SYSTEM_S4_LOCAL,
    "S4F": ms.SYSTEM_S4F,
    "S4R": ms.SYSTEM_S4R,
    "S4R_noabs": ms.SYSTEM_S4R_NOABS,
}
GATED = ("S1g", "S3g", "S4g")
# Routed frame systems: "tau" is (tau_ref, tau_abs), tuned on dev like GATED.
FRAME_TUNED = ("S4R", "S4R_noabs")
TUNED = GATED + FRAME_TUNED
FRAME_GRID = tuple((r, a) for r in (0.3, 0.4, 0.5, 0.6) for a in (0.3, 0.4, 0.5))
FRAME_GRID_NOABS = tuple((r, 1.01) for r in (0.3, 0.4, 0.5, 0.6))
V3_MCQ_SYSTEMS = ["S0", "S1", "S1g", "S2", "S3", "S3g", "S4_triple_retrieval", "S4",
                  "S4L", "S4F", "S4R", "S4R_noabs"]
V3_OPEN_SYSTEMS = ["S0", "S1", "S3g", "S4", "S4L", "S4R"]
# Pre-registered v3 primary family (docs/EVAL_V3_PLAN.md), Holm over these only.
V3_PRIMARY = [("S4R", "S0"), ("S4R", "S1"), ("S4R", "S3g"), ("S4R", "S4"),
              ("S4R", "S4R_noabs"), ("S4F", "S4"), ("S4L", "S4")]
MCQ_SYSTEMS = ["S0", "S1", "S2", "S3", "S4", "S4_triple_retrieval",
               "S4_no_shared_pivot", "S1g", "S3g", "S4g"]
OPEN_SYSTEMS = ["S0", "S1", "S3", "S4", "S4g"]
RETRIEVAL_SYSTEMS = [s for s in MCQ_SYSTEMS if s != "S0"]
TAU_GRID = (0.15, 0.25, 0.35, 0.45)
SEEDS = (42, 43, 44)
LETTERS = ("A", "B", "C", "D")

PARAPHRASE_MAX_TOKENS = 400
MCQ_MAX_TOKENS = 512          # = LLM_MAX_TOKENS used by the CulturalBench runs
OPEN_MAX_TOKENS = 300


# ===========================================================================
# Helpers
# ===========================================================================

def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@contextlib.contextmanager
def gated_config(label: str, tau: Optional[float]):
    """Apply the relevance gate for ``*g`` labels; always restore config."""
    old = (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE,
           config.FRAME_TAU_REF, config.FRAME_TAU_ABS)
    try:
        if label in GATED:
            if tau is None:
                raise ValueError(f"{label} needs a tau")
            config.RETRIEVAL_STRICT_CULTURE = True
            config.RETRIEVAL_MIN_SCORE = float(tau)
        elif label in FRAME_TUNED:
            if tau is None:
                raise ValueError(f"{label} needs (tau_ref, tau_abs)")
            config.FRAME_TAU_REF, config.FRAME_TAU_ABS = (float(x) for x in tau)
        yield
    finally:
        (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE,
         config.FRAME_TAU_REF, config.FRAME_TAU_ABS) = old


@contextlib.contextmanager
def masked(ctx, instance_ids):
    """Temporarily mask evidence derived from ``instance_ids``."""
    if ctx is None:
        yield
        return
    ctx.set_mask(instance_ids)
    try:
        yield
    finally:
        ctx.set_mask(None)


def _norm(text: str) -> str:
    return normalize_text(text).lower()


def leak_count(evidence_blocks: list[str], items: list[dict]) -> int:
    """Evidence strings containing their own question's eval_question or
    eval_whole_desc text (normalized, case-insensitive). Must be 0."""
    n = 0
    for block, it in zip(evidence_blocks, items):
        if not block:
            continue
        ev = _norm(block)
        for key in ("eval_question", "eval_whole_desc"):
            needle = _norm(it.get(key) or "")
            if needle and needle in ev:
                n += 1
                break
    return n


# ===========================================================================
# MCQ construction
# ===========================================================================

def item_stem(it: dict) -> str:
    return prompts.CB_MCQ_STEM.format(
        cultural_group=it["cultural_group"],
        context=it.get("context") or "(not specified)",
        eval_question=it["eval_question"],
    )


def item_options(it: dict, paraphrases: Optional[dict] = None) -> list[str]:
    """[gold, d1, d2, d3], paraphrased when available (original otherwise)."""
    orig = [it["gold"], *it["distractors"]]
    para = (paraphrases or {}).get(it["qid"])
    if not para or len(para) != len(orig):
        return orig
    return [p if (p and p.strip()) else o for p, o in zip(para, orig)]


def build_mcq(it: dict, seed: int, paraphrases: Optional[dict] = None) -> MCQQuestion:
    opts = item_options(it, paraphrases)
    order = list(range(len(opts)))
    random.Random(seed * 1_000_003 + zlib.crc32(it["qid"].encode())).shuffle(order)
    options = {LETTERS[k]: opts[j] for k, j in enumerate(order)}
    answer_key = LETTERS[order.index(0)]
    return MCQQuestion(
        question_id=f"{it['qid']}__s{seed}", variant=MCQVariant.EASY, group_id=it["qid"],
        question=item_stem(it), cultural_group=it["cultural_group"],
        options=options, answer_key=answer_key,
    )


def open_question(it: dict) -> MCQQuestion:
    """Carrier for retrieval only (query = persona question + cultural group)."""
    return MCQQuestion(question_id=it["qid"], variant=MCQVariant.EASY, group_id=it["qid"],
                       question=it["eval_question"], cultural_group=it["cultural_group"])


# ===========================================================================
# Evidence (cached per label / tau / mask)
# ===========================================================================

def compute_evidence(label: str, questions: list[MCQQuestion], ctx, top_k: int, *,
                     tau: Optional[float] = None, cache: Optional[dict] = None,
                     cache_tag: str = "") -> list[tuple[str, list[str], dict]]:
    system = SYSTEM_IDS[label]
    out: list = []
    with gated_config(label, tau):
        for q in questions:
            key = (label, tuple(tau) if isinstance(tau, (list, tuple)) else
                   (tau if label in TUNED else None), cache_tag, q.group_id, q.question)
            if cache is not None and key in cache:
                out.append(cache[key])
                continue
            ev = ms.evidence_for(system, ctx, q, top_k)
            if cache is not None:
                cache[key] = ev
            out.append(ev)
    return out


def _mask_tag(ctx) -> str:
    ids = getattr(ctx, "masked_instance_ids", None)
    fr = getattr(ctx, "frame_retriever", None)
    kg = f"_kg{fr.kg_dir.name}" if fr is not None else ""
    return "" if not ids else f"mask{len(ids)}_{zlib.crc32(','.join(sorted(ids)).encode())}{kg}"


def _tau_value(label: str, v):
    """tau.json value -> the form gated_config expects."""
    return tuple(v) if label in FRAME_TUNED and v is not None else v


def answer_mcq(label: str, items: list[dict], seed: int, backend, ctx, top_k: int, *,
               paraphrases: Optional[dict], tau: Optional[float] = None,
               cache: Optional[dict] = None, leak_log: Optional[dict] = None,
               leak_key: Optional[str] = None) -> list[dict]:
    qs = [build_mcq(it, seed, paraphrases) for it in items]
    ev = compute_evidence(label, qs, ctx, top_k, tau=tau, cache=cache, cache_tag=_mask_tag(ctx))
    if leak_log is not None:
        key = leak_key or label
        leak_log[key] = leak_log.get(key, 0) + leak_count([e[0] for e in ev], items)
    answers = ms.run_system(SYSTEM_IDS[label], qs, backend, ctx, top_k=top_k,
                            evidence=ev, max_tokens=MCQ_MAX_TOKENS)
    rows = []
    for it, q, a, e in zip(items, qs, answers, ev):
        rows.append({
            "qid": it["qid"], "system": label, "seed": seed, "tau": tau,
            "answer_key": q.answer_key, "predicted": a.predicted, "correct": int(a.correct),
            "n_evidence": len(e[1]), "retrieved_ids": e[1], "route": e[2].get("route"),
            "raw_output": a.raw_output[-300:],
        })
    return rows


def _acc(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


# ===========================================================================
# Stage: paraphrase
# ===========================================================================

_PARA_LINE = re.compile(r"^\s*\**\(?([1-9])\)?\s*[:.)\-]\s*\**\s*(.+?)\s*$")


def parse_paraphrases(text: str, n: int) -> list[str]:
    """Parse ``k: text`` lines; missing entries are ''."""
    out = [""] * n
    for line in (text or "").splitlines():
        m = _PARA_LINE.match(line)
        if m:
            k = int(m.group(1)) - 1
            if 0 <= k < n and not out[k]:
                out[k] = m.group(2).strip().strip('"').strip()
    return out


def run_paraphrase(backend, ctx, data: dict, out_dir: Path, *,
                   cache_path: Path, **cfg) -> dict:
    """Paraphrase all four options of every dev+test item with one prompt each."""
    items = data["dev"] + data["test"]
    cached = {r["qid"]: r for r in _read_jsonl(cache_path)}
    todo = [it for it in items if it["qid"] not in cached]
    LOGGER.info("paraphrase: %d cached, %d to generate", len(cached), len(todo))
    if todo:
        user_prompts, orders = [], []
        for it in todo:
            opts = [it["gold"], *it["distractors"]]
            order = list(range(len(opts)))
            # Shuffle within the prompt so the gold is never systematically first.
            random.Random(zlib.crc32(("para" + it["qid"]).encode())).shuffle(order)
            numbered = "\n".join(f"{k + 1}: {opts[j]}" for k, j in enumerate(order))
            user_prompts.append(prompts.CB_PARAPHRASE_USER.format(numbered=numbered, n=len(opts)))
            orders.append(order)
        outs = backend.generate(user_prompts, system_prompt=prompts.CB_PARAPHRASE_SYSTEM,
                                max_tokens=PARAPHRASE_MAX_TOKENS)
        for it, order, out in zip(todo, orders, outs):
            opts = [it["gold"], *it["distractors"]]
            parsed = parse_paraphrases(out, len(opts))
            para = [""] * len(opts)
            for k, j in enumerate(order):
                para[j] = parsed[k]
            fallback = [i for i, p in enumerate(para) if not p]
            para = [p or o for p, o in zip(para, opts)]
            cached[it["qid"]] = {"qid": it["qid"], "options": para,
                                 "fallback_indices": fallback, "raw": out[:1500]}
        _write_jsonl(cache_path, [cached[it["qid"]] for it in items if it["qid"] in cached])
    n_fb_items = sum(1 for it in items if cached[it["qid"]]["fallback_indices"])
    n_fb_opts = sum(len(cached[it["qid"]]["fallback_indices"]) for it in items)
    res = {"items": len(items), "generated": len(todo), "items_with_fallback": n_fb_items,
           "options_fallback": n_fb_opts, "cache": str(cache_path)}
    _write_json(out_dir / "paraphrase_summary.json", res)
    return res


def load_paraphrases(cache_path: Path) -> dict:
    return {r["qid"]: r["options"] for r in _read_jsonl(cache_path)}


# ===========================================================================
# tau tuning (dev only)
# ===========================================================================

def tune_tau(label: str, backend, ctx, data: dict, out_dir: Path, *, top_k: int,
             paraphrases: dict, cache: Optional[dict] = None,
             leak_log: Optional[dict] = None, tau_grid=TAU_GRID) -> float:
    """Pick tau for a gated system on dev (seed 42, best accuracy; ties -> lower tau).
    Frozen in ``out_dir/tau.json`` and reused thereafter."""
    tau_path = out_dir / "tau.json"
    taus = json.loads(tau_path.read_text()) if tau_path.exists() else {}
    if label in taus:
        return _tau_value(label, taus[label]["tau"]) if label in FRAME_TUNED \
            else float(taus[label]["tau"])
    if label in FRAME_TUNED:
        tau_grid = FRAME_GRID if label == "S4R" else FRAME_GRID_NOABS
    dev_rows, grid = [], {}
    for tau in tau_grid:
        rows = answer_mcq(label, data["dev"], 42, backend, ctx, top_k, paraphrases=paraphrases,
                          tau=tau, cache=cache, leak_log=leak_log,
                          leak_key=f"dev_tau:{label}")
        grid[str(tau)] = round(_acc(rows), 4)
        dev_rows.extend(rows)
    # Ties -> the stricter (higher) refinement gate for frame systems, the lower
    # tau for the v2 gated systems (unchanged).
    if label in FRAME_TUNED:
        best = max(tau_grid, key=lambda t: (grid[str(t)], t[0], t[1]))
    else:
        best = max(tau_grid, key=lambda t: (grid[str(t)], -t))
    taus[label] = {"tau": best, "dev_accuracy_by_tau": grid, "n_dev": len(data["dev"])}
    _write_json(tau_path, taus)
    _write_jsonl(out_dir / f"dev_tau_{label}.jsonl", dev_rows)
    LOGGER.info("tau(%s) = %s  dev grid=%s", label, best, grid)
    return tuple(best) if label in FRAME_TUNED else float(best)


# ===========================================================================
# Stage: open_gen (runs early: a judge job polls for open_answers.DONE)
# ===========================================================================

def run_open_gen(backend, ctx, data: dict, out_dir: Path, *, top_k: int = 8,
                 paraphrases: Optional[dict] = None, cache: Optional[dict] = None,
                 leak_log: Optional[dict] = None, systems=OPEN_SYSTEMS, **cfg) -> dict:
    items = [it for it in data["test"] if it.get("open_ended")]
    rows, per_sys = [], {}
    for label in systems:
        tau = None
        if label in TUNED:
            tau = tune_tau(label, backend, ctx, data, out_dir, top_k=top_k,
                           paraphrases=paraphrases or {}, cache=cache, leak_log=leak_log)
        qs = [open_question(it) for it in items]
        ev = compute_evidence(label, qs, ctx, top_k, tau=tau, cache=cache)
        leaks = leak_count([e[0] for e in ev], items)
        if leak_log is not None:
            leak_log[f"open:{label}"] = leaks
        user = [prompts.CB_OPEN_USER.format(evidence_block=e[0], question=it["eval_question"])
                for e, it in zip(ev, items)]
        outs = backend.generate(user, system_prompt=prompts.CB_OPEN_SYSTEM,
                                max_tokens=OPEN_MAX_TOKENS)
        for it, out in zip(items, outs):
            rows.append({"qid": it["qid"], "system": label, "cultural_group": it["cultural_group"],
                         "question": it["eval_question"], "answer": (out or "").strip(),
                         "gold_desc": it["eval_whole_desc"]})
        per_sys[label] = {"n": len(items), "tau": tau, "leaks": leaks,
                          "empty_answers": sum(1 for o in outs if not (o or "").strip())}
    path = out_dir / "open_answers.jsonl"
    _write_jsonl(path, rows)
    (out_dir / "open_answers.DONE").write_text("")
    res = {"n_questions": len(items), "n_rows": len(rows), "systems": per_sys, "path": str(path)}
    _write_json(out_dir / "open_gen_summary.json", res)
    return res


# ===========================================================================
# Stage: mcq (dev tau tuning + test, 3 option-order seeds)
# ===========================================================================

def run_mcq(backend, ctx, data: dict, out_dir: Path, *, top_k: int = 8,
            paraphrases: Optional[dict] = None, cache: Optional[dict] = None,
            leak_log: Optional[dict] = None, systems=MCQ_SYSTEMS, seeds=SEEDS, **cfg) -> dict:
    paraphrases = paraphrases or {}
    taus = {}
    for label in systems:
        if label in TUNED:
            taus[label] = tune_tau(label, backend, ctx, data, out_dir, top_k=top_k,
                                   paraphrases=paraphrases, cache=cache, leak_log=leak_log)
    test_leaks: dict = {}
    all_rows = []
    for label in systems:
        for seed in seeds:
            t0 = time.time()
            rows = answer_mcq(label, data["test"], seed, backend, ctx, top_k,
                              paraphrases=paraphrases, tau=taus.get(label), cache=cache,
                              leak_log=test_leaks if seed == seeds[0] else None)
            all_rows.extend(rows)
            LOGGER.info("mcq %s seed=%d acc=%.3f (%.0fs)", label, seed, _acc(rows),
                        time.time() - t0)
    if leak_log is not None:
        for k, v in test_leaks.items():
            leak_log[f"mcq_test:{k}"] = v
    _write_jsonl(out_dir / "mcq_test_answers.jsonl", all_rows)
    acc = _mcq_accuracy(all_rows, systems, seeds)
    res = {"n_test": len(data["test"]), "seeds": list(seeds), "tau": taus, "accuracy": acc,
           "leaks_test": test_leaks}
    _write_json(out_dir / "mcq_summary.json", res)
    return res


def _mcq_accuracy(rows: list[dict], systems, seeds) -> dict:
    out = {}
    for label in systems:
        per_seed = {}
        for seed in seeds:
            r = [x for x in rows if x["system"] == label and x["seed"] == seed]
            if r:
                per_seed[str(seed)] = round(_acc(r), 4)
        vals = list(per_seed.values())
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        out[label] = {"mean": round(mean, 4), "sd": round(sd, 4), "per_seed": per_seed}
    return out


# ===========================================================================
# Stage: heldout (near / far masks, seed 42)
# ===========================================================================

def run_heldout(backend, ctx, data: dict, out_dir: Path, *, top_k: int = 8,
                paraphrases: Optional[dict] = None, cache: Optional[dict] = None,
                leak_log: Optional[dict] = None, systems=MCQ_SYSTEMS, **cfg) -> dict:
    items = [it for it in data["test"] if it.get("heldout")]
    taus = {}
    for label in systems:
        if label in TUNED:
            taus[label] = tune_tau(label, backend, ctx, data, out_dir, top_k=top_k,
                                   paraphrases=paraphrases or {}, cache=cache, leak_log=leak_log)
    lookup = build_lookup(ctx)
    table, all_rows, gold_hits = {}, [], {}
    s0_rows = None
    full_frames = getattr(ctx, "frame_retriever", None)
    heldout_frames = getattr(ctx, "heldout_frame_retrievers", None) or {}
    for mask_name in ("near", "far"):
        ids = data["masks"][mask_name]
        # Frame systems use a KG *built* without the masked records (norm types
        # and culture profiles never saw them); other systems mask at query time.
        if mask_name in heldout_frames:
            ctx.frame_retriever = heldout_frames[mask_name]
        try:
            with masked(ctx, ids):
                for label in systems:
                    if label == "S0" and s0_rows is not None:
                        rows = [dict(r, mask=mask_name) for r in s0_rows]
                    else:
                        rows = answer_mcq(label, items, 42, backend, ctx, top_k,
                                          paraphrases=paraphrases, tau=taus.get(label), cache=cache,
                                          leak_log=leak_log, leak_key=f"heldout_{mask_name}:{label}")
                        for r in rows:
                            r["mask"] = mask_name
                        if label == "S0":
                            s0_rows = rows
                    # Sanity: no masked instance may appear in evidence.
                    if label != "S0":
                        masked_ids = set(ids)
                        hits = 0
                        for r in rows:
                            for rid in r["retrieved_ids"]:
                                if lookup["instances"](label, rid) & masked_ids:
                                    hits += 1
                                    break
                        gold_hits[f"{mask_name}:{label}"] = hits
                    table.setdefault(label, {})[mask_name] = round(_acc(rows), 4)
                    all_rows.extend(rows)
        finally:
            ctx.frame_retriever = full_frames
    bad = {k: v for k, v in gold_hits.items() if v}
    if bad:
        LOGGER.error("HELD-OUT MASK VIOLATION (masked evidence retrieved): %s", bad)
    _write_jsonl(out_dir / "heldout_answers.jsonl", all_rows)
    res = {"n": len(items), "mask_sizes": {k: len(v) for k, v in data["masks"].items()},
           "accuracy": table, "tau": taus, "mask_violations": bad}
    _write_json(out_dir / "heldout_summary.json", res)
    return res


# ===========================================================================
# Stage: retrieval_metrics (no LLM)
# ===========================================================================

def build_lookup(ctx) -> dict:
    """Map retrieved ids (passage / triple / pathway) to instance_ids and culture."""
    if ctx is None:
        return {"instances": lambda _l, _i: set(), "culture": lambda _l, _i: ""}
    cached = getattr(ctx, "_cb_lookup", None)
    if cached is not None:
        return cached
    passage_inst, passage_cult = {}, {}
    if ctx.passage_retriever:
        for d in ctx.passage_retriever.documents:
            pid = str(d.get("passage_id", ""))
            passage_inst[pid] = str(d.get("instance_id") or pid)
            passage_cult[pid] = d.get("cultural_group") or ""
    t2i = ctx.triple_to_instance()
    triple_cult = {}
    for r in (ctx.triple_retriever, ctx.refined_triple_retriever):
        if r is not None:
            for t in r.triples:
                raw = t.raw_fields or {}
                triple_cult.setdefault(t.triple_id, raw.get("cultural_group") or raw.get("cultural group") or "")
    pw_inst, pw_cult = {}, {}
    if ctx.pathway_retriever:
        for pw in ctx.pathway_retriever.pathways:
            pw_inst[pw.pathway_id] = {t2i[t] for t in pw.triple_ids if t in t2i}
            pw_cult[pw.pathway_id] = pw.culture

    lpw_inst, lpw_cult = {}, {}
    if getattr(ctx, "local_pathway_retriever", None):
        for pw in ctx.local_pathway_retriever.pathways:
            lpw_inst[pw.pathway_id] = set(pw.instance_ids)
            lpw_cult[pw.pathway_id] = pw.culture
    frame_cult = {}
    if getattr(ctx, "frame_retriever", None):
        frame_cult = {f.frame_id: f.culture_label for f in ctx.frame_retriever.frames}

    def kind(label: str) -> str:
        sid = SYSTEM_IDS[label]
        if sid == ms.SYSTEM_S1:
            return "passage"
        if sid in (ms.SYSTEM_S4, ms.SYSTEM_S4_NO_SHARED):
            return "pathway"
        if sid == ms.SYSTEM_S4_LOCAL:
            return "local_pathway"
        if sid in (ms.SYSTEM_S4F, ms.SYSTEM_S4R, ms.SYSTEM_S4R_NOABS):
            return "frame"
        return "triple"

    def instances(label: str, rid: str) -> set:
        k = kind(label)
        if k == "passage":
            return {passage_inst.get(rid, rid)}
        if k == "pathway":
            return pw_inst.get(rid, set())
        if k == "local_pathway":
            return lpw_inst.get(rid, set())
        if k == "frame":
            return {rid}
        return {t2i[rid]} if rid in t2i else set()

    def culture(label: str, rid: str) -> str:
        k = kind(label)
        if k == "passage":
            return passage_cult.get(rid, "")
        if k == "pathway":
            return pw_cult.get(rid, "")
        if k == "local_pathway":
            return lpw_cult.get(rid, "")
        if k == "frame":
            return frame_cult.get(rid, "")
        return triple_cult.get(rid, "")

    lk = {"instances": instances, "culture": culture}
    try:
        ctx._cb_lookup = lk
    except Exception:  # noqa: BLE001
        pass
    return lk


def retrieval_scores(label: str, items: list[dict], evidence: list, lookup: dict,
                     ks=(1, 3, 8)) -> dict:
    n = len(items)
    recall = {k: 0 for k in ks}
    rr, prec, prec_n, chars, empty = 0.0, 0.0, 0, 0, 0
    for it, (block, ids, _meta) in zip(items, evidence):
        gold = it["instance_id"]
        rank = None
        for i, rid in enumerate(ids, 1):
            if gold in lookup["instances"](label, rid):
                rank = i
                break
        for k in ks:
            if rank is not None and rank <= k:
                recall[k] += 1
        rr += 1.0 / rank if rank else 0.0
        if ids:
            qc = canonical_culture(it["cultural_group"]) or _norm(it["cultural_group"])
            same = sum(1 for rid in ids
                       if (canonical_culture(lookup["culture"](label, rid))
                           or _norm(lookup["culture"](label, rid))) == qc)
            prec += same / len(ids)
            prec_n += 1
        else:
            empty += 1
        chars += len(block or "")
    return {
        "n": n,
        **{f"recall@{k}": round(recall[k] / n, 4) if n else 0.0 for k in ks},
        "mrr": round(rr / n, 4) if n else 0.0,
        "same_culture_precision": round(prec / prec_n, 4) if prec_n else 0.0,
        "no_evidence_rate": round(empty / n, 4) if n else 0.0,
        "evidence_tokens_per_q_approx": round(chars / 4 / n, 1) if n else 0.0,
    }


def run_retrieval_metrics(backend, ctx, data: dict, out_dir: Path, *, top_k: int = 8,
                          cache: Optional[dict] = None, leak_log: Optional[dict] = None,
                          systems=RETRIEVAL_SYSTEMS, **cfg) -> dict:
    tau_path = out_dir / "tau.json"
    taus = json.loads(tau_path.read_text()) if tau_path.exists() else {}
    lookup = build_lookup(ctx)
    table: dict = {}
    for split in ("dev", "test"):
        items = data[split]
        qs = [build_mcq(it, 42) for it in items]
        for label in systems:
            tau = _tau_value(label, taus.get(label, {}).get("tau")) if label in TUNED else None
            if label in TUNED and tau is None:
                LOGGER.warning("retrieval_metrics: no frozen tau for %s; skipped", label)
                continue
            ev = compute_evidence(label, qs, ctx, top_k, tau=tau, cache=cache)
            sc = retrieval_scores(label, items, ev, lookup)
            sc["leaks"] = leak_count([e[0] for e in ev], items)
            routes = [e[2].get("route") for e in ev if e[2].get("route")]
            if routes:
                sc["routes"] = {m: round(routes.count(m) / len(ev), 4)
                                for m in ("refine", "abstract", "none")}
            sc["tau"] = tau
            if leak_log is not None:
                leak_log[f"retrieval_{split}:{label}"] = sc["leaks"]
            table.setdefault(split, {})[label] = sc
    _write_json(out_dir / "retrieval_metrics.json", table)
    return table


# ===========================================================================
# Summary
# ===========================================================================

def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = round(running, 6)
    return out


def paired_delta_bootstrap(a: list[float], b: list[float], n_boot: int = 10000,
                           seed: int = 42) -> dict:
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    d = (a[idx] - b[idx]).mean(axis=1)
    delta = float((a - b).mean())
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": round(delta, 4), "ci_low": round(float(np.percentile(d, 2.5)), 4),
            "ci_high": round(float(np.percentile(d, 97.5)), 4), "p_value": round(min(1.0, p), 6)}


def comparisons(rows: list[dict], systems, seeds, n_boot: int = 10000,
                pairs: Optional[list] = None) -> dict:
    """McNemar (seed 42 + per seed) and paired bootstrap on the per-item
    seed-averaged correctness; Holm-Bonferroni over the primary family
    (each system vs S0, S4 vs S1, S4g vs S4) using the seed-42 McNemar p."""
    by = {}
    for r in rows:
        by.setdefault((r["system"], r["seed"]), {})[r["qid"]] = r["correct"]
    qids = sorted(by.get(("S0", seeds[0]), {}))
    if pairs is None:
        pairs = [(s, "S0") for s in systems if s != "S0"] + [("S4", "S1"), ("S4g", "S4")]
    out = {}
    for a, b in pairs:
        if (a, seeds[0]) not in by or (b, seeds[0]) not in by:
            continue
        mc = {str(s): mcnemar([by[(a, s)][q] for q in qids], [by[(b, s)][q] for q in qids])
              for s in seeds if (a, s) in by and (b, s) in by}
        avg = lambda sys_: [sum(by[(sys_, s)][q] for s in seeds if (sys_, s) in by)
                            / sum(1 for s in seeds if (sys_, s) in by) for q in qids]
        out[f"{a}__vs__{b}"] = {"mcnemar": mc, "bootstrap": paired_delta_bootstrap(
            avg(a), avg(b), n_boot=n_boot)}
    p = {k: v["mcnemar"][str(seeds[0])]["p_value"] for k, v in out.items()}
    adj = holm(p)
    for k in out:
        out[k]["p_holm"] = adj[k]
    return out


def write_summary(out_dir: Path, *, run_meta: dict, n_boot: int = 10000) -> dict:
    """Assemble summary.json / summary.md from the stage outputs present."""
    def load(name):
        p = out_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    mcq = load("mcq_summary.json")
    held = load("heldout_summary.json")
    retr = load("retrieval_metrics.json")
    taus = load("tau.json")
    opn = load("open_gen_summary.json")
    para = load("paraphrase_summary.json")
    leaks = load("leakage_guard.json")
    comps, comps_v3, ci = None, None, None
    if mcq:
        rows = _read_jsonl(out_dir / "mcq_test_answers.jsonl")
        systems = [s for s in dict.fromkeys(MCQ_SYSTEMS + V3_MCQ_SYSTEMS) if s in mcq["accuracy"]]
        seeds = mcq["seeds"]
        comps = comparisons(rows, systems, seeds, n_boot=n_boot)
        if any(s in mcq["accuracy"] for s in ("S4L", "S4F", "S4R")):
            comps_v3 = comparisons(rows, systems, seeds, n_boot=n_boot, pairs=[
                (a, b) for a, b in V3_PRIMARY if a in systems and b in systems])
        s42 = {}
        for r in rows:
            if r["seed"] == seeds[0]:
                s42.setdefault(r["system"], []).append(r["correct"])
        ci = paired_bootstrap(s42, n_boot=min(n_boot, 2000)) if s42 else None
    summary = {"run_meta": run_meta, "paraphrase": para, "tau": taus, "mcq": mcq,
               "mcq_ci_seed42": ci, "comparisons": comps,
               "comparisons_v3_primary": comps_v3, "heldout": held,
               "retrieval": retr, "open_gen": opn, "leakage_guard": leaks}
    _write_json(out_dir / "summary.json", summary)

    L = [f"# {run_meta.get('title', 'Eval v2 - Evaluation 1: CultureBank grounded QA')}", "",
         f"Model: {run_meta.get('model')}  provider: {run_meta.get('provider')}  "
         f"top_k: {run_meta.get('top_k')}  smoke: {run_meta.get('smoke')}", ""]
    if leaks:
        L += [f"**Leakage guard:** total leaks = {leaks.get('total')} "
              f"({'PASS' if leaks.get('total') == 0 else 'FAIL'})", ""]
    if taus:
        L += ["## Frozen tau (dev)", "", "| system | tau | dev acc by tau |", "|---|---|---|"]
        for k, v in taus.items():
            L.append(f"| {k} | {v['tau']} | {v['dev_accuracy_by_tau']} |")
        L.append("")
    if mcq:
        L += [f"## MCQ accuracy (test n={mcq['n_test']}, seeds {mcq['seeds']})", "",
              "| system | mean | sd | 95% CI (seed 42) |", "|---|---|---|---|"]
        for s, v in mcq["accuracy"].items():
            c = (ci or {}).get(s, {})
            L.append(f"| {s} | {v['mean']*100:.1f}% | {v['sd']*100:.1f} | "
                     f"[{c.get('ci_low', 0)*100:.1f}, {c.get('ci_high', 0)*100:.1f}] |")
        L += ["", "## Comparisons (McNemar seed 42, Holm over family; bootstrap on seed-mean)", "",
              "| comparison | delta | boot 95% CI | McNemar p | Holm p |", "|---|---|---|---|---|"]
        for k, v in (comps or {}).items():
            b = v["bootstrap"]
            p42 = next(iter(v["mcnemar"].values()))["p_value"]
            L.append(f"| {k.replace('__vs__', ' vs ')} | {b['delta']*100:+.1f} | "
                     f"[{b['ci_low']*100:+.1f}, {b['ci_high']*100:+.1f}] | {p42} | {v['p_holm']} |")
        L.append("")
        if comps_v3:
            L += ["## KG v3 primary family (pre-registered; Holm over these 7 only)", "",
                  "| comparison | delta | boot 95% CI | McNemar p | Holm p |",
                  "|---|---|---|---|---|"]
            for k, v in comps_v3.items():
                b = v["bootstrap"]
                p42 = next(iter(v["mcnemar"].values()))["p_value"]
                L.append(f"| {k.replace('__vs__', ' vs ')} | {b['delta']*100:+.1f} | "
                         f"[{b['ci_low']*100:+.1f}, {b['ci_high']*100:+.1f}] | {p42} | "
                         f"{v['p_holm']} |")
            L.append("")
    if held:
        L += [f"## Held-out (n={held['n']}, seed 42; mask sizes {held['mask_sizes']})", "",
              "| system | near | far |", "|---|---|---|"]
        for s, v in held["accuracy"].items():
            L.append(f"| {s} | {v.get('near', 0)*100:.1f}% | {v.get('far', 0)*100:.1f}% |")
        if held.get("mask_violations"):
            L.append(f"\n**Mask violations:** {held['mask_violations']}")
        L.append("")
    if retr:
        for split, tab in retr.items():
            L += [f"## Retrieval ({split})", "",
                  "| system | R@1 | R@3 | R@8 | MRR | same-culture P@k | no-evidence | ~tokens/q | leaks |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for s, v in tab.items():
                L.append(f"| {s} | {v['recall@1']:.3f} | {v['recall@3']:.3f} | {v['recall@8']:.3f} | "
                         f"{v['mrr']:.3f} | {v['same_culture_precision']:.3f} | "
                         f"{v['no_evidence_rate']:.3f} | {v['evidence_tokens_per_q_approx']} | {v['leaks']} |")
            L.append("")
            routed = {s_: v["routes"] for s_, v in tab.items() if v.get("routes")}
            if routed:
                L += [f"Routes ({split}): " + "; ".join(
                    f"{s_} {r}" for s_, r in routed.items()), ""]
    if opn:
        L += [f"## Open-ended generation: {opn['n_rows']} answers "
              f"({opn['n_questions']} questions); judged by a separate job.", ""]
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    return summary
