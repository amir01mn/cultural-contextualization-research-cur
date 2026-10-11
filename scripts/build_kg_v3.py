"""Build KG v3 (frame KG + record-local pathways). See docs/EVAL_V3_PLAN.md.

Stages (run in order; each is resumable or cheap to redo):

    frames      CultureBank CSV rows -> frames (descriptor fields only)
    label       LLM names each frame's norm, norm type and values   [GPU]
    kg          norm-type clustering + culture profiles + embedding index for
                the full KG and two build-time held-out KGs (eval v2 masks)
    local       record-local pathways over the August refined triples (S4L)
    check       dev-only diagnostics: reachability, retrieval recall, routes

Writes only under data/kg_v3/ (never data/processed or data/outputs).

    python scripts/build_kg_v3.py --stages frames,label,kg,local,check --provider vllm
    python scripts/build_kg_v3.py --stages frames,kg,local,check --smoke   # CPU, fake labels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STAGES = ["frames", "label", "kg", "local", "check", "audit"]


class SmokeLabeler:
    """Deterministic stand-in for the LLM labeler (smoke mode only)."""

    model = "smoke-fake"

    def generate(self, prompts_, system_prompt=None, *, temperature=None, max_tokens=None):
        out = []
        for p in prompts_:
            topic = p.split("Topic: ", 1)[-1].split(".", 1)[0].lower() if "Topic: " in p else "custom"
            vals = ["tradition"] if len(p) % 2 else ["benevolence (caring)", "face"]
            out.append(json.dumps({"norm": "People follow this custom.",
                                   "norm_type": f"{topic} customs", "values": vals}))
        return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--provider", default="vllm", choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--masks", default=str(ROOT / "data" / "eval_v2" / "culturebank" / "masks.json"))
    p.add_argument("--dev", default=str(ROOT / "data" / "eval_v2" / "culturebank" / "dev.jsonl"))
    p.add_argument("--threshold", type=float, default=None, help="norm-type clustering cosine")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N rows (smoke)")
    args = p.parse_args(argv)
    if args.smoke:
        os.environ["EMBEDDING_BACKEND"] = "tfidf"

    from src.construction import frame_builder as fb
    from src.construction.record_pathways import build_record_pathways
    from src.core import config
    from src.core.embedding_utils import backend_name, embed_texts
    from src.core.schemas import CulturalFrame
    from src.core.utils import get_logger
    from src.loaders.culturebank_eval_loader import find_culturebank_csvs, rows_with_instance_ids
    from src.retrieval.frame_retriever import build_frame_index

    log = get_logger("build_kg_v3")
    if args.smoke:
        config.EMBEDDING_BACKEND = "tfidf"
    out = Path(args.out_dir) if args.out_dir else (
        config.KG_V3_DIR.parent / "kg_v3_smoke" if args.smoke else config.KG_V3_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in STAGES if s in {x.strip() for x in args.stages.split(",")}]
    threshold = args.threshold or fb.NORM_TYPE_SIM_THRESHOLD
    timings: dict = {}
    frames_path = out / "frames_unlabeled.jsonl"

    def load_frames() -> list:
        with frames_path.open(encoding="utf-8") as f:
            return [CulturalFrame(**json.loads(line)) for line in f if line.strip()]

    if "frames" in stages:
        t0 = time.time()
        hf_home = os.environ.get("HF_HOME") or str(ROOT.parent / "hf_cache")
        rows = rows_with_instance_ids(find_culturebank_csvs(hf_home))
        if args.limit:
            rows = rows[:args.limit]
        frames = fb.frames_from_rows(rows)
        with frames_path.open("w", encoding="utf-8") as f:
            for fr in frames:
                f.write(json.dumps(fr.model_dump(), ensure_ascii=False) + "\n")
        log.info("frames: %d of %d rows (%d cultures)", len(frames), len(rows),
                 len({f.culture for f in frames}))
        timings["frames"] = round(time.time() - t0, 1)

    labels_path = out / "frame_labels.jsonl"
    if "label" in stages:
        t0 = time.time()
        frames = load_frames()
        if args.smoke:
            backend = SmokeLabeler()
        else:
            from src.llm.llm_backend import get_backend

            backend = get_backend(args.provider, model=args.model)
            if hasattr(backend, "warmup"):
                backend.warmup()
        fb.label_frames(backend, frames, labels_path)
        timings["label"] = round(time.time() - t0, 1)

    if "kg" in stages:
        t0 = time.time()
        frames = load_frames()
        labels = {}
        if labels_path.exists():
            for line in labels_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    labels[r["frame_id"]] = r
        status = fb.apply_labels(frames, labels)
        log.info("label status: %s", status)
        masks = json.loads(Path(args.masks).read_text()) if Path(args.masks).exists() else {}
        variants = {"full": None, **{f"heldout_{k}": v for k, v in masks.items()}}

        def embed_fn(texts):
            return embed_texts(texts, fit=True)

        for name, excl in variants.items():
            kg = fb.build_frame_kg(frames, embed_fn, exclude_ids=excl, threshold=threshold)
            fb.save_frame_kg(kg, out / name)
            build_frame_index(out / name)
            log.info("kg %s: %s", name, kg["meta"])
        timings["kg"] = round(time.time() - t0, 1)

    if "local" in stages:
        t0 = time.time()
        from eval_culturalbench import _read_rows, _to_triples
        from src.retrieval.pathway_retriever import build_pathway_index

        refined = _to_triples(_read_rows(Path(config.CORPUS_TRIPLES_REFINED_PATH)))
        if args.limit:
            keep = {f.frame_id for f in load_frames()}
            refined = [t for t in refined if t.instance_id in keep]
        pathways, stats = build_record_pathways(refined)
        build_pathway_index(pathways, out / "local_pathways_index")
        (out / "local_pathways_stats.json").write_text(json.dumps(stats, indent=2))
        timings["local"] = round(time.time() - t0, 1)

    if "check" in stages:
        t0 = time.time()
        res = run_check(out, Path(args.dev), log)
        (out / "check_dev.json").write_text(json.dumps(res, indent=2))
        timings["check"] = round(time.time() - t0, 1)

    if "audit" in stages:
        write_audit_samples(out, log)

    meta_path = out / "build_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("runs", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": stages, "seconds": timings,
        "embedding_backend": backend_name(), "smoke": args.smoke, "threshold": threshold,
        "model": args.model or (None if args.smoke else config.VLLM_MODEL
                                if hasattr(config, "VLLM_MODEL") else None)})
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Done %s in %s -> %s", stages, timings, out)
    return 0


def _stratified(items: list, key, n: int, per: int, seed: int = 42) -> list:
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    cultures = sorted(c for c, v in by.items() if len(v) >= per)
    rng.shuffle(cultures)
    out = []
    for c in cultures:
        out += rng.sample(by[c], per)
        if len(out) >= n:
            break
    return out[:n]


def write_audit_samples(out: Path, log, n: int = 200, per: int = 4) -> None:
    """200 frame chains and 200 record-local pathways (4 per culture, seed 42) in
    the pathway-audit format of src/evaluation/intrinsic_eval.py."""
    from src.retrieval.frame_retriever import FrameRetriever
    from src.retrieval.pathway_retriever import PathwayRetriever

    fr = FrameRetriever(out / "full")
    frames = [f for f in fr.frames if f.label_status == "ok"]
    rows = []
    for f in _stratified(frames, lambda f: f.culture, n, per):
        nt = fr.nt_by_id.get(f.norm_type_id)
        rows.append({
            "pathway_id": f.frame_id, "culture": f.culture_label, "has_shared_pivot":
            bool(nt and len(nt.cultures) >= 2),
            "nodes": [{"label": ", ".join(f.values), "layer": "LAYER_3_DEEP_VALUE"},
                      {"label": nt.label if nt else f.norm_type_label, "layer": "LAYER_2_SOCIAL"},
                      {"label": f.norm, "layer": "LAYER_2_SOCIAL"},
                      {"label": f.text, "layer": "LAYER_1_SURFACE"}],
            "predicates": ["is expressed through the norm type", "takes this form here",
                           "is practised as"],
            "sources": [f"{f.culture_label}. {f.text}"]})
    with (out / "audit_frames.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lp = PathwayRetriever(out / "local_pathways_index")
    rows = []
    for pw in _stratified(lp.pathways, lambda p: p.culture.lower(), n, per):
        d = pw.model_dump(mode="json")
        d["sources"] = [pw.source_text or ""]
        rows.append(d)
    with (out / "audit_local_pathways.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("audit samples written to %s", out)


def run_check(out: Path, dev_path: Path, log) -> dict:
    """Dev-only: is the gold record reachable and retrieved? Which route fires?"""
    from src.core.culture_normalizer import canonical_culture
    from src.retrieval.frame_retriever import FrameRetriever
    from src.retrieval.pathway_retriever import PathwayRetriever

    res: dict = {}
    fr = FrameRetriever(out / "full")
    res["frames"] = len(fr.frames)
    res["label_status"] = dict(Counter(f.label_status for f in fr.frames))
    res["norm_types"] = len(fr.norm_types)
    res["norm_type_size_hist"] = dict(Counter(
        "1" if len(n.frame_ids) == 1 else "2-5" if len(n.frame_ids) <= 5 else
        "6-50" if len(n.frame_ids) <= 50 else ">50" for n in fr.norm_types))
    res["norm_types_multi_culture"] = sum(1 for n in fr.norm_types if len(n.cultures) >= 2)
    res["top_norm_types"] = [(n.label, len(n.frame_ids), len(n.cultures)) for n in fr.norm_types[:25]]
    res["value_counts"] = dict(Counter(v for f in fr.frames for v in f.values).most_common())
    lp_dir = out / "local_pathways_index"
    lp = PathwayRetriever(lp_dir) if lp_dir.exists() else None
    if not dev_path.exists():
        return res
    dev = [json.loads(line) for line in dev_path.open() if line.strip()]
    from src.llm import prompts

    def q_of(it):
        return prompts.CB_MCQ_STEM.format(cultural_group=it["cultural_group"],
                                          context=it.get("context") or "(not specified)",
                                          eval_question=it["eval_question"])

    frame_ids = {f.frame_id for f in fr.frames}
    res["dev_n"] = len(dev)
    res["dev_gold_reachable_frames"] = sum(it["instance_id"] in frame_ids for it in dev) / len(dev)
    if lp:
        lp_ids = {i for p in lp.pathways for i in p.instance_ids}
        res["dev_gold_reachable_local_pathways"] = sum(it["instance_id"] in lp_ids for it in dev) / len(dev)
    ranks_f, ranks_l, best_same, best_nt, gold_score = [], [], [], [], []
    for it in dev:
        q = q_of(it)
        hits = fr.retrieve(q, top_k=8, question_culture=it["cultural_group"])
        ids = [f.frame_id for f, _ in hits]
        ranks_f.append(ids.index(it["instance_id"]) + 1 if it["instance_id"] in ids else None)
        if lp:
            ph = lp.retrieve(q, top_k=8, question_culture=it["cultural_group"])
            pids = [i for pw, _ in ph for i in pw.instance_ids]
            ranks_l.append(pids.index(it["instance_id"]) + 1 if it["instance_id"] in pids else None)
        r = fr.route(q, it["cultural_group"], top_k=8, tau_ref=-1.0, tau_abs=-1.0)
        best_same.append(r["best_same_culture"])
        scores = fr._scores(q)
        gi = fr.frame_index.get(it["instance_id"])
        gold_score.append(float(scores[gi]) if gi is not None else None)
        qv = __import__("src.core.embedding_utils", fromlist=["x"]).embed_texts([q])[0]
        best_nt.append(float((fr.nt_emb @ qv).max()) if len(fr.norm_types) else None)

    def recall(ranks, k):
        return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 4) if ranks else None

    res["dev_recall_frames"] = {f"@{k}": recall(ranks_f, k) for k in (1, 3, 8)}
    if ranks_l:
        res["dev_recall_local_pathways"] = {f"@{k}": recall(ranks_l, k) for k in (1, 3, 8)}

    def quantiles(xs):
        import numpy as np
        xs = [x for x in xs if x is not None]
        return {q: round(float(np.quantile(xs, q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if xs else {}

    res["dev_best_same_culture_score"] = quantiles(best_same)
    res["dev_gold_frame_score"] = quantiles(gold_score)
    res["dev_best_norm_type_score"] = quantiles(best_nt)
    res["dev_cultures_in_kg"] = sum(
        1 for it in dev if (canonical_culture(it["cultural_group"]) or it["cultural_group"].lower())
        in fr.profiles) / len(dev)
    log.info("check: %s", {k: v for k, v in res.items() if k.startswith("dev_")})
    return res


if __name__ == "__main__":
    sys.exit(main())
