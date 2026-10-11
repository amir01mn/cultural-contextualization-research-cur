#!/usr/bin/env python
"""CPU prep for Evaluation 3 (intrinsic KG quality); see docs/EVAL_V2_PLAN.md.

Reads the August KG artifacts in data/processed/ (read-only) and writes to
data/eval_v2/intrinsic/:

  records.jsonl   300 non-fallback records, stratified by canonical culture
                  (equal allocation over the 50 largest cultures, seed 42), each
                  with eval_whole_desc, its clean raw triples, clean refined
                  triples and the pathways (as indexed for S4, generated ones
                  included but sorted last) containing any of its triples.
  triples.jsonl   500 refined triples stratified by culture, with source_text.
  pathways.jsonl  200 FULL non-generated pathways stratified by culture, with
                  the source texts of their triples.
  prep_meta.json  counts, culture spread, corpus-wide fallback/generated rates.

"Clean" = neither ``llm_failed`` nor ``generated``. The triple audit samples from
all refined triples (the S3/A3 retrieval corpus) and keeps the generated flag so
rates can be split by it.

    python scripts/prep_intrinsic_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import config  # noqa: E402
from src.core.culture_normalizer import canonical_culture  # noqa: E402
from src.core.utils import get_logger, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.evaluation.intrinsic_eval import stratified_sample  # noqa: E402

LOGGER = get_logger("prep_intrinsic_v2")

TRIPLE_COLS = ["triple_id", "subject", "predicate", "object", "subject_layer",
               "predicate_layer", "object_layer", "confidence", "source_text",
               "instance_id", "generated", "llm_failed", "extraction_method"]
KEEP_IN_RECORD = ["triple_id", "subject", "predicate", "object",
                  "subject_layer", "object_layer"]


def read_triples(path: Path, with_raw_fields: bool) -> list[dict]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = TRIPLE_COLS + (["raw_fields"] if with_raw_fields else [])
    table = pq.read_table(path, columns=cols)
    if with_raw_fields:
        rf = table.column("raw_fields")
        extra = {name: pc.struct_field(rf, name).to_pylist()
                 for name in ("eval_whole_desc", "cultural_group", "topic")}
        table = table.drop(["raw_fields"])
    rows = table.to_pylist()
    if with_raw_fields:
        for i, r in enumerate(rows):
            for name, vals in extra.items():
                r[name] = vals[i]
    LOGGER.info("Read %d rows from %s", len(rows), path.name)
    return rows


def _clean(t: dict) -> bool:
    return not t.get("generated") and not t.get("llm_failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_records", type=int, default=300)
    ap.add_argument("--n_triples", type=int, default=500)
    ap.add_argument("--n_pathways", type=int, default=200)
    ap.add_argument("--max_strata", type=int, default=50)
    ap.add_argument("--max_pathways_per_record", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "data" / "eval_v2" / "intrinsic")
    args = ap.parse_args()
    t0 = time.time()
    proc = config.PROCESSED_DIR

    passages = {p["instance_id"]: p for p in read_jsonl(proc / "corpus_passages.jsonl")}
    raw = read_triples(proc / "corpus_triples.parquet", with_raw_fields=True)
    refined = read_triples(proc / "corpus_triples_refined.parquet", with_raw_fields=False)
    pathways = list(read_jsonl(proc / "corpus_pathways.jsonl"))
    LOGGER.info("Loaded %d passages, %d pathways (%.0fs)", len(passages), len(pathways),
                time.time() - t0)

    def culture_of(iid: str, fallback: str = "") -> tuple[str, str]:
        group = (passages.get(iid) or {}).get("cultural_group") or fallback or ""
        return group, canonical_culture(group)

    # ---- corpus-wide fallback / generated rates -------------------------------
    raw_by_rec: dict[str, list[dict]] = defaultdict(list)
    for t in raw:
        raw_by_rec[t["instance_id"]].append(t)
    ref_by_rec: dict[str, list[dict]] = defaultdict(list)
    for t in refined:
        ref_by_rec[t["instance_id"]].append(t)
    n_rec = len(raw_by_rec)
    rec_any_failed = sum(any(t.get("llm_failed") for t in ts) for ts in raw_by_rec.values())
    rec_no_clean = sum(not any(_clean(t) for t in ts) for ts in raw_by_rec.values())
    fallback = {
        "raw_triples": len(raw),
        "raw_generated_rate": round(sum(bool(t["generated"]) for t in raw) / len(raw), 4),
        "raw_llm_failed_rate": round(sum(bool(t["llm_failed"]) for t in raw) / len(raw), 4),
        "refined_triples": len(refined),
        "refined_generated_rate": round(sum(bool(t["generated"]) for t in refined) / len(refined), 4),
        "refined_llm_failed_rate": round(sum(bool(t["llm_failed"]) for t in refined) / len(refined), 4),
        "refined_critique_suggested_rate": round(
            sum(t.get("extraction_method") == "critique_suggested" for t in refined) / len(refined), 4),
        "pathways": len(pathways),
        "pathway_generated_rate": round(sum(bool(p.get("generated")) for p in pathways) / len(pathways), 4),
        "records_with_triples": n_rec,
        "records_with_any_llm_failed_rate": round(rec_any_failed / n_rec, 4),
        "records_without_clean_raw_triple_rate": round(rec_no_clean / n_rec, 4),
        "passages": len(passages),
        "passages_without_triples": sum(1 for iid in passages if iid not in raw_by_rec),
    }
    LOGGER.info("Fallback/generated rates: %s", fallback)

    # ---- pathway index --------------------------------------------------------
    pw_by_triple: dict[str, list[dict]] = defaultdict(list)
    for p in pathways:
        for tid in p.get("triple_ids") or []:
            pw_by_triple[tid].append(p)

    def slim_pathway(p: dict) -> dict:
        return {"pathway_id": p["pathway_id"], "culture": p.get("culture", ""),
                "completeness": p.get("completeness"),
                "has_shared_pivot": p.get("has_shared_pivot"),
                "nodes": [{"label": n.get("label", ""), "layer": n.get("layer", "")}
                          for n in p.get("nodes") or []],
                "predicates": p.get("predicates") or [],
                "generated": bool(p.get("generated")),
                "triple_ids": p.get("triple_ids") or []}

    # ---- records --------------------------------------------------------------
    candidates = []
    for iid, ts in raw_by_rec.items():
        if any(t.get("llm_failed") for t in ts):
            continue
        clean_raw = [t for t in ts if _clean(t)]
        desc = next((t.get("eval_whole_desc") for t in ts if t.get("eval_whole_desc")), None)
        if not clean_raw or not desc:
            continue
        group, canon = culture_of(iid, ts[0].get("cultural_group") or "")
        candidates.append({"instance_id": iid, "cultural_group": group, "culture": canon,
                           "_desc": desc})
    LOGGER.info("%d eligible (non-fallback, with eval_whole_desc) records", len(candidates))
    sampled = stratified_sample(candidates, lambda r: r["culture"], args.n_records,
                                seed=args.seed, max_strata=args.max_strata)
    records = []
    for c in sampled:
        iid = c["instance_id"]
        clean_raw = [t for t in raw_by_rec[iid] if _clean(t)]
        clean_ref = [t for t in ref_by_rec.get(iid, []) if _clean(t)]
        # Pathways: every pathway of the S4 index that contains any of the
        # record's triples (S4 serves generated pathways too, penalized).
        tids = {t["triple_id"] for t in raw_by_rec[iid] + ref_by_rec.get(iid, [])}
        pws, seen = [], set()
        for tid in sorted(tids):
            for p in pw_by_triple.get(tid, []):
                if p["pathway_id"] not in seen:
                    seen.add(p["pathway_id"])
                    pws.append(p)
        pws.sort(key=lambda p: (bool(p.get("generated")), p.get("completeness") != "FULL",
                                p["pathway_id"]))
        records.append({
            "instance_id": iid, "cultural_group": c["cultural_group"], "culture": c["culture"],
            "topic": (passages.get(iid) or {}).get("topic", ""),
            "eval_whole_desc": c["_desc"],
            "raw_triples": [{k: t[k] for k in KEEP_IN_RECORD} for t in clean_raw],
            "refined_triples": [{k: t[k] for k in KEEP_IN_RECORD} for t in clean_ref],
            "n_pathways_total": len(pws),
            "pathways": [slim_pathway(p) for p in pws[: args.max_pathways_per_record]],
        })

    # ---- refined triples for the audit ---------------------------------------
    for t in refined:
        t["cultural_group"], t["culture"] = culture_of(t["instance_id"])
    audit_triples = stratified_sample([t for t in refined if not t.get("llm_failed")],
                                      lambda t: t["culture"], args.n_triples,
                                      seed=args.seed, max_strata=args.max_strata)

    # ---- FULL pathways for the audit -----------------------------------------
    src_by_tid = {t["triple_id"]: t.get("source_text", "") for t in raw}
    src_by_tid.update({t["triple_id"]: t.get("source_text", "") for t in refined})
    full = [p for p in pathways if p.get("completeness") == "FULL" and not p.get("generated")]
    for p in full:
        p["_canon"] = canonical_culture(p.get("culture", ""))
    audit_pws = []
    for p in stratified_sample(full, lambda p: p["_canon"], args.n_pathways,
                               seed=args.seed, max_strata=args.max_strata):
        sp = slim_pathway(p)
        sources = []
        for tid in sp["triple_ids"]:
            s = src_by_tid.get(tid, "")
            if s and s not in sources:
                sources.append(s)
        sp["sources"] = sources
        sp["culture_canonical"] = p["_canon"]
        audit_pws.append(sp)

    # ---- write ----------------------------------------------------------------
    out = args.out_dir
    write_jsonl(out / "records.jsonl", records)
    write_jsonl(out / "triples.jsonl", audit_triples)
    write_jsonl(out / "pathways.jsonl", audit_pws)
    rec_cult = Counter(r["culture"] for r in records)
    meta = {
        "seed": args.seed, "max_strata": args.max_strata,
        "sampling": "equal allocation, round-robin over the largest canonical cultures",
        "n_records": len(records), "n_record_cultures": len(rec_cult),
        "record_cultures": dict(rec_cult.most_common()),
        "records_with_pathways": sum(1 for r in records if r["pathways"]),
        "records_with_refined": sum(1 for r in records if r["refined_triples"]),
        "mean_raw_triples_per_record": round(sum(len(r["raw_triples"]) for r in records) / max(1, len(records)), 2),
        "n_triples": len(audit_triples),
        "n_triple_cultures": len({t["culture"] for t in audit_triples}),
        "triples_generated": sum(bool(t["generated"]) for t in audit_triples),
        "n_pathways": len(audit_pws),
        "n_pathway_cultures": len({p["culture_canonical"] for p in audit_pws}),
        "pathways_without_sources": sum(1 for p in audit_pws if not p["sources"]),
        "n_full_nongenerated_pathways": len(full),
        "eligible_records": len(candidates),
        "fallback": fallback,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(out / "prep_meta.json", meta)
    LOGGER.info("Wrote %s: %s", out, json.dumps({k: v for k, v in meta.items()
                                                  if k not in ("record_cultures",)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
