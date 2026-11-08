#!/usr/bin/env python3
"""Train-dev go/no-go gate (preregistration §4): gold log-likelihood, S1 vs S4R.

A seeded sample of train questions (free-text answers) is scored with
Qwen3-32B under each condition, using the full train index with the question's
own source passage left out of retrieval (leave-own-source-out). The metric is
the mean per-token log-likelihood of the gold answer after the assistant
``Answer:`` prefix. The S1 tuning queries (the first 400 dev rows used by
tune_kazculture_retrieval.py) are excluded from the sample by default.
Output: <out>/<condition>.jsonl checkpoints and <out>/gate_decision.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_corpus import read_rows
from src.evaluation.kazculture_final import (
    DEFAULT_FRAME_ADAPTER, FRAME_CONDITIONS, append_jsonl, embedding_run_settings, file_sha256, format_labels,
    frame_evidence, free_text_prompt, gate_decision, load_frame_adapter, parse_route_modes,
    read_jsonl_checkpoint, s1_evidence,
)

GATE_CONDITIONS = ("S0", "S1", "S4R", "S4R-noabs", "S1+L")


def make_index(path: Path):
    from src.retrieval.kazculture_index import MultilingualIndex
    return MultilingualIndex(path, device="cpu")


def make_scorer(args):
    from src.evaluation.kazculture_hf_scorer_v2 import HFQwenLetterScorerV2
    return HFQwenLetterScorerV2(args.model, model_revision=args.model_revision,
                                max_context_tokens=args.max_context_tokens)


def sample_questions(rows: list[dict], *, n: int, seed: int, exclude_first_n: int,
                     exclude_ids: set, indexed_pids: set) -> list[dict]:
    tuning = {r["train_row_id"] for r in rows[:exclude_first_n]}
    pool = [r for r in rows
            if r["train_row_id"] not in tuning and r["train_row_id"] not in exclude_ids
            and r["passage_id"] in indexed_pids
            and str(r.get("question") or "").strip() and str(r.get("answer") or "").strip()]
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} eligible train questions for n={n}")
    picked = random.Random(seed).sample(pool, n)
    return sorted(picked, key=lambda r: r["train_row_id"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--dev", type=Path, required=True, help="train-dev QA JSONL (train_row_id, question, answer, passage_id)")
    p.add_argument("--index", type=Path, required=True, help="FULL train S1 index")
    p.add_argument("--s1_config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conditions", default="S1,S4R,S4R-noabs,S1+L")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_first_n", type=int, default=400,
                   help="Skip the S1 threshold-tuning queries (first N dev rows)")
    p.add_argument("--exclude_ids_file", type=Path,
                   help="JSON list of train_row_ids to exclude (e.g. routing-tuning queries)")
    p.add_argument("--frame_adapter", default=DEFAULT_FRAME_ADAPTER)
    p.add_argument("--frame_adapter_kwargs", default=None)
    p.add_argument("--route_modes", default=None)
    p.add_argument("--setting", default="full", help="setting name passed to the adapter")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    p.add_argument("--max_context_tokens", type=int, default=16384)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--max_minutes", type=float, default=165.0)
    a = p.parse_args(argv)
    t0 = time.monotonic()

    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    if not conditions or set(conditions) - set(GATE_CONDITIONS) or \
            len(set(conditions)) != len(conditions) or not {"S1", "S4R"} <= set(conditions):
        raise ValueError(f"Conditions must include S1 and S4R, from {GATE_CONDITIONS}")
    route_modes = parse_route_modes(a.route_modes)
    cfg = json.loads(a.s1_config.read_text(encoding="utf-8"))
    if cfg.get("source") != "train_dev_only":
        raise ValueError("S1 config must originate from train-dev tuning")
    top_k, min_sim = int(cfg["top_k"]), float(cfg["min_similarity"])
    budget = int(cfg.get("evidence_budget_chars", 5000))

    index = make_index(a.index)
    indexed = {u["passage_id"] for u in index.units if u["dataset"] == "KazCulture"}
    exclude_ids = set(json.loads(a.exclude_ids_file.read_text())) if a.exclude_ids_file else set()
    sample = sample_questions(read_rows(a.dev), n=a.n, seed=a.seed,
                              exclude_first_n=a.exclude_first_n,
                              exclude_ids=exclude_ids, indexed_pids=indexed)
    a.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "dev": str(a.dev.resolve()), "dev_sha256": file_sha256(a.dev),
        "index": str(a.index.resolve()), "s1_config_sha256": file_sha256(a.s1_config),
        "n": a.n, "seed": a.seed, "exclude_first_n": a.exclude_first_n,
        "exclude_ids_n": len(exclude_ids), "model": a.model, "model_revision": a.model_revision,
        "top_k": top_k, "min_similarity": min_sim, "budget_chars": budget,
        "frame_adapter": a.frame_adapter, "route_modes": route_modes, "setting": a.setting,
        "frame_adapter_kwargs": a.frame_adapter_kwargs,
        "max_context_tokens": a.max_context_tokens,
        **embedding_run_settings(),
        "sample_train_row_ids": [r["train_row_id"] for r in sample],
    }
    sp = a.out / "settings.json"
    if sp.exists() and json.loads(sp.read_text(encoding="utf-8")) != settings:
        raise ValueError("Gate settings changed; use a new --out")
    sp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    adapter = None
    if set(conditions) & (set(FRAME_CONDITIONS) | {"S1+L"}):
        kwargs = json.loads(a.frame_adapter_kwargs) if a.frame_adapter_kwargs else None
        adapter = load_frame_adapter(a.frame_adapter, kwargs=kwargs)
    scorer = None
    results = {}
    for cond in conditions:
        path = a.out / f"{cond}.jsonl"
        done = {r["train_row_id"]: r for r in read_jsonl_checkpoint(path)}
        if set(done) - {r["train_row_id"] for r in sample}:
            raise ValueError(f"{path} has rows outside the sample")
        with path.open("a", encoding="utf-8") as fh:
            for q in sample:
                if q["train_row_id"] in done:
                    continue
                if (time.monotonic() - t0) / 60 > a.max_minutes:
                    print(f"Time guard at {cond}: rerun to resume", flush=True)
                    return 0
                own = {q["passage_id"]}
                excl_mode = "index_exclude"
                if cond == "S0":
                    blocks_text, meta = "", []
                elif cond in ("S1", "S1+L"):
                    hits = index.search(q["question"], system="S1", k=top_k,
                                        min_similarity=min_sim, exclude_passage_ids=own)
                    ann = (lambda u: format_labels(adapter.labels(u["passage_id"]))) \
                        if cond == "S1+L" else None
                    ev = s1_evidence(hits, top_k=top_k, max_chars=budget, annotate=ann)
                    blocks_text, meta = ev.text(), ev.meta
                else:
                    items, excl_mode = adapter.retrieve(
                        q["question"], None, setting=a.setting, top_k=top_k,
                        budget_chars=budget, route_mode=route_modes[cond],
                        exclude_passage_ids=own)
                    ev = frame_evidence(items, max_chars=budget, sep=adapter.separator)
                    blocks_text, meta = ev.text(), ev.meta
                if any(m["id"] in own and not m.get("indirect") for m in meta):
                    raise ValueError(f"Own source leaked for train row {q['train_row_id']}")
                if scorer is None:
                    scorer = make_scorer(a)
                prompt = free_text_prompt(scorer.tokenizer, q["question"], blocks_text)
                res = scorer.score_gold_loglik([prompt], [q["answer"]])[0]
                row = {"train_row_id": q["train_row_id"], "condition": cond,
                       "passage_id": q["passage_id"], **res,
                       "retrieved_ids": [m["id"] for m in meta],
                       "rendered_chars": len(blocks_text),
                       "own_source_exclusion": excl_mode}
                append_jsonl(fh, row)
                done[q["train_row_id"]] = row
        results[cond] = list(done.values())
        print(json.dumps({"condition": cond, "n": len(done),
                          "mean_loglik": sum(r["mean_loglik"] for r in done.values()) / len(done)}),
              flush=True)
    decision = gate_decision(results, n_boot=a.n_boot, seed=0)
    decision["settings"] = {k: v for k, v in settings.items() if k != "sample_train_row_ids"}
    decision["exclusion_modes"] = {
        c: sorted({r["own_source_exclusion"] for r in rows}) for c, rows in results.items()}
    decision["limitations"] = [
        "Own source is removed from retrieval only; frame-KG aggregates (norm-type clusters, "
        "culture profiles, routing thresholds) may still include the own passage's frame.",
        "Mean per-token log-likelihood, not accuracy; gate is a direction check, no significance rule.",
    ]
    decision["written_utc"] = datetime.now(timezone.utc).isoformat()
    decision["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    (a.out / "gate_decision.json").write_text(json.dumps(decision, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(decision["gate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
