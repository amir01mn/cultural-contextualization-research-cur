#!/usr/bin/env python3
"""Tune KazCulture S4R routing thresholds (tau_ref, tau_abs) on TRAIN questions only.

Objective: mean (over questions) of the per-token gold-answer log-likelihood
under Qwen3-32B (non-thinking) given the S4R evidence: ``score_gold_loglik``
of ``HFQwenLetterScorerV2`` (src/evaluation/kazculture_hf_scorer_v2.py), with
the evidence numbered and packed by ``kazculture_final.frame_evidence`` and the
prompt built by ``kazculture_final.free_text_prompt`` -- the exact rendering the
train-dev gate (scripts/eval_kazculture_traindev_gate.py) uses. Both are
imported lazily; tests inject stand-ins.

* Questions: a seeded (seed 0) sample of 400 rows of train_dev_qa.jsonl, drawn
  from rows 400+ so it is disjoint from the first 400 rows that
  tune_kazculture_retrieval.py reads (``read_rows(dev)[:400]``).
* Leave-source-out: each question's own source passage is masked from the full
  frame index, so tuning mimics the strict (source-absent) setting.
* Test questions are never read.

Writes ``<frames_dir>/routing_thresholds.json`` (refuses to overwrite), a
per-question detail file, and ``routing_tuning_train_row_ids.json`` (pass it to
the gate's ``--exclude_ids_file`` so gate questions are not tuning questions).

    python scripts/tune_kazculture_routing.py
    python scripts/tune_kazculture_routing.py --dry_run    # routes only, no LLM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = ROOT / "data" / "kazculture_pipeline"
TAU_REF_GRID = (0.3, 0.4, 0.5, 0.6)
TAU_ABS_GRID = (0.3, 0.4, 0.5)
RETRIEVAL_TUNER_ROWS = 400      # tune_kazculture_retrieval.py: read_rows(dev)[:400]


def sample_questions(rows: list[dict], *, n: int = 400, seed: int = 0,
                     skip_first: int = RETRIEVAL_TUNER_ROWS) -> list[dict]:
    """Seeded sample from rows[skip_first:], returned in file order."""
    pool = rows[skip_first:]
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} train questions after the first {skip_first}")
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(int(i) for i in idx)]


def make_scorer(model: str, revision: str, max_context_tokens: int):
    from src.evaluation.kazculture_hf_scorer_v2 import HFQwenLetterScorerV2
    return HFQwenLetterScorerV2(model, model_revision=revision,
                                max_context_tokens=max_context_tokens)


def gate_rendering():
    """(frame_evidence, free_text_prompt) shared with the train-dev gate."""
    from src.evaluation.kazculture_final import frame_evidence, free_text_prompt
    return frame_evidence, free_text_prompt


def main(argv=None, *, scorer=None, rendering=None, retriever=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frames_dir", type=Path, default=BASE / "frames_v3k")
    p.add_argument("--train_qa", type=Path, default=BASE / "corpus" / "train_dev_qa.jsonl")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--budget_chars", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--embedding_backend", default="bge-m3", choices=["bge-m3", "tfidf"])
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    p.add_argument("--max_context_tokens", type=int, default=16384)
    p.add_argument("--dry_run", action="store_true", help="routes only, no LLM")
    args = p.parse_args(argv)

    from src.core import config
    from src.loaders.kazculture_corpus import read_rows
    from src.retrieval.kazculture_frame_retriever import KazCultureFrameRetriever

    if "test" in args.train_qa.name:
        raise ValueError("routing is tuned on train questions only")
    config.EMBEDDING_BACKEND = args.embedding_backend
    out = args.out or args.frames_dir / "routing_thresholds.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen thresholds are never overwritten")
    rows = read_rows(args.train_qa)
    sample = sample_questions(rows, n=args.n, seed=args.seed)
    kr = retriever or KazCultureFrameRetriever(args.frames_dir, settings=("full",),
                                               tau_ref=0.0, tau_abs=0.0)
    known = set(kr.kazakh_frames("full"))
    missing = [q["passage_id"] for q in sample if q["passage_id"] not in known]
    if missing:
        raise ValueError(f"{len(missing)} sampled questions have no source frame")

    grid = [(tr, ta) for tr in TAU_REF_GRID for ta in TAU_ABS_GRID]
    t0 = time.time()
    cells = []          # (config, question index, adapter items)
    for qi, q in enumerate(sample):
        for tr, ta in grid:
            items = kr.retrieve_s4r(q["question"], None, "full", args.top_k,
                                    args.budget_chars, "full",
                                    exclude_passage_ids={q["passage_id"]},
                                    tau_ref=tr, tau_abs=ta)
            if any(b["source_id"] == q["passage_id"] for b in items):
                raise AssertionError("own source leaked into tuning evidence")
            cells.append(((tr, ta), qi, items))
    route_secs = round(time.time() - t0, 1)

    lls: list = [None] * len(cells)
    shown_chars = [None] * len(cells)
    n_unique = 0
    if not args.dry_run:
        frame_evidence, free_text_prompt = rendering or gate_rendering()
        scorer = scorer or make_scorer(args.model, args.model_revision,
                                       args.max_context_tokens)
        texts = [frame_evidence(items, max_chars=args.budget_chars).text()
                 for _c, _qi, items in cells]
        shown_chars = [len(t) for t in texts]
        unique: dict[tuple[str, int], int] = {}
        for (_c, qi, _items), t in zip(cells, texts):
            unique.setdefault((t, qi), len(unique))
        keys = list(unique)
        n_unique = len(keys)
        prompts_ = [free_text_prompt(scorer.tokenizer, sample[qi]["question"], t)
                    for t, qi in keys]
        res = scorer.score_gold_loglik(prompts_, [sample[qi]["answer"] for _t, qi in keys])
        vals = [float(r["mean_loglik"]) for r in res]
        if len(vals) != len(keys) or not all(np.isfinite(vals)):
            raise ValueError("scorer returned a wrong number of / non-finite log-likelihoods")
        lls = [vals[unique[(t, qi)]] for (_c, qi, _i), t in zip(cells, texts)]

    results = []
    for tr, ta in grid:
        idx = [k for k, c in enumerate(cells) if c[0] == (tr, ta)]
        results.append({
            "tau_ref": tr, "tau_abs": ta, "n": len(idx),
            "mean_gold_loglik": None if args.dry_run else
            round(float(np.mean([lls[k] for k in idx])), 6),
            "routes": dict(Counter(cells[k][2][0]["route"] if cells[k][2] else "none"
                                   for k in idx)),
            "mean_adapter_blocks": round(float(np.mean([len(cells[k][2]) for k in idx])), 2),
        })
    best = None
    if not args.dry_run:
        # Highest objective; ties go to the stricter (higher) thresholds.
        best = max(results, key=lambda r: (r["mean_gold_loglik"], r["tau_ref"], r["tau_abs"]))
    row_ids = [q["train_row_id"] for q in sample]
    payload = {
        "source": "train_only", "protocol": "leave_source_out_full_index",
        "objective": "mean per-token gold-answer log-likelihood (score_gold_loglik), "
                     "Qwen3-32B non-thinking, train-dev gate prompt",
        "dry_run": args.dry_run, "model": args.model, "model_revision": args.model_revision,
        "tau_ref": best["tau_ref"] if best else None,
        "tau_abs": best["tau_abs"] if best else None,
        "best_mean_gold_loglik": best["mean_gold_loglik"] if best else None,
        "grid": results, "top_k": args.top_k, "budget_chars": args.budget_chars,
        "n_questions": len(sample), "seed": args.seed,
        "skipped_first_rows": RETRIEVAL_TUNER_ROWS,
        "train_row_ids_sha256": hashlib.sha256(
            "\n".join(map(str, row_ids)).encode()).hexdigest(),
        "train_qa": str(args.train_qa), "frames_dir": str(args.frames_dir),
        "n_unique_prompts": n_unique, "route_seconds": route_secs,
        "note": "Own source passage masked at query time; it still shaped the norm-type "
                "clusters and the Kazakh profile (a build-time held-out KG per question "
                "is too costly). Test never read.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    ids_path = out.with_name("routing_tuning_train_row_ids.json")
    ids_path.write_text(json.dumps(row_ids) + "\n", encoding="utf-8")
    if not args.dry_run:
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    detail = out.with_name(out.stem + ("_dryrun" if args.dry_run else "") + "_details.jsonl")
    with detail.open("w", encoding="utf-8") as fh:
        for k, ((tr, ta), qi, items) in enumerate(cells):
            fh.write(json.dumps({
                "train_row_id": sample[qi]["train_row_id"], "tau_ref": tr, "tau_abs": ta,
                "route": items[0]["route"] if items else "none",
                "source_ids": [b["source_id"] for b in items],
                "shown_chars": shown_chars[k], "gold_loglik": lls[k]}) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "grid"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
