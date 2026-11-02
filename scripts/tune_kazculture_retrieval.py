#!/usr/bin/env python3
"""Tune KazCulture retrieval thresholds on TRAIN-only QA-to-passage recall.

This is a self-source retrieval sanity test (query answer comes from an indexed
train passage). It is NOT an independent held-out passage generalization test.
No KazCulture test labels/questions are used to tune the threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_corpus import read_rows
from src.retrieval.kazculture_index import MultilingualIndex


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max_queries", type=int, default=400)
    p.add_argument("--target_recall", type=float, default=0.75)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    index = MultilingualIndex(args.index, device="cpu")
    qa = read_rows(args.dev)[:args.max_queries]
    if not qa:
        raise ValueError("No training-development questions")
    source_ids = {u["passage_id"] for u in index.units
                  if u["dataset"] == "KazCulture"}
    samples = [q for q in qa if q["passage_id"] in source_ids]
    if len(samples) < 30:
        raise ValueError("Insufficient same-source development questions")
    # Encode all train-dev queries in batches once. The previous implementation
    # called index.search() separately for every query, causing 400 independent
    # BGE-M3 forward passes on CPU. This keeps the same S1 cosine ranking while
    # making the diagnostic practical on a compute node.
    model = index._load_model()
    questions = [q["question"] for q in samples]
    vectors = np.asarray(model.encode(
        questions,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ), dtype=np.float32)
    if vectors.shape[0] != len(samples) or not np.isfinite(vectors).all():
        raise ValueError("Invalid train-development query embeddings")

    s1_indices = [
        i for i, u in enumerate(index.units)
        if u["system"] == "S1" and u["dataset"] == "KazCulture"
        and not u.get("indirect", False)
    ]
    if not s1_indices:
        raise ValueError("Index contains no KazCulture S1 units")
    s1_units = [index.units[i] for i in s1_indices]
    s1_emb = np.asarray(index.emb[s1_indices], dtype=np.float32)
    scores = np.asarray(vectors @ s1_emb.T, dtype=np.float32)

    retrieved = []
    for q, row_scores in zip(samples, scores):
        order = sorted(
            range(len(s1_units)),
            key=lambda j: (-float(row_scores[j]), s1_units[j]["unit_id"]),
        )
        hits = []
        for j in order:
            score = round(float(row_scores[j]), 6)
            if score < 0.0:
                continue
            hits.append({**s1_units[j], "similarity": score})
            if len(hits) >= 8:
                break
        retrieved.append((q, hits))
    candidates = []
    for top_k in (3, 5, 8):
        for threshold in (0, 0.15, 0.25, 0.35, 0.45, 0.55):
            hits = sum(
                q["passage_id"] in {
                    h["passage_id"] for h in row[:top_k]
                    if h["similarity"] >= threshold
                } for q, row in retrieved
            )
            candidates.append({
                "top_k": top_k, "min_similarity": threshold,
                "matched": hits, "n": len(samples),
                "recall_at_k": hits / len(samples),
            })
    viable = [c for c in candidates if c["recall_at_k"] >= args.target_recall]
    if not viable:
        raise ValueError("Train development retrieval recall below gate for every threshold: "
                         + json.dumps(candidates[-6:]))
    # Prefer fewer passages then a higher threshold, without tuning on test.
    best = min(viable, key=lambda c: (c["top_k"], -c["min_similarity"]))
    config = {
        "source": "train_dev_only", "protocol": "self_source_retrieval_not_holdout",
        "top_k": best["top_k"], "min_similarity": best["min_similarity"],
        "dev_n": len(samples), "train_dev_recall": best["recall_at_k"],
        "target_recall": args.target_recall, "candidate_grid": candidates,
        "note": "Gold source comes from training; test never used to tune."
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("Frozen development config exists; choose a new versioned path")
    args.out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
