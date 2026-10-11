"""CPU prep for eval v2 / Evaluation 2 (NormAd). Runs on the login node.

Writes ``data/eval_v2/normad/``:
  - ``stories_dev.jsonl``  : 300 stories stratified by country (seed 42), for tau tuning
  - ``stories_test.jsonl`` : the remaining stories
  - ``prep_meta.json``     : embedding method, coverage stats per country, covered fraction

Coverage (pre-registered, docs/EVAL_V2_PLAN.md): the max embedding similarity
between a story's gold Rule-of-Thumb and every CultureBank descriptor
(``data/processed/corpus_passages.jsonl``, read-only) of that story's country;
covered <=> sim >= 0.5. The threshold was set for MiniLM (all-MiniLM-L6-v2);
if MiniLM cannot be loaded the script falls back to TF-IDF and records it.

Usage:
    source slurm/load_modules.sh && load_login_modules
    source slurm/env_hf.sh && setup_hf_offline
    .venv/bin/python scripts/prep_normad_v2.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.core import config  # noqa: E402
from src.core.culture_normalizer import canonical_culture  # noqa: E402
from src.core.utils import get_logger  # noqa: E402
from src.loaders.normad_loader import load_normad  # noqa: E402

LOGGER = get_logger("prep_normad_v2")

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "eval_v2" / "normad"
COVERAGE_THRESHOLD = 0.5
DEV_SIZE = 300
SEED = 42


def descriptor_bin(n: int) -> str:
    """Per-country coverage strata (same cut points as the CulturalBench re-score)."""
    if n < 20:
        return "<20"
    if n < 100:
        return "20-99"
    return ">=100"


def stratified_dev_split(stories: list[dict], dev_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Proportional (largest-remainder) allocation of ``dev_size`` over countries."""
    by_country: dict[str, list[dict]] = defaultdict(list)
    for s in stories:
        by_country[s["country"]].append(s)
    total = len(stories)
    countries = sorted(by_country)
    exact = {c: dev_size * len(by_country[c]) / total for c in countries}
    quota = {c: int(exact[c]) for c in countries}
    remaining = dev_size - sum(quota.values())
    for c in sorted(countries, key=lambda c: (-(exact[c] - quota[c]), c))[:remaining]:
        quota[c] += 1
    rng = random.Random(seed)
    dev_ids: set[str] = set()
    for c in countries:
        items = sorted(by_country[c], key=lambda s: s["id"])
        rng.shuffle(items)
        dev_ids.update(s["id"] for s in items[:quota[c]])
    dev = [s for s in stories if s["id"] in dev_ids]
    test = [s for s in stories if s["id"] not in dev_ids]
    return dev, test


def _read_passages(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append({"passage_id": d.get("passage_id"),
                             "culture": canonical_culture(d.get("cultural_group")),
                             "text": d.get("text", "")})
    return rows


def _embedder():
    """Return (encode_fn, method). MiniLM if loadable, else TF-IDF."""
    from src.core.embedding_utils import get_sentence_transformer

    try:  # login nodes cap CPU; torch's default thread count thrashes there
        import torch

        torch.set_num_threads(int(os.getenv("PREP_TORCH_THREADS", "4")))
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    model = get_sentence_transformer()
    if model is not None:
        LOGGER.info("MiniLM loaded in %.1fs", time.time() - t0)

        def enc(texts):
            return np.asarray(model.encode(texts, batch_size=128, normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)
        return enc, "sentence-transformers/all-MiniLM-L6-v2"
    LOGGER.warning("MiniLM unavailable; falling back to TF-IDF (threshold 0.5 was set for MiniLM).")
    return None, "tfidf"


def compute_coverage(stories: list[dict], passages: list[dict]) -> str:
    """Annotate each story with coverage_sim / covered / best passage. Returns method."""
    by_culture: dict[str, list[dict]] = defaultdict(list)
    for p in passages:
        by_culture[p["culture"]].append(p)
    cultures = sorted({s["culture"] for s in stories})
    rel_passages = [p for c in cultures for p in by_culture.get(c, [])]
    enc, method = _embedder()
    rots = [s["rule_of_thumb"] for s in stories]
    ptexts = [p["text"] for p in rel_passages]
    if enc is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        vec.fit(ptexts + rots)
        P = normalize(vec.transform(ptexts)).toarray().astype(np.float32) if ptexts else np.zeros((0, 1))
        R = normalize(vec.transform(rots)).toarray().astype(np.float32)
    else:
        t0 = time.time()
        P = enc(ptexts) if ptexts else np.zeros((0, 384), dtype=np.float32)
        R = enc(rots)
        LOGGER.info("Embedded %d passages + %d RoTs in %.1fs", len(ptexts), len(rots), time.time() - t0)
    idx_by_culture: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(rel_passages):
        idx_by_culture[p["culture"]].append(i)
    for j, s in enumerate(stories):
        idxs = idx_by_culture.get(s["culture"], [])
        s["n_descriptors"] = len(idxs)
        s["descriptor_bin"] = descriptor_bin(len(idxs))
        if not idxs:
            s["coverage_sim"] = 0.0
            s["coverage_passage_id"] = None
        else:
            sims = P[idxs] @ R[j]
            k = int(np.argmax(sims))
            s["coverage_sim"] = round(float(sims[k]), 4)
            s["coverage_passage_id"] = rel_passages[idxs[k]]["passage_id"]
        s["covered"] = s["coverage_sim"] >= COVERAGE_THRESHOLD
    return method


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Prepare NormAd for eval v2.")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--passages", default=str(config.PROCESSED_DIR / "corpus_passages.jsonl"))
    p.add_argument("--limit", type=int, default=None, help="debug only")
    args = p.parse_args(argv)
    t_start = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stories = load_normad(limit=args.limit)
    passages = _read_passages(Path(args.passages))
    LOGGER.info("%d stories, %d passages", len(stories), len(passages))
    method = compute_coverage(stories, passages)
    dev_size = min(DEV_SIZE, len(stories) // 2) if args.limit else DEV_SIZE
    dev, test = stratified_dev_split(stories, dev_size, SEED)
    for s in dev:
        s["split"] = "dev"
    for s in test:
        s["split"] = "test"
    _write_jsonl(out / "stories_dev.jsonl", dev)
    _write_jsonl(out / "stories_test.jsonl", test)

    per_country = {}
    for c in sorted({s["country"] for s in stories}):
        items = [s for s in stories if s["country"] == c]
        per_country[c] = {
            "culture": items[0]["culture"],
            "n_stories": len(items),
            "n_descriptors": items[0]["n_descriptors"],
            "descriptor_bin": items[0]["descriptor_bin"],
            "covered_fraction": round(sum(s["covered"] for s in items) / len(items), 4),
            "mean_coverage_sim": round(float(np.mean([s["coverage_sim"] for s in items])), 4),
        }

    def frac(rows):
        return round(sum(s["covered"] for s in rows) / len(rows), 4) if rows else 0.0

    meta = {
        "dataset": "akhilayerukola/NormAd",
        "n_stories": len(stories),
        "n_dev": len(dev),
        "n_test": len(test),
        "seed": SEED,
        "label_counts": dict(Counter(s["gold"] for s in stories)),
        "embedding_method": method,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "coverage_threshold_note": "threshold pre-registered for MiniLM all-MiniLM-L6-v2",
        "covered_fraction": {"all": frac(stories), "dev": frac(dev), "test": frac(test)},
        "n_covered_test": sum(s["covered"] for s in test),
        "descriptor_bin_counts_test": dict(Counter(s["descriptor_bin"] for s in test)),
        "unmatched_countries": sorted(c for c, v in per_country.items() if v["n_descriptors"] == 0),
        "per_country": per_country,
        "wall_time_s": round(time.time() - t_start, 1),
    }
    (out / "prep_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    LOGGER.info("Covered fraction: %s (method=%s) -> %s", meta["covered_fraction"], method, out)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_country"}, indent=2))


if __name__ == "__main__":
    main()
