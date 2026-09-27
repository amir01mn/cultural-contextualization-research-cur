"""CPU prep for eval v2, Evaluation 1 (CultureBank grounded QA).

See docs/EVAL_V2_PLAN.md ("Evaluation 1"). Writes to data/eval_v2/culturebank/:

    dev.jsonl, test.jsonl   MCQ items (gold + 3 distractors, flags heldout/open_ended)
    masks.json              held-out evidence masks {"near": [...], "far": [...]}
    prep_meta.json          counts, similarity method, gold-mapping verification

Runs on the login node (no GPU, no LLM):

    python scripts/prep_culturebank_eval_v2.py            # --sim auto
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import config  # noqa: E402
from src.core.culture_normalizer import canonical_culture  # noqa: E402
from src.core.utils import get_logger, normalize_text  # noqa: E402
from src.loaders.culturebank_eval_loader import (  # noqa: E402
    find_culturebank_csvs,
    rows_with_instance_ids,
)

LOGGER = get_logger("prep_culturebank_eval_v2")

OUT_DIR = ROOT / "data" / "eval_v2" / "culturebank"
MINILM_CACHE = OUT_DIR / "descriptor_emb_minilm.npy"

AGREEMENT_MIN = 0.6
PER_CULTURE = 20
N_DEV, N_TEST = 300, 1500
N_HELDOUT, N_OPEN = 300, 500
DISTRACTOR_MAX_SIM = 0.75
FAR_MASK_SIM = 0.85
SEED = 42


def culture_key(group: str) -> str:
    return canonical_culture(group) or normalize_text(group).lower()


# --------------------------------------------------------------------------
# Descriptor similarity (MiniLM preferred, TF-IDF fallback)
# --------------------------------------------------------------------------

def descriptor_embeddings(texts: list[str], method: str) -> tuple[object, str]:
    """Return (L2-normalized matrix, method used). Rows align with ``texts``."""
    if method in ("auto", "minilm"):
        if MINILM_CACHE.exists():
            emb = np.load(MINILM_CACHE)
            if emb.shape[0] == len(texts):
                LOGGER.info("Using cached MiniLM descriptor embeddings %s", emb.shape)
                return emb.astype(np.float32), "minilm(all-MiniLM-L6-v2, cached)"
        try:
            from sentence_transformers import SentenceTransformer

            t0 = time.time()
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(texts, normalize_embeddings=True, batch_size=256,
                               show_progress_bar=False).astype(np.float32)
            np.save(MINILM_CACHE, emb)
            LOGGER.info("MiniLM embeddings in %.0fs", time.time() - t0)
            return emb, "minilm(all-MiniLM-L6-v2)"
        except Exception as exc:  # noqa: BLE001
            if method == "minilm":
                raise
            LOGGER.warning("MiniLM unavailable (%s); using TF-IDF cosine.", exc)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(texts)  # rows are L2-normalized by default
    return mat, "tfidf-cosine"


def _sim(emb, rows_a: list[int], rows_b: list[int]) -> np.ndarray:
    a = emb[rows_a]
    b = emb[rows_b]
    out = a @ b.T
    if hasattr(out, "toarray"):
        out = out.toarray()
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------
# Gold-mapping verification
# --------------------------------------------------------------------------

def verify_gold_mapping(rows: list[dict], n_sample: int = 2000) -> dict:
    """Check reproduced ids against passages (all rows) and triples (eval_question join)."""
    result: dict = {}
    passages_path = config.PROCESSED_DIR / "corpus_passages.jsonl"
    passages = {}
    with passages_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            passages[d["passage_id"]] = d
    n_found = sum(1 for r in rows if r["instance_id"] in passages)
    n_culture_ok = sum(
        1 for r in rows
        if r["instance_id"] in passages
        and normalize_text(passages[r["instance_id"]].get("cultural_group")) == r["cultural group"])
    result["passages"] = {"rows": len(rows), "passages": len(passages),
                          "row_ids_found_in_passages": n_found,
                          "cultural_group_agrees": n_culture_ok}

    by_id = {r["instance_id"]: r for r in rows}
    trip = {"checked": 0, "eval_question_match": 0, "instance_ids_with_triples": 0}
    try:
        import pyarrow.parquet as pq

        tbl = pq.read_table(config.PROCESSED_DIR / "corpus_triples.parquet",
                            columns=["instance_id", "raw_fields"])
        iids = tbl.column("instance_id").to_pylist()
        trip["instance_ids_with_triples"] = len(set(iids))
        trip["instance_ids_with_triples_known"] = len(set(iids) & set(by_id))
        rng = random.Random(SEED)
        idx = rng.sample(range(len(iids)), min(n_sample, len(iids)))
        raw_col = tbl.column("raw_fields")
        for i in idx:
            raw = raw_col[i].as_py()
            if isinstance(raw, str):
                raw = json.loads(raw)
            row = by_id.get(iids[i])
            trip["checked"] += 1
            if row and normalize_text((raw or {}).get("eval_question")) == row["eval_question"]:
                trip["eval_question_match"] += 1
    except Exception as exc:  # noqa: BLE001
        trip["error"] = str(exc)
    result["triples"] = trip
    return result


# --------------------------------------------------------------------------
# Sampling + distractors
# --------------------------------------------------------------------------

def build(rows: list[dict], emb, *, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    for r in rows:
        r["culture_key"] = culture_key(r.get("cultural group", ""))

    has_beh = [r for r in rows if r.get("actor_behavior")]
    by_culture_all: dict[str, list[int]] = defaultdict(list)
    by_topic_all: dict[str, list[int]] = defaultdict(list)
    for r in has_beh:
        by_culture_all[r["culture_key"]].append(r["row_index"])
        by_topic_all[r.get("topic", "")].append(r["row_index"])

    eligible = [
        r for r in has_beh
        if r["agreement_f"] is not None and r["agreement_f"] >= AGREEMENT_MIN
        and r["culture_key"] and r.get("eval_question") and r.get("eval_whole_desc")
        and r.get("topic")
    ]
    elig_by_culture: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        elig_by_culture[r["culture_key"]].append(r)

    cache: dict[tuple[str, str], tuple[list[int], np.ndarray]] = {}

    def distractor_pool(ckey: str, topic: str) -> tuple[list[int], np.ndarray]:
        key = (ckey, topic)
        if key not in cache:
            cands = [i for i in by_topic_all.get(topic, [])
                     if rows[i]["culture_key"] != ckey]
            if not cands:
                cache[key] = ([], np.zeros(0))
            else:
                sims = _sim(emb, cands, by_culture_all[ckey]).max(axis=1)
                keep = sims < DISTRACTOR_MAX_SIM
                cache[key] = ([c for c, k in zip(cands, keep) if k], sims[keep])
        return cache[key]

    def pick_distractors(r: dict) -> list[tuple[int, float]] | None:
        cands, sims = distractor_pool(r["culture_key"], r["topic"])
        order = list(range(len(cands)))
        rng.shuffle(order)
        chosen: list[tuple[int, float]] = []
        seen_cultures, seen_text = set(), {r["actor_behavior"].lower()}
        for j in order:
            c = rows[cands[j]]
            t = c["actor_behavior"].lower()
            if c["culture_key"] in seen_cultures or t in seen_text:
                continue
            chosen.append((cands[j], float(sims[j])))
            seen_cultures.add(c["culture_key"])
            seen_text.add(t)
            if len(chosen) == 3:
                return chosen
        return None

    pool: list[dict] = []
    n_no_distractors = 0
    for ckey in sorted(elig_by_culture):
        items = list(elig_by_culture[ckey])
        rng.shuffle(items)
        by_topic: dict[str, list[dict]] = defaultdict(list)
        for r in items:
            by_topic[r["topic"]].append(r)
        topics = sorted(by_topic)
        rng.shuffle(topics)
        # Round-robin over topics so each culture's quota is spread over topics.
        ordered: list[dict] = []
        while any(by_topic[t] for t in topics):
            for t in topics:
                if by_topic[t]:
                    ordered.append(by_topic[t].pop(0))
        taken = 0
        for r in ordered:
            if taken >= PER_CULTURE:
                break
            d = pick_distractors(r)
            if d is None:
                n_no_distractors += 1
                continue
            pool.append({**r, "_distractors": d})
            taken += 1

    rng.shuffle(pool)
    if len(pool) < N_DEV + N_TEST:
        LOGGER.warning("Pool has only %d items (< %d).", len(pool), N_DEV + N_TEST)
    dev_rows = pool[:N_DEV]
    test_rows = pool[N_DEV:N_DEV + N_TEST]

    def to_item(r: dict, split: str) -> dict:
        d = r["_distractors"]
        return {
            "qid": f"cbq_{r['instance_id']}",
            "instance_id": r["instance_id"],
            "row_index": r["row_index"],
            "split": split,
            "cultural_group": r["cultural group"],
            "culture_key": r["culture_key"],
            "topic": r["topic"],
            "context": r.get("context", ""),
            "eval_question": r["eval_question"],
            "eval_whole_desc": r["eval_whole_desc"],
            "eval_persona": r.get("eval_persona", ""),
            "agreement": r["agreement_f"],
            "gold": r["actor_behavior"],
            "distractors": [rows[i]["actor_behavior"] for i, _ in d],
            "distractor_instance_ids": [rows[i]["instance_id"] for i, _ in d],
            "distractor_cultures": [rows[i]["cultural group"] for i, _ in d],
            "distractor_max_sim_to_gold_culture": [round(s, 4) for _, s in d],
            "heldout": False,
            "open_ended": False,
        }

    dev = [to_item(r, "dev") for r in dev_rows]
    test = [to_item(r, "test") for r in test_rows]

    heldout_idx = set(random.Random(seed).sample(range(len(test)), min(N_HELDOUT, len(test))))
    open_idx = set(random.Random(seed + 1).sample(range(len(test)), min(N_OPEN, len(test))))
    for i, it in enumerate(test):
        it["heldout"] = i in heldout_idx
        it["open_ended"] = i in open_idx

    near = sorted(it["instance_id"] for it in test if it["heldout"])
    row_of = {r["instance_id"]: r["row_index"] for r in rows}
    near_rows = [row_of[i] for i in near]
    beh_rows = [r["row_index"] for r in has_beh]
    far_extra: set[str] = set()
    if near_rows:
        sims = _sim(emb, beh_rows, near_rows).max(axis=1)
        far_extra = {rows[i]["instance_id"] for i, s in zip(beh_rows, sims) if s > FAR_MASK_SIM}
    far = sorted(set(near) | far_extra)

    stats = {
        "rows_total": len(rows),
        "rows_with_behavior": len(has_beh),
        "eligible_agreement_ge_0.6": len(eligible),
        "cultures_eligible": len(elig_by_culture),
        "pool_size": len(pool),
        "pool_cultures": len({r["culture_key"] for r in pool}),
        "rows_skipped_no_valid_distractors": n_no_distractors,
        "dev": len(dev), "test": len(test),
        "dev_cultures": len({it["culture_key"] for it in dev}),
        "test_cultures": len({it["culture_key"] for it in test}),
        "max_items_per_culture_dev_plus_test": max(
            (sum(1 for it in dev + test if it["culture_key"] == c)
             for c in {it["culture_key"] for it in dev + test}), default=0),
        "heldout": len(near), "open_ended": sum(it["open_ended"] for it in test),
        "mask_near": len(near), "mask_far": len(far),
        "dev_test_overlap": len({it["instance_id"] for it in dev} & {it["instance_id"] for it in test}),
    }
    return {"dev": dev, "test": test, "masks": {"near": near, "far": far}, "stats": stats}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sim", default="auto", choices=["auto", "minilm", "tfidf"])
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--skip_verify", action="store_true")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    hf_home = os.environ.get("HF_HOME") or str(ROOT.parent / "hf_cache")
    csvs = find_culturebank_csvs(hf_home)
    rows = rows_with_instance_ids(csvs)
    LOGGER.info("Read %d CultureBank rows from %s", len(rows), [str(c) for c in csvs])

    verification = {} if args.skip_verify else verify_gold_mapping(rows)
    LOGGER.info("Gold mapping verification: %s", verification)

    texts = [r.get("actor_behavior", "") for r in rows]
    emb, method = descriptor_embeddings(texts, args.sim)
    built = build(rows, emb)

    for split in ("dev", "test"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for it in built[split]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out_dir / "masks.json").write_text(json.dumps(built["masks"]), encoding="utf-8")
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "docs/EVAL_V2_PLAN.md Evaluation 1",
        "seed": SEED,
        "agreement_min": AGREEMENT_MIN,
        "per_culture_max": PER_CULTURE,
        "culture_key": "canonical_culture(cultural group) or lowercased group",
        "similarity_method": method,
        "descriptor": "actor_behavior",
        "distractor_rule": f"other culture, same topic, max sim to every gold-culture descriptor < {DISTRACTOR_MAX_SIM}; distinct cultures",
        "far_mask_rule": f"near + any descriptor with sim > {FAR_MASK_SIM} to a masked one",
        "instance_id_rule": "stable_id('culturebank', row_index, prefix='cb_') over tiktok then reddit rows",
        "csvs": [str(c) for c in csvs],
        "gold_mapping_verification": verification,
        "stats": built["stats"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "prep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Prep done: %s", json.dumps(built["stats"]))


if __name__ == "__main__":
    main()
