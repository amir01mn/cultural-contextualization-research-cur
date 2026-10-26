"""Hard-distractor variant of the eval v2 CultureBank MCQ (docs/EVAL_V3_PLAN.md).

Same dev/test questions, gold answers, held-out flags and masks as eval v2.
Only the distractors change: from the same valid pool (other culture, same
topic, max similarity to every gold-culture descriptor < 0.75, distinct
cultures and texts), take the three *most similar to the gold behaviour*
instead of three at random. With random distractors the closed-book model
already scores 78.6%, which leaves little room to measure evidence.

Run on a GPU node (MiniLM); writes data/eval_v3/culturebank_hard/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prep_culturebank_eval_v2 import (  # noqa: E402
    DISTRACTOR_MAX_SIM,
    _sim,
    culture_key,
    descriptor_embeddings,
)
from src.loaders.culturebank_eval_loader import (  # noqa: E402
    find_culturebank_csvs,
    read_jsonl_items,
    rows_with_instance_ids,
)

SRC_DIR = ROOT / "data" / "eval_v2" / "culturebank"
OUT_DIR = ROOT / "data" / "eval_v3" / "culturebank_hard"


def hard_distractors(item: dict, rows: list[dict], emb, by_topic: dict,
                     by_culture: dict) -> tuple[list[int], list[float]]:
    gold_row = item["row_index"]
    ckey = item["culture_key"]
    cands = [i for i in by_topic.get(item["topic"], []) if rows[i]["culture_key"] != ckey]
    if not cands:
        return [], []
    max_to_culture = _sim(emb, cands, by_culture[ckey]).max(axis=1)
    valid = [c for c, m in zip(cands, max_to_culture) if m < DISTRACTOR_MAX_SIM]
    if not valid:
        return [], []
    to_gold = _sim(emb, valid, [gold_row])[:, 0]
    chosen, sims, seen_c = [], [], set()
    seen_t = {rows[gold_row]["actor_behavior"].lower()}
    for j in np.argsort(-to_gold):
        r = rows[valid[j]]
        t = r["actor_behavior"].lower()
        if r["culture_key"] in seen_c or t in seen_t:
            continue
        chosen.append(valid[j])
        sims.append(float(to_gold[j]))
        seen_c.add(r["culture_key"])
        seen_t.add(t)
        if len(chosen) == 3:
            break
    return chosen, sims


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sim", default="auto", choices=["auto", "minilm", "tfidf"])
    p.add_argument("--src_dir", default=str(SRC_DIR))
    p.add_argument("--out_dir", default=str(OUT_DIR))
    args = p.parse_args(argv)
    src, out = Path(args.src_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    hf_home = os.environ.get("HF_HOME") or str(ROOT.parent / "hf_cache")
    rows = rows_with_instance_ids(find_culturebank_csvs(hf_home))
    for r in rows:
        r["culture_key"] = culture_key(r.get("cultural group", ""))
    emb, method = descriptor_embeddings([r.get("actor_behavior", "") for r in rows], args.sim)
    row_of = {r["instance_id"]: r["row_index"] for r in rows}
    by_topic, by_culture = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("actor_behavior"):
            by_topic[r.get("topic", "")].append(r["row_index"])
            by_culture[r["culture_key"]].append(r["row_index"])

    stats = {"similarity_method": method, "kept_v2_distractors": 0, "items": 0,
             "mean_sim_to_gold_v2": [], "mean_sim_to_gold_hard": []}
    for split in ("dev", "test"):
        items = read_jsonl_items(src / f"{split}.jsonl")
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                stats["items"] += 1
                chosen, sims = hard_distractors(it, rows, emb, by_topic, by_culture)
                if len(chosen) < 3:
                    stats["kept_v2_distractors"] += 1
                    f.write(json.dumps({**it, "distractor_rule": "v2_random_fallback"},
                                       ensure_ascii=False) + "\n")
                    continue
                v2_rows = [row_of[i] for i in it["distractor_instance_ids"] if i in row_of]
                if v2_rows:
                    stats["mean_sim_to_gold_v2"].append(
                        float(_sim(emb, v2_rows, [it["row_index"]])[:, 0].mean()))
                stats["mean_sim_to_gold_hard"].append(float(np.mean(sims)))
                f.write(json.dumps({
                    **it,
                    "distractors": [rows[i]["actor_behavior"] for i in chosen],
                    "distractor_instance_ids": [rows[i]["instance_id"] for i in chosen],
                    "distractor_cultures": [rows[i]["cultural group"] for i in chosen],
                    "distractor_sim_to_gold": [round(x, 4) for x in sims],
                    "distractor_rule": "hard: valid v2 pool, top-3 by similarity to gold",
                    "v2_distractors": it["distractors"],
                }, ensure_ascii=False) + "\n")
    (out / "masks.json").write_text((src / "masks.json").read_text())
    for k in ("mean_sim_to_gold_v2", "mean_sim_to_gold_hard"):
        v = stats[k]
        stats[k] = round(float(np.mean(v)), 4) if v else None
    stats["seconds"] = round(time.time() - t0, 1)
    stats["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "prep_meta.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
