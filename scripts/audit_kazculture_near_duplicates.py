#!/usr/bin/env python3
"""Near-duplicate audit of the strict S1 index against KazCulture test passages.

Exact-match removal (1,262 train passages) does not remove paraphrases or
supersets (preregistration §6). For every unique test source passage this
reports the maximum BGE-M3 cosine (pinned revision) and the maximum
character-5-gram containment |G(test) ∩ G(train)| / |G(test)| over the strict
index, with the nearest train passage for each. Outputs in --out:

    near_duplicate_audit.json      counts at cosine >= 0.90 / 0.95 and containment >= 0.8
    near_duplicate_passages.jsonl  one row per unique test passage
    near_dedup_list.json           flagged passage/question IDs (strict-near-dedup subset)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_loader import load_test
from src.loaders.kazculture_corpus import normalized_passage, passage_id, read_rows

COSINE_THRESHOLDS = (0.90, 0.95)
CONTAINMENT_THRESHOLD = 0.8


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = normalized_passage(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def max_containment(test_texts: list[str], train_texts: list[str], n: int = 5
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per test text: max containment over train texts and the argmax index."""
    from scipy import sparse

    vocab: dict[str, int] = {}
    def rows(texts, grow):
        indptr, indices = [0], []
        sizes = []
        for t in texts:
            grams = char_ngrams(t, n)
            sizes.append(len(grams))
            for g in grams:
                j = vocab.get(g)
                if j is None and grow:
                    j = vocab[g] = len(vocab)
                if j is not None:
                    indices.append(j)
            indptr.append(len(indices))
        return indptr, indices, np.asarray(sizes, dtype=float)

    tr_ptr, tr_idx, _ = rows(train_texts, True)
    te_ptr, te_idx, te_sizes = rows(test_texts, False)
    shape_v = max(len(vocab), 1)
    train = sparse.csr_matrix((np.ones(len(tr_idx), dtype=np.float32), tr_idx, tr_ptr),
                              shape=(len(train_texts), shape_v))
    test = sparse.csr_matrix((np.ones(len(te_idx), dtype=np.float32), te_idx, te_ptr),
                             shape=(len(test_texts), shape_v))
    inter = (test @ train.T).toarray()
    with np.errstate(divide="ignore", invalid="ignore"):
        cont = np.where(te_sizes[:, None] > 0, inter / te_sizes[:, None], 0.0)
    return cont.max(axis=1), cont.argmax(axis=1)


def encode_test(index, texts: list[str], cache: Path | None, batch_size: int) -> np.ndarray:
    digest = hashlib.sha256("\n\x00".join(texts).encode("utf-8")).hexdigest()
    if cache and cache.exists():
        meta = json.loads(cache.with_suffix(".json").read_text())
        if meta["sha256"] == digest:
            return np.load(cache)
    vec = np.asarray(index._load_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False), dtype=np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vec)
        cache.with_suffix(".json").write_text(json.dumps({"sha256": digest, "n": len(texts)}))
    return vec


def make_index(path: Path):
    from src.retrieval.kazculture_index import MultilingualIndex
    return MultilingualIndex(path, device="cpu")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--index_strict", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--embeddings_cache", type=Path)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ngram", type=int, default=5)
    p.add_argument("--flag_cosine", type=float, default=0.95)
    p.add_argument("--flag_containment", type=float, default=CONTAINMENT_THRESHOLD)
    p.add_argument("--expected_size", type=int, default=1334)
    a = p.parse_args(argv)

    questions, _ = load_test(local_file=str(a.test_file), expected_size=a.expected_size)
    rows = read_rows(a.test_file)
    by_pid: dict[str, dict] = {}
    for q, r in zip(questions, rows):
        text = str(r.get("item_description") or "").strip()
        pid = passage_id(text)
        by_pid.setdefault(pid, {"text": text, "question_ids": []})["question_ids"].append(
            q.question_id)
    pids = sorted(by_pid)
    texts = [by_pid[x]["text"] for x in pids]

    index = make_index(a.index_strict)
    sel = [i for i, u in enumerate(index.units)
           if u["system"] == "S1" and u["dataset"] == "KazCulture"]
    train_units = [index.units[i] for i in sel]
    if {u["passage_id"] for u in train_units} & set(pids):
        raise ValueError("Index is not strict: exact test passages present")
    train_emb = np.asarray(index.emb[sel], dtype=np.float32)
    test_emb = encode_test(index, texts, a.embeddings_cache, a.batch_size)
    cos = test_emb @ train_emb.T
    cos_max, cos_arg = cos.max(axis=1), cos.argmax(axis=1)
    cont_max, cont_arg = max_containment(texts, [u["text"] for u in train_units], a.ngram)

    a.out.mkdir(parents=True, exist_ok=True)
    per = []
    with (a.out / "near_duplicate_passages.jsonl").open("w", encoding="utf-8") as fh:
        for i, pid in enumerate(pids):
            row = {"test_passage_id": pid, "question_ids": by_pid[pid]["question_ids"],
                   "max_cosine": round(float(cos_max[i]), 6),
                   "nearest_cosine_train_id": train_units[int(cos_arg[i])]["passage_id"],
                   "max_containment": round(float(cont_max[i]), 6),
                   "nearest_containment_train_id": train_units[int(cont_arg[i])]["passage_id"]}
            per.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def count(pred):
        hit = [r for r in per if pred(r)]
        return {"passages": len(hit), "questions": sum(len(r["question_ids"]) for r in hit)}

    counts = {f"cosine_ge_{t:.2f}": count(lambda r, t=t: r["max_cosine"] >= t)
              for t in COSINE_THRESHOLDS}
    counts[f"containment_ge_{CONTAINMENT_THRESHOLD}"] = count(
        lambda r: r["max_containment"] >= CONTAINMENT_THRESHOLD)
    flag = lambda r: r["max_cosine"] >= a.flag_cosine or r["max_containment"] >= a.flag_containment
    counts["flagged_union"] = count(flag)
    flagged = [r for r in per if flag(r)]
    criteria = {"cosine_ge": a.flag_cosine, "containment_ge": a.flag_containment,
                "ngram": a.ngram, "rule": "flag if max cosine >= cosine_ge OR "
                                          "max char-n-gram containment >= containment_ge"}
    report = {"n_test_questions": len(questions), "n_unique_test_passages": len(pids),
              "n_strict_index_passages": len(train_units),
              "embedding_model": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
              "counts": counts, "criteria": criteria,
              "cosine_quantiles": {q: round(float(np.quantile(cos_max, q)), 4)
                                   for q in (0.5, 0.9, 0.99)},
              "containment_quantiles": {q: round(float(np.quantile(cont_max, q)), 4)
                                        for q in (0.5, 0.9, 0.99)}}
    (a.out / "near_duplicate_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.out / "near_dedup_list.json").write_text(json.dumps({
        "criteria": criteria,
        "flagged_passage_ids": [r["test_passage_id"] for r in flagged],
        "flagged_question_ids": sorted(q for r in flagged for q in r["question_ids"]),
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
