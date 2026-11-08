#!/usr/bin/env python3
"""KazCulture v2 development harness: MCQ accuracy on held-out *train* questions.

Train questions carry the same A-D options and answer key as the test split, so
v2 is developed on the test metric without touching the test set. Each dev
question is answered with its own source passage, and every passage that is a
near-duplicate of it (BGE-M3 cosine >= --neardup_cosine), hidden from retrieval,
which mirrors the strict test setting.

Dev questions come from scripts/build_kazculture_dev_mcq.py (train rows not used for
tuning or gating in v1, turned into test-format MCQ; a disclosed synthetic proxy).

Conditions (identical V2 prompt and scorer as scripts/eval_kazculture_final.py):
    S0       no evidence
    S1       frozen S1 retrieval (top_k / min_similarity / budget from the S1 config)
    S1+L     S1 passages with each passage's own norm / value labels attached
    S4R      v1 frame retrieval + routing (BGE-M3), as preregistered
    S4R-PR   S1's ranked passages rendered as frames (v1 layout): the frame layers
             with retrieval held fixed to S1, isolating the layers from ranking
  v2 option-grounded conditions (src/retrieval/kazculture_option_grounded.py):
    S1-OG    flat S1 blocks, pooled hybrid ranking (dense question + BM25 question+options)
    S1-OG+L  S1-OG with each passage's own labels (S1+L annotation)
    S1-OGpo  flat S1 blocks, per-option hybrid ranking interleaved (control for S4R-OG)
    S4R-OG   per-option grounded frames, grouped by option, layered (values/norm type/norm)
    S4R-OGR  S4R-OG with grounding routing: an option no retrieved passage mentions
             in full (grounding < 1.0) also gets the abstract route (norm type + sibling norms)

Output: <out>/<condition>.jsonl (resumable), <out>/summary.json, <out>/sample.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.kazculture_final import (  # noqa: E402
    Evidence, append_jsonl, compare_paired, condition_summary, format_labels,
    frame_evidence, prediction_row, read_jsonl_checkpoint, s1_evidence)
from src.evaluation.kazculture_phase1 import choose_highest  # noqa: E402
from src.loaders.kazculture_loader import CHOICES, KazCultureQuestion  # noqa: E402

QWEN = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
BASE = ROOT / "data" / "kazculture_pipeline"
CONDITIONS = ("S0", "S1", "S1+L", "S4R", "S4R-PR",
              "S1-OG", "S1-OG+L", "S1-OGpo", "S4R-OG", "S4R-OGR")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--dev_mcq", type=Path, default=BASE / "v2_dev" / "dev_mcq_v1.jsonl",
                   help="scripts/build_kazculture_dev_mcq.py output (train rows only)")
    p.add_argument("--index", type=Path, default=BASE / "index" / "s1_full")
    p.add_argument("--frames_dir", type=Path, default=BASE / "frames_v3k")
    p.add_argument("--s1_config", type=Path, default=BASE / "s1_full_dev_thresholds_v3.json")
    p.add_argument("--n", type=int, default=1000, help="first N rows of --dev_mcq")
    p.add_argument("--neardup_cosine", type=float, default=0.95)
    p.add_argument("--conditions", default="S0,S1,S1+L,S4R,S4R-PR")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max_context_tokens", type=int, default=16384)
    p.add_argument("--max_minutes", type=float, default=165.0)
    p.add_argument("--retrieval_only", action="store_true",
                   help="Build evidence and report retrieval statistics without the LLM")
    p.add_argument("--limit", type=int, default=None, help="smoke runs only")
    return p


# ---------------------------------------------------------------------------
# Dev sample
# ---------------------------------------------------------------------------

def to_question(row: dict) -> KazCultureQuestion:
    options = {c: str(row.get(c.lower()) or "").strip() for c in CHOICES}
    answer = str(row.get("answer_label") or "").strip().upper()
    question = str(row.get("question") or "").strip()
    if not question or answer not in CHOICES or not all(options.values()):
        raise ValueError(f"train row {row.get('train_row_id')}: incomplete MCQ")
    return KazCultureQuestion(question_id=f"kazculture_train_{row['train_row_id']:05d}",
                              question=question, options=options, answer_key=answer,
                              category=str(row.get("category_english") or "Unknown").strip())


def neardup_exclusions(index, own_pids: dict[int, str], threshold: float) -> dict[int, set[str]]:
    """Own passage plus every KazCulture passage with cosine >= threshold to it."""
    units = [(k, u["passage_id"]) for k, u in enumerate(index.units)
             if u["system"] == "S1" and u["dataset"] == "KazCulture"]
    row_of = {pid: k for k, pid in units}
    pid_arr = [pid for _k, pid in units]
    emb = np.asarray(index.emb[[k for k, _ in units]], dtype=np.float32)
    out = {}
    for rid, pid in own_pids.items():
        if pid not in row_of:
            raise ValueError(f"source passage {pid} of train row {rid} is not in the index")
        sims = emb @ np.asarray(index.emb[row_of[pid]], dtype=np.float32)
        out[rid] = {pid} | {pid_arr[j] for j in np.nonzero(sims >= threshold)[0]}
    return out


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class Evidencer:
    def __init__(self, args, cfg: dict):
        from src.retrieval.kazculture_frame_retriever import (
            KazCultureFrameRetriever, check_embedding_backend)
        from src.retrieval.kazculture_index import MultilingualIndex
        self.args = args
        self.top_k = int(cfg["top_k"])
        self.min_sim = float(cfg["min_similarity"])
        self.budget = int(cfg.get("evidence_budget_chars", 5000))
        self.index = MultilingualIndex(args.index, device="cpu")
        check_embedding_backend(args.frames_dir / "full")
        self.kr = KazCultureFrameRetriever(args.frames_dir, settings=("full",))
        self._s1: dict[str, list[dict]] = {}
        self._og = None
        self._dense: dict[str, np.ndarray] = {}

    # -- option-grounded (v2) ------------------------------------------------
    def og(self):
        if self._og is None:
            from src.retrieval.kazculture_option_grounded import OptionGroundedRetriever
            rows = [k for k, u in enumerate(self.index.units)
                    if u["system"] == "S1" and u["dataset"] == "KazCulture" and not u["indirect"]]
            self._og_rows = np.asarray(rows)
            units = [self.index.units[k] for k in rows]
            self._og = OptionGroundedRetriever(units, self.dense_scores)
            frames = self.kr.kazakh_frames("full")
            self._frames = frames
            nt = {}
            for pid, f in frames.items():
                if f.norm_type_id:
                    nt.setdefault(f.norm_type_id, []).append(pid)
            self._nt_members = nt
        return self._og

    def dense_scores(self, question: str) -> np.ndarray:
        if question not in self._dense:
            vec = np.asarray(self.index._load_model().encode(
                [question], normalize_embeddings=True, convert_to_numpy=True)[0],
                dtype=np.float32)
            self._dense[question] = np.asarray(self.index.emb[self._og_rows] @ vec, dtype=float)
        return self._dense[question]

    def og_evidence(self, condition: str, q, excl: set[str]) -> Evidence:
        from src.retrieval.kazculture_option_grounded import (
            norm_abstraction, render_option_grounded)
        og = self.og()
        if condition in ("S1-OG", "S1-OG+L"):
            annotate = None
            if condition == "S1-OG+L":
                annotate = lambda u: format_labels(self.kr.labels_for_passage(u["passage_id"]))
            return s1_evidence(og.pooled(q.question, q.options, exclude=excl),
                               top_k=self.top_k, max_chars=self.budget, annotate=annotate)
        per = og.per_option(q.question, q.options, exclude=excl, k=4)
        if condition == "S1-OGpo":
            return s1_evidence(og.interleave(per, self.top_k), top_k=self.top_k,
                               max_chars=self.budget)
        abstraction = tau = None
        if condition == "S4R-OGR":
            tau = 1.0
            abstraction = norm_abstraction(self._frames, self._nt_members, exclude=excl)
        blocks, meta = render_option_grounded(
            q.options, per, self._frames, budget_chars=self.budget, per_option=2,
            tau_ground=tau, abstraction=abstraction)
        for m in meta:
            m.setdefault("unit_id", None)
            m["route"] = "abstract" if m["layer"] == "abstract" else "refine"
            m.setdefault("indirect", False)
        keep = list(zip(blocks, meta))
        return Evidence([b for b, _ in keep], [m for _, m in keep], sep="\n",
                        routes=sorted({str(m.get("grounded")) for m in meta if m.get("layer") == "option"}))

    def s1_hits(self, q, excl: set[str]) -> list[dict]:
        if q.question_id not in self._s1:
            self._s1[q.question_id] = self.index.search(
                q.question, system="S1", k=40, min_similarity=self.min_sim,
                exclude_passage_ids=excl, combined=False)
        return self._s1[q.question_id]

    def evidence(self, condition: str, q, excl: set[str]) -> Evidence:
        if condition == "S0":
            return Evidence([], [])
        if condition in ("S1", "S1+L"):
            annotate = None
            if condition == "S1+L":
                annotate = lambda u: format_labels(self.kr.labels_for_passage(u["passage_id"]))
            return s1_evidence(self.s1_hits(q, excl), top_k=self.top_k,
                               max_chars=self.budget, annotate=annotate)
        if condition == "S4R":
            items = self.kr.retrieve_s4r(q.question, setting="full", top_k=self.top_k,
                                         budget_chars=self.budget, route_mode="routed",
                                         exclude_passage_ids=excl)
            return frame_evidence(items, max_chars=self.budget, sep="\n")
        if condition in ("S1-OG", "S1-OG+L", "S1-OGpo", "S4R-OG", "S4R-OGR"):
            return self.og_evidence(condition, q, excl)
        if condition == "S4R-PR":
            return frame_evidence(self.passage_ranked_items(q, excl),
                                  max_chars=self.budget, sep="\n")
        raise ValueError(condition)

    def passage_ranked_items(self, q, excl: set[str]) -> list[dict]:
        """S1's top_k passages as a refine route: the v1 frame layout, S1's ranking."""
        from src.retrieval.frame_retriever import ROUTE_REFINE
        from src.retrieval.kazculture_frame_retriever import fit_blocks, route_blocks
        frames = self.kr.kazakh_frames("full")
        hits = [h for h in self.s1_hits(q, excl)[:self.top_k] if h["passage_id"] in frames]
        if not hits:
            return []
        r = {"mode": ROUTE_REFINE,
             "frames": [(frames[h["passage_id"]], h["similarity"]) for h in hits]}
        return fit_blocks(route_blocks(r, self.kr._fr("full"), None), self.budget)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    t0 = time.monotonic()
    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise ValueError(f"unknown conditions {bad}")
    cfg = json.loads(a.s1_config.read_text(encoding="utf-8"))
    if cfg.get("source") != "train_dev_only":
        raise ValueError("S1 config must come from train-dev tuning")

    if "test" in a.dev_mcq.name:
        raise ValueError("v2 development never reads the test split")
    dev = read_jsonl_checkpoint(a.dev_mcq, repair=False)[:a.limit or a.n]
    if any("train_row_id" not in r for r in dev):
        raise ValueError("dev MCQ rows must come from the train split")
    ids = [r["train_row_id"] for r in dev]
    questions = {r["train_row_id"]: to_question(r) for r in dev}
    a.out.mkdir(parents=True, exist_ok=True)

    ev = Evidencer(a, cfg)
    own = {r["train_row_id"]: r["passage_id"] for r in dev}
    excl = neardup_exclusions(ev.index, own, a.neardup_cosine)
    sample = {"train_row_ids": ids, "n": len(ids), "dev_mcq": str(a.dev_mcq),
              "neardup_cosine": a.neardup_cosine,
              "mean_excluded_passages": float(np.mean([len(v) for v in excl.values()])),
              "categories": dict(Counter(q.category for q in questions.values()))}
    sp = a.out / "sample.json"
    if sp.exists() and json.loads(sp.read_text())["train_row_ids"] != ids:
        raise ValueError("dev sample changed; use a new --out")
    sp.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in sample.items() if k != "train_row_ids"},
                     ensure_ascii=False), flush=True)

    if a.retrieval_only:
        report = {}
        for c in conditions:
            if c == "S0":
                continue
            stats = Counter()
            for rid, q in questions.items():
                e = ev.evidence(c, q, excl[rid])
                stats["questions"] += 1
                stats["blocks"] += len(e)
                stats["chars"] += len(e.text())
                stats["empty"] += not len(e)
                if any(m["id"] in excl[rid] for m in e.meta):
                    raise AssertionError(f"{c}: excluded passage leaked for row {rid}")
            report[c] = {k: v / stats["questions"] if k != "questions" else v
                         for k, v in stats.items()}
        (a.out / "retrieval_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0

    from src.evaluation.kazculture_rag import V2EvidenceScorer
    scorer = V2EvidenceScorer(QWEN[0], model_revision=QWEN[1],
                              max_context_tokens=a.max_context_tokens)
    status = {}
    for c in conditions:
        path = a.out / f"{c}.jsonl"
        done = {r["question_id"] for r in read_jsonl_checkpoint(path)}
        with path.open("a", encoding="utf-8") as fh:
            for k, (rid, q) in enumerate(questions.items(), 1):
                if q.question_id in done:
                    continue
                if (time.monotonic() - t0) / 60 > a.max_minutes:
                    break
                e = ev.evidence(c, q, excl[rid])
                if any(m["id"] in excl[rid] for m in e.meta):
                    raise AssertionError(f"{c}: excluded passage leaked for row {rid}")
                scores = scorer.score_rendered(scorer.render(q, e.text()))
                append_jsonl(fh, prediction_row(
                    q, scores, condition=c, setting="dev", evidence_text=e.text(),
                    meta=e.meta, info={"train_row_id": rid}, prediction=choose_highest(scores)))
                if k % 100 == 0:
                    print(f"{c}: {k}/{len(questions)} "
                          f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
        n = len(read_jsonl_checkpoint(path))
        status[c] = "complete" if n == len(questions) else f"partial {n}/{len(questions)}"
    rows_by = {c: read_jsonl_checkpoint(a.out / f"{c}.jsonl") for c in conditions}
    summary = {"written_utc": datetime.now(timezone.utc).isoformat(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "status": status,
               "sample": {k: v for k, v in sample.items() if k != "train_row_ids"},
               "conditions": {c: condition_summary(r) for c, r in rows_by.items() if r}}
    if "S1" in rows_by and status.get("S1") == "complete":
        summary["vs_S1"] = {c: compare_paired(rows_by["S1"], r)
                            for c, r in rows_by.items()
                            if c != "S1" and status.get(c) == "complete"}
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
