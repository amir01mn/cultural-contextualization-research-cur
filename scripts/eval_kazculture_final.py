#!/usr/bin/env python3
"""Final KazCulture test evaluation: S0/S1/S4R/S4R-noabs/S1+L/CTRL-LEN/CTRL-SHUF.

One frozen scorer load serves every condition passed in ``--conditions``; each
(setting, condition) is run over all 1,334 test questions with the identical
answering template, only the numbered evidence block changing
(docs/KAZCULTURE_PREREGISTRATION.md §3). Output layout under ``--out``:

    settings.json                    invariant experiment settings (checked on resume)
    runs.jsonl                       one record per invocation (audit trail, criterion 3)
    retrieval/<setting>_S1.jsonl     cached S1 candidates (ranked, k = --s1_candidates)
    retrieval/<setting>_<cond>.jsonl cached frame-adapter items for S4R-type conditions
    closed_book/S0.jsonl             S0 (setting-independent)
    <setting>/<condition>.jsonl      per-question predictions (checkpoint; resumable)
    <setting>/<condition>_summary.json
    reproduction_<setting>_<cond>.json   comparison with a frozen run, if given

``--retrieval_only`` fills the retrieval caches on CPU without loading the
answer model and writes ``retrieval_report_<setting>.json`` (including how
many questions the frozen first-overflow ``break`` affects).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_loader import load_test
from src.loaders.kazculture_corpus import passage_id, read_rows
from src.evaluation.kazculture_phase1 import choose_highest
from src.evaluation.kazculture_final import (
    CLOSED_BOOK, CONDITIONS, CTRL_LEN_MODES, DEFAULT_FRAME_ADAPTER,
    DEFAULT_ROUTE_MODES, DOCUMENTED, FRAME_CONDITIONS, check_documented_s1_config,
    documented_count_check, FRAME_RENDER_MODES, FROZEN_S1, SETTINGS, Evidence,
    append_jsonl, condition_summary, ctrl_len_evidence, file_sha256, fit_blocks,
    format_labels, frame_evidence, load_frame_adapter, prediction_row,
    parse_route_modes, read_jsonl_checkpoint, reproduction_check, s1_evidence,
)

QWEN = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
OLMO_ALLOWED = {"conditions": {"S0", "S1", "S4R"}, "settings": {"strict"}}
NEEDS_S4R_CHARS = {"CTRL-LEN"}


from src.evaluation.kazculture_og_evidence import OG_CONDITIONS  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def make_index(path: Path):
    from src.retrieval.kazculture_index import MultilingualIndex
    return MultilingualIndex(path, device="cpu")


def make_scorer(args):
    if args.model_family == "olmo2":
        from src.evaluation.kazculture_hf_scorer_olmo import HFOlmoLetterScorer
        return HFOlmoLetterScorer(args.model, model_revision=args.model_revision,
                                  max_context_tokens=args.max_context_tokens)
    from src.evaluation.kazculture_rag import V2EvidenceScorer
    return V2EvidenceScorer(args.model, model_revision=args.model_revision,
                            max_context_tokens=args.max_context_tokens)


def parse_list(text: str, allowed, what: str) -> list[str]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if not items or bad or len(items) != len(set(items)):
        raise ValueError(f"Invalid/duplicate {what}: {items} (allowed {list(allowed)})")
    return items


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--index_strict", type=Path)
    p.add_argument("--index_full", type=Path)
    p.add_argument("--s1_config", type=Path, required=True,
                   help="data/kazculture_pipeline/s1_full_dev_thresholds_v3.json")
    p.add_argument("--conditions", required=True, help="Comma list of " + ",".join(CONDITIONS + OG_CONDITIONS))
    p.add_argument("--settings", default="strict", help="Comma list of strict,full")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model_family", choices=("qwen3", "olmo2"), default="qwen3")
    p.add_argument("--model", default=None)
    p.add_argument("--model_revision", default=None)
    p.add_argument("--max_context_tokens", type=int, default=None,
                   help="qwen3 default 16384 (raise only; frozen S1 prompts are <4096); "
                        "olmo2 default 4096 with block truncation")
    p.add_argument("--frame_adapter", default=DEFAULT_FRAME_ADAPTER)
    p.add_argument("--frame_adapter_kwargs", default=None, help="JSON passed to configure()")
    p.add_argument("--route_modes", default=None,
                   help="Override adapter route_mode per condition, e.g. S4R=routed,CTRL-SHUF=shuffled")
    p.add_argument("--skip_oversize", action="store_true",
                   help="Opt-in: skip a block that overflows the budget instead of stopping "
                        "(frozen S1 stops; leave off to reproduce 831/1334)")
    p.add_argument("--ctrl_len_mode", choices=CTRL_LEN_MODES, default="append")
    p.add_argument("--frame_render", choices=FRAME_RENDER_MODES, default="adapter",
                   help="adapter: frame-KG v3 layout verbatim (default); numbered: S1-style headers")
    p.add_argument("--s1_candidates", type=int, default=40,
                   help="Ranked S1 candidates cached per question (S1 uses the first top_k)")
    p.add_argument("--reference_s0", type=Path)
    p.add_argument("--reference_s1_strict", type=Path)
    p.add_argument("--reference_s1_full", type=Path)
    p.add_argument("--require_reproduction", action="store_true",
                   help="Refuse non-baseline conditions until S0/S1 (and the S1 config and "
                        "index sizes) reproduce exactly: against --reference_* prediction files "
                        "when they exist, else against the documented counts")
    p.add_argument("--accept_reproduction_mismatch", metavar="REASON", default=None,
                   help="Proceed despite a reproduction mismatch; the reason is recorded")
    p.add_argument("--max_minutes", type=float, default=165.0,
                   help="Stop cleanly (resumable) after this wall time")
    p.add_argument("--limit", type=int, default=None, help="Smoke runs only")
    p.add_argument("--expected_size", type=int, default=1334,
                   help="Required number of test rows (tests use tiny fixtures)")
    p.add_argument("--retrieval_only", action="store_true")
    return p


class Runner:
    def __init__(self, args):
        self.args = args
        self.t0 = time.monotonic()
        self.scorer = None
        self.indexes: dict = {}
        self.adapter = None
        self.s1_cache: dict = {}
        self.frame_cache: dict = {}
        self.og_builders: dict = {}

    # ---------------- setup ----------------
    def out_of_time(self) -> bool:
        return (time.monotonic() - self.t0) / 60 > self.args.max_minutes

    def load_data(self):
        a = self.args
        cfg = json.loads(a.s1_config.read_text(encoding="utf-8"))
        if cfg.get("source") != "train_dev_only":
            raise ValueError("S1 config must originate from train-dev tuning")
        self.top_k = int(cfg["top_k"])
        self.min_sim = float(cfg["min_similarity"])
        self.budget = int(cfg.get("evidence_budget_chars", 5000))
        self.frozen_config = (self.top_k, self.min_sim, self.budget) == (
            FROZEN_S1["top_k"], FROZEN_S1["min_similarity"], FROZEN_S1["evidence_budget_chars"])
        if a.s1_candidates < self.top_k:
            raise ValueError("--s1_candidates must be >= top_k")
        self.questions, self.qmeta = load_test(local_file=str(a.test_file), limit=a.limit,
                                            expected_size=a.expected_size)
        rows = read_rows(a.test_file)
        self.test_pid = {q.question_id: passage_id(rows[i].get("item_description") or "")
                         for i, q in enumerate(self.questions)}
        self.all_test_pids = {passage_id(r.get("item_description") or "") for r in rows}

    def settings_record(self) -> dict:
        a = self.args
        rec = {
            "protocol": "kazculture-final-v1 (identical V2 template; only evidence varies)",
            "model_family": a.model_family, "model": a.model,
            "model_revision": a.model_revision, "max_context_tokens": a.max_context_tokens,
            "truncate_to_fit": a.model_family == "olmo2",
            "test_file": str(a.test_file.resolve()), "test_file_sha256": file_sha256(a.test_file),
            "n_questions": len(self.questions),
            "s1_config": str(a.s1_config.resolve()), "s1_config_sha256": file_sha256(a.s1_config),
            "top_k": self.top_k, "min_similarity": self.min_sim,
            "evidence_budget_chars": self.budget, "frozen_s1_config": self.frozen_config,
            "s1_candidates": a.s1_candidates, "skip_oversize": a.skip_oversize,
            "ctrl_len_mode": a.ctrl_len_mode, "frame_render": a.frame_render,
            "frame_adapter": a.frame_adapter,
            "frame_adapter_kwargs": a.frame_adapter_kwargs,
            "route_modes": self.route_modes,
            "embedding_model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        }
        for s in SETTINGS:
            ix = getattr(a, f"index_{s}")
            if ix is not None:
                rec[f"index_{s}"] = str(ix.resolve())
                meta = ix / "metadata.json"
                rec[f"index_{s}_units_sha256"] = (
                    json.loads(meta.read_text(encoding="utf-8")).get("units_sha256")
                    if meta.exists() else None)
        return rec

    def check_settings(self):
        out = self.args.out
        out.mkdir(parents=True, exist_ok=True)
        rec = self.settings_record()
        path = out / "settings.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old != rec:
                diff = sorted(k for k in set(old) | set(rec) if old.get(k) != rec.get(k))
                raise ValueError(f"Settings changed ({diff}); use a new --out directory")
        else:
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (out / "runs.jsonl").open("a", encoding="utf-8") as fh:
            append_jsonl(fh, {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "host": os.uname().nodename, "conditions": self.conditions,
                "settings": self.settings, "retrieval_only": self.args.retrieval_only,
                "argv": sys.argv[1:],
            })

    def index(self, setting: str):
        if setting not in self.indexes:
            path = getattr(self.args, f"index_{setting}")
            if path is None:
                raise ValueError(f"--index_{setting} is required")
            ix = make_index(path)
            manifest_path = path / "build_manifest.json"
            manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest_path.exists() else {})
            strict = bool(manifest.get("strict_passage_disjoint"))
            if strict != (setting == "strict"):
                raise ValueError(f"Index {path} strict_passage_disjoint={strict} "
                                 f"does not match setting {setting}")
            excluded = set()
            if strict:
                ids = {u["passage_id"] for u in ix.units if u["dataset"] == "KazCulture"}
                if ids & self.all_test_pids:
                    raise ValueError("Strict test-overlap source leakage detected in index")
                excluded = set(self.all_test_pids)  # identical to the frozen S1 run
            self.indexes[setting] = (ix, excluded, {u["unit_id"]: u for u in ix.units})
        return self.indexes[setting]

    def frame_adapter(self):
        if self.adapter is None:
            kwargs = json.loads(self.args.frame_adapter_kwargs) \
                if self.args.frame_adapter_kwargs else None
            self.adapter = load_frame_adapter(self.args.frame_adapter, kwargs=kwargs)
        return self.adapter

    # ---------------- retrieval caches ----------------
    def _cache_rows(self, path: Path) -> dict:
        return {r["question_id"]: r for r in read_jsonl_checkpoint(path)}

    def s1_candidates(self, setting: str, q) -> list[dict]:
        path = self.args.out / "retrieval" / f"{setting}_S1.jsonl"
        if setting not in self.s1_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.s1_cache[setting] = self._cache_rows(path)
        ix, excluded, by_unit = self.index(setting)
        qsha = hashlib.sha256(q.question.encode("utf-8")).hexdigest()
        row = self.s1_cache[setting].get(q.question_id)
        if row is not None:
            if (row["query_sha256"], row["k"], row["min_similarity"]) != (
                    qsha, self.args.s1_candidates, self.min_sim):
                raise ValueError(f"Stale S1 retrieval cache row for {q.question_id}")
            return [{**by_unit[h["unit_id"]], "similarity": h["similarity"]}
                    for h in row["hits"]]
        # Exactly the frozen S1 call (question text only; k only truncates).
        hits = ix.search(q.question, system="S1", k=self.args.s1_candidates,
                         min_similarity=self.min_sim, exclude_passage_ids=excluded,
                         combined=False)
        row = {"question_id": q.question_id, "query_sha256": qsha,
               "k": self.args.s1_candidates, "min_similarity": self.min_sim,
               "hits": [{"unit_id": h["unit_id"], "similarity": h["similarity"]} for h in hits]}
        with path.open("a", encoding="utf-8") as fh:
            append_jsonl(fh, row)
        self.s1_cache[setting][q.question_id] = row
        return hits

    def frame_items(self, setting: str, condition: str, q) -> list[dict]:
        key = (setting, condition)
        path = self.args.out / "retrieval" / f"{setting}_{condition}.jsonl"
        if key not in self.frame_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.frame_cache[key] = self._cache_rows(path)
        mode = self.route_modes[condition]
        row = self.frame_cache[key].get(q.question_id)
        if row is not None:
            if (row["route_mode"], row["top_k"], row["budget_chars"]) != (
                    mode, self.top_k, self.budget):
                raise ValueError(f"Stale frame cache row for {q.question_id}")
            return row["items"]
        items, _ = self.frame_adapter().retrieve(
            q.question, dict(q.options), setting=setting, top_k=self.top_k,
            budget_chars=self.budget, route_mode=mode)
        if setting == "strict":
            leaked = {it["source_id"] for it in items
                      if not it.get("indirect")} & self.all_test_pids
            if leaked:
                raise ValueError(f"Strict leakage: adapter returned test sources {sorted(leaked)[:3]}")
        row = {"question_id": q.question_id, "route_mode": mode, "top_k": self.top_k,
               "budget_chars": self.budget, "items": items}
        with path.open("a", encoding="utf-8") as fh:
            append_jsonl(fh, row)
        self.frame_cache[key][q.question_id] = row
        return items

    # ---------------- evidence ----------------
    def frame_ev(self, items: list[dict]) -> Evidence:
        sep = self.frame_adapter().separator if self.args.frame_render == "adapter" else "\n\n"
        return frame_evidence(items, max_chars=self.budget, skip_oversize=self.args.skip_oversize,
                              render=self.args.frame_render, sep=sep)

    def og_builder(self, setting: str):
        """v2 option-grounded evidence (same builder as the dev harness)."""
        if setting not in self.og_builders:
            from src.evaluation.kazculture_og_evidence import OGEvidenceBuilder
            from src.retrieval import kazculture_frame_retriever as kfr
            adapter = self.frame_adapter()
            ix, _excluded, _ = self.index(setting)
            self.og_builders[setting] = OGEvidenceBuilder(
                ix, kfr._get().kazakh_frames(setting), labels_fn=adapter.labels)
        return self.og_builders[setting]

    def evidence(self, setting: str, condition: str, q) -> Evidence:
        a = self.args
        if condition == "S0":
            return Evidence([], [])
        if condition in OG_CONDITIONS:
            _ix, excluded, _ = self.index(setting)
            ev = self.og_builder(setting).evidence(
                condition, q.question, dict(q.options), excluded,
                top_k=self.top_k, budget=self.budget)
            if setting == "strict":
                leaked = {m["id"] for m in ev.meta if m.get("id")} & self.all_test_pids
                if leaked:
                    raise ValueError(f"Strict leakage in {condition}: {sorted(leaked)[:3]}")
            return ev
        if condition in FRAME_CONDITIONS:
            return self.frame_ev(self.frame_items(setting, condition, q))
        cands = self.s1_candidates(setting, q)
        annotate = None
        if condition == "S1+L":
            adapter = self.frame_adapter()
            annotate = lambda unit: format_labels(adapter.labels(unit["passage_id"]))
        s1 = s1_evidence(cands, top_k=self.top_k, max_chars=self.budget,
                         skip_oversize=a.skip_oversize, annotate=annotate)
        if condition != "CTRL-LEN":
            if condition == "S1+L":
                s1.info["labels_attached"] = sum(
                    "\nProvisional labels (model-generated" in b for b in s1.blocks)
            return s1
        s4r = self.frame_ev(self.frame_items(setting, "S4R", q))
        s1_chars, s4r_chars = len(s1.text()), len(s4r.text())
        ev = ctrl_len_evidence(cands[:a.s1_candidates], s1,
                               extra_chars=max(0, s4r_chars - s1_chars), top_k=self.top_k,
                               base_budget=self.budget, mode=a.ctrl_len_mode,
                               skip_oversize=a.skip_oversize)
        ev.info.update(ctrl_len_s1_chars=s1_chars, ctrl_len_s4r_chars=s4r_chars)
        return ev

    # ---------------- scoring ----------------
    def score(self, q, ev: Evidence) -> tuple[dict, str, dict]:
        scorer = self.scorer
        rendered = scorer.render(q, ev.text())
        extra = {}
        if self.args.model_family == "olmo2":
            limit = self.args.max_context_tokens
            n_tok = scorer.count_tokens(rendered)
            if n_tok > limit:
                k = fit_blocks(len(ev), lambda k: scorer.count_tokens(
                    scorer.render(q, ev.text(k))) <= limit)
                extra = {"truncated": True, "blocks_before_truncation": len(ev),
                         "tokens_before_truncation": n_tok}
                ev.blocks, ev.meta = ev.blocks[:k], ev.meta[:k]
                rendered = scorer.render(q, ev.text())
            else:
                extra = {"truncated": False}
            extra["prompt_tokens"] = scorer.count_tokens(rendered)
        return scorer.score_rendered(rendered), ev.text(), extra

    def output_path(self, setting: str, condition: str) -> Path:
        folder = self.args.out / (CLOSED_BOOK if condition == "S0" else setting)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{condition}.jsonl"

    def reference_for(self, setting: str, condition: str) -> Path | None:
        a = self.args
        ref = (a.reference_s0 if condition == "S0" else
               getattr(a, f"reference_s1_{setting}") if condition == "S1" else None)
        return ref if ref is not None and ref.exists() else None

    def write_reproduction(self, name: str, rep: dict) -> None:
        (self.args.out / f"reproduction_{name}.json").write_text(
            json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"reproduction": name, "exact": rep["exact"],
                          "expected": rep.get("expected"), "observed": rep.get("observed")}),
              flush=True)

    def refuse_or_accept(self, failed: list[str]) -> None:
        """Exit 3 on reproduction failures unless a recorded reason accepts them."""
        reason = self.args.accept_reproduction_mismatch
        if not reason:
            print(f"Reproduction gate: {failed} missing or not exact; refusing downstream "
                  "conditions (pass --accept_reproduction_mismatch REASON to override)",
                  flush=True)
            raise SystemExit(3)
        with (self.args.out / "reproduction_acceptance.jsonl").open("a", encoding="utf-8") as fh:
            append_jsonl(fh, {"utc": datetime.now(timezone.utc).isoformat(),
                              "accepted_failures": failed, "reason": reason,
                              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                              "git_commit": git_commit()})
        print(f"WARNING: accepting reproduction failures {failed}: {reason}", flush=True)

    def preflight_reproduction(self) -> None:
        """Documented S1 config and index sizes (artifacts were regenerated)."""
        if not self.args.require_reproduction:
            return
        cfg = json.loads(self.args.s1_config.read_text(encoding="utf-8"))
        self.write_reproduction("config", check_documented_s1_config(cfg))
        for setting in self.settings:
            if setting == "strict" or getattr(self.args, f"index_{setting}") is not None:
                ix = self.index(setting)[0]
                n = len({u["passage_id"] for u in ix.units
                         if u["system"] == "S1" and u["dataset"] == "KazCulture"})
                exp = DOCUMENTED[f"{setting}_index_passages"]
                self.write_reproduction(f"index_{setting}", {
                    "check": f"{setting}_index_passages", "mode": "documented",
                    "expected": exp, "observed": n, "exact": n == exp})

    def check_reproduction_gate(self, setting: str):
        if not self.args.require_reproduction:
            return
        names = ["config", f"index_{setting}", f"{CLOSED_BOOK}_S0", f"{setting}_S1"]
        failed = []
        for name in names:
            rep = self.args.out / f"reproduction_{name}.json"
            if not (rep.exists() and json.loads(rep.read_text(encoding="utf-8")).get("exact")):
                failed.append(name)
        if failed and failed != self._accepted.get(setting):
            self.refuse_or_accept(failed)
            self._accepted[setting] = failed

    def run_condition(self, setting: str, condition: str) -> str:
        path = self.output_path(setting, condition)
        sett = CLOSED_BOOK if condition == "S0" else setting
        by_id = {q.question_id: q for q in self.questions}
        done = {}
        for r in read_jsonl_checkpoint(path):
            qid = r["question_id"]
            if qid not in by_id or qid in done or r["condition"] != condition \
                    or r["setting"] != sett or r["answer_key"] != by_id[qid].answer_key:
                raise ValueError(f"Corrupted checkpoint {path} at {qid}")
            done[qid] = r
        remaining = [q for q in self.questions if q.question_id not in done]
        if remaining and condition not in ("S0", "S1"):
            self.check_reproduction_gate(setting)
        if remaining and self.scorer is None:
            self.scorer = make_scorer(self.args)
        with path.open("a", encoding="utf-8") as fh:
            for i, q in enumerate(remaining, 1):
                if self.out_of_time():
                    print(f"Time guard: stopping {sett}/{condition} at "
                          f"{len(done)}/{len(self.questions)} (resumable)", flush=True)
                    return "incomplete"
                ev = self.evidence(setting, condition, q)
                scores, text, extra = self.score(q, ev)
                row = prediction_row(q, scores, condition=condition, setting=sett,
                                     evidence_text=text, meta=ev.meta, info=ev.info,
                                     prediction=choose_highest(scores), extra=extra)
                append_jsonl(fh, row)
                done[q.question_id] = row
                if i % 50 == 0:
                    print(f"{sett}/{condition}: {len(done)}/{len(self.questions)} "
                          f"({(time.monotonic() - self.t0) / 60:.1f} min)", flush=True)
        ordered = [done[q.question_id] for q in self.questions]
        summary = condition_summary(ordered)
        summary.update(condition=condition, setting=sett)
        path.with_name(f"{condition}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ref = self.reference_for(setting, condition)
        status = "complete"
        rep = None
        if ref is not None:
            rep = reproduction_check(ordered, read_jsonl_checkpoint(ref, repair=False))
            rep.update(mode="frozen_predictions", reference=str(ref),
                       expected={"correct": rep["reference_correct_on_same_ids"]},
                       observed={"correct": rep["correct"]})
        elif condition in ("S0", "S1") and (
                self.args.require_reproduction or f"{sett}/{condition}" in DOCUMENTED["counts"]):
            rep = documented_count_check(ordered, f"{sett}/{condition}")
            rep["note"] = "frozen prediction file unavailable; checked against documented counts"
        if rep is not None:
            self.write_reproduction(f"{sett}_{condition}", rep)
            if not rep["exact"]:
                status = "reproduction_mismatch"
        print(json.dumps({"condition": condition, "setting": sett, "n": summary["n"],
                          "accuracy_pct": summary["accuracy_pct"]}), flush=True)
        return status

    def retrieval_only(self, setting: str):
        cands_needed = any(c in ("S1", "S1+L", "CTRL-LEN") for c in self.conditions)
        frames = [c for c in self.conditions if c in FRAME_CONDITIONS]
        if "CTRL-LEN" in self.conditions and "S4R" not in frames:
            frames.append("S4R")
        affected, n = [], 0
        for q in self.questions:
            if self.out_of_time():
                print("Time guard during retrieval_only (resumable)", flush=True)
                return
            if cands_needed:
                cands = self.s1_candidates(setting, q)
                ev = s1_evidence(cands, top_k=self.top_k, max_chars=self.budget)
                if ev.info["oversize_break_affected"]:
                    affected.append(q.question_id)
            for c in frames:
                self.frame_items(setting, c, q)
            n += 1
        report = {"setting": setting, "n_questions": n,
                  "oversize_break_affected_n": len(affected) if cands_needed else None,
                  "oversize_break_affected_ids": affected,
                  "definition": "frozen first-overflow break hid >=1 later top-k hit that "
                                "would fit the 5,000-char budget"}
        (self.args.out / f"retrieval_report_{setting}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "oversize_break_affected_ids"}),
              flush=True)

    def main(self) -> int:
        a = self.args
        self.conditions = parse_list(a.conditions, CONDITIONS + OG_CONDITIONS, "conditions")
        self.settings = parse_list(a.settings, SETTINGS, "settings")
        self.route_modes = parse_route_modes(a.route_modes)
        if a.model_family == "olmo2":
            from src.evaluation.kazculture_hf_scorer_olmo import (
                OLMO_MAX_CONTEXT_TOKENS, OLMO_MODEL, OLMO_REVISION)
            if set(self.conditions) - OLMO_ALLOWED["conditions"] or \
                    set(self.settings) - OLMO_ALLOWED["settings"]:
                raise ValueError("OLMo replication is S0,S1,S4R in strict only")
            a.model = a.model or OLMO_MODEL
            a.model_revision = a.model_revision or OLMO_REVISION
            a.max_context_tokens = a.max_context_tokens or OLMO_MAX_CONTEXT_TOKENS
        else:
            a.model = a.model or QWEN[0]
            a.model_revision = a.model_revision or QWEN[1]
            a.max_context_tokens = a.max_context_tokens or 16384
        self.load_data()
        self.check_settings()
        self._accepted: dict = {}
        if a.retrieval_only:
            for s in self.settings:
                self.retrieval_only(s)
            return 0
        self.preflight_reproduction()
        statuses = {}
        jobs = []
        if "S0" in self.conditions:
            jobs.append((self.settings[0], "S0"))
        jobs += [(s, c) for s in self.settings for c in self.conditions if c != "S0"]
        for setting, cond in jobs:
            status = self.run_condition(setting, cond)
            statuses[f"{CLOSED_BOOK if cond == 'S0' else setting}/{cond}"] = status
            if status == "incomplete":
                break
            if status == "reproduction_mismatch" and a.require_reproduction \
                    and not a.accept_reproduction_mismatch:
                print("Reproduction mismatch: stopping before further conditions", flush=True)
                break
        print(json.dumps({"statuses": statuses}), flush=True)
        if any(v == "reproduction_mismatch" for v in statuses.values()) \
                and a.require_reproduction and not a.accept_reproduction_mismatch:
            return 3
        return 0


def main(argv=None) -> int:
    return Runner(build_parser().parse_args(argv)).main()


if __name__ == "__main__":
    raise SystemExit(main())
