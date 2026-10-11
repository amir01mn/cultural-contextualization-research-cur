#!/usr/bin/env python
"""GPU driver for Evaluation 3 (intrinsic KG quality) + the open-ended judge.

Loads the judge (``microsoft/phi-4``, pre-registered in docs/EVAL_V2_PLAN.md)
once and runs the requested stages of ``src/evaluation/intrinsic_eval.py`` over
the prep written by ``scripts/prep_intrinsic_v2.py``.

    python scripts/run_intrinsic_v2.py --provider vllm \
        --stages facts,retention,triple_audit,pathway_audit,open_judge

``open_judge`` waits (polling every ``--poll_seconds``) for
``data/outputs/eval_v2/culturebank/open_answers.DONE`` up to ``--wait_minutes``;
on timeout it logs, records the status and exits 0 with the other results written.

``--smoke`` runs every stage on CPU with a deterministic fake judge, a tiny limit,
a synthetic open_answers file, and writes only to ``data/outputs/eval_v2/intrinsic_smoke/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.utils import ensure_dir, get_logger, read_jsonl, write_jsonl  # noqa: E402
from src.evaluation import intrinsic_eval as ie  # noqa: E402

LOGGER = get_logger("run_intrinsic_v2")

DATA_DIR = ROOT / "data" / "eval_v2" / "intrinsic"
OUT_DIR = ROOT / "data" / "outputs" / "eval_v2" / "intrinsic"
CB_OUT_DIR = ROOT / "data" / "outputs" / "eval_v2" / "culturebank"
SMOKE_DIR = ROOT / "data" / "outputs" / "eval_v2" / "intrinsic_smoke"
ALL_STAGES = ["facts", "retention", "triple_audit", "pathway_audit", "open_judge"]
JUDGE_MODEL = "microsoft/phi-4"
PROMPT_BUDGET = 12000


class FakeJudgeBackend:
    """Deterministic stand-in for the judge (smoke tests only)."""

    name = "fake"
    model = "fake-judge"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def _bit(prompt: str) -> bool:
        return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 2 == 0

    def generate(self, prompts, system_prompt=None, *, temperature=None, max_tokens=None):
        assert max_tokens, "every judge call must pass an explicit max_tokens"
        self.calls.append({"n": len(prompts), "max_tokens": max_tokens,
                           "temperature": temperature})
        out = []
        for p in prompts:
            y = "yes" if self._bit(p) else "no"
            if "atomic facts" in p:
                m = re.search(r'"""(.*?)"""', p, flags=re.S)
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", m.group(1) if m else "")
                         if s.strip()]
                out.append(json.dumps(sents[:6] or ["Fact."]))
            elif '"faithful"' in p:
                out.append('```json\n{"faithful": "%s", "layers_correct": "yes", '
                           '"culture_specific": "no", "stereotyping": "no"}\n```' % y)
            elif '"coherent"' in p:
                out.append('{"coherent": "%s", "culture_consistent": "yes", "correct": "%s"}'
                           % (y, y))
            else:
                out.append(f"{y.capitalize()}.")
        return out

    def generate_one(self, prompt, system_prompt=None, **kw):
        return self.generate([prompt], system_prompt, **kw)[0]

    def status(self):
        return {"available": True, "model": self.model}


def _synthetic_open_answers(path: Path, records: list[dict]) -> None:
    rows = []
    for i, r in enumerate(records[:6]):
        for system in ("S0_no_kg", "S1_raw_text_rag", "S3_layered_kg",
                       "S4_abstraction_refinement_kg", "S4g"):
            rows.append({"qid": f"q{i}", "system": system,
                         "cultural_group": r.get("cultural_group", ""),
                         "question": "What should I know?",
                         "answer": f"{system} answer {i}: {r['eval_whole_desc'][:80]}",
                         "gold_desc": r["eval_whole_desc"]})
    write_jsonl(path, rows)
    path.with_suffix(".DONE").write_text("smoke\n")


def _wait_for(flag: Path, minutes: float, poll_seconds: float) -> bool:
    deadline = time.time() + minutes * 60
    while True:
        if flag.exists():
            return True
        if time.time() >= deadline:
            return False
        LOGGER.info("Waiting for %s (%.0f min left)", flag, (deadline - time.time()) / 60)
        time.sleep(min(poll_seconds, max(1.0, deadline - time.time())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default="vllm", choices=["vllm", "ollama", "fake"])
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--open_answers", type=Path, default=CB_OUT_DIR / "open_answers.jsonl")
    ap.add_argument("--open_out_dir", type=Path, default=None)
    ap.add_argument("--wait_minutes", type=float, default=100.0)
    ap.add_argument("--poll_seconds", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=None, help="per-stage item cap")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    # KG v3: audit other pathway files; v3 open-ended comparison family.
    ap.add_argument("--pathway_files", default=None,
                    help="comma list of name=path JSONL files to audit (one pathway_audit each)")
    ap.add_argument("--v3_comparisons", action="store_true")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    if args.smoke:
        args.provider = "fake"
        args.limit = args.limit or 4
        args.out_dir = args.out_dir or SMOKE_DIR
        args.open_out_dir = args.open_out_dir or (SMOKE_DIR / "culturebank")
        args.n_boot = min(args.n_boot, 200)
    out_dir = ensure_dir(args.out_dir or OUT_DIR)
    open_out_dir = args.open_out_dir or CB_OUT_DIR

    def _maybe(name):
        f = args.data_dir / name
        return list(read_jsonl(f)) if f.exists() else []
    records, triples, pathways = (_maybe("records.jsonl"), _maybe("triples.jsonl"),
                                  _maybe("pathways.jsonl"))
    LOGGER.info("Prep: %d records, %d triples, %d pathways", len(records), len(triples),
                len(pathways))

    if args.smoke:  # exercise pathway evidence: records that have pathways first
        records.sort(key=lambda r: not r.get("pathways"))
        args.open_answers = ensure_dir(open_out_dir) / "open_answers.jsonl"
        _synthetic_open_answers(args.open_answers, records)

    if args.provider == "fake":
        backend = FakeJudgeBackend()
    else:
        from src.llm.llm_backend import get_backend

        if args.model != JUDGE_MODEL:
            LOGGER.warning("Judge model %s differs from the pre-registered %s",
                           args.model, JUDGE_MODEL)
        backend = get_backend(args.provider, model=args.model)
        if hasattr(backend, "warmup"):
            t = time.time()
            LOGGER.info("Loading judge %s: %s", args.model, backend.warmup())
            LOGGER.info("Judge load took %.0fs", time.time() - t)

    meta = {"judge_model": getattr(backend, "model", args.model), "provider": args.provider,
            "data_dir": str(args.data_dir), "limit": args.limit, "smoke": args.smoke}
    cfg = {"limit": args.limit}
    results: dict = {}
    timings: dict = {}

    def _run(stage, fn):
        t = time.time()
        LOGGER.info("=== stage %s ===", stage)
        results[stage] = fn()
        timings[stage] = round(time.time() - t, 1)
        LOGGER.info("stage %s done in %.1fs", stage, timings[stage])
        ie.write_summary(out_dir, results, {**meta, "stage_seconds": timings})

    if "facts" in stages:
        _run("facts", lambda: ie.run_facts(backend, records, out_dir, **cfg))
    if "retention" in stages:
        facts_path = out_dir / "facts.jsonl"
        if not facts_path.exists():
            LOGGER.error("retention needs %s (run the facts stage first); skipping", facts_path)
        else:
            facts_by_id = {r["instance_id"]: r["facts"] for r in read_jsonl(facts_path)}
            _run("retention", lambda: ie.run_retention(backend, records, facts_by_id,
                                                       out_dir, **cfg))
    if "triple_audit" in stages:
        _run("triple_audit", lambda: ie.run_triple_audit(backend, triples, out_dir, **cfg))
    if "pathway_audit" in stages and args.pathway_files:
        for spec in args.pathway_files.split(","):
            name, path = spec.split("=", 1)
            pws = list(read_jsonl(Path(path)))
            sub = ensure_dir(out_dir / f"pathway_audit_{name}")
            _run(f"pathway_audit_{name}",
                 lambda pws=pws, sub=sub: ie.run_pathway_audit(backend, pws, sub, **cfg))
    elif "pathway_audit" in stages:
        _run("pathway_audit", lambda: ie.run_pathway_audit(backend, pathways, out_dir, **cfg))

    if "open_judge" in stages:
        done_flag = args.open_answers.with_suffix(".DONE")
        if not _wait_for(done_flag, args.wait_minutes, args.poll_seconds):
            LOGGER.warning("open_judge: %s did not appear within %.0f min; skipping "
                           "(other results are written).", done_flag, args.wait_minutes)
            results["open_judge"] = {"status": f"timed out waiting for {done_flag.name}"}
            ie.write_summary(out_dir, results, {**meta, "stage_seconds": timings})
        else:
            answers = list(read_jsonl(args.open_answers))
            LOGGER.info("open_judge: %d answers from %s", len(answers), args.open_answers)
            comps = ie.V3_COMPARISONS if args.v3_comparisons else ie.PRIMARY_COMPARISONS
            _run("open_judge", lambda: ie.run_open_judge(
                backend, answers, open_out_dir, n_boot=args.n_boot, comparisons=comps, **cfg))

    n_prompts = sum(c["n"] for c in getattr(backend, "calls", []))
    LOGGER.info("Stage wall times (s): %s", timings)
    if n_prompts:
        LOGGER.info("Judge prompts issued: %d (budget ~%d)", n_prompts, PROMPT_BUDGET)
    LOGGER.info("Summary: %s", out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
