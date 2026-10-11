"""GPU driver for eval v2, Evaluation 1 (CultureBank grounded QA).

Loads the answering model and the August KG RetrievalContext once, then runs the
stages of ``src/evaluation/culturebank_grounded.py`` in a fixed order
(paraphrase -> open_gen -> mcq -> heldout -> retrieval_metrics). open_gen runs
early because the judge job polls for ``open_answers.DONE``.

Needs the prep outputs in data/eval_v2/culturebank/ (scripts/prep_culturebank_eval_v2.py).

    # full run (GPU node, via SLURM)
    python scripts/run_culturebank_eval_v2.py --provider vllm \\
        --stages paraphrase,open_gen,mcq,heldout,retrieval_metrics

    # CPU smoke test: fake backend, TF-IDF, tiny subset, separate output dir
    python scripts/run_culturebank_eval_v2.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STAGE_ORDER = ["paraphrase", "open_gen", "mcq", "heldout", "retrieval_metrics"]


class SmokeBackend:
    """Deterministic stand-in for vLLM (smoke mode only)."""

    name = "smoke-fake"
    model = "smoke-fake"

    def __init__(self):
        self.n_prompts = 0

    def generate(self, prompts, system_prompt=None, *, temperature=None, max_tokens=None):
        from src.llm import prompts as P

        assert max_tokens is not None, "eval v2 must pass explicit max_tokens"
        self.n_prompts += len(prompts)
        out = []
        for i, p in enumerate(prompts):
            if system_prompt == P.CB_PARAPHRASE_SYSTEM:
                n = p.count("\n", 0, p.find("\n\nReturn")) if "\n\nReturn" in p else 4
                lines = [ln for ln in p.split("\n") if ln[:2] in {"1:", "2:", "3:", "4:"}]
                # Leave one option empty on every 5th prompt to exercise the fallback.
                out.append("\n".join(
                    f"{ln[:2]} {'' if (i % 5 == 0 and j == 1) else 'Paraphrased: ' + ln[3:]}"
                    for j, ln in enumerate(lines[:max(n, 4)])))
            elif system_prompt == P.CB_OPEN_SYSTEM:
                out.append("People there usually follow local custom; ask politely.")
            else:
                out.append("Reasoning...\nANSWER: " + "ABCD"[(len(p) + i) % 4])
        return out

    def generate_one(self, prompt, system_prompt=None, **kw):
        return self.generate([prompt], system_prompt, **kw)[0]

    def status(self):
        return {"available": True, "message": "smoke fake backend"}


def _smoke_subset(data: dict, limit: int) -> dict:
    test = data["test"]
    held = [it for it in test if it.get("heldout")][:limit]
    opn = [it for it in test if it.get("open_ended") and it not in held][:limit]
    rest = [it for it in test if it not in held and it not in opn][:limit]
    return {"dev": data["dev"][:limit], "test": held + opn + rest, "masks": data["masks"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Eval v2 / CultureBank grounded QA driver.")
    p.add_argument("--provider", default="vllm", choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument("--stages", default=",".join(STAGE_ORDER))
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--smoke", action="store_true",
                   help="Fake backend + TF-IDF + tiny subset, output to culturebank_smoke/.")
    p.add_argument("--limit", type=int, default=None,
                   help="Items per group (dev / heldout / open / other test). Smoke default 6.")
    args = p.parse_args(argv)

    if args.smoke:
        os.environ["EMBEDDING_BACKEND"] = "tfidf"

    from src.core import config
    from src.core.utils import get_logger
    from src.evaluation import culturebank_grounded as cg
    from src.loaders.culturebank_eval_loader import read_jsonl_items
    from eval_v2_common import EVAL_V2_DATA_DIR, EVAL_V2_OUT_DIR, build_retrieval_context

    log = get_logger("run_culturebank_eval_v2")
    if args.smoke:
        config.EMBEDDING_BACKEND = "tfidf"

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGE_ORDER]
    if bad:
        log.error("Unknown stages %s; valid: %s", bad, STAGE_ORDER)
        return 2
    stages = [s for s in STAGE_ORDER if s in stages]
    top_k = args.top_k or config.RETRIEVAL_TOP_K

    data_dir = Path(args.data_dir) if args.data_dir else EVAL_V2_DATA_DIR / "culturebank"
    out_dir = Path(args.out_dir) if args.out_dir else (
        EVAL_V2_OUT_DIR / ("culturebank_smoke" if args.smoke else "culturebank"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:  # always start the smoke run clean
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
    para_cache = (out_dir if args.smoke else data_dir) / "mcq_paraphrased.jsonl"

    data = {
        "dev": read_jsonl_items(data_dir / "dev.jsonl"),
        "test": read_jsonl_items(data_dir / "test.jsonl"),
        "masks": json.loads((data_dir / "masks.json").read_text()),
    }
    limit = args.limit if args.limit is not None else (6 if args.smoke else None)
    if limit:
        data = _smoke_subset(data, limit)
    log.info("Items: dev=%d test=%d (heldout=%d open=%d); masks near=%d far=%d",
             len(data["dev"]), len(data["test"]),
             sum(bool(i.get("heldout")) for i in data["test"]),
             sum(bool(i.get("open_ended")) for i in data["test"]),
             len(data["masks"]["near"]), len(data["masks"]["far"]))

    timings: dict = {}
    t0 = time.time()
    ctx = build_retrieval_context()
    timings["load_context"] = round(time.time() - t0, 1)
    log.info("RetrievalContext ready in %.0fs", timings["load_context"])

    if args.smoke:
        backend = SmokeBackend()
    else:
        from src.llm.llm_backend import get_backend

        backend = get_backend(args.provider, model=args.model)
        status = backend.status()
        if not status.get("available"):
            log.error("Backend unavailable: %s", status.get("message"))
            return 2
        if hasattr(backend, "warmup"):
            t0 = time.time()
            backend.warmup()
            timings["model_load"] = round(time.time() - t0, 1)

    cache: dict = {}
    leak_log: dict = {}
    common = dict(top_k=top_k, cache=cache, leak_log=leak_log)
    results = {}
    for stage in stages:
        t0 = time.time()
        log.info("=== stage %s ===", stage)
        paraphrases = cg.load_paraphrases(para_cache)
        if stage == "paraphrase":
            results[stage] = cg.run_paraphrase(backend, ctx, data, out_dir, cache_path=para_cache)
        elif stage == "open_gen":
            results[stage] = cg.run_open_gen(backend, ctx, data, out_dir,
                                             paraphrases=paraphrases, **common)
        elif stage == "mcq":
            results[stage] = cg.run_mcq(backend, ctx, data, out_dir,
                                        paraphrases=paraphrases, **common)
        elif stage == "heldout":
            results[stage] = cg.run_heldout(backend, ctx, data, out_dir,
                                            paraphrases=paraphrases, **common)
        elif stage == "retrieval_metrics":
            results[stage] = cg.run_retrieval_metrics(backend, ctx, data, out_dir, **common)
        timings[stage] = round(time.time() - t0, 1)
        log.info("stage %s done in %.0fs", stage, timings[stage])

    total_leaks = sum(leak_log.values())
    (out_dir / "leakage_guard.json").write_text(
        json.dumps({"total": total_leaks, "counts": leak_log}, indent=2), encoding="utf-8")
    if total_leaks:
        log.error("LEAKAGE GUARD FAILED: %d evidence strings contain eval text: %s",
                  total_leaks, {k: v for k, v in leak_log.items() if v})
    else:
        log.info("Leakage guard: 0 leaks over %d checks", len(leak_log))

    run_meta = {
        "provider": "smoke-fake" if args.smoke else args.provider,
        "model": getattr(backend, "model", None), "top_k": top_k, "smoke": args.smoke,
        "limit": limit, "stages": stages, "stage_seconds": timings,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": {"paraphrase": cg.PARAPHRASE_MAX_TOKENS, "mcq": cg.MCQ_MAX_TOKENS,
                       "open": cg.OPEN_MAX_TOKENS},
        "data_dir": str(data_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    cg.write_summary(out_dir, run_meta=run_meta, n_boot=200 if args.smoke else args.n_boot)
    log.info("Stage wall times (s): %s", timings)
    log.info("Outputs in %s", out_dir)
    return 3 if total_leaks else 0


if __name__ == "__main__":
    sys.exit(main())
