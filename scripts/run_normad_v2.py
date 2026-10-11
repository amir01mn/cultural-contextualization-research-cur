"""GPU driver for eval v2 / Evaluation 2 (NormAd). See docs/EVAL_V2_PLAN.md.

Needs ``scripts/prep_normad_v2.py`` to have written ``data/eval_v2/normad/``.
Loads the model and the RetrievalContext once, tunes tau for the gated systems
on dev, then runs every condition on test (one batched generate per condition).

GPU (inside a SLURM job, offline):
    python scripts/run_normad_v2.py --provider vllm

CPU smoke test (fake backend, TF-IDF, tiny limit, separate out dir):
    export EMBEDDING_BACKEND=tfidf
    python scripts/run_normad_v2.py --smoke
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core import config  # noqa: E402
from src.core.utils import get_logger  # noqa: E402

LOGGER = get_logger("run_normad_v2")

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "eval_v2" / "normad"
OUT_DIR = REPO / "data" / "outputs" / "eval_v2" / "normad"
SMOKE_OUT_DIR = REPO / "data" / "outputs" / "eval_v2" / "normad_smoke"


class FakeBackend:
    """CPU stand-in for smoke runs: answers deterministically from the prompt."""

    name = "fake"
    model = "fake-normad"

    def __init__(self):
        self.calls = 0

    def status(self):
        return {"available": True, "message": "fake backend"}

    def generate(self, prompts, system_prompt=None, *, temperature=None, max_tokens=None):
        assert max_tokens, "explicit max_tokens required"
        self.calls += 1
        labels = ("Yes", "No", "Neither")
        return [f"Brief reasoning.\nANSWER: {labels[len(p) % 3]}" for p in prompts]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run NormAd (eval v2, Evaluation 2).")
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument("--smoke", action="store_true", help="fake backend, tiny limit, smoke out dir")
    p.add_argument("--limit", type=int, default=None, help="cap test stories (debug)")
    p.add_argument("--dev_limit", type=int, default=None, help="cap dev stories (debug)")
    p.add_argument("--conditions", default=None, help="comma-separated subset of conditions")
    p.add_argument("--taus", default=None,
                   help="JSON dict of frozen taus, e.g. '{\"S4g\": 0.25}' (skips tuning for those)")
    p.add_argument("--top_k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--kg_v3", default=None, help="KG v3 dir (frames + local pathways)")
    p.add_argument("--drop_eval_leak_triples", action="store_true")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    if args.smoke:
        os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")
        config.EMBEDDING_BACKEND = os.environ["EMBEDDING_BACKEND"]
        args.limit = args.limit or 12
        args.dev_limit = args.dev_limit or 6
    out_dir = Path(args.out_dir or (SMOKE_OUT_DIR if args.smoke else OUT_DIR))
    if not args.smoke and (out_dir / "summary.json").exists():
        LOGGER.error("%s already has results; refusing to overwrite. Pass a new --out_dir.", out_dir)
        sys.exit(2)

    from eval_v2_common import build_retrieval_context
    from src.evaluation import normad_eval as ne

    t_all = time.time()
    stage_times: dict[str, float] = {}
    data_dir = Path(args.data_dir)
    test = _read_jsonl(data_dir / "stories_test.jsonl")
    dev = _read_jsonl(data_dir / "stories_dev.jsonl")
    if args.limit:
        test = test[: args.limit]
    if args.dev_limit:
        dev = dev[: args.dev_limit]
    assert not ({s["id"] for s in test} & {s["id"] for s in dev}), "dev/test overlap"
    conditions = ([c.strip() for c in args.conditions.split(",") if c.strip()]
                  if args.conditions else list(ne.ALL_CONDITIONS))
    bad = [c for c in conditions if c not in ne.KNOWN_CONDITIONS]
    if bad:
        LOGGER.error("Unknown conditions %s; valid: %s", bad, ne.KNOWN_CONDITIONS)
        sys.exit(2)
    LOGGER.info("NormAd: %d test, %d dev stories; conditions=%s", len(test), len(dev), conditions)

    t0 = time.time()
    ctx = build_retrieval_context(kg_v3_dir=Path(args.kg_v3) if args.kg_v3 else None,
                                  drop_eval_leak_triples=args.drop_eval_leak_triples)
    stage_times["retrieval_context_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    if args.smoke:
        backend = FakeBackend()
    else:
        from src.llm.llm_backend import get_backend

        backend = get_backend(args.provider, model=args.model)
        status = backend.status()
        if not status.get("available"):
            LOGGER.error("Backend unavailable: %s", status.get("message"))
            sys.exit(2)
        if hasattr(backend, "warmup"):
            backend.warmup()
    stage_times["model_load_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    run_meta = {
        "provider": "fake" if args.smoke else args.provider,
        "model": getattr(backend, "model", None),
        "smoke": args.smoke,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "data_dir": str(data_dir),
        "prep_meta": json.loads((data_dir / "prep_meta.json").read_text()).get("embedding_method")
        if (data_dir / "prep_meta.json").exists() else None,
        "kg_v3": args.kg_v3, "drop_eval_leak_triples": args.drop_eval_leak_triples,
        "title": args.title,
    }
    kwargs = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    summary = ne.run_normad(backend, ctx, test, out_dir, dev_stories=dev, conditions=conditions,
                            taus=json.loads(args.taus) if args.taus else None,
                            top_k=args.top_k, run_meta=run_meta, **kwargs)
    stage_times["normad_stage_s"] = round(time.time() - t0, 1)
    stage_times["total_s"] = round(time.time() - t_all, 1)
    (out_dir / "wall_times.json").write_text(json.dumps(stage_times, indent=2))
    LOGGER.info("Wall times: %s", stage_times)
    for cond, m in summary["subsets"]["all"]["conditions"].items():
        LOGGER.info("  %-32s acc=%.4f macroF1=%.4f gap=%s", cond, m["accuracy"], m["macro_f1"],
                    m.get("gap_closed"))
    LOGGER.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
