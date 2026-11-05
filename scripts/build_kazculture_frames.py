#!/usr/bin/env python3
"""Build KazCulture frames v3k (CultureBank KG v3 method). See docs/KAZCULTURE_FRAMES_V3K.md.

Stages (each resumable or refuses to overwrite):

    label   Qwen3-32B labels norm / norm type / values per passage      [2 GPU]
            --shard i/N labels one round-robin shard into labels/shard_*.jsonl
    merge   shards -> labels.jsonl (one row per passage, gaps are errors)
    kg      KazCulture + CultureBank v3 frames -> norm-type clusters, culture
            profiles and BGE-M3 frame index, for full/ and strict/        [1 GPU]
            (strict drops the test-overlap passages via the S1 index's
            select_passages, before clustering)

Writes only under --out (default data/kazculture_pipeline/frames_v3k/).

    python scripts/build_kazculture_frames.py --stages label --shard 0/4
    python scripts/build_kazculture_frames.py --stages merge,kg
    python scripts/build_kazculture_frames.py --stages label,merge,kg --fake_labeler \\
        --embedding_backend tfidf --out /tmp/x     # CPU smoke, fake labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BASE = ROOT / "data" / "kazculture_pipeline"
STAGES = ("label", "merge", "kg")


class FakeLabeler:
    """Deterministic stand-in for the LLM (smoke runs and tests only)."""

    model = "fake-labeler"

    def generate(self, prompts_, system_prompt=None, *, temperature=None, max_tokens=None):
        out = []
        for p in prompts_:
            topic = p.split("topic: ", 1)[-1].split(")", 1)[0].lower()
            if "descriptive" in p.split("Fill in three fields")[0]:
                out.append('{"norm": "none", "norm_type": "none", "values": []}')
                continue
            vals = ["tradition"] if len(p) % 2 else ["benevolence (caring)", "face"]
            out.append(json.dumps({"norm": f"Kazakhs follow {topic} customs.",
                                   "norm_type": f"{topic} customs", "values": vals}))
        return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_shard(text: str) -> tuple[int, int]:
    i, n = (int(x) for x in text.split("/"))
    return i, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stages", default="label,merge,kg")
    p.add_argument("--passages", type=Path, default=BASE / "corpus" / "passages.jsonl")
    p.add_argument("--overlap_file", type=Path, default=BASE / "corpus" / "test_overlap.json")
    p.add_argument("--culturebank_kg", type=Path, default=ROOT / "data" / "kg_v3" / "full")
    p.add_argument("--out", type=Path, default=BASE / "frames_v3k")
    p.add_argument("--shard", default="0/1", help="i/N for the label stage")
    p.add_argument("--n_shards", type=int, default=None,
                   help="merge: expected N (default: the shard files present)")
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--max_model_len", type=int, default=6144)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--embedding_backend", default="bge-m3", choices=["bge-m3", "tfidf"])
    p.add_argument("--fake_labeler", action="store_true", help="smoke/test only")
    p.add_argument("--limit", type=int, default=None, help="first N passages (smoke)")
    p.add_argument("--overwrite_kg", action="store_true",
                   help="replace an existing full/ strict/ build in --out")
    args = p.parse_args(argv)

    from src.construction import kazculture_frames_v3k as kf
    from src.core import config
    from src.core.embedding_utils import backend_name, embed_texts
    from src.core.utils import get_logger
    from src.loaders.kazculture_corpus import read_rows

    log = get_logger("build_kazculture_frames")
    config.EMBEDDING_BACKEND = args.embedding_backend
    os.environ["EMBEDDING_BACKEND"] = args.embedding_backend
    model = args.model or ("fake-labeler" if args.fake_labeler else kf.LABEL_MODEL)
    revision = args.revision or ("none" if args.fake_labeler else kf.LABEL_REVISION)
    stages = [s for s in STAGES if s in {x.strip() for x in args.stages.split(",")}]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    passages = read_rows(args.passages)
    passages.sort(key=lambda r: r["passage_id"])
    if args.limit:
        passages = passages[:args.limit]
    if len({r["passage_id"] for r in passages}) != len(passages):
        raise ValueError("duplicate passage ids in the corpus")
    timings: dict = {}

    if "label" in stages:
        t0 = time.time()
        i, n = parse_shard(args.shard)
        part = kf.shard(passages, i, n)
        tokenizer = None
        if args.fake_labeler:
            backend = FakeLabeler()
        else:
            from src.llm.llm_backend import VLLMBackend

            backend = VLLMBackend(
                model=model, revision=revision, max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                chat_template_kwargs={"enable_thinking": False})
            backend.warmup()
            tokenizer = backend._tokenizer
            if tokenizer is None:
                raise RuntimeError("vLLM tokenizer unavailable; refusing char-based truncation")
        rows = kf.label_passages(
            backend, part, kf.shard_path(out, i, n), model=model, revision=revision,
            tokenizer=tokenizer, max_model_len=None if args.fake_labeler else args.max_model_len,
            batch_size=args.batch_size)
        log.info("shard %d/%d: %d rows", i, n, len(rows))
        timings[f"label_{i}_of_{n}"] = round(time.time() - t0, 1)

    if "merge" in stages:
        t0 = time.time()
        paths = sorted((out / "labels").glob("shard_*_of_*.jsonl"))
        ns = {int(p.stem.split("_of_")[1]) for p in paths}
        if len(ns) != 1:
            raise ValueError(f"shard files from different N: {sorted(ns)}")
        n = ns.pop()
        if args.n_shards is not None and args.n_shards != n:
            raise ValueError(f"expected {args.n_shards} shards, files say {n}")
        if len(paths) != n:
            raise ValueError(f"{len(paths)} of {n} shard files present")
        merged, stats = kf.merge_label_shards(paths, passages, model=model, revision=revision)
        with (out / "labels.jsonl").open("w", encoding="utf-8") as fh:
            for r in merged:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "labels_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        log.info("merge: %s", stats)
        timings["merge"] = round(time.time() - t0, 1)

    if "kg" in stages:
        t0 = time.time()
        from build_kazculture_s1_index import select_passages
        from src.construction.frame_builder import save_frame_kg
        from src.retrieval.frame_retriever import build_frame_index

        for name in ("full", "strict"):
            if (out / name / "kg_meta.json").exists() and not args.overwrite_kg:
                raise FileExistsError(f"{out / name} already built; use a new --out "
                                      "or --overwrite_kg")
        label_rows = read_rows(out / "labels.jsonl")
        overlap = json.loads(args.overlap_file.read_text(encoding="utf-8"))
        if args.limit:   # smoke: keep only overlap ids inside the subset
            have = {r["passage_id"] for r in passages}
            overlap = {**overlap, "exclude_train_passage_ids":
                       [x for x in overlap["exclude_train_passage_ids"] if x in have]}
        kept, excluded = select_passages(passages, strict=True, overlap=overlap)
        frames = kf.frames_from_passages(passages)
        label_stats = kf.apply_label_rows(frames, label_rows)
        cb_frames, cb_stats = kf.load_culturebank_frames(args.culturebank_kg)
        log.info("labels: %s; CultureBank: %s", label_stats, cb_stats)

        def embed_fn(texts):
            return embed_texts(texts, fit=True)

        metas = {}
        for name, excl in (("full", set()), ("strict", set(excluded))):
            kg = kf.build_kazculture_kg(frames, cb_frames, embed_fn, exclude_ids=excl)
            save_frame_kg(kg, out / name)
            build_frame_index(out / name)
            metas[name] = kg["meta"]
            if name == "full":
                with (out / "frames.jsonl").open("w", encoding="utf-8") as fh:
                    for f in kg["frames"]:
                        if f.culture == kf.KAZAKH:
                            fh.write(json.dumps(f.model_dump(), ensure_ascii=False) + "\n")
            log.info("kg %s: %s", name, kg["meta"])
        if len(kept) != metas["strict"]["n_kazculture_frames"]:
            raise AssertionError("strict KG frame count differs from select_passages")
        manifest = {
            "schema": kf.SCHEMA_VERSION,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label_model": model, "label_revision": revision, "thinking": False,
            "label_prompt_sha256": kf.prompt_sha256(),
            "label_max_tokens": kf.LABEL_MAX_TOKENS,
            "practice_max_tokens": kf.PRACTICE_MAX_TOKENS,
            "label_status": label_stats,
            "embedding_backend": backend_name(),
            "passages": str(args.passages), "passages_sha256": sha256_file(args.passages),
            "n_passages": len(passages), "limit": args.limit,
            "overlap_file": str(args.overlap_file),
            "overlap_sha256": sha256_file(args.overlap_file),
            "strict_excluded_passages": len(excluded), "strict_kept_passages": len(kept),
            "culturebank_kg": str(args.culturebank_kg),
            "culturebank_frames_sha256": sha256_file(args.culturebank_kg / "frames.jsonl"),
            **cb_stats,
            "kg": metas,
            "outputs": {"frames": "frames.jsonl (KazCulture frames, full-KG norm types)",
                        "labels": "labels.jsonl", "full": "full/", "strict": "strict/",
                        "norm_types": "{full,strict}/norm_types.jsonl",
                        "culture_profiles": "{full,strict}/culture_profiles.json"},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                                           + "\n", encoding="utf-8")
        timings["kg"] = round(time.time() - t0, 1)

    log.info("Done %s in %s -> %s", stages, timings, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
