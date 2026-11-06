#!/usr/bin/env python3
"""Turn held-out KazCulture *train* questions into test-format 4-option MCQ.

The train split has free-text answers only (its a/b/c/d/answer_label columns are
empty strings). Test options are short (mean 14 chars), same-form alternatives
such as "Шыбыртқымен айдайды." / "Арқанмен айдайды." / ... For v2 development we
ask Qwen3-32B (non-thinking, temperature 0) to (1) shorten the reference answer to
that form and (2) write three plausible but wrong alternatives of the same form.
Option order is shuffled deterministically per row. The source passage is not
shown to the generator. This is a disclosed synthetic proxy for the test format;
it never touches the test split.

Rows used for tuning/gating in v1 are excluded (see eval_kazculture_v2_dev.py).

Output (JSONL, one per kept row): train_row_id, question, a, b, c, d, answer_label,
category_english, passage_id, reference_answer, generator_raw.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_corpus import passage_id, read_rows  # noqa: E402

QWEN = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
BASE = ROOT / "data" / "kazculture_pipeline"
RETRIEVAL_TUNER_ROWS = 400
LETTERS = ("a", "b", "c", "d")

SYSTEM = ("You write multiple-choice options for a Kazakh cultural knowledge quiz. "
          "Reply with JSON only.")
USER = """Question (Kazakh): {question}
Reference answer (Kazakh): {answer}

Write, in Kazakh:
1. "answer": the reference answer shortened to a few words, in the same grammatical form
   a short quiz option would take (e.g. "Шыбыртқымен айдайды." or "Сүйінші").
2. "distractors": exactly three different options of the same form and similar length that
   are plausible for this question but WRONG according to the reference answer.

Reply as JSON: {{"answer": "...", "distractors": ["...", "...", "..."]}}"""


def norm(s: str) -> str:
    return re.sub(r"[\s\.\,\!\?«»\"']+", " ", s).strip().lower()


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ans = str(d.get("answer") or "").strip()
    dis = [str(x).strip() for x in (d.get("distractors") or []) if str(x).strip()]
    opts = [ans] + dis
    if not ans or len(dis) != 3 or len({norm(o) for o in opts}) != 4:
        return None
    if any(len(o) > 120 for o in opts):
        return None
    return {"answer": ans, "distractors": dis}


def load_used_ids(paths: list[Path]) -> set[int]:
    used = set(range(RETRIEVAL_TUNER_ROWS))
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        used |= set(d["sample_train_row_ids"] if isinstance(d, dict) else d)
    return used


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--train_file", type=Path, default=BASE / "corpus" / "train.parquet")
    p.add_argument("--used_ids", type=Path, nargs="*", default=[
        BASE / "frames_v3k" / "routing_tuning_train_row_ids.json",
        BASE / "evaluation" / "traindev_gate_v1" / "settings.json"])
    p.add_argument("--n_generate", type=int, default=1600)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=BASE / "v2_dev" / "dev_mcq_v1.jsonl")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    p.add_argument("--fake", action="store_true", help="CPU smoke test without an LLM")
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; dev sets are never overwritten")

    rows = read_rows(a.train_file)
    used = load_used_ids(a.used_ids)
    eligible = [i for i, r in enumerate(rows) if i not in used
                and str(r.get("question") or "").strip() and str(r.get("answer") or "").strip()]
    ids = sorted(random.Random(a.seed).sample(eligible, min(a.n_generate, len(eligible))))
    prompts = [USER.format(question=rows[i]["question"].strip(), answer=rows[i]["answer"].strip())
               for i in ids]

    t0 = time.time()
    if a.fake:
        raws = [json.dumps({"answer": f"ans{i}", "distractors": [f"x{i}", f"y{i}", f"z{i}"]})
                for i in ids]
    else:
        from src.llm.llm_backend import VLLMBackend
        backend = VLLMBackend(model=QWEN[0], revision=QWEN[1], max_model_len=a.max_model_len,
                              tensor_parallel_size=a.tensor_parallel_size,
                              gpu_memory_utilization=a.gpu_memory_utilization,
                              chat_template_kwargs={"enable_thinking": False})
        raws = backend.generate(prompts, system_prompt=SYSTEM, temperature=0.0, max_tokens=200)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    kept = failed = 0
    with a.out.open("w", encoding="utf-8") as fh:
        for i, raw in zip(ids, raws):
            d = parse(raw)
            if d is None:
                failed += 1
                continue
            opts = [d["answer"]] + d["distractors"]
            order = list(range(4))
            random.Random(f"{a.seed}:{i}").shuffle(order)
            shuffled = [opts[k] for k in order]
            gold = LETTERS[order.index(0)]
            r = rows[i]
            fh.write(json.dumps({
                "train_row_id": i, "question": r["question"].strip(),
                **dict(zip(LETTERS, shuffled)), "answer_label": gold,
                "category_english": r.get("category_english"),
                "passage_id": passage_id(r.get("item_description") or ""),
                "reference_answer": r["answer"].strip(), "generator_raw": raw,
            }, ensure_ascii=False) + "\n")
            kept += 1
    meta = {"n_generated": len(ids), "kept": kept, "failed_parse": failed, "seed": a.seed,
            "model": QWEN, "prompt_system": SYSTEM, "prompt_user": USER,
            "seconds": round(time.time() - t0, 1), "fake": a.fake}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
