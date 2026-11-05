"""KazCulture final experiment: conditions, evidence blocks, checkpoints, statistics.

Implements the evidence side of docs/KAZCULTURE_PREREGISTRATION.md §3. Every
condition uses the same answering template (``kazculture_rag.evidence_prompt``);
only the numbered evidence block differs:

    S0         no evidence (closed-book V2 prompt)
    S1         frozen passage retrieval (top_k 8, min_sim 0.45, 5,000 chars)
    S4R        v3 frame retrieval with routing (refine / abstract / none)
    S4R-noabs  S4R with the abstract route disabled
    S1+L       S1 with each passage's own norm/value labels attached
    CTRL-LEN   S1 plus further S1 passages filling the characters S4R added
    CTRL-SHUF  S4R layout with labels/analogs from non-matching passages

The frame KG retrieval adapter belongs to the frame-KG port and is imported
lazily by module name (``load_frame_adapter``); it must expose
``retrieve_s4r(question_text, options, setting, top_k, budget_chars, route_mode)``
returning ``[{text, source_id, layer, route, indirect, similarity}]`` and
``labels_for_passage(passage_id) -> {norm, norm_type, values}``.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from ..retrieval.kazculture_index import pack_numbered_blocks, render_s1_blocks, s1_block

CONDITIONS = ("S0", "S1", "S4R", "S4R-noabs", "S1+L", "CTRL-LEN", "CTRL-SHUF")
SETTINGS = ("strict", "full")
CLOSED_BOOK = "closed_book"  # S0 does not depend on the retrieval setting
FRAME_CONDITIONS = ("S4R", "S4R-noabs", "CTRL-SHUF")
S1_CONDITIONS = ("S1", "S1+L", "CTRL-LEN")
# route_mode values passed to the adapter; overridable from the CLI.
DEFAULT_ROUTE_MODES = {"S4R": "routed", "S4R-noabs": "noabs", "CTRL-SHUF": "shuffled"}
DEFAULT_FRAME_ADAPTER = os.environ.get(
    "KAZCULTURE_FRAME_ADAPTER", "src.retrieval.kazculture_frame_retriever")
FROZEN_S1 = {"top_k": 8, "min_similarity": 0.45, "evidence_budget_chars": 5000}
CTRL_LEN_MODES = ("append", "budget")


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def read_jsonl_checkpoint(path: Path, *, repair: bool = True) -> list[dict]:
    """Read an append-only JSONL checkpoint, tolerating a truncated last line.

    A job killed mid-write can leave a partial final record. That record is
    dropped (and, with ``repair``, cut from the file so later appends start on
    a clean line). Corruption anywhere else is an error, never skipped.
    A complete final record missing only its newline gets the newline back.
    """
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    rows: list[dict] = []
    offset = 0
    lines = data.split(b"\n")
    for i, raw in enumerate(lines):
        last = i == len(lines) - 1
        if raw.strip():
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not last:
                    raise ValueError(f"Corrupt checkpoint line {i + 1} in {path}")
                if repair:
                    with path.open("r+b") as fh:
                        fh.truncate(offset)
                print(f"WARNING: dropped truncated final line of {path}", flush=True)
                return rows
            if last and repair:  # complete record without trailing newline
                with path.open("ab") as fh:
                    fh.write(b"\n")
        offset += len(raw) + 1
    return rows


def append_jsonl(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Frame adapter (owned by the frame-KG port; imported lazily)
# --------------------------------------------------------------------------

class FrameAdapter:
    """Thin validated wrapper around the frame-KG retrieval module."""

    def __init__(self, module, *, kwargs: dict | None = None):
        self.module = module
        self.name = getattr(module, "__name__", str(module))
        configure = getattr(module, "configure", None)
        if kwargs and configure is None:
            raise ValueError(f"{self.name} has no configure(); cannot apply adapter kwargs")
        if configure is not None and kwargs is not None:
            configure(**kwargs)
        self._retrieve = getattr(module, "retrieve_s4r")
        # The adapter's own layout joins its blocks with SEPARATOR (frame KG v3
        # render_route); "adapter" rendering keeps that text verbatim.
        self.separator = getattr(module, "SEPARATOR", "\n")
        self._labels = getattr(module, "labels_for_passage", None)
        params = inspect.signature(self._retrieve).parameters
        self.exclude_kw = next((k for k in ("exclude_passage_ids", "exclude_source_ids")
                                if k in params), None)

    def retrieve(self, question_text: str, options, *, setting: str, top_k: int,
                 budget_chars: int, route_mode: str,
                 exclude_passage_ids: set[str] | None = None) -> tuple[list[dict], str]:
        """Return validated items and how own-source exclusion was applied."""
        kwargs = dict(setting=setting, top_k=top_k, budget_chars=budget_chars,
                      route_mode=route_mode)
        mode = "none"
        if exclude_passage_ids:
            if self.exclude_kw:
                kwargs[self.exclude_kw] = set(exclude_passage_ids)
                mode = "adapter"
            else:
                mode = "post_filter"
        items = list(self._retrieve(question_text, options, **kwargs) or [])
        for it in items:
            if not isinstance(it, dict) or not str(it.get("text") or "").strip():
                raise ValueError(f"Adapter item lacks text: {it!r}")
        if mode == "post_filter":
            items = [it for it in items if it["source_id"] not in exclude_passage_ids]
        return items, mode

    def labels(self, passage_id: str) -> dict | None:
        if self._labels is None:
            raise ValueError(f"{self.name} has no labels_for_passage(); S1+L unavailable")
        try:
            return self._labels(passage_id)
        except KeyError:
            return None


def parse_route_modes(text: str | None) -> dict:
    modes = dict(DEFAULT_ROUTE_MODES)
    for part in (text or "").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            if k.strip() not in FRAME_CONDITIONS or not v.strip():
                raise ValueError(f"Bad --route_modes entry {part!r}")
            modes[k.strip()] = v.strip()
    return modes


def load_frame_adapter(spec: str = DEFAULT_FRAME_ADAPTER, *,
                       kwargs: dict | None = None) -> FrameAdapter:
    return FrameAdapter(importlib.import_module(spec), kwargs=kwargs)


# --------------------------------------------------------------------------
# Evidence blocks
# --------------------------------------------------------------------------

def format_labels(labels: dict | None) -> str | None:
    """S1+L annotation: the passage's own provisional labels, clearly marked."""
    if not labels:
        return None
    values = labels.get("values") or []
    if isinstance(values, str):
        values = [values]
    parts = []
    if labels.get("norm_type"):
        parts.append(f"norm type: {labels['norm_type']}")
    if labels.get("norm"):
        parts.append(f"norm: {labels['norm']}")
    if values:
        parts.append("values: " + ", ".join(str(v) for v in values))
    if not parts:
        return None
    return "Provisional labels (model-generated, not source text): " + " | ".join(parts)


def frame_block(number: int, item: dict) -> str:
    label = "INDIRECT COMPARISON" if item.get("indirect") else "KAZAKH TRAINING SOURCE"
    return (f"[{number}] {label} | layer={item.get('layer') or '-'}"
            f" | route={item.get('route') or '-'} | id={item['source_id']}\n{item['text']}")


def s1_meta(unit: dict) -> dict:
    return {"id": unit["passage_id"], "unit_id": unit["unit_id"],
            "similarity": unit.get("similarity"), "indirect": bool(unit.get("indirect"))}


def frame_meta(item: dict) -> dict:
    sid = item.get("source_id") or "-"
    return {"id": sid, "unit_id": item.get("unit_id") or sid,
            "similarity": item.get("similarity"), "indirect": bool(item.get("indirect")),
            "layer": item.get("layer"), "route": item.get("route")}


class Evidence:
    """Numbered evidence blocks plus metadata describing exactly what is shown."""

    def __init__(self, blocks: list[str], meta: list[dict], *, sep: str = "\n\n", **info):
        if len(blocks) != len(meta):
            raise ValueError("blocks/meta mismatch")
        self.blocks = blocks
        self.meta = meta
        self.sep = sep
        self.info = info

    def text(self, n: int | None = None) -> str:
        return self.sep.join(self.blocks if n is None else self.blocks[:n])

    def __len__(self):
        return len(self.blocks)


def s1_evidence(candidates: list[dict], *, top_k: int, max_chars: int,
                skip_oversize: bool = False, annotate=None) -> Evidence:
    """Frozen S1: first ``top_k`` ranked hits, rendered under the char budget."""
    blocks, used = render_s1_blocks(candidates[:top_k], max_chars=max_chars,
                                    skip_oversize=skip_oversize, annotate=annotate)
    frozen_used = render_s1_blocks(candidates[:top_k], max_chars=max_chars)[1]
    skip_used = render_s1_blocks(candidates[:top_k], max_chars=max_chars,
                                 skip_oversize=True)[1]
    return Evidence(blocks, [s1_meta(u) for u in used],
                    oversize_break_affected=len(skip_used) > len(frozen_used))


FRAME_RENDER_MODES = ("adapter", "numbered")


def frame_evidence(items: list[dict], *, max_chars: int, skip_oversize: bool = False,
                   render: str = "adapter", sep: str = "\n") -> Evidence:
    """S4R-type evidence from adapter items.

    ``adapter`` (default): the adapter's block texts verbatim, joined by its
    separator (the frame-KG v3 layout, already numbered); kept as the longest
    in-order prefix within ``max_chars`` (a no-op when the adapter budgeted).
    ``numbered``: each item wrapped in an S1-style ``[n] LABEL | layer | route``
    header and packed like S1.
    """
    if render == "adapter":
        blocks, used = [], []
        for it in items:
            if len(sep.join(blocks + [it["text"]])) > max_chars:
                if skip_oversize:
                    continue
                break
            blocks.append(it["text"])
            used.append(it)
    elif render == "numbered":
        blocks, used = pack_numbered_blocks(items, frame_block, max_chars=max_chars,
                                            skip_oversize=skip_oversize)
        sep = "\n\n"
    else:
        raise ValueError(f"Unknown frame render mode {render!r}")
    return Evidence(blocks, [frame_meta(x) for x in used], sep=sep,
                    frame_render=render, frame_items_dropped=len(items) - len(used),
                    routes_used=sorted({str(x.get("route")) for x in used}))


def ctrl_len_evidence(candidates: list[dict], s1: Evidence, *, extra_chars: int,
                      top_k: int, base_budget: int, mode: str = "append",
                      skip_oversize: bool = False) -> Evidence:
    """CTRL-LEN: S1 given exactly ``extra_chars`` more characters (deterministic).

    ``append`` (default): the S1 evidence verbatim, followed by further S1
    candidates in rank order (any not already shown) numbered continuously,
    each kept if it fits within ``len(S1 text) + extra_chars`` total. With
    ``extra_chars == 0`` this is S1 exactly; the total never exceeds S1+extra.
    ``budget``: re-render the extended ranking with budget
    ``base_budget + extra_chars`` and no top-k cap (S1 exactly when extra is 0).
    """
    if extra_chars < 0:
        raise ValueError("extra_chars must be non-negative")
    if mode not in CTRL_LEN_MODES:
        raise ValueError(f"Unknown CTRL-LEN mode {mode!r}")
    info = {"ctrl_len_target_extra_chars": extra_chars, "ctrl_len_mode": mode}
    if extra_chars == 0:
        return Evidence(list(s1.blocks), list(s1.meta), ctrl_len_added_chars=0, **info)
    s1_chars = len(s1.text())
    if mode == "budget":
        blocks, used = render_s1_blocks(candidates, max_chars=base_budget + extra_chars,
                                        skip_oversize=skip_oversize)
        meta = [s1_meta(u) for u in used]
    else:
        shown = {m["unit_id"] for m in s1.meta}
        further = [c for c in candidates if c["unit_id"] not in shown]
        new_blocks, new_used = pack_numbered_blocks(
            further, s1_block, max_chars=s1_chars + extra_chars, skip_oversize=True,
            start=len(s1.blocks) + 1, prefix_chars=s1_chars if s1.blocks else None)
        blocks = list(s1.blocks) + new_blocks
        meta = list(s1.meta) + [s1_meta(u) for u in new_used]
    ev = Evidence(blocks, meta, **info)
    ev.info["ctrl_len_added_chars"] = len(ev.text()) - s1_chars
    return ev


def fit_blocks(n_blocks: int, fits: Callable[[int], bool]) -> int:
    """Largest prefix length k <= n_blocks with fits(k); 0 if nothing fits."""
    for k in range(n_blocks, -1, -1):
        if fits(k):
            return k
    return 0


# --------------------------------------------------------------------------
# Rows and summaries
# --------------------------------------------------------------------------

def prediction_row(q, scores: dict, *, condition: str, setting: str,
                   evidence_text: str, meta: list[dict], info: dict,
                   prediction: str, extra: dict | None = None) -> dict:
    row = {
        "question_id": q.question_id, "category": q.category,
        "condition": condition, "setting": setting,
        "prediction": prediction, "gold": q.answer_key, "answer_key": q.answer_key,
        "correct": prediction == q.answer_key,
        "letter_logprobs": scores,
        "retrieved_ids": [m["id"] for m in meta],
        "retrieved_unit_ids": [m["unit_id"] for m in meta],
        "retrieval_scores": [m.get("similarity") for m in meta],
        "n_blocks": len(meta),
        "rendered_chars": len(evidence_text),
        "evidence_sha256": hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
    }
    if any("route" in m for m in meta):
        row["routes"] = [m.get("route") for m in meta]
        row["layers"] = [m.get("layer") for m in meta]
        row["indirect"] = [m.get("indirect") for m in meta]
    row.update(info)
    row.update(extra or {})
    return row


def condition_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    correct = sum(bool(r["correct"]) for r in rows)
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r)
    out = {
        "n": n, "correct": correct, "accuracy": correct / n,
        "accuracy_pct": round(100 * correct / n, 4),
        "no_evidence_n": sum(not r["n_blocks"] for r in rows),
        "mean_blocks": sum(r["n_blocks"] for r in rows) / n,
        "mean_rendered_chars": sum(r["rendered_chars"] for r in rows) / n,
        "category_accuracy": {
            k: {"n": len(v), "correct": sum(bool(x["correct"]) for x in v),
                "accuracy": sum(bool(x["correct"]) for x in v) / len(v)}
            for k, v in sorted(cats.items())},
    }
    for key in ("oversize_break_affected", "truncated"):
        if any(key in r for r in rows):
            out[f"{key}_n"] = sum(bool(r.get(key)) for r in rows)
    if any("ctrl_len_target_extra_chars" in r for r in rows):
        tgt = [r["ctrl_len_target_extra_chars"] for r in rows]
        add = [r["ctrl_len_added_chars"] for r in rows]
        out["ctrl_len"] = {
            "mean_target_extra_chars": sum(tgt) / n,
            "mean_added_chars": sum(add) / n,
            "questions_with_extra_target": sum(t > 0 for t in tgt),
            "mean_shortfall_chars": sum(t - a for t, a in zip(tgt, add)) / n,
        }
    return out


def reproduction_check(rows: list[dict], reference: list[dict]) -> dict:
    """Per-question comparison against a frozen run (exact match expected)."""
    ref = {r["question_id"]: r for r in reference}
    mism = {"missing": [], "prediction": [], "evidence_sha256": [], "retrieved_unit_ids": []}
    for r in rows:
        f = ref.get(r["question_id"])
        if f is None:
            mism["missing"].append(r["question_id"])
            continue
        if f["prediction"] != r["prediction"]:
            mism["prediction"].append(r["question_id"])
        for key in ("evidence_sha256", "retrieved_unit_ids"):
            if key in f and f[key] != r[key]:
                mism[key].append(r["question_id"])
    ours = sum(bool(r["correct"]) for r in rows)
    theirs = sum(bool(ref[r["question_id"]]["correct"]) for r in rows if r["question_id"] in ref)
    return {
        "n": len(rows), "correct": ours, "reference_correct_on_same_ids": theirs,
        "exact": not any(mism.values()) and ours == theirs,
        "mismatch_counts": {k: len(v) for k, v in mism.items()},
        "mismatch_examples": {k: v[:20] for k, v in mism.items()},
    }


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------

def mcnemar_exact(baseline: Iterable[bool], system: Iterable[bool]) -> dict:
    """Exact two-sided McNemar (binomial on discordant pairs, integer arithmetic)."""
    b = [bool(x) for x in baseline]
    c = [bool(x) for x in system]
    if len(b) != len(c):
        raise ValueError("Paired vectors differ in length")
    rescued = sum((not u) and v for u, v in zip(b, c))
    harmed = sum(u and (not v) for u, v in zip(b, c))
    d = rescued + harmed
    if d:
        tail = sum(math.comb(d, k) for k in range(min(rescued, harmed) + 1))
        p = min(1.0, (2 * tail) / (1 << d))
    else:
        p = 1.0
    return {"rescued": rescued, "harmed": harmed, "discordant": d, "p": p}


def paired_bootstrap_ci(diffs: list[float], *, n_boot: int = 10000, seed: int = 0,
                        chunk: int = 500) -> tuple[float, float]:
    """Percentile 95% CI of the mean paired difference (numpy, chunked)."""
    import numpy as np

    x = np.asarray(diffs, dtype=float)
    if not len(x):
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        means.append(x[rng.integers(0, len(x), (m, len(x)))].mean(axis=1))
        done += m
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return (float(lo), float(hi))


def compare_paired(baseline_rows: list[dict], system_rows: list[dict], *,
                   n_boot: int = 10000, seed: int = 0,
                   question_ids: set[str] | None = None) -> dict:
    """McNemar + paired bootstrap on accuracy; requires identical question IDs."""
    b = {r["question_id"]: r for r in baseline_rows}
    s = {r["question_id"]: r for r in system_rows}
    if set(b) != set(s) or not b:
        raise ValueError("Paired comparison requires exactly matching question IDs")
    ids = sorted(b if question_ids is None else set(b) & set(question_ids))
    bc = [bool(b[i]["correct"]) for i in ids]
    sc = [bool(s[i]["correct"]) for i in ids]
    n = len(ids)
    diffs = [int(v) - int(u) for u, v in zip(bc, sc)]
    lo, hi = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    mc = mcnemar_exact(bc, sc)
    cats: dict[str, list[int]] = defaultdict(list)
    for i, dlt in zip(ids, diffs):
        cats[s[i]["category"]].append(dlt)
    return {
        "n": n, "baseline_correct": sum(bc), "system_correct": sum(sc),
        "baseline_accuracy": sum(bc) / n if n else None,
        "system_accuracy": sum(sc) / n if n else None,
        "delta_pp": 100 * sum(diffs) / n if n else None,
        "net_questions": sum(diffs),
        "bootstrap_95_ci_pp": [100 * lo, 100 * hi],
        "bootstrap": {"n_boot": n_boot, "seed": seed},
        "mcnemar_exact_p": mc["p"], "rescued": mc["rescued"], "harmed": mc["harmed"],
        "category_delta": {
            k: {"n": len(v), "delta_pp": 100 * sum(v) / len(v), "net": sum(v)}
            for k, v in sorted(cats.items())},
    }


# --------------------------------------------------------------------------
# Train-dev gate (free-text train QA; preregistration §4)
# --------------------------------------------------------------------------

def free_text_prompt(tokenizer, question: str, evidence: str, *,
                     template_kwargs: dict | None = None) -> str:
    """Train-dev prompt: same evidence header and assistant prefix, no options."""
    from .kazculture_rag import EVIDENCE_HEADER, QWEN3_TEMPLATE_KWARGS
    kwargs = QWEN3_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    user = "Question:\n" + question
    if evidence.strip():
        user = EVIDENCE_HEADER + evidence + "\n\n" + user
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, **kwargs) + "Answer:"


def gate_decision(rows_by_condition: dict[str, list[dict]], *, n_boot: int = 10000,
                  seed: int = 0, primary: str = "S4R", baseline: str = "S1") -> dict:
    """Paired mean gold log-likelihood comparison; gate = S4R mean > S1 mean."""
    by = {c: {r["train_row_id"]: r for r in rows} for c, rows in rows_by_condition.items()}
    if baseline not in by or primary not in by:
        raise ValueError(f"Gate needs {baseline} and {primary}")
    means = {c: sum(r["mean_loglik"] for r in v.values()) / len(v) for c, v in by.items() if v}
    paired = {}
    for c in by:
        if c == baseline:
            continue
        ids = sorted(set(by[c]) & set(by[baseline]))
        if len(ids) != len(by[baseline]) or len(ids) != len(by[c]):
            raise ValueError(f"{c} and {baseline} cover different train questions")
        d = [by[c][i]["mean_loglik"] - by[baseline][i]["mean_loglik"] for i in ids]
        lo, hi = paired_bootstrap_ci(d, n_boot=n_boot, seed=seed)
        paired[f"{c}_minus_{baseline}"] = {
            "n": len(ids), "mean_delta": sum(d) / len(d), "bootstrap_95_ci": [lo, hi],
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d),
            "ties": sum(x == 0 for x in d),
        }
    key = f"{primary}_minus_{baseline}"
    proceed = paired[key]["mean_delta"] > 0
    return {
        "metric": "mean per-token gold-answer log-likelihood (Qwen3-32B, teacher forced)",
        "condition_means": means, "paired": paired,
        "gate": {
            "rule": "preregistration §4: proceed with S4R as primary iff it improves mean "
                    "gold log-likelihood over S1 on train-dev (mean paired delta > 0)",
            "s4r_minus_s1_mean_delta": paired[key]["mean_delta"],
            "decision": "proceed_s4r_primary" if proceed else
                        "not_met_run_test_once_report_negative_no_iteration",
            "passed": proceed,
        },
    }


# --------------------------------------------------------------------------
# Documented baselines (regeneration without the frozen artifacts)
# --------------------------------------------------------------------------
# docs/KAZCULTURE_PREREGISTRATION.md §2 and KAZCULTURE_BRANCH_HANDOFF.md §14-15.
# The original prediction files/indexes/config are not accessible, so they are
# regenerated and must reproduce these documented numbers exactly.
DOCUMENTED = {
    "counts": {"closed_book/S0": (552, 1334), "strict/S1": (831, 1334),
               "full/S1": (1221, 1334)},
    "strict_index_passages": 5487,
    "full_index_passages": 6749,
    "s1_config": {"top_k": 8, "min_similarity": 0.45, "evidence_budget_chars": 5000,
                  "dev_n": 400, "matched": 285},
}


def check_documented_s1_config(cfg: dict) -> dict:
    """Compare a tuned S1 config with the documented frozen one (exact)."""
    exp = DOCUMENTED["s1_config"]
    chosen = next((c for c in cfg.get("candidate_grid", [])
                   if c.get("top_k") == cfg.get("top_k")
                   and c.get("min_similarity") == cfg.get("min_similarity")), {})
    obs = {"top_k": cfg.get("top_k"), "min_similarity": cfg.get("min_similarity"),
           "evidence_budget_chars": cfg.get("evidence_budget_chars"),
           "dev_n": cfg.get("dev_n"),
           "matched": chosen.get("matched", round((cfg.get("train_dev_recall") or 0)
                                                  * (cfg.get("dev_n") or 0)))}
    mism = {k: {"expected": exp[k], "observed": obs[k]} for k in exp if obs[k] != exp[k]}
    if cfg.get("source") != "train_dev_only":
        mism["source"] = {"expected": "train_dev_only", "observed": cfg.get("source")}
    return {"check": "s1_config", "mode": "documented", "expected": exp, "observed": obs,
            "mismatches": mism, "exact": not mism}


def documented_count_check(rows: list[dict], key: str) -> dict:
    exp_correct, exp_n = DOCUMENTED["counts"][key]
    obs_correct, obs_n = sum(bool(r["correct"]) for r in rows), len(rows)
    return {"check": key, "mode": "documented_counts",
            "expected": {"correct": exp_correct, "n": exp_n},
            "observed": {"correct": obs_correct, "n": obs_n},
            "difference_correct": obs_correct - exp_correct,
            "exact": (obs_correct, obs_n) == (exp_correct, exp_n)}
