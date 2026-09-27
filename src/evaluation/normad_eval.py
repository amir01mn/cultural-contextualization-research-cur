"""Eval v2 / Evaluation 2: NormAd (docs/EVAL_V2_PLAN.md).

Stories are loaded and prepared by ``scripts/prep_normad_v2.py``; this module
receives plain dicts (keys: id, country_display, culture, value, rule_of_thumb,
story, gold in {yes,no,neither}, covered, descriptor_bin) and never loads data.

Conditions (only the context block of the prompt changes):
    C1  country only (== S0, closed-book)
    C2  country + gold Value           (~ L3)
    C3  country + gold Rule-of-Thumb   (~ L2; the upper reference)
    S1..S4, S4_triple_retrieval, S4_no_shared_pivot
        country + evidence retrieved with the CulturalBench retrieval code
        (``mcq_systems._evidence_for``; query = story + country)
    S1g, S3g, S4g
        same with RETRIEVAL_STRICT_CULTURE=True and RETRIEVAL_MIN_SCORE=tau,
        tau tuned per system on dev from TAU_GRID, then frozen for test.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from ..core import config
from ..core.schemas import MCQQuestion, MCQVariant
from ..core.utils import get_logger
from ..llm import prompts
from . import mcq_systems as ms
from .mcq_evaluator import mcnemar

LOGGER = get_logger("normad_eval")

LABELS = ("yes", "no", "neither")
MAX_TOKENS = 256  # room for "two sentences" of reasoning before the ANSWER line
TAU_GRID = (0.15, 0.25, 0.35, 0.45)
BOOTSTRAP_N = 10000
SEED = 42

C1, C2, C3 = "C1_country", "C2_value", "C3_rule_of_thumb"
CONTEXT_CONDITIONS = [C1, C2, C3]
RETRIEVAL_SYSTEMS = [
    ms.SYSTEM_S1, ms.SYSTEM_S2, ms.SYSTEM_S3, ms.SYSTEM_S4,
    ms.SYSTEM_S4_TRIPLE, ms.SYSTEM_S4_NO_SHARED,
]
GATED_SYSTEMS = {"S1g": ms.SYSTEM_S1, "S3g": ms.SYSTEM_S3, "S4g": ms.SYSTEM_S4}
ALL_CONDITIONS = CONTEXT_CONDITIONS + RETRIEVAL_SYSTEMS + list(GATED_SYSTEMS)
DESCRIPTOR_BINS = ("<20", "20-99", ">=100")


# ---------------------------------------------------------------------------
# Prompting and parsing
# ---------------------------------------------------------------------------

def story_question(story: dict) -> MCQQuestion:
    """Wrap a story as an MCQQuestion so the CulturalBench retrieval code applies.

    ``_query_for`` builds the query as question + cultural_group, i.e. story + country.
    The canonical culture key is used (e.g. 'turkey' for 'türkiye') so that the
    retrieval culture match, and hence the strict-culture gate, sees the country.
    """
    return MCQQuestion(
        question_id=str(story["id"]),
        variant=MCQVariant.EASY,
        group_id=str(story["id"]),
        question=story["story"],
        cultural_group=story.get("culture") or story.get("country_display") or story.get("country"),
    )


def build_context(condition: str, story: dict, evidence_block: str = "") -> str:
    country = story.get("country_display") or story.get("country", "")
    if condition == C2:
        return prompts.NORMAD_CONTEXT_VALUE.format(country=country, value=story.get("value", ""))
    if condition == C3:
        return prompts.NORMAD_CONTEXT_ROT.format(
            country=country, rule_of_thumb=story.get("rule_of_thumb", ""))
    if condition == C1:
        return prompts.NORMAD_CONTEXT_COUNTRY.format(country=country)
    return prompts.NORMAD_CONTEXT_EVIDENCE.format(country=country, evidence_block=evidence_block)


def build_user_prompt(condition: str, story: dict, evidence_block: str = "") -> str:
    return prompts.NORMAD_USER.format(
        context_block=build_context(condition, story, evidence_block),
        story=story["story"],
    )


_LABEL_WORD = r"(yes|no|neither|neutral)"
_ANSWER_RE = re.compile(r"ANSWER\s*[:\-]?\s*[\s<\[\(\*\"'`]*" + _LABEL_WORD + r"\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"^[\s<\[\(\*\"'`]*(?:\d\)\s*)?" + _LABEL_WORD + r"\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\b" + _LABEL_WORD + r"\b", re.IGNORECASE)


def _norm_label(word: str) -> str:
    w = word.lower()
    return "neither" if w == "neutral" else w


def parse_normad_answer(text: Optional[str]) -> Optional[str]:
    """Return yes/no/neither, or None when the output has no clear answer.

    Priority: the last ``ANSWER: X`` line; else an answer word opening the output;
    else a single unambiguous answer word on the last non-empty line.
    """
    if not text:
        return None
    matches = _ANSWER_RE.findall(text)
    if matches:
        return _norm_label(matches[-1])
    m = _LEADING_RE.match(text)
    if m:
        return _norm_label(m.group(1))
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        words = {_norm_label(w) for w in _WORD_RE.findall(lines[-1])}
        if len(words) == 1:
            return words.pop()
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(correct: list[int]) -> float:
    return sum(correct) / len(correct) if correct else 0.0


def macro_f1(golds: list[str], preds: list[Optional[str]], labels: Iterable[str] = LABELS) -> float:
    """Macro-F1 over the gold label set; unparsed predictions (None) count as wrong."""
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def gap_closed(acc: float, acc_c1: float, acc_c3: float) -> Optional[float]:
    """(acc - acc_C1) / (acc_C3 - acc_C1); None when C3 does not beat C1."""
    denom = acc_c3 - acc_c1
    if denom <= 0:
        return None
    return (acc - acc_c1) / denom


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def paired_bootstrap_delta(vec_a: list[int], vec_b: list[int],
                           n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> dict:
    """95% CI and two-sided bootstrap p for acc(a) - acc(b) on paired items."""
    import numpy as np

    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_boot": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": round(float(a.mean() - b.mean()), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "p_boot": round(float(min(1.0, p)), 6)}


def primary_comparisons(present: Iterable[str]) -> list[tuple[str, str]]:
    """Pre-registered family: each system vs C1 (== S0); S4 vs S1; S4g vs S4."""
    present = set(present)
    pairs = [(s, C1) for s in RETRIEVAL_SYSTEMS + list(GATED_SYSTEMS) if s in present]
    for a, b in ((ms.SYSTEM_S4, ms.SYSTEM_S1), ("S4g", ms.SYSTEM_S4)):
        if a in present and b in present:
            pairs.append((a, b))
    return pairs


def subset_metrics(records_by_cond: dict[str, list[dict]], keep) -> dict:
    """Metrics for every condition over the items where ``keep(record)`` holds."""
    out: dict = {"conditions": {}}
    vecs: dict[str, list[int]] = {}
    for cond, recs in records_by_cond.items():
        sel = [r for r in recs if keep(r)]
        vec = [int(r["correct"]) for r in sel]
        vecs[cond] = vec
        out["conditions"][cond] = {
            "n": len(sel),
            "accuracy": round(accuracy(vec), 4),
            "macro_f1": round(macro_f1([r["gold"] for r in sel], [r["pred"] for r in sel]), 4),
            "unparsed": sum(1 for r in sel if r["pred"] is None),
            "evidence_nonempty": sum(1 for r in sel if r.get("evidence")),
        }
    conds = out["conditions"]
    if C1 in conds and C3 in conds:
        for cond, m in conds.items():
            g = gap_closed(accuracy(vecs[cond]), accuracy(vecs[C1]), accuracy(vecs[C3]))
            m["gap_closed"] = None if g is None else round(g, 4)
    tests = {}
    for a, b in primary_comparisons(conds):
        if not vecs.get(a) or len(vecs[a]) != len(vecs.get(b, [])):
            continue
        t = mcnemar(vecs[a], vecs[b])
        t.update({f"boot_{k}": v for k, v in paired_bootstrap_delta(vecs[a], vecs[b]).items()
                  if k != "delta"})
        tests[f"{a} vs {b}"] = t
    adj = holm({k: t["p_value"] for k, t in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = round(adj[k], 6)
    # Reference (not in the Holm family): gold-context conditions vs C1.
    refs = {}
    for c in (C2, C3):
        if c in vecs and C1 in vecs and vecs[c] and len(vecs[c]) == len(vecs[C1]):
            refs[f"{c} vs {C1}"] = mcnemar(vecs[c], vecs[C1])
    out["tests_primary_holm"] = tests
    out["tests_reference"] = refs
    return out


# ---------------------------------------------------------------------------
# Running conditions
# ---------------------------------------------------------------------------

@contextmanager
def gated_retrieval(tau: float):
    """Temporarily enable the relevance gate; config is always restored."""
    old = (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE)
    config.RETRIEVAL_STRICT_CULTURE = True
    config.RETRIEVAL_MIN_SCORE = float(tau)
    try:
        yield
    finally:
        config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE = old


def _generate(backend, user_prompts: list[str], max_tokens: int) -> list[str]:
    return backend.generate(user_prompts, system_prompt=prompts.NORMAD_SYSTEM,
                            temperature=config.LLM_TEMPERATURE, max_tokens=max_tokens)


def run_condition(condition: str, backend, ctx, stories: list[dict], *,
                  top_k: int = 8, tau: Optional[float] = None,
                  max_tokens: int = MAX_TOKENS) -> tuple[list[dict], dict]:
    """Run one condition over ``stories`` in a single ``backend.generate`` call.

    Returns (per-item records, timing dict).
    """
    t0 = time.time()
    base_system = GATED_SYSTEMS.get(condition, condition)
    is_retrieval = base_system in RETRIEVAL_SYSTEMS
    evid: list[tuple[str, list[str]]] = []
    if is_retrieval:
        def _retrieve():
            return [ms._evidence_for(base_system, ctx, story_question(s), top_k)[:2]
                    for s in stories]
        if condition in GATED_SYSTEMS:
            if tau is None:
                raise ValueError(f"{condition} needs a tau")
            with gated_retrieval(tau):
                evid = _retrieve()
        else:
            evid = _retrieve()
    else:
        evid = [("", []) for _ in stories]
    user_prompts = [build_user_prompt(condition, s, e[0]) for s, e in zip(stories, evid)]
    t1 = time.time()
    outputs = _generate(backend, user_prompts, max_tokens) if user_prompts else []
    t2 = time.time()
    records = []
    for s, (block, ids), out in zip(stories, evid, outputs):
        pred = parse_normad_answer(out)
        records.append({
            "id": s["id"], "condition": condition, "country": s.get("country"),
            "culture": s.get("culture"), "gold": s["gold"], "pred": pred,
            "correct": pred == s["gold"], "covered": bool(s.get("covered", False)),
            "descriptor_bin": s.get("descriptor_bin"), "tau": tau,
            "retrieved_ids": ids, "evidence": block, "raw_output": out,
        })
    timing = {"retrieval_s": round(t1 - t0, 2), "generate_s": round(t2 - t1, 2),
              "n_prompts": len(user_prompts)}
    LOGGER.info("NormAd %s: n=%d acc=%.4f (retrieve %.1fs, generate %.1fs)", condition,
                len(records), accuracy([int(r["correct"]) for r in records]),
                timing["retrieval_s"], timing["generate_s"])
    return records, timing


def tune_taus(backend, ctx, dev_stories: list[dict], *, top_k: int = 8,
              grid: Iterable[float] = TAU_GRID, gated: Iterable[str] = tuple(GATED_SYSTEMS),
              max_tokens: int = MAX_TOKENS) -> dict:
    """Pick tau per gated system by dev accuracy (ties -> smaller tau), then freeze."""
    result = {}
    for g in gated:
        scores = {}
        for tau in grid:
            recs, _ = run_condition(g, backend, ctx, dev_stories, top_k=top_k, tau=tau,
                                    max_tokens=max_tokens)
            scores[tau] = accuracy([int(r["correct"]) for r in recs])
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        result[g] = {"tau": best, "dev_accuracy": {str(k): round(v, 4) for k, v in scores.items()}}
        LOGGER.info("NormAd tau for %s: %.2f  (dev %s)", g, best, result[g]["dev_accuracy"])
    return result


def run_normad(backend, ctx, test_stories: list[dict], out_dir: Path, *,
               dev_stories: Optional[list[dict]] = None,
               conditions: Optional[list[str]] = None,
               taus: Optional[dict[str, float]] = None,
               top_k: int = config.RETRIEVAL_TOP_K,
               max_tokens: int = MAX_TOKENS,
               run_meta: Optional[dict] = None) -> dict:
    """Stage entry point. Writes results.jsonl, summary.json, summary.md to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = conditions or list(ALL_CONDITIONS)
    timings: dict[str, dict] = {}
    tau_info: dict = {}
    gated_needed = [c for c in conditions if c in GATED_SYSTEMS]
    taus = dict(taus or {})
    missing = [g for g in gated_needed if g not in taus]
    if missing:
        if not dev_stories:
            raise ValueError(f"gated systems {missing} need dev_stories or explicit taus")
        t0 = time.time()
        tau_info = tune_taus(backend, ctx, dev_stories, top_k=top_k, gated=missing,
                             max_tokens=max_tokens)
        taus.update({g: v["tau"] for g, v in tau_info.items()})
        timings["tau_tuning"] = {"total_s": round(time.time() - t0, 2)}
        (out_dir / "taus.json").write_text(json.dumps(tau_info, indent=2))

    records_by_cond: dict[str, list[dict]] = {}
    with (out_dir / "results.jsonl").open("w") as f:
        for cond in conditions:
            recs, timing = run_condition(cond, backend, ctx, test_stories, top_k=top_k,
                                         tau=taus.get(cond), max_tokens=max_tokens)
            timings[cond] = timing
            records_by_cond[cond] = recs
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    subsets = {"all": subset_metrics(records_by_cond, lambda r: True),
               "covered": subset_metrics(records_by_cond, lambda r: r["covered"]),
               "uncovered": subset_metrics(records_by_cond, lambda r: not r["covered"])}
    for b in DESCRIPTOR_BINS:
        subsets[f"descriptors {b}"] = subset_metrics(
            records_by_cond, lambda r, b=b: r.get("descriptor_bin") == b)
    summary = {
        "evaluation": "normad",
        "n_test": len(test_stories),
        "n_covered": sum(1 for s in test_stories if s.get("covered")),
        "conditions": conditions,
        "taus": {g: taus[g] for g in gated_needed},
        "tau_tuning": tau_info,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": config.LLM_TEMPERATURE,
        "timings": timings,
        "subsets": subsets,
        "run_meta": run_meta or {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "summary.md").write_text(render_summary_md(summary))
    return summary


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def render_summary_md(summary: dict) -> str:
    lines = ["# NormAd (eval v2, Evaluation 2)", "",
             f"Test stories: {summary['n_test']} (covered: {summary['n_covered']}). "
             f"top_k={summary['top_k']}, max_tokens={summary['max_tokens']}, "
             f"temperature={summary['temperature']}.",
             f"Gated taus (tuned on dev, frozen): {summary['taus'] or 'n/a'}", ""]
    for name, sub in summary["subsets"].items():
        lines += [f"## {name}", "",
                  "| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |",
                  "|---|---|---|---|---|---|---|"]
        for cond, m in sub["conditions"].items():
            lines.append(f"| {cond} | {m['n']} | {_fmt(m['accuracy'])} | {_fmt(m['macro_f1'])} | "
                         f"{_fmt(m.get('gap_closed'))} | {m['unparsed']} | {m['evidence_nonempty']} |")
        if sub["tests_primary_holm"]:
            lines += ["", "| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |",
                      "|---|---|---|---|---|---|"]
            for k, t in sub["tests_primary_holm"].items():
                lines.append(f"| {k} | {t['delta_accuracy']:+.4f} | {t['b_a_only']}/{t['c_b_only']} | "
                             f"{t['p_value']:.4g} | {t['p_holm']:.4g} | "
                             f"[{t['boot_ci_low']:+.4f}, {t['boot_ci_high']:+.4f}] |")
        for k, t in sub.get("tests_reference", {}).items():
            lines.append(f"\nReference (not in Holm family): {k}: delta {t['delta_accuracy']:+.4f}, "
                         f"McNemar p {t['p_value']:.4g}")
        lines.append("")
    lines += ["## Wall time", "", "| stage | retrieval s | generate s | prompts |", "|---|---|---|---|"]
    for k, t in summary["timings"].items():
        lines.append(f"| {k} | {t.get('retrieval_s', t.get('total_s', ''))} | "
                     f"{t.get('generate_s', '')} | {t.get('n_prompts', '')} |")
    return "\n".join(lines) + "\n"
