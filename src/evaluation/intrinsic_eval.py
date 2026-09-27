"""Evaluation 3 (intrinsic KG quality, LLM-judged) and the open-ended judge.

Protocol: ``docs/EVAL_V2_PLAN.md`` ("Evaluation 3", and the Format-B judge of
Evaluation 1). The judge is ``microsoft/phi-4``; this module never loads data or
models itself. ``scripts/prep_intrinsic_v2.py`` builds the samples and
``scripts/run_intrinsic_v2.py`` loads the judge once and calls the stages:

    run_facts          eval_whole_desc -> 3-6 atomic facts (JSON list)
    run_retention      fact inferable from the record's raw triples / refined
                       triples / pathways? (yes/no)  -> retention rates
    run_triple_audit   faithful / layers_correct / culture_specific / stereotyping
    run_pathway_audit  coherent / culture_consistent / correct (FULL pathways)
    run_open_judge     CultureBank §6 knowledge entailment of open answers

Every stage takes ``(backend, <loaded inputs>, out_dir, **cfg)``, writes its
per-item JSONL and a JSON summary under ``out_dir`` and returns the summary.
Judge calls use temperature 0 and an explicit ``max_tokens`` per stage.
Unparseable judge outputs are recorded as ``None`` and reported, never guessed.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..core.utils import ensure_dir, get_logger, write_json, write_jsonl
from ..llm import prompts

LOGGER = get_logger("intrinsic_eval")

# Explicit per-stage generation budgets (never a global budget; AGENTS.md).
STAGE_MAX_TOKENS = {
    "facts": 400,
    "retention": 8,
    "triple_audit": 120,
    "pathway_audit": 120,
    "open_judge": 8,
}
JUDGE_TEMPERATURE = 0.0
# The judge job runs with VLLM_MAX_MODEL_LEN=4096; keep prompts well under it.
MAX_EVIDENCE_CHARS = 7000
MAX_SOURCE_CHARS = 1200
MAX_ANSWER_CHARS = 3000

RETENTION_SOURCES = ("raw_triples", "refined_triples", "pathways")
TRIPLE_AUDIT_KEYS = ("faithful", "layers_correct", "culture_specific", "stereotyping")
PATHWAY_AUDIT_KEYS = ("coherent", "culture_consistent", "correct")

_LAYER_SHORT = {
    "LAYER_1_SURFACE": "L1", "LAYER_2_SOCIAL": "L2", "LAYER_3_DEEP_VALUE": "L3",
}


# ===========================================================================
# Parsing (robust to prose, code fences and minor JSON damage)
# ===========================================================================

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().strip(".!\"'").lower()
        if v in _YES:
            return True
        if v in _NO:
            return False
    return None


def parse_yes_no(text: Optional[str]) -> Optional[bool]:
    """First standalone yes/no in the output (after an optional 'Answer:')."""
    if not text:
        return None
    t = text.strip().lower()
    t = re.sub(r"^[`*\s]*(final\s+)?answer\s*[:\-]\s*", "", t)
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1) == "yes"
    m = re.search(r"\b(true|false)\b", t)
    if m:
        return m.group(1) == "true"
    return None


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text or "").strip()


def parse_json_list(text: Optional[str]) -> Optional[list[str]]:
    """Extract a list of non-empty strings from the model output."""
    if not text:
        return None
    t = _strip_fences(text)
    start, end = t.find("["), t.rfind("]")
    if start != -1:
        # A missing closing bracket means the output was truncated; salvage it.
        chunk = t[start:end + 1] if end > start else t[start:]
        for candidate in (chunk, re.sub(r",\s*]", "]", chunk)):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                items = [str(x).strip() for x in data
                         if isinstance(x, (str, int, float)) and str(x).strip()]
                return items or None
        # Damaged JSON: pull the quoted strings out of the bracketed chunk.
        quoted = re.findall(r'"((?:[^"\\]|\\.)+)"', chunk)
        if quoted:
            return [q.strip() for q in quoted if q.strip()]
    # Fallback: numbered / bulleted lines.
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln).strip()
             for ln in t.splitlines()
             if re.match(r"^\s*(?:\d+[.)]|[-*•])\s+\S", ln)]
    return lines or None


def parse_json_flags(text: Optional[str], keys: Iterable[str]) -> dict[str, Optional[bool]]:
    """Parse ``{"key": "yes"|"no"|true|false}`` for each key; missing -> None."""
    keys = list(keys)
    out: dict[str, Optional[bool]] = {k: None for k in keys}
    if not text:
        return out
    t = _strip_fences(text)
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(t[start:end + 1])
            if isinstance(data, dict):
                lowered = {str(k).strip().lower(): v for k, v in data.items()}
                for k in keys:
                    out[k] = _as_bool(lowered.get(k))
        except json.JSONDecodeError:
            pass
    for k in keys:  # regex fallback for keys the JSON pass missed
        if out[k] is None:
            m = re.search(rf'(?<![\w])"?{re.escape(k)}"?\s*[:=]\s*"?(yes|no|true|false)\b',
                          t, flags=re.IGNORECASE)
            if m:
                out[k] = _as_bool(m.group(1))
    return out


def normalize_facts(facts: Optional[list[str]], max_facts: int = 6) -> list[str]:
    """Dedupe (case-insensitive) and cap at ``max_facts``."""
    seen, out = set(), []
    for f in facts or []:
        key = re.sub(r"\s+", " ", f.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(f.strip())
    return out[:max_facts]


# ===========================================================================
# Sampling (used by the prep script; pure so it is testable)
# ===========================================================================

def stratified_sample(items: list[dict],
                      key: Callable[[dict], str],
                      n: int,
                      *,
                      seed: int = 42,
                      max_strata: Optional[int] = 50) -> list[dict]:
    """Equal-allocation stratified sample: round-robin over strata.

    Strata are the ``max_strata`` largest non-empty keys (ties broken by name);
    within each stratum items are shuffled with ``seed``. A stratum that runs
    out simply drops from the rotation, so small strata never block ``n``.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        k = key(it)
        if k:
            groups[k].append(it)
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    if max_strata:
        order = order[:max_strata]
    rng = random.Random(seed)
    pools = {}
    for k in order:
        pool = list(groups[k])
        rng.shuffle(pool)
        pools[k] = pool
    out: list[dict] = []
    depth = 0
    while len(out) < n and any(depth < len(pools[k]) for k in order):
        for k in order:
            if depth < len(pools[k]) and len(out) < n:
                out.append(pools[k][depth])
        depth += 1
    return out


# ===========================================================================
# Formatting helpers
# ===========================================================================

def _layer(v: Any) -> str:
    return _LAYER_SHORT.get(str(v), str(v or "UNKNOWN"))


def _clip(text: Any, n: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def format_triples(triples: list[dict], culture: str = "") -> str:
    lines = [f"Cultural group: {culture}"] if culture else []
    for t in triples:
        lines.append(f"- ({t.get('subject', '')}) --[{t.get('predicate', '')}]--> "
                     f"({t.get('object', '')})")
    return _clip_block("\n".join(lines))


def format_pathway(pw: dict) -> str:
    nodes = pw.get("nodes") or []
    preds = pw.get("predicates") or []
    parts = []
    for i, node in enumerate(nodes):
        parts.append(f"[{_layer(node.get('layer'))}] {node.get('label', '')}")
        if i < len(nodes) - 1:
            parts.append(f"--{preds[i] if i < len(preds) else 'related to'}-->")
    return " ".join(parts)


def format_pathways(pathways: list[dict]) -> str:
    lines = [f"- ({pw.get('culture', '')}) {format_pathway(pw)}" for pw in pathways]
    return _clip_block("\n".join(lines))


def _clip_block(text: str, n: int = MAX_EVIDENCE_CHARS) -> str:
    if len(text) <= n:
        return text
    cut = text[:n]
    return cut[: cut.rfind("\n")] if "\n" in cut else cut


def _generate(backend, prompt_list: list[str], stage: str, max_tokens: Optional[int]) -> list[str]:
    if not prompt_list:
        return []
    return backend.generate(prompt_list, prompts.INTRINSIC_JUDGE_SYSTEM,
                            temperature=JUDGE_TEMPERATURE,
                            max_tokens=max_tokens or STAGE_MAX_TOKENS[stage])


def _rate(values: list[Optional[bool]]) -> dict:
    valid = [v for v in values if v is not None]
    return {"rate": round(sum(valid) / len(valid), 4) if valid else None,
            "n": len(valid), "n_unparsed": len(values) - len(valid)}


# ===========================================================================
# Stage: facts
# ===========================================================================

def run_facts(backend, records: list[dict], out_dir: Path, *,
              max_tokens: Optional[int] = None, limit: Optional[int] = None,
              **_: Any) -> dict:
    records = records[:limit] if limit else records
    outs = _generate(backend, [prompts.INTRINSIC_FACTS_USER.format(desc=r["eval_whole_desc"])
                               for r in records], "facts", max_tokens)
    rows, n_fail, n_short = [], 0, 0
    for r, raw in zip(records, outs):
        facts = normalize_facts(parse_json_list(raw))
        n_fail += not facts
        n_short += 0 < len(facts) < 3
        rows.append({"instance_id": r["instance_id"], "culture": r.get("culture", ""),
                     "facts": facts, "raw_output": raw})
    write_jsonl(Path(out_dir) / "facts.jsonl", rows)
    n_facts = [len(x["facts"]) for x in rows if x["facts"]]
    summary = {"n_records": len(rows), "n_parse_failed": n_fail,
               "n_fewer_than_3": n_short, "n_facts_total": sum(n_facts),
               "mean_facts_per_record": round(sum(n_facts) / len(n_facts), 3) if n_facts else None}
    write_json(Path(out_dir) / "facts_summary.json", summary)
    return summary


# ===========================================================================
# Stage: retention
# ===========================================================================

def retention_evidence(record: dict) -> dict[str, str]:
    """Evidence text per KG source for one record ('' when the source is empty)."""
    culture = record.get("cultural_group") or record.get("culture", "")
    raw = record.get("raw_triples") or []
    ref = record.get("refined_triples") or []
    pws = record.get("pathways") or []
    return {
        "raw_triples": format_triples(raw, culture) if raw else "",
        "refined_triples": format_triples(ref, culture) if ref else "",
        "pathways": format_pathways(pws) if pws else "",
    }


_SOURCE_NAMES = {"raw_triples": "raw extracted triples",
                 "refined_triples": "critique-refined triples",
                 "pathways": "value -> norm -> practice pathways"}


def aggregate_retention(rows: list[dict]) -> dict:
    """Fact-level retention rates per source, overall and per culture.

    ``rows``: {instance_id, culture, fact_idx, source, inferable: bool|None}.
    Also reports a record-level macro mean (mean over records of per-record rate).
    """
    by_source: dict[str, list] = defaultdict(list)
    by_culture: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    by_record: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r["inferable"])
        by_culture[r.get("culture", "")][r["source"]].append(r["inferable"])
        by_record[(r["source"], r["instance_id"])].append(r["inferable"])
    overall = {}
    for src in sorted(by_source, key=lambda s: (RETENTION_SOURCES + (s,)).index(s)):
        entry = _rate(by_source[src])
        per_rec = [sum(v for v in vals if v is not None) / len([v for v in vals if v is not None])
                   for (s, _), vals in by_record.items()
                   if s == src and any(v is not None for v in vals)]
        entry["record_macro"] = round(sum(per_rec) / len(per_rec), 4) if per_rec else None
        entry["n_records"] = len(per_rec)
        overall[src] = entry
    per_culture = {c: {s: _rate(v) for s, v in srcs.items()}
                   for c, srcs in sorted(by_culture.items())}
    return {"overall": overall, "per_culture": per_culture}


def run_retention(backend, records: list[dict], facts_by_id: dict[str, list[str]],
                  out_dir: Path, *, max_tokens: Optional[int] = None,
                  limit: Optional[int] = None, **_: Any) -> dict:
    records = [r for r in records if facts_by_id.get(r["instance_id"])]
    records = records[:limit] if limit else records
    rows, prompt_list, prompt_rows = [], [], []
    n_empty = defaultdict(int)
    for r in records:
        ev = retention_evidence(r)
        for i, fact in enumerate(facts_by_id[r["instance_id"]]):
            for src in RETENTION_SOURCES:
                row = {"instance_id": r["instance_id"], "culture": r.get("culture", ""),
                       "fact_idx": i, "fact": fact, "source": src,
                       "inferable": None, "empty_evidence": not ev[src]}
                rows.append(row)
                if not ev[src]:
                    row["inferable"] = False  # nothing stored -> not retained
                    n_empty[src] += 1
                    continue
                prompt_list.append(prompts.INTRINSIC_RETENTION_USER.format(
                    source_name=_SOURCE_NAMES[src], evidence=ev[src], fact=fact))
                prompt_rows.append(row)
    outs = _generate(backend, prompt_list, "retention", max_tokens)
    for row, raw in zip(prompt_rows, outs):
        row["inferable"] = parse_yes_no(raw)
        row["raw_output"] = raw
    write_jsonl(Path(out_dir) / "retention.jsonl", rows)
    summary = aggregate_retention(rows)
    summary.update({"n_records": len(records), "n_judge_prompts": len(prompt_list),
                    "n_empty_evidence_facts": dict(n_empty)})
    write_json(Path(out_dir) / "retention_summary.json", summary)
    return summary


# ===========================================================================
# Stage: triple audit
# ===========================================================================

def run_triple_audit(backend, triples: list[dict], out_dir: Path, *,
                     max_tokens: Optional[int] = None, limit: Optional[int] = None,
                     **_: Any) -> dict:
    triples = triples[:limit] if limit else triples
    prompt_list = [prompts.INTRINSIC_TRIPLE_AUDIT_USER.format(
        culture=t.get("cultural_group") or t.get("culture", ""),
        source_text=_clip(t.get("source_text"), MAX_SOURCE_CHARS),
        subject=t.get("subject", ""), predicate=t.get("predicate", ""),
        object=t.get("object", ""), subject_layer=_layer(t.get("subject_layer")),
        object_layer=_layer(t.get("object_layer"))) for t in triples]
    outs = _generate(backend, prompt_list, "triple_audit", max_tokens)
    rows = []
    for t, raw in zip(triples, outs):
        flags = parse_json_flags(raw, TRIPLE_AUDIT_KEYS)
        rows.append({"triple_id": t["triple_id"], "culture": t.get("culture", ""),
                     "generated": bool(t.get("generated")), **flags, "raw_output": raw})
    write_jsonl(Path(out_dir) / "triple_audit.jsonl", rows)
    summary = {"n": len(rows),
               "overall": {k: _rate([r[k] for r in rows]) for k in TRIPLE_AUDIT_KEYS},
               "by_generated": {
                   str(g): {k: _rate([r[k] for r in rows if r["generated"] == g])
                            for k in TRIPLE_AUDIT_KEYS}
                   for g in (False, True) if any(r["generated"] == g for r in rows)}}
    write_json(Path(out_dir) / "triple_audit_summary.json", summary)
    return summary


# ===========================================================================
# Stage: pathway audit
# ===========================================================================

def run_pathway_audit(backend, pathways: list[dict], out_dir: Path, *,
                      max_tokens: Optional[int] = None, limit: Optional[int] = None,
                      **_: Any) -> dict:
    pathways = pathways[:limit] if limit else pathways
    prompt_list = []
    for pw in pathways:
        sources = "\n".join(f"- {_clip(s, 500)}" for s in (pw.get("sources") or [])[:6])
        prompt_list.append(prompts.INTRINSIC_PATHWAY_AUDIT_USER.format(
            culture=pw.get("culture", ""), pathway=format_pathway(pw),
            sources=_clip_block(sources or "(none)", 4000)))
    outs = _generate(backend, prompt_list, "pathway_audit", max_tokens)
    rows = []
    for pw, raw in zip(pathways, outs):
        flags = parse_json_flags(raw, PATHWAY_AUDIT_KEYS)
        rows.append({"pathway_id": pw["pathway_id"], "culture": pw.get("culture", ""),
                     "has_shared_pivot": bool(pw.get("has_shared_pivot")),
                     **flags, "raw_output": raw})
    write_jsonl(Path(out_dir) / "pathway_audit.jsonl", rows)
    summary = {"n": len(rows),
               "overall": {k: _rate([r[k] for r in rows]) for k in PATHWAY_AUDIT_KEYS}}
    write_json(Path(out_dir) / "pathway_audit_summary.json", summary)
    return summary


# ===========================================================================
# Stage: open-ended judge (Evaluation 1, Format B)
# ===========================================================================

_CANONICAL_SYSTEMS = {"S0": "S0_no_kg", "S1": "S1_raw_text_rag", "S2": "S2_flat_kg",
                      "S3": "S3_layered_kg", "S4": "S4_abstraction_refinement_kg"}
PRIMARY_COMPARISONS = [("S1", "S0"), ("S3", "S0"), ("S4", "S0"), ("S4g", "S0"),
                       ("S4", "S1"), ("S4g", "S4")]


def resolve_system(label: str, systems: Iterable[str]) -> Optional[str]:
    """Map a short label (S0, S4g, ...) onto the system names present."""
    systems = list(systems)
    if label in systems:
        return label
    if _CANONICAL_SYSTEMS.get(label) in systems:
        return _CANONICAL_SYSTEMS[label]
    cands = [s for s in systems if s.split("_")[0] == label]
    return cands[0] if len(cands) == 1 else None


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = round(running, 6)
    return out


def paired_delta_ci(a: list[int], b: list[int], *, n_boot: int = 10000,
                    seed: int = 42) -> tuple[float, float]:
    import numpy as np

    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    deltas = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def compare_systems(scores: dict[str, dict[str, int]], *, n_boot: int = 10000,
                    comparisons=PRIMARY_COMPARISONS) -> dict:
    """Per-system mean + paired tests over the pre-registered primary family.

    ``scores``: system -> {qid: 0/1}. Pairs use the qids both systems answered.
    """
    from .mcq_evaluator import mcnemar  # lazy: keeps the other stages import-light

    per_system = {s: {"mean_entailment": round(sum(v.values()) / len(v), 4) if v else None,
                      "n": len(v)} for s, v in scores.items()}
    tests = {}
    for a_lbl, b_lbl in comparisons:
        a, b = resolve_system(a_lbl, scores), resolve_system(b_lbl, scores)
        if not a or not b:
            continue
        qids = sorted(set(scores[a]) & set(scores[b]))
        va, vb = [scores[a][q] for q in qids], [scores[b][q] for q in qids]
        res = mcnemar(va, vb)
        res["n_paired"] = len(qids)
        res["delta_ci95"] = list(paired_delta_ci(va, vb, n_boot=n_boot))
        tests[f"{a_lbl} vs {b_lbl}"] = {"system_a": a, "system_b": b, **res}
    adj = holm({k: v["p_value"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    return {"per_system": per_system, "comparisons": tests}


def run_open_judge(backend, answers: list[dict], out_dir: Path, *,
                   max_tokens: Optional[int] = None, limit: Optional[int] = None,
                   n_boot: int = 10000, **_: Any) -> dict:
    if limit:  # cap by question (all systems of the first ``limit`` qids) to keep pairs
        keep = list(dict.fromkeys(str(a["qid"]) for a in answers))[:limit]
        answers = [a for a in answers if str(a["qid"]) in set(keep)]
    prompt_list = [prompts.OPEN_JUDGE_USER.format(
        cultural_group=a.get("cultural_group", ""), gold_desc=a.get("gold_desc", ""),
        question=_clip(a.get("question"), 1500),
        answer=_clip(a.get("answer"), MAX_ANSWER_CHARS)) for a in answers]
    outs = _generate(backend, prompt_list, "open_judge", max_tokens)
    rows, scores = [], defaultdict(dict)
    n_unparsed = defaultdict(int)
    for a, raw in zip(answers, outs):
        verdict = parse_yes_no(raw)
        if verdict is None:
            n_unparsed[a["system"]] += 1
        # Unparseable verdicts count as "no" (reported); keeps pairing complete.
        score = 1 if verdict else 0
        scores[a["system"]][str(a["qid"])] = score
        rows.append({"qid": a["qid"], "system": a["system"],
                     "cultural_group": a.get("cultural_group", ""),
                     "entails": verdict, "score": score, "raw_output": raw})
    out_dir = ensure_dir(out_dir)
    write_jsonl(out_dir / "open_judge.jsonl", rows)
    summary = compare_systems(dict(scores), n_boot=n_boot)
    summary.update({"n_answers": len(rows), "n_unparsed_by_system": dict(n_unparsed),
                    "protocol": "CultureBank §6 knowledge entailment; judge yes=1/no=0; "
                                "unparseable counted as 0; McNemar + paired bootstrap; "
                                "Holm over the primary family"})
    write_json(out_dir / "open_judge_summary.json", summary)
    (out_dir / "open_judge_summary.md").write_text(open_judge_markdown(summary))
    return summary


def open_judge_markdown(summary: dict) -> str:
    lines = ["# CultureBank open-ended answers: judge entailment (phi-4)", "",
             f"Answers judged: {summary.get('n_answers', 0)}. "
             f"Unparsed verdicts (scored 0): {summary.get('n_unparsed_by_system', {})}", "",
             "| System | Mean entailment | n |", "|---|---|---|"]
    for s, v in summary["per_system"].items():
        m = v["mean_entailment"]
        lines.append(f"| {s} | {'—' if m is None else f'{m:.1%}'} | {v['n']} |")
    lines += ["", "## Primary comparisons (McNemar exact, Holm-adjusted)", "",
              "| Comparison | Δ | 95% CI (bootstrap) | b/c | p | p (Holm) |",
              "|---|---|---|---|---|---|"]
    for k, t in summary["comparisons"].items():
        lo, hi = t["delta_ci95"]
        lines.append(f"| {k} | {t['delta_accuracy']:+.1%} | [{lo:+.1%}, {hi:+.1%}] | "
                     f"{t['b_a_only']}/{t['c_b_only']} | {t['p_value']:.4g} | {t['p_holm']:.4g} |")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Combined summary
# ===========================================================================

def _fmt_rate(entry: Optional[dict]) -> str:
    if not entry or entry.get("rate") is None:
        return "—"
    return f"{entry['rate']:.1%} (n={entry['n']}, unparsed={entry['n_unparsed']})"


def write_summary(out_dir: Path, results: dict, meta: Optional[dict] = None) -> dict:
    """Write ``summary.json`` + ``summary.md`` for the stages that ran."""
    out_dir = ensure_dir(out_dir)
    payload = {"meta": meta or {}, "stages": results}
    write_json(out_dir / "summary.json", payload)
    L = ["# Evaluation 3: intrinsic KG quality (judge: phi-4)", ""]
    for k, v in (meta or {}).items():
        L.append(f"- {k}: {v}")
    if "facts" in results:
        f = results["facts"]
        L += ["", "## Atomic facts", "",
              f"{f['n_records']} records, {f['n_facts_total']} facts "
              f"(mean {f['mean_facts_per_record']}/record); parse failures "
              f"{f['n_parse_failed']}, records with <3 facts {f['n_fewer_than_3']}."]
    if "retention" in results:
        r = results["retention"]
        L += ["", "## Fact retention (fact inferable from the source)", "",
              "| Source | Fact-level rate | Record macro | Empty-evidence facts |",
              "|---|---|---|---|"]
        for src, e in r["overall"].items():
            macro = e.get("record_macro")
            L.append(f"| {src} | {_fmt_rate(e)} | {'—' if macro is None else f'{macro:.1%}'} | "
                     f"{r.get('n_empty_evidence_facts', {}).get(src, 0)} |")
        L += ["", "Empty evidence (e.g. no pathway contains any of the record's triples) "
                  "counts as not retained. Per-culture rates are in summary.json."]
    if "triple_audit" in results:
        t = results["triple_audit"]
        L += ["", f"## Triple audit ({t['n']} refined triples)", "",
              "| Criterion | All | Non-generated | Generated |", "|---|---|---|---|"]
        bg = t.get("by_generated", {})
        for k in TRIPLE_AUDIT_KEYS:
            L.append(f"| {k} | {_fmt_rate(t['overall'][k])} | "
                     f"{_fmt_rate(bg.get('False', {}).get(k))} | "
                     f"{_fmt_rate(bg.get('True', {}).get(k))} |")
    if "pathway_audit" in results:
        p = results["pathway_audit"]
        L += ["", f"## Pathway audit ({p['n']} FULL pathways)", "",
              "| Criterion | Rate |", "|---|---|"]
        for k in PATHWAY_AUDIT_KEYS:
            L.append(f"| {k} | {_fmt_rate(p['overall'][k])} |")
    if "open_judge" in results:
        oj = results["open_judge"]
        L += ["", "## Open-ended judge (CultureBank Format B)", ""]
        if isinstance(oj, dict) and oj.get("status"):
            L.append(f"Status: {oj['status']}")
        else:
            L += ["| System | Mean entailment | n |", "|---|---|---|"]
            for s, v in oj.get("per_system", {}).items():
                m = v["mean_entailment"]
                L.append(f"| {s} | {'—' if m is None else f'{m:.1%}'} | {v['n']} |")
            L += ["", "Paired tests: `open_judge_summary.md` next to `open_answers.jsonl`."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    return payload
