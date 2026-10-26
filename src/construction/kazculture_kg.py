"""Passage-grounded KazCulture frame, triple and pathway candidates.

LLM labels are never treated as ground truth: all inferred norms/values are
marked provisional, and every candidate triple requires a verbatim passage
quote plus provenance. The original source text is never replaced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

from ..core.utils import stable_id
from ..construction.frame_builder import SCHWARTZ_VALUES

VALUE_NAMES = tuple(SCHWARTZ_VALUES)
MAX_TRIPLES = 6

EXTRACTION_SYSTEM = """You extract evidence-grounded cultural relations from a Kazakh-language source passage.
Respond with ONE JSON object, no markdown or commentary. Never use knowledge
outside the supplied passage. Preserve Kazakh wording in subjects/objects.
If the passage does not support a relation, output fewer triples.
Norm, norm_type and values are INTERPRETATIONS, not source-proven facts.
Return exactly keys:
triples: [{"subject": "...", "predicate": "...", "object": "...", "evidence": "EXACT verbatim substring"}]
norm: short tentative social norm or empty string
norm_type: short general norm category or empty string
values: list of at most two items from the allowed names, or [].
Do not invent a quote; copy a contiguous substring from the passage.
"""

def extraction_user(passage: dict) -> str:
    allowed = ", ".join(VALUE_NAMES)
    return (
        "Culture: Kazakh\n"
        f"Source: {', '.join(passage['sources'])}\n"
        f"Category: {', '.join(passage['categories'])}\n"
        f"Allowed values: {allowed}\n"
        "Source passage (only evidence permitted):\n"
        + passage["text"]
    )


def _candidate_json(raw: str) -> dict | None:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def extract_frame(passage: dict, raw: str, *, model: str) -> dict:
    """Validate grounded quotes and preserve source/LLM provenance."""
    obj = _candidate_json(raw)
    triples = []
    if obj:
        for item in obj.get("triples", [])[:MAX_TRIPLES]:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            dest = str(item.get("object") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            # A quote verifies grounding location, NOT factual interpretation.
            if not (subj and predicate and dest and evidence and evidence in passage["text"]):
                continue
            triples.append({
                "triple_id": stable_id(passage["passage_id"], subj, predicate, dest, prefix="kzt_"),
                "subject": subj, "predicate": predicate, "object": dest,
                "evidence": evidence, "passage_id": passage["passage_id"],
                "sources": passage["sources"], "dataset": passage["dataset"],
                "generated": True, "extraction_method": "llm_verbatim_quote_candidate",
                "quote_verified": True, "review_status": "unreviewed",
            })
    allowed_values = obj.get("values", []) if obj else []
    if not isinstance(allowed_values, list):
        allowed_values = []
    values = [v for v in allowed_values if isinstance(v, str) and v in VALUE_NAMES][:2]
    norm = str(obj.get("norm") or "").strip()[:400] if obj else ""
    norm_type = str(obj.get("norm_type") or "").strip()[:180] if obj else ""
    status = "candidate" if triples else "failed"
    return {
        **passage, "triples": triples,
        "norm": norm if triples else "", "norm_type": norm_type if triples else "",
        "values": values if triples else [],
        "label_status": status, "interpretation_status": "provisional",
        "extraction_model": model,
    }


def pathways(frame: dict) -> list[dict]:
    """Each pathway is tied to ONE source passage; no cross-record stitching."""
    if frame["label_status"] != "candidate":
        return []
    return [{
        "pathway_id": stable_id(frame["passage_id"], t["triple_id"], prefix="kzw_"),
        "passage_id": frame["passage_id"],
        "culture": frame["culture"], "source": frame["sources"],
        "values": frame["values"], "norm_type": frame["norm_type"],
        "norm": frame["norm"], "triple_id": t["triple_id"],
        "evidence": t["evidence"], "provisional": True,
    } for t in frame["triples"]]


def audit_rows(frames: list[dict], n: int, seed: int = 42) -> list[dict]:
    """A source-grounded sample requiring human review; no automatic approval."""
    candidates = [(f, t) for f in frames for t in f["triples"]]
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} candidate triples, need {n} for audit")
    rnd = random.Random(seed)
    selected = rnd.sample(candidates, n)
    return [{
        "triple_id": t["triple_id"], "passage_id": f["passage_id"],
        "category": ", ".join(f["categories"]),
        "source": ", ".join(f["sources"]),
        "passage": f["text"],
        "subject": t["subject"], "predicate": t["predicate"],
        "object": t["object"], "verbatim_quote": t["evidence"],
        "norm_type_provisional": f["norm_type"],
        "norm_provisional": f["norm"],
        "values_provisional": "; ".join(f["values"]),
        "triple_correct_0_or_1": "",
        "pathway_correct_0_or_1": "",
        "reviewer_notes": "",
    } for f, t in selected]


def write_audit_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_audit(path: Path, *, min_rows: int = 50, min_agreement: float = 0.8) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manual audit required: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < min_rows:
        raise ValueError(f"Need at least {min_rows} audited rows, found {len(rows)}")
    for r in rows:
        if r["triple_correct_0_or_1"] not in ("0", "1") or r["pathway_correct_0_or_1"] not in ("0", "1"):
            raise ValueError("Audit has unanswered 0/1 quality fields; manual review is required")
    triple = sum(int(r["triple_correct_0_or_1"]) for r in rows) / len(rows)
    pathway = sum(int(r["pathway_correct_0_or_1"]) for r in rows) / len(rows)
    report = {
        "audited": len(rows), "triple_approval_rate": triple,
        "pathway_approval_rate": pathway, "threshold": min_agreement,
        "passed": triple >= min_agreement and pathway >= min_agreement,
    }
    if not report["passed"]:
        raise ValueError(f"Audit quality gate failed: {report}")
    return report
