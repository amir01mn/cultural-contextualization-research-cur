"""KazCulture KG pilot v2: separately auditable facts and cultural norms.

The schema is deliberately narrower than the October pilot. A relation can be
a useful FACT without an L2/L3 pathway. Norms/values require distinct quoted
social-behavior evidence; model-proposed labels are provisional, not truth.
No source title, category, train answer, or external knowledge becomes an edge.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from ..core.utils import stable_id
from .frame_builder import SCHWARTZ_VALUES
from .kazculture_kg import _candidate_json

SCHEMA_VERSION = "kazculture-kg-factual-normative-v2"
ALLOWED_PREDICATES = frozenset({
    "IS_A", "MADE_FROM", "USED_FOR", "PART_OF", "HAS_PROPERTY",
    "WORN_BY", "LOCATED_IN", "CREATED_BY", "USED_IN", "ASSOCIATED_WITH",
    "PERFORMED_DURING", "SYMBOLIZES", "PRACTICED_BY", "REQUIRES",
    "OTHER_SUPPORTED_RELATION",
})
VALUES = frozenset(SCHWARTZ_VALUES)
MAX_TRIPLES = 5

SYSTEM_V2 = """You are extracting structured, source-grounded Kazakh cultural knowledge.
You MUST use only the SOURCE PASSAGE, not metadata, the book title, or outside facts.
Produce one strict JSON object with the two keys "triples" and "pathways".
Do not output explanations or Markdown. Use Kazakh names where possible.

FACT EXTRACTION: For each supported relation return:
{"subject":"entity actually acting or described","predicate":"ALLOWED_PREDICATE",
"object":"actual target/material/purpose","evidence":"exact contiguous substring from SOURCE PASSAGE"}
Allowed predicates: IS_A, MADE_FROM, USED_FOR, PART_OF, HAS_PROPERTY, WORN_BY,
LOCATED_IN, CREATED_BY, USED_IN, ASSOCIATED_WITH, PERFORMED_DURING,
SYMBOLIZES, PRACTICED_BY, REQUIRES, OTHER_SUPPORTED_RELATION.
Use no more than 5 triples, or [] if unsupported. A description can be
valid factual knowledge without a social norm. Ensure the relationship
direction is correct. Example: if a BOWSTRING is made of tendon, the
BOWSTRING is the subject of MADE_FROM, NOT the whole bow. A book title or
source label is never an entity unless it is literally part of the passage.

NORM EXTRACTION: Return "pathways":[] unless the passage explicitly describes
a SOCIAL EXPECTATION, socially prescribed BEHAVIOR, duty, taboo, or custom.
Descriptions of objects, materials, games, clothing, and recipes alone do not
imply tradition, benevolence, conformity, or any other value.
Only for a fact associated with an explicit social norm, add:
{"triple_index":0,"norm":"short source-supported social expectation",
"norm_type":"short general social rule type",
"values":["one or two exact Schwartz value names"],
"evidence":"exact contiguous quote supporting the social expectation"}
triple_index is the ZERO-based index into your triples array. Norm evidence
must be in the source and distinct from a mere technical description.
Use the provided allowed values; prefer [] when the underlying value is
not supported. Never force a factual triple into an L3->L2->L1 pathway.
"""

def extraction_user_v2(passage: dict) -> str:
    """No source title, category, record ID or test-related metadata in prompt."""
    allowed = ", ".join(sorted(VALUES))
    return (
        "Culture: Kazakh.\n"
        f"Allowed Schwartz value labels: {allowed}\n"
        "SOURCE PASSAGE (only source of facts and norms):\n"
        + passage["text"]
    )


def parse_v2_chunk(passage: dict, raw: str, *, model: str) -> dict:
    """Parse one model-visible chunk into quote-verified, still-unreviewed data."""
    obj = _candidate_json(raw)
    reject: Counter = Counter()
    facts: list[dict] = []
    original_indices = {}
    if not isinstance(obj, dict) or not isinstance(obj.get("triples"), list):
        reject["invalid_json_or_schema"] += 1
        proposals = []
        candidate_paths = []
    else:
        proposals = obj["triples"][:MAX_TRIPLES]
        candidate_paths = obj.get("pathways")
        if not isinstance(candidate_paths, list):
            candidate_paths = []
            reject["invalid_pathway_schema"] += 1
        if len(obj["triples"]) > MAX_TRIPLES:
            reject["over_max_triples"] += len(obj["triples"]) - MAX_TRIPLES

    for idx, item in enumerate(proposals):
        if not isinstance(item, dict):
            reject["invalid_triple_schema"] += 1
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip().upper()
        target = str(item.get("object") or "").strip()
        quote = str(item.get("evidence") or "").strip()
        if not subject or not target or not quote:
            reject["incomplete_triple"] += 1
            continue
        if predicate not in ALLOWED_PREDICATES:
            reject["invalid_predicate"] += 1
            continue
        if quote not in passage["text"]:
            reject["unverified_source_quote"] += 1
            continue
        if any(subject.casefold() == str(source).strip().casefold() for source in
               passage.get("sources", [])) and subject.casefold() not in quote.casefold():
            reject["source_metadata_as_subject"] += 1
            continue
        if any(target.casefold() == str(source).strip().casefold() for source in
               passage.get("sources", [])) and target.casefold() not in quote.casefold():
            reject["source_metadata_as_object"] += 1
            continue
        triple = {
            "triple_id": stable_id(passage["passage_id"], subject,
                                    predicate, target, prefix="kzt2_"),
            "passage_id": passage["passage_id"],
            "subject": subject, "predicate": predicate, "object": target,
            "evidence": quote,
            "source": list(passage.get("sources", [])),
            "dataset": "issai/KazCulture", "generated": True,
            "quote_verified": True,
            "subject_in_quote": subject.casefold() in quote.casefold(),
            "object_in_quote": target.casefold() in quote.casefold(),
            "extraction_method": SCHEMA_VERSION,
            "review_status": "unreviewed",
            "knowledge_kind": "factual",
            "pathway": None,
        }
        original_indices[idx] = len(facts)
        facts.append(triple)

    used = set()
    for path in candidate_paths:
        if not isinstance(path, dict):
            reject["invalid_pathway_schema"] += 1
            continue
        i = path.get("triple_index")
        if type(i) is not int or i not in original_indices:
            reject["orphan_pathway"] += 1
            continue
        if i in used:
            reject["duplicate_pathway"] += 1
            continue
        norm = str(path.get("norm") or "").strip()[:400]
        norm_type = str(path.get("norm_type") or "").strip()[:180]
        evidence = str(path.get("evidence") or "").strip()
        values = path.get("values")
        if not isinstance(values, list):
            values = []
        values = list(dict.fromkeys(v for v in values
                                   if isinstance(v, str) and v in VALUES))[:2]
        if not (norm and norm_type and evidence and evidence in passage["text"] and values):
            reject["unsupported_or_incomplete_pathway"] += 1
            continue
        j = original_indices[i]
        facts[j]["knowledge_kind"] = "normative"
        facts[j]["pathway"] = {
            "pathway_id": stable_id(facts[j]["triple_id"], norm, prefix="kzw2_"),
            "triple_id": facts[j]["triple_id"],
            "passage_id": passage["passage_id"],
            "norm": norm, "norm_type": norm_type, "values": values,
            "evidence": evidence, "quote_verified": True,
            "rule_kind": str(path.get("rule_kind") or ""),
            "review_status": "unreviewed", "provisional": True,
        }
        used.add(i)

    return {
        "triples": facts,
        "parse_status": "parsed" if isinstance(obj, dict) else "invalid_json",
        "rejections": dict(reject),
        "extraction_model": model,
    }


def aggregate_v2(passage: dict, chunks: list[dict], responses: list[str],
                 *, model: str) -> dict:
    """Merge overlapping original-text chunks without conflating local norms."""
    if not chunks or len(chunks) != len(responses):
        raise ValueError("One response per non-empty chunk required")
    if any(p["passage_id"] != passage["passage_id"] for p in chunks):
        raise ValueError("Cross-passage merge forbidden")
    triples, seen, counts = [], set(), Counter()
    states = []
    for idx, (part, raw) in enumerate(zip(chunks, responses)):
        parsed = parse_v2_chunk(part, raw, model=model)
        counts.update(parsed["rejections"])
        states.append(parsed["parse_status"])
        for t in parsed["triples"]:
            if t["triple_id"] in seen:
                continue
            seen.add(t["triple_id"])
            triples.append({
                **t, "source_chunk_index": idx,
                "source_start_char": part["_chunk_start"],
                "source_end_char": part["_chunk_end"],
            })
    paths = [t["pathway"] for t in triples if t["pathway"] is not None]
    return {
        **passage, "schema_version": SCHEMA_VERSION,
        "triples": triples, "pathways": paths,
        # Backward-compatible index fields remain empty unless a single
        # unambiguous normative pathway exists. Fact-only records have none.
        "norm": paths[0]["norm"] if len(paths) == 1 else "",
        "norm_type": paths[0]["norm_type"] if len(paths) == 1 else "",
        "values": paths[0]["values"] if len(paths) == 1 else [],
        "label_status": "candidate" if triples else "failed",
        "chunk_count": len(chunks), "chunk_status": states,
        "rejections": dict(counts), "interpretation_status": "provisional",
        "extraction_model": model,
    }


def make_v2_audit(frames: list[dict], *, n: int = 60, seed: int = 42) -> list[dict]:
    """Sample all triples equally. Human fields intentionally stay empty."""
    population = [(f, t) for f in frames for t in f["triples"]]
    if len(population) < n:
        raise ValueError(f"Only {len(population)} triples, requested {n}")
    selected = random.Random(seed).sample(population, n)
    result = []
    for frame, t in selected:
        path = t.get("pathway")
        result.append({
            "triple_id": t["triple_id"], "passage_id": frame["passage_id"],
            "category": "; ".join(frame["categories"]),
            "source": "; ".join(frame["sources"]),
            "passage": frame["text"],
            "subject": t["subject"], "predicate": t["predicate"],
            "object": t["object"], "verbatim_quote": t["evidence"],
            "subject_in_quote_automated": int(t["subject_in_quote"]),
            "object_in_quote_automated": int(t["object_in_quote"]),
            "model_proposed_pathway_0_or_1": int(path is not None),
            "rule_kind_provisional": path.get("rule_kind", "") if path else "",
            "norm_type_provisional": path["norm_type"] if path else "",
            "norm_provisional": path["norm"] if path else "",
            "values_provisional": "; ".join(path["values"]) if path else "",
            "norm_evidence_quote": path["evidence"] if path else "",
            "triple_correct_0_or_1": "",
            "normative_eligible_0_or_1": "",
            "pathway_correct_0_or_1_if_proposed": "",
            "reviewer_notes": "",
        })
    return result


def save_v2_audit(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No audit rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Review audit already exists; never overwrite decisions")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_v2_audit(path: Path, *, min_rows: int = 40,
                       min_correctness: float = 0.8,
                       min_proposed_paths: int = 8) -> dict:
    """Separate triple precision, pathway precision and eligible coverage.

    Absence of a proposed pathway is NOT counted as an incorrect pathway.
    Human normative eligibility detects missed paths independently.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < min_rows:
        raise ValueError(f"Need >= {min_rows} reviewed rows, got {len(rows)}")
    for row in rows:
        if row["triple_correct_0_or_1"] not in ("0", "1"):
            raise ValueError("Every triple requires independent human 0/1 review")
        if row["normative_eligible_0_or_1"] not in ("0", "1"):
            raise ValueError("Every triple requires human normative eligibility review")
        proposed = row["model_proposed_pathway_0_or_1"] == "1"
        correct = row["pathway_correct_0_or_1_if_proposed"]
        if proposed and correct not in ("0", "1"):
            raise ValueError("Proposed pathways require human 0/1 correctness review")
        if not proposed and correct not in ("",):
            raise ValueError("Do not score a non-existent pathway as correct or incorrect")

    n = len(rows)
    n_good = sum(int(r["triple_correct_0_or_1"]) for r in rows)
    proposed = [r for r in rows if r["model_proposed_pathway_0_or_1"] == "1"]
    true_eligible = sum(int(r["normative_eligible_0_or_1"]) for r in rows)
    accepted = sum(r["pathway_correct_0_or_1_if_proposed"] == "1"
                   and r["normative_eligible_0_or_1"] == "1"
                   and r["triple_correct_0_or_1"] == "1" for r in proposed)
    precision = accepted / len(proposed) if proposed else None
    coverage = accepted / true_eligible if true_eligible else None
    verdict = ("pass" if n_good / n >= min_correctness
               and len(proposed) >= min_proposed_paths
               and precision is not None and precision >= min_correctness
               else "not_ready_to_scale")
    return {
        "audited_triples": n,
        "triple_precision": n_good / n,
        "model_proposed_pathways": len(proposed),
        "human_normatively_eligible": true_eligible,
        "pathway_precision_on_proposed": precision,
        "eligible_pathway_coverage": coverage,
        "minimum_required_proposals": min_proposed_paths,
        "pilot_gate": verdict,
        "note": "Automated labels are not human-reviewed. No extrapolation to full corpus.",
    }
