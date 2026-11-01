"""KG v2.2: norm annotations nested in their source-grounded practice triple.

Research-dev protocol is frozen before running held-out pilot passages
150:300. Human judgments are still required; quote matching is necessary,
not sufficient, for either factual or normative correctness.

Crucial change from v2.1: no model-created array indices can become orphan
pathways after an unsupported triple is rejected. Unwarranted Schwartz value
links are never fabricated just to populate L3.
"""
from __future__ import annotations

from collections import Counter

from ..core.utils import stable_id
from .frame_builder import SCHWARTZ_VALUES
from .kazculture_kg import _candidate_json
from .kazculture_kg_v2 import (
    ALLOWED_PREDICATES, parse_v2_chunk, extraction_user_v2,
)
from .kazculture_kg_v21 import ALLOWED_RULE_KINDS

SCHEMA_VERSION = "kazculture-kg-nested-norms-v2.2"
MAX_MODEL_TRIPLES = 3
VALUES = frozenset(SCHWARTZ_VALUES)
FACT_ONLY_PREDICATES = frozenset({
    "MADE_FROM", "IS_A", "PART_OF", "HAS_PROPERTY",
    "LOCATED_IN", "CREATED_BY",
})

SYSTEM_V22 = """Extract ONLY evidence-supported facts about Kazakh culture from
the given SOURCE PASSAGE. Do not use outside knowledge or source metadata.
Return ONE short JSON object only. No commentary or Markdown.

Output:
{"triples":[
 {"subject":"Kazakh entity or behavior",
  "predicate":"USED_FOR",
  "object":"actual object, target, actor or practice",
  "evidence":"short EXACT contiguous quotation from SOURCE PASSAGE",
  "norm":null}
]}
Make 0 to 3 triples, NEVER more. If no well-supported relation, triples=[].
Possible predicates (USE EXACTLY): IS_A, MADE_FROM, USED_FOR, PART_OF,
HAS_PROPERTY, WORN_BY, LOCATED_IN, CREATED_BY, USED_IN, ASSOCIATED_WITH,
PERFORMED_DURING, SYMBOLIZES, PRACTICED_BY, REQUIRES,
OTHER_SUPPORTED_RELATION.

Relationship rules:
- If bowSTRING is made from tendon, only the bowSTRING is MADE_FROM tendon.
- A material, equipment part, or gift does not itself perform a custom.
- Distinguish a physical nose peg from a similarly named wedding ritual.
- A festival sport being played is a description, NOT an obligation.
- A foreign community's practice must not be generalized to Kazakhstan.
- Actor, action, object and quote must all describe the SAME relationship.
- Use exact text quotation from the SOURCE PASSAGE; keep it short.
- Never invent facts, motivations, duties or values.

Only if the triple expresses an explicit SOCIAL BEHAVIOR AND the source
states a social expectation, custom, taboo, or etiquette, its norm may be:
{"behavior":"the SAME behavior as this triple",
 "norm":"short cultural expectation supported by quoted text",
 "norm_type":"etiquette / taboo / duty / custom",
 "rule_kind":"explicit_custom",
 "evidence":"short EXACT quote for this social rule",
 "values":[]}
Otherwise its norm MUST be null. Allowed rule_kind: explicit_obligation,
explicit_prohibition, explicit_etiquette, explicit_custom.
A rule about game mechanics, object construction, food recipes, kinship
definitions or racing to a finish line IS NOT a social norm.

Only put a Schwartz value into values when the source gives DIRECT evidence
for that value, not merely because the text mentions a tradition, sport,
game rule or cooperation. Otherwise values=[] and the norm remains a
norm-only candidate, NOT an L3 pathway. Choose 0-2 labels from the allowed
values. Never invent L3 merely to build a complete three-layer pathway.

CRITICAL: norm belongs INSIDE its directly supporting triple. There is no
separate pathways array, no triple_index, and no cross-fact norm links.
"""


def aggregate_v22(passage: dict, chunks: list[dict], responses: list[str],
                  *, model: str) -> dict:
    """Conservatively validate nested factual triples/norms with exact quotes."""
    if not chunks or len(chunks) != len(responses):
        raise ValueError("Expected nonempty aligned chunk responses")
    if any(chunk["passage_id"] != passage["passage_id"] for chunk in chunks):
        raise ValueError("Cannot merge across passages")
    accepted = []
    seen = set()
    rejections = Counter()
    statuses = []
    norm_candidates = []
    for chunk_idx, (chunk, raw) in enumerate(zip(chunks, responses)):
        obj = _candidate_json(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("triples"), list):
            statuses.append("invalid_json")
            rejections["invalid_json_or_schema"] += 1
            continue
        statuses.append("parsed")
        propositions = obj["triples"]
        if len(propositions) > MAX_MODEL_TRIPLES:
            rejections["over_max_triples"] += len(propositions) - MAX_MODEL_TRIPLES
        for item in propositions[:MAX_MODEL_TRIPLES]:
            if not isinstance(item, dict):
                rejections["invalid_triple_schema"] += 1
                continue
            subj = str(item.get("subject") or "").strip()
            pred = str(item.get("predicate") or "").strip().upper()
            target = str(item.get("object") or "").strip()
            quote = str(item.get("evidence") or "").strip()
            if not (subj and target and quote):
                rejections["incomplete_triple"] += 1
                continue
            if pred not in ALLOWED_PREDICATES:
                rejections["invalid_predicate"] += 1
                continue
            if quote not in chunk["text"]:
                rejections["unverified_source_quote"] += 1
                continue
            if any(subj.casefold() == str(x).strip().casefold()
                   for x in chunk.get("sources", [])) and subj.casefold() not in quote.casefold():
                rejections["metadata_subject"] += 1
                continue
            if any(target.casefold() == str(x).strip().casefold()
                   for x in chunk.get("sources", [])) and target.casefold() not in quote.casefold():
                rejections["metadata_object"] += 1
                continue
            tid = stable_id(passage["passage_id"], subj, pred, target, prefix="kzt22_")
            if tid in seen:
                rejections["duplicate_triple"] += 1
                continue
            seen.add(tid)
            fact = {
                "triple_id": tid, "passage_id": passage["passage_id"],
                "subject": subj, "predicate": pred, "object": target,
                "evidence": quote, "quote_verified": True,
                "source": list(passage.get("sources", [])),
                "dataset": passage["dataset"],
                "source_chunk_index": chunk_idx,
                "source_start_char": chunk["_chunk_start"],
                "source_end_char": chunk["_chunk_end"],
                "generated": True, "review_status": "unreviewed",
                "extraction_method": SCHEMA_VERSION, "knowledge_kind": "factual",
                "pathway": None,
            }
            norm_obj = item.get("norm")
            if norm_obj is None:
                pass
            elif not isinstance(norm_obj, dict):
                rejections["bad_norm_schema"] += 1
            elif pred in FACT_ONLY_PREDICATES:
                rejections["norm_on_descriptive_predicate"] += 1
            else:
                behavior = str(norm_obj.get("behavior") or "").strip()
                norm = str(norm_obj.get("norm") or "").strip()[:400]
                norm_type = str(norm_obj.get("norm_type") or "").strip()[:180]
                kind = str(norm_obj.get("rule_kind") or "").strip()
                nquote = str(norm_obj.get("evidence") or "").strip()
                vals = norm_obj.get("values")
                if not isinstance(vals, list):
                    vals = []
                clean_values = list(dict.fromkeys(
                    v for v in vals if isinstance(v, str) and v in VALUES
                ))[:2]
                if kind not in ALLOWED_RULE_KINDS:
                    rejections["bad_rule_kind"] += 1
                elif not (behavior and norm and norm_type and nquote and nquote in chunk["text"]):
                    rejections["unsupported_norm_quote_or_schema"] += 1
                else:
                    norm_candidate = {
                        "triple_id": tid, "passage_id": passage["passage_id"],
                        "behavior": behavior, "norm": norm, "norm_type": norm_type,
                        "rule_kind": kind, "evidence": nquote,
                        "values": clean_values, "quote_verified": True,
                        "review_status": "unreviewed", "provisional": True,
                    }
                    fact["norm_candidate"] = norm_candidate
                    norm_candidates.append(norm_candidate)
                    if clean_values:
                        path = dict(norm_candidate)
                        path["pathway_id"] = stable_id(tid, norm, prefix="kzw22_")
                        fact["pathway"] = path
                        fact["knowledge_kind"] = "normative"
                    else:
                        fact["knowledge_kind"] = "norm_only"
            accepted.append(fact)

    pathways = [t["pathway"] for t in accepted if t["pathway"] is not None]
    return {
        **passage,
        "schema_version": SCHEMA_VERSION, "triples": accepted,
        "pathways": pathways, "norm_candidates": norm_candidates,
        "norm": pathways[0]["norm"] if len(pathways) == 1 else "",
        "norm_type": pathways[0]["norm_type"] if len(pathways) == 1 else "",
        "values": pathways[0]["values"] if len(pathways) == 1 else [],
        "chunk_count": len(chunks), "chunk_status": statuses,
        "rejections": dict(rejections), "label_status": "candidate" if accepted else "failed",
        "interpretation_status": "provisional", "extraction_model": model,
    }
