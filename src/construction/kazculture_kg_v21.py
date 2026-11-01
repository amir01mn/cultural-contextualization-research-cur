"""KazCulture KG v2.1: conservative norms and better grounded relation direction.

This is a NEW, deliberately versioned prompt/validator. V2 original results
remain immutable, and automated validation is not a human correctness score.
"""
from __future__ import annotations

from collections import Counter

from .kazculture_kg_v2 import (
    aggregate_v2, parse_v2_chunk, extraction_user_v2,
    ALLOWED_PREDICATES, SCHEMA_VERSION as V2_SCHEMA,
)

SCHEMA_VERSION = "kazculture-kg-factual-normative-v2.1"
MAX_MODEL_TRIPLES = 3
ALLOWED_RULE_KINDS = frozenset({
    "explicit_obligation", "explicit_prohibition",
    "explicit_etiquette", "explicit_custom",
})
SYSTEM_V21 = """Extract Kazakh cultural facts and ONLY explicitly supported
social norms from a provided source passage. Never use outside knowledge.
Output ONE COMPACT JSON object only; no Markdown or explanatory prose.

JSON: {"triples":[{"subject":"...", "predicate":"...", "object":"...",
"evidence":"EXACT short source quote"}],
"pathways":[{"triple_index":0,"norm":"...", "norm_type":"...",
"values":["..."],"evidence":"EXACT short social-rule quote",
"rule_kind":"explicit_custom"}]}

Make 0-3 triples MAX. Keep each quote short (ideally <= 120 characters),
but include the span necessary to directly justify the complete relation.
Copy the source's exact words/spacing into each evidence quote.
Keep subjects/objects concise, grounded in the same quote, and in Kazakh.
Do NOT output the source title, source category or invented concepts.

Relationship direction MUST be accurate:
- BOWSTRING made from tendon: subject=BOWSTRING, predicate=MADE_FROM,
  object=tendon. Do NOT use BOW as subject.
- A wedding token or custom is NOT the same entity as a similarly named
  physical object. Do NOT conflate homonyms or related-sounding words.
- "X is performed at festivals" is descriptive (USED_IN), not "people
  must perform X" (REQUIRES). Never manufacture obligatory wording.
- "X may be given as a gift" is not "X is MADE_FROM those gifts."
- Do not write publication names as subjects/objects.
Use only these predicates: IS_A, MADE_FROM, USED_FOR, PART_OF,
HAS_PROPERTY, WORN_BY, LOCATED_IN, CREATED_BY, USED_IN, ASSOCIATED_WITH,
PERFORMED_DURING, SYMBOLIZES, PRACTICED_BY, REQUIRES,
OTHER_SUPPORTED_RELATION. If subject/predicate/object direction is uncertain,
omit the relation. The extra generic predicate is a last resort.

Add a pathway ONLY if the source explicitly states a SOCIAL expectation,
duty, taboo, etiquette rule, or names a custom as a practice.
Never infer a social obligation from a simple factual description, recipe,
garment, game rule, classification, or object construction.
Never infer a Schwartz value from general positive language, cooperation,
a game's rules, or the presence of the word 'traditional' alone.
A normative pathway must attach to the SAME cultural behavior as its
triple_index and be grounded by a separate, exact social-rule quote.
rule_kind MUST be one of: explicit_obligation, explicit_prohibition,
explicit_etiquette, explicit_custom.
Select 0-2 values only from the provided allowed Schwartz value labels.
Prefer pathways=[] if no justified cultural rule/value can be identified.
A legitimate factual triple does not need a pathway.
"""

def aggregate_v21(passage: dict, chunks: list[dict], responses: list[str],
                  *, model: str) -> dict:
    """V2 quote verification plus V2.1 normative-shape guard.

    We use the tested v2 parser's triple direction as MODEL PROPOSAL.
    Human review is still required; this guard only checks explicit output
    structure and source quote presence.
    """
    import json
    from .kazculture_kg import _candidate_json

    sanitized = []
    removed = Counter()
    for raw in responses:
        obj = _candidate_json(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("triples"), list):
            sanitized.append(raw)
            continue
        original_triples = obj["triples"]
        if len(original_triples) > MAX_MODEL_TRIPLES:
            removed["extra_triples_over_3"] += len(original_triples) - MAX_MODEL_TRIPLES
        proposed = obj.get("pathways", [])
        if not isinstance(proposed, list):
            proposed = []
        kept = []
        for path in proposed:
            if not isinstance(path, dict):
                removed["malformed_pathway"] += 1
                continue
            kind = path.get("rule_kind")
            if kind not in ALLOWED_RULE_KINDS:
                removed["missing_or_invalid_rule_kind"] += 1
                continue
            idx = path.get("triple_index")
            if type(idx) is not int or not 0 <= idx < MAX_MODEL_TRIPLES:
                removed["orphan_pathway"] += 1
                continue
            kept.append(path)
        sanitized.append(json.dumps(
            {"triples": original_triples[:MAX_MODEL_TRIPLES],
             "pathways": kept}, ensure_ascii=False,
        ))
    frame = aggregate_v2(passage, chunks, sanitized, model=model)
    frame["schema_version"] = SCHEMA_VERSION
    frame["v21_rejections"] = dict(removed)
    for triple in frame["triples"]:
        triple["extraction_method"] = SCHEMA_VERSION
        path = triple.get("pathway")
        if path is not None:
            path["review_status"] = "unreviewed"
            path["schema_version"] = SCHEMA_VERSION
    return frame
