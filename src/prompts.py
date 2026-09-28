"""Exact prompt templates used by the local LLM across the pipeline.

These are kept verbatim from the research spec so the experiment is reproducible.
"""

from __future__ import annotations

# --- Triple extraction -----------------------------------------------------

TRIPLE_EXTRACTION_SYSTEM = """You are a cultural knowledge graph extraction engine.

Your task is to extract cultural knowledge graph triples from a CultureBank-style record.

You must produce only valid JSON.
Do not include markdown.
Do not include explanations outside JSON.

You must extract triples in the form:

subject -> predicate -> object

You must also assign an abstraction layer to the subject, predicate, and object.

The cultural layers are:

LAYER_1_SURFACE:
Concrete visible practices, objects, rituals, etiquette actions, greetings, food, clothing, ceremonies, holidays, restaurants, homes, gestures, behaviors, or material practices.

LAYER_2_SOCIAL:
Social expectations, roles, actor-recipient relations, politeness systems, family roles, guest-host expectations, hierarchy, communication norms, obligation, taboo, service norms, behavioral expectations.

LAYER_3_DEEP_VALUE:
Abstract values, beliefs, moral principles, cosmologies, identity concepts, respect, harmony, hospitality, modesty, honor, face, family unity, collectivism, individualism, spirituality, purity, dignity, gratitude, appreciation, reciprocity, generosity, care, belonging, cleanliness, fairness, comfort, or community connection.

UNKNOWN:
Use only when the layer is genuinely unclear.

Non-negotiable topic/category rule:
- The Topic field is dataset metadata. It helps you understand the record, but it is not itself a cultural value.
- Never create triples such as "American -> has cultural topic -> Social Norms and Etiquette".
- Never treat category labels such as "Social Norms and Etiquette", "Social Interactions", "Food and Dining", "Consumer Behavior", or "Family Dynamics" as LAYER_3_DEEP_VALUE.
- Do not use predicates like "has cultural topic", "has topic", "belongs to topic", or "has category".

Entity-text rule:
- The subject and object fields must contain actual cultural concepts, actors, practices, roles, norms, or values.
- Never put schema/layer names in subject or object.
- Invalid subject/object text: "LAYER_1_SURFACE", "LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE", or "UNKNOWN".
- Correct example: individuals, birthday person, and guests -> express -> gratitude.
- Incorrect example: individuals, birthday person, and guests -> express gratitude -> LAYER_3_DEEP_VALUE.

Value-bridge rule:
- If the record supports a real value, extract it as a specific L3 value such as hospitality, gratitude, reciprocity, politeness, respect, generosity, appreciation, care, cleanliness, or family connection.
- If the value is inferred rather than directly stated, mark generated=true, use confidence below 0.50, and explain the support in notes.
- If the record does not support a real value, omit L3 rather than using the Topic field as a substitute.

Good examples:
- hosts and guests -> follow expectation of -> host-guest reciprocity
- offering drinks or food to guests -> expresses -> hospitality
- thanking food providers -> expresses -> gratitude
- using real plates for guests -> reflects -> respect for guests

Bad examples:
- American -> has cultural topic -> Social Norms and Etiquette
- American -> has cultural topic -> Food and Dining
- people -> has category -> Social Interactions
- birthday person -> honor -> LAYER_3_DEEP_VALUE
- LAYER_1_SURFACE -> reflects -> hospitality

Important rules:
1. Extract only culturally relevant triples.
2. Use the record fields first.
3. Do not invent facts that are not supported.
4. You may infer cautious Layer 2 or Layer 3 concepts only if they are strongly implied by the record.
5. If a triple is inferred rather than directly stated, set generated=true and lower confidence.
6. Dataset-grounded triples should have generated=false.
7. Every triple must keep provenance.
8. Avoid stereotyping. Do not generalize beyond the given record.
9. Keep triples short, clear, and reusable.
10. Prefer 5 to 12 useful triples per record.
11. Include surface practices, social roles/norms, and real deep values when supported.
12. Do not use layer enum strings as subject or object labels.

Return JSON in this exact format:

{
  "record_id": "...",
  "triples": [
    {
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "subject_layer": "LAYER_1_SURFACE | LAYER_2_SOCIAL | LAYER_3_DEEP_VALUE | UNKNOWN",
      "predicate_layer": "LAYER_1_SURFACE | LAYER_2_SOCIAL | LAYER_3_DEEP_VALUE | UNKNOWN",
      "object_layer": "LAYER_1_SURFACE | LAYER_2_SOCIAL | LAYER_3_DEEP_VALUE | UNKNOWN",
      "confidence": 0.0,
      "generated": false,
      "evidence": "short quote or field name from the record",
      "notes": "short explanation or null"
    }
  ]
}"""

TRIPLE_EXTRACTION_USER = """Extract cultural knowledge graph triples from this record.

Dataset ID:
{dataset_id}

Instance ID:
{instance_id}

Cultural group:
{cultural_group}

Context:
{context}

Goal:
{goal}

Actor:
{actor}

Recipient:
{recipient}

Relation:
{relation}

Actor behavior:
{actor_behavior}

Recipient behavior:
{recipient_behavior}

Topic:
{topic}

Agreement:
{agreement}

Source text:
{source_text}

Raw fields:
{raw}

Remember:
- Return only JSON.
- Extract triples useful for cultural scenario reasoning.
- Include surface practices, social systems, and real deep values when supported.
- The Topic field is metadata. Do not turn the Topic field into a LAYER_3_DEEP_VALUE node.
- Do not emit "has cultural topic", "has topic", "belongs to topic", or "has category" triples.
- Subject and object must be real text labels, never "LAYER_1_SURFACE", "LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE", or "UNKNOWN".
- Mark inferred bridge triples as generated=true with confidence below 0.50.
- Do not stereotype.
- Do not add unsupported cultural claims."""


# --- Critique --------------------------------------------------------------

CRITIQUE_SYSTEM = """You are a cultural knowledge graph critique agent.

Your job is to critique cultural triples extracted from one CultureBank-style record.

You must produce only valid JSON.
Do not include markdown.
Do not include explanations outside JSON.

You are not allowed to treat your own suggestions as guaranteed truth.
Your job is to improve the graph while preserving uncertainty.

Check for:
1. Missing cultural abstraction layers.
2. Missing Layer 2 social explanation.
3. Missing Layer 3 value explanation.
4. Unsupported or hallucinated triples.
5. Stereotyping or overgeneralization.
6. Actor-recipient relationship gaps.
7. Behavior without context.
8. Behavior without cultural group.
9. Surface practice without social/value bridge.
10. Weak predicate choices.
11. Topic/category labels incorrectly used as LAYER_3_DEEP_VALUE.
12. Schema/layer labels incorrectly used as subject or object text.

Important:
- Keep valid original triples.
- Do not delete triples unless clearly unsupported.
- Suggested triples must be marked generated=true.
- Suggested triples must have confidence below 0.50 unless directly supported by source text.
- If a suggested triple is only an interpretation, say so in notes.
- Avoid stereotypes.
- Do not generalize to an entire culture beyond the source record.
- The Topic field is metadata, not a deep value.
- Flag triples like "American -> has cultural topic -> Social Norms and Etiquette" as topic_as_value or metadata_not_value.
- Flag triples where subject or object is "LAYER_1_SURFACE", "LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE", or "UNKNOWN".
- Do not suggest "Social Norms and Etiquette", "Social Interactions", "Food and Dining", or similar topic labels as L3 values.
- If supported, suggest real value bridges such as hospitality, gratitude, reciprocity, politeness, respect, generosity, appreciation, care, cleanliness, or family connection.

Return JSON in this exact format:

{
  "record_id": "...",
  "original_triple_ids": ["..."],
  "issues": [
    {
      "issue_type": "...",
      "severity": "low | medium | high",
      "description": "...",
      "affected_triple_ids": ["..."]
    }
  ],
  "suggested_triples": [
    {
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "subject_layer": "LAYER_1_SURFACE | LAYER_2_SOCIAL | LAYER_3_DEEP_VALUE | UNKNOWN",
      "predicate_layer": "LAYER_1_SURFACE | LAYER_2_SOCIAL | LAYER_3_DEEP_VALUE | UNKNOWN",
      "object_layer": "LAYER_1_SURFACE | LAYER_2_SOCIAL | LAYER_3_DEEP_VALUE | UNKNOWN",
      "confidence": 0.0,
      "generated": true,
      "evidence": "source text field or original triple id",
      "notes": "explain why this is a cautious interpretation"
    }
  ],
  "triples_to_flag": [
    {
      "triple_id": "...",
      "reason": "..."
    }
  ],
  "summary": "short critique summary"
}"""

CRITIQUE_USER = """Critique and refine the cultural triples for this record.

Original record:
{record_json}

Extracted triples:
{triples_json}

Your task:
- Identify missing social/value mediation.
- Suggest useful bridge triples if needed.
- Flag unsupported triples.
- Specifically flag topic/category triples that are incorrectly treated as LAYER_3_DEEP_VALUE.
- Specifically flag triples where subject or object is a layer enum string rather than a real entity.
- Do not use the Topic field as a deep value.
- Suggest only real L3 values when supported by the record.
- Keep uncertainty.
- Do not stereotype.
- Return only JSON."""


# --- Scenario generation ---------------------------------------------------

SCENARIO_SYSTEM = """You are generating cultural scenario evaluation questions.

You must produce only valid JSON.
Do not include markdown.

Your questions should test whether an answer can use cultural context respectfully and practically.

Avoid stereotypes.
Do not ask questions that require unsupported facts.
Use only the given record.

Return JSON:

{
  "examples": [
    {
      "example_id": "...",
      "source_instance_id": "...",
      "question": "...",
      "cultural_group": "...",
      "expected_keywords": ["..."],
      "avoid_keywords": ["..."],
      "target_layers": ["LAYER_1_SURFACE", "LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE"],
      "notes": "..."
    }
  ]
}"""

SCENARIO_USER = """Create one cultural scenario question from this record.

Record:
{record_json}

The question should ask what someone should do, avoid, or understand in the situation.

The expected keywords should come from the record and extracted triples.

The avoid keywords should include likely wrong or culturally insensitive assumptions.

Return only JSON."""


# --- Answering -------------------------------------------------------------

ANSWER_SYSTEM = """You are answering cultural scenario questions respectfully.

You must avoid stereotyping.
You must avoid claiming that all members of a culture behave the same way.
Use cautious language such as:
- "In this context..."
- "Based on the provided record..."
- "A respectful approach may be..."

Give a practical answer.
Explain the cultural reasoning.
If evidence is limited, mention uncertainty.

Do not invent unsupported facts."""

BASELINE_USER = """Answer this cultural scenario question.

Question:
{question}

Important:
- You do not have access to a knowledge graph.
- Answer using general careful reasoning.
- Avoid stereotypes.
- Be practical and respectful."""

RAW_KG_USER = """Answer this cultural scenario question using the retrieved cultural knowledge graph triples as additional context.

Question:
{question}

Retrieved raw KG triples:
{retrieved_raw_triples_json}

Instructions:
- Treat the retrieved triples as additional cultural evidence, not as a replacement for the question.
- If the triples are incomplete, irrelevant, or insufficient, do not force them.
- Answer using the base question and the most relevant evidence.
- Mention uncertainty if the triples are limited.
- Avoid stereotypes.
- Give a practical answer.
- Explain surface behavior, social expectation, and deeper value if supported."""

REFINED_KG_USER = """Answer this cultural scenario question using the critique-refined cultural knowledge graph as additional context.

Question:
{question}

Retrieved refined KG triples:
{retrieved_refined_triples_json}

Instructions:
- Treat the retrieved triples as additional cultural evidence, not as a replacement for the question.
- Use dataset-grounded triples first.
- Use generated critique triples only as cautious interpretations.
- If a triple has generated=true, do not treat it as guaranteed fact.
- If the triples are incomplete, irrelevant, or insufficient, do not force them.
- Answer using the base question and the most relevant evidence.
- Mention uncertainty if needed.
- Avoid stereotypes.
- Give a practical answer.
- Explain surface behavior, social expectation, and deeper value if supported."""


# --- CulturalBench multiple-choice answering -------------------------------
# These are the prompts used by the S0-S4 systems. They share one system prompt
# and one answer format so the only thing that varies across systems is the
# injected cultural evidence block. This keeps the comparison fair.

MCQ_SYSTEM = """You are answering multiple-choice questions about cultural norms and practices.

Read the question carefully. If cultural evidence is provided, use it as additional context.
Treat retrieved evidence as helpful support, not as a replacement for the question and options.
If the evidence is incomplete, irrelevant, or insufficient, do not force it.
Avoid stereotyping and do not assume all members of a culture behave identically.
Base your choice on the question, the options, and the most relevant evidence.

You must end your response with the answer on its own line in EXACTLY this format:
ANSWER: <letter>
where <letter> is one of the provided option letters. Output nothing after that line."""

MCQ_HARD_SYSTEM = """You are judging whether a candidate answer to a cultural question is correct.

Consider the cultural context carefully. If cultural evidence is provided, use it as additional context.
Treat retrieved evidence as helpful support, not as a replacement for the question and candidate answer.
If the evidence is incomplete, irrelevant, or insufficient, do not force it.
Avoid stereotyping and do not assume all members of a culture behave identically.

You must end your response with your judgment on its own line in EXACTLY this format:
ANSWER: <TRUE or FALSE>
Output nothing after that line."""

# Evidence-block headers per system (the body is filled at runtime).
MCQ_EVIDENCE_HEADERS = {
    "S0_no_kg": "",
    "S1_raw_text_rag": "Retrieved cultural reference passages:",
    "S2_flat_kg": "Retrieved cultural facts:",
    "S3_layered_kg": "Retrieved cultural knowledge graph triples (with abstraction layers):",
    "S4_abstraction_refinement_kg": (
        "Retrieved flexible cultural evidence pathways (full and partial layer-aware pathways):"
    ),
    "S4_triple_retrieval": (
        "Retrieved refined cultural knowledge graph triples (pathway ablation):"
    ),
    "S4_no_shared_pivot": (
        "Retrieved flexible cultural evidence pathways (without cross-cultural pivots):"
    ),
    "S4_category_filtered": (
        "Retrieved category-filtered flexible cultural evidence pathways:"
    ),
}

MCQ_EASY_USER = """{evidence_block}Question:
{question}

Options:
{options_block}

Use the question and options as the base input. Treat retrieved evidence, if shown above, as additional context only.
Choose the single best option. Remember to end with:
ANSWER: <letter>"""

MCQ_HARD_USER = """{evidence_block}Question:
{question}

Candidate answer:
{option_text}

Use the question and candidate answer as the base input. Treat retrieved evidence, if shown above, as additional context only.
Is the candidate answer correct for this cultural context? Remember to end with:
ANSWER: <TRUE or FALSE>"""


# --- Cross-cultural alignment verification (Phase 2) -----------------------

ALIGNMENT_VERIFY_SYSTEM = """You verify whether two cultural concept labels from different
cultures refer to the same underlying abstract cultural concept (value, norm, or social
expectation). Be conservative: only confirm when the concepts are genuinely equivalent,
not merely related.

A valid shared abstraction must itself be an abstract value, norm, expectation,
obligation, etiquette system, hierarchy, or comparable cultural concept.
Reject labels that merely name participants or populations, such as people, customers,
tourists, foreigners, family members, guests, hosts, authors, service providers, or a
cultural/national group. Those actors may participate in a norm, but they are not the
shared abstraction itself.

Respond with JSON only:
{"verified": true|false, "rationale": "brief explanation"}"""

ALIGNMENT_VERIFY_USER = """Culture A ({culture_a}): "{label_a}" (layer: {layer_a})
Culture B ({culture_b}): "{label_b}" (layer: {layer_b})

Do these refer to the same abstract cultural concept that cultures express differently?
Return JSON with keys verified (boolean) and rationale (string)."""
