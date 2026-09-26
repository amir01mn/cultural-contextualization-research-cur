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
Abstract values, beliefs, moral principles, cosmologies, identity concepts, respect, harmony, hospitality, modesty, honor, face, family unity, collectivism, individualism, spirituality, purity, dignity.

UNKNOWN:
Use only when the layer is genuinely unclear.

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
- Include surface practices, social systems, and deep values when supported.
- Mark inferred bridge triples as generated=true.
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

Important:
- Keep valid original triples.
- Do not delete triples unless clearly unsupported.
- Suggested triples must be marked generated=true.
- Suggested triples must have confidence below 0.50 unless directly supported by source text.
- If a suggested triple is only an interpretation, say so in notes.
- Avoid stereotypes.
- Do not generalize to an entire culture beyond the source record.

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

RAW_KG_USER = """Answer this cultural scenario question using the retrieved cultural knowledge graph triples.

Question:
{question}

Retrieved raw KG triples:
{retrieved_raw_triples_json}

Instructions:
- Use only the retrieved triples as cultural evidence.
- Mention uncertainty if the triples are limited.
- Avoid stereotypes.
- Give a practical answer.
- Explain surface behavior, social expectation, and deeper value if supported."""

REFINED_KG_USER = """Answer this cultural scenario question using the critique-refined cultural knowledge graph.

Question:
{question}

Retrieved refined KG triples:
{retrieved_refined_triples_json}

Instructions:
- Use dataset-grounded triples first.
- Use generated critique triples only as cautious interpretations.
- If a triple has generated=true, do not treat it as guaranteed fact.
- Mention uncertainty if needed.
- Avoid stereotypes.
- Give a practical answer.
- Explain surface behavior, social expectation, and deeper value if supported."""


# --- CulturalBench multiple-choice answering -------------------------------
# These are the prompts used by the S0-S3 systems. They share one system prompt
# and one answer format so the only thing that varies across systems is the
# injected cultural evidence block. This keeps the comparison fair.

MCQ_SYSTEM = """You are answering multiple-choice questions about cultural norms and practices.

Read the question carefully.

Retrieved cultural evidence may be provided. It is retrieved automatically and is
NOT guaranteed to be relevant:
- Use it only when it directly addresses the question and the culture being asked about.
- If it concerns a different culture, a different topic, or does not help decide
  between the options, IGNORE it and answer from your own knowledge.
- Never pick an option merely because its wording resembles the evidence.

Avoid stereotyping and do not assume all members of a culture behave identically.
Base your choice on the most likely culturally appropriate answer.

You must end your response with the answer on its own line in EXACTLY this format:
ANSWER: <letter>
where <letter> is one of the provided option letters. Output nothing after that line."""

MCQ_HARD_SYSTEM = """You are judging whether a candidate answer to a cultural question is correct.

Consider the cultural context carefully.

Retrieved cultural evidence may be provided. It is retrieved automatically and is
NOT guaranteed to be relevant:
- Use it only when it directly addresses the question and the culture being asked about.
- If it concerns a different culture, a different topic, or does not help decide,
  IGNORE it and judge from your own knowledge.
- Never judge TRUE merely because the statement's wording resembles the evidence.

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
        "Retrieved abstraction→refinement pathways (L3→L2→L1):"
    ),
    "S4_triple_retrieval": (
        "Retrieved refined cultural knowledge graph triples (pathway ablation):"
    ),
    "S4_no_shared_pivot": (
        "Retrieved abstraction→refinement pathways (without cross-cultural pivots):"
    ),
}

MCQ_EASY_USER = """{evidence_block}Question:
{question}

Options:
{options_block}

Choose the single best option. Remember to end with:
ANSWER: <letter>"""

MCQ_HARD_USER = """{evidence_block}Question:
{question}

Candidate answer:
{option_text}

Is the candidate answer correct for this cultural context? Remember to end with:
ANSWER: <TRUE or FALSE>"""


# --- Cross-cultural alignment verification (Phase 2) -----------------------

ALIGNMENT_VERIFY_SYSTEM = """You verify whether two cultural concept labels from different
cultures refer to the same underlying abstract cultural concept (value, norm, or social
expectation). Be conservative: only confirm when the concepts are genuinely equivalent,
not merely related.

Respond with JSON only:
{"verified": true|false, "rationale": "brief explanation"}"""

ALIGNMENT_VERIFY_USER = """Culture A ({culture_a}): "{label_a}" (layer: {layer_a})
Culture B ({culture_b}): "{label_b}" (layer: {layer_b})

Do these refer to the same abstract cultural concept that cultures express differently?
Return JSON with keys verified (boolean) and rationale (string)."""
