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


# --- Critique -------------------------------------------------------------

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
    # KG v3 (docs/EVAL_V3_PLAN.md)
    "S4_local_pathways": (
        "Retrieved abstraction→refinement pathways (value → norm → practice), "
        "each with the description it was extracted from:"
    ),
    "S4F_frame_kg": (
        "Retrieved cultural frames (value → norm type → this group's norm → practice):"
    ),
    "S4R_routed_frame_kg": "Cultural knowledge graph evidence:",
    "S4R_no_abstraction": "Cultural knowledge graph evidence:",
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


# === EVAL_V2: CULTUREBANK GROUNDED (Evaluation 1) ===========================
# (owned by the CultureBank-grounded evaluation; insert prompts below this line)

# MCQ stem (Format A). The answer scaffold is the shared MCQ_SYSTEM /
# MCQ_EASY_USER, so only the question text is specific to this evaluation.
CB_MCQ_STEM = """Cultural group: {cultural_group}
Situation: {context}

Someone asks: "{eval_question}"

In this situation, which of the following behaviors is most typical of {cultural_group}?"""

# Option paraphrasing: all four options of an item go through the same prompt
# in one call, so no option carries a distinct style cue.
CB_PARAPHRASE_SYSTEM = """You rewrite short descriptions of behaviors.
Paraphrase each numbered statement in plain, neutral English. Keep the meaning,
keep it to one sentence of similar length, do not add or remove details, and do
not mention any country, nationality or cultural group. Apply exactly the same
style to every statement."""

CB_PARAPHRASE_USER = """Statements:
{numbered}

Return exactly {n} lines, one per statement, in this format:
1: <paraphrase>
2: <paraphrase>
..."""

# Open-ended answering (Format B): the persona question from CultureBank.
CB_OPEN_SYSTEM = """You are a helpful assistant answering people's questions about cultural practices.

Retrieved cultural evidence may be provided. It is retrieved automatically and is
NOT guaranteed to be relevant: use it only when it directly addresses the question
and the culture being asked about; otherwise ignore it and answer from your own knowledge.

Avoid stereotyping and do not assume all members of a culture behave identically.
Answer directly and concisely (at most about 150 words)."""

CB_OPEN_USER = """{evidence_block}Question:
{question}"""
# === END EVAL_V2: CULTUREBANK GROUNDED =======================================


# === EVAL_V2: NORMAD (Evaluation 2) =========================================
# (owned by the NormAd evaluation; insert prompts below this line)
# Modeled on NormAd's official prompt (Rao et al. 2024): story + context, then
# "Is what [character] did socially acceptable?" with Yes / No / Neither. NormAd
# stories already end with that question. Only {context_block} differs between
# conditions (C1 country / C2 +Value / C3 +Rule-of-Thumb / systems +evidence).
NORMAD_SYSTEM = (
    "You are a careful assistant that judges whether actions in short stories "
    "are socially acceptable in a given country."
)

NORMAD_USER = (
    "Task: You will be given a short story involving two or more characters about "
    "an everyday situation, together with cultural context. Read the story and "
    "reason whether the action by the character in the story is socially "
    "acceptable under the given context. Do not make extra inferences about "
    "actions outside of the given context and country. Only align to the country "
    "given.\n\n"
    "{context_block}"
    "Story: {story}\n\n"
    "Options: 1) Yes 2) No 3) Neither\n"
    "Think briefly (at most two sentences), then give your final answer on the "
    "last line exactly as\n"
    "ANSWER: <Yes|No|Neither>"
)

NORMAD_CONTEXT_COUNTRY = "Country: {country}\n\n"
NORMAD_CONTEXT_VALUE = "Country: {country}\nValue: {value}\n\n"
NORMAD_CONTEXT_ROT = "Country: {country}\nRule-of-Thumb: {rule_of_thumb}\n\n"
# {evidence_block} is the header + numbered evidence from mcq_systems._evidence_for
# (already ends with a blank line), or empty when retrieval returned nothing.
NORMAD_CONTEXT_EVIDENCE = "Country: {country}\n\n{evidence_block}"
# === END EVAL_V2: NORMAD =====================================================


# === EVAL_V2: INTRINSIC + JUDGE (Evaluation 3) ==============================
# (owned by the intrinsic/judge evaluation; insert prompts below this line)
# Judge model: microsoft/phi-4 (docs/EVAL_V2_PLAN.md). Every prompt demands a
# short, machine-parseable answer; the parsers in src/evaluation/intrinsic_eval.py
# tolerate extra prose around it.

INTRINSIC_JUDGE_SYSTEM = """You are a careful, impartial annotator of cultural knowledge.
Follow the output format exactly. Do not add explanations unless asked."""

INTRINSIC_FACTS_USER = """Split the following cultural description into 3 to 6 atomic facts.
Each fact must be one short, self-contained sentence that names the cultural group and
states exactly one piece of cultural knowledge (a practice, a norm, a value, or who does
what in which context). Do not add information that is not in the description.
Skip statements that only say how widely the behavior is agreed upon or refer to
"the sampled population".

Description:
\"\"\"{desc}\"\"\"

Return ONLY a JSON list of strings, for example:
["Fact one.", "Fact two.", "Fact three."]"""

INTRINSIC_RETENTION_USER = """Below is cultural knowledge stored in a knowledge graph ({source_name}),
followed by one fact.

Knowledge graph content:
{evidence}

Fact: "{fact}"

Can the fact be inferred from the knowledge graph content above (the same cultural knowledge
is stated or clearly implied; paraphrase is fine; the cultural group must not contradict)?
Answer with exactly one word: yes or no."""

INTRINSIC_TRIPLE_AUDIT_USER = """Audit one knowledge-graph triple extracted from a source text.

Source text ({culture}): \"\"\"{source_text}\"\"\"

Triple: ({subject}) --[{predicate}]--> ({object})
Layer labels: subject = {subject_layer}; object = {object_layer}
Layer definitions: L1 = observable practice or behavior; L2 = social norm, rule,
expectation, role or social context; L3 = deep value or belief.

Answer four questions:
1. faithful: is the triple supported by the source text (no invented content)?
2. layers_correct: are the subject and object layer labels both correct by the definitions?
3. culture_specific: does the triple express something specific to this culture rather
   than a near-universal human behavior?
4. stereotyping: does the triple overgeneralize or stereotype the group beyond the source?

Return ONLY a JSON object with yes/no values:
{{"faithful": "yes", "layers_correct": "yes", "culture_specific": "yes", "stereotyping": "no"}}"""

INTRINSIC_PATHWAY_AUDIT_USER = """Audit one cultural knowledge pathway for the culture "{culture}".
A pathway should run from a deep value (L3) through a social norm (L2) to an observable
practice (L1).

Pathway:
{pathway}

Source texts the pathway was built from:
{sources}

Answer three questions:
1. coherent: does the chain form a coherent value -> norm -> practice explanation
   (each step plausibly motivates the next)?
2. culture_consistent: is every step consistent with the named culture and the sources
   (no content imported from a different culture)?
3. correct: is the pathway as a whole a correct statement of this culture's knowledge,
   given the sources?

Return ONLY a JSON object with yes/no values:
{{"coherent": "yes", "culture_consistent": "yes", "correct": "yes"}}"""

OPEN_JUDGE_USER = """You are checking whether an answer reflects a piece of cultural knowledge.

Cultural descriptor ({cultural_group}):
\"\"\"{gold_desc}\"\"\"

Question: {question}

Answer to evaluate:
\"\"\"{answer}\"\"\"

Does the answer entail the cultural descriptor, i.e. is it consistent with and does it
reflect the cultural knowledge in the descriptor? Answer with exactly one word: yes or no."""
# === END EVAL_V2: INTRINSIC + JUDGE ==========================================



# === KG_V3: FRAME LABELING (construction) ====================================
# One call per CultureBank descriptor. The prompt sees descriptor fields only,
# never CultureBank's eval_* columns (src/core/leakage.py). The LLM names the
# norm and picks values from a fixed list; it does not extract triples.

FRAME_LABEL_SYSTEM = """You annotate cultural descriptors for a knowledge graph.
Reply with one JSON object and nothing else."""

FRAME_LABEL_USER = """Cultural descriptor (from CultureBank):
{descriptor}

Fill in three fields:
- "norm": one sentence stating the social norm or expectation this practice follows
  among {culture}. Keep the group's specific stance (what is expected, avoided or
  valued), but state it so it would also cover similar situations. At most 30 words.
- "norm_type": a short culture-neutral name for the kind of norm, 2-6 words, that
  would fit any group's version of it (e.g. "tipping service staff",
  "removing shoes indoors", "greeting elders"). Do not name the group.
- "values": one or two values the norm expresses, copied exactly from this list:
{value_list}

Return exactly: {{"norm": "...", "norm_type": "...", "values": ["..."]}}"""

FRAME_LABEL_RETRY_SUFFIX = """

Your previous reply was not valid. Return ONLY the JSON object, with "values" chosen
from the list above."""
# === END KG_V3: FRAME LABELING ===============================================

# === KAZCULTURE FRAMES V3K: FRAME LABELING ===================================
# KazCulture variant of FRAME_LABEL_USER (docs/KAZCULTURE_FRAMES_V3K.md). Same
# system prompt (FRAME_LABEL_SYSTEM), retry suffix, JSON schema and token cap
# (frame_builder.LABEL_MAX_TOKENS), so frame_builder.parse_label and everything
# downstream are unchanged. Differences from the CultureBank prompt: the source
# is a Kazakh-language passage (possibly truncated for the prompt only), the
# group is fixed to Kazakh, the output must be English, and a purely descriptive
# passage may return "none". parse_label rejects "none" exactly as it rejects
# an empty field in v3, so such a passage stays a refine-only frame without a
# norm (label_status "none"); nothing is invented.
KAZCULTURE_FRAME_LABEL_USER = """Cultural passage (from KazCulture, written in Kazakh; topic: {topic}):
{passage}

The passage describes {culture} culture. Fill in three fields, written in English:
- "norm": one sentence stating the social norm or expectation this practice follows
  among {culture} people. Keep the group's specific stance (what is expected, avoided
  or valued), but state it so it would also cover similar situations. At most 30 words.
  Use only what the passage says or directly implies.
- "norm_type": a short culture-neutral name for the kind of norm, 2-6 words, that
  would fit any group's version of it (e.g. "tipping service staff",
  "removing shoes indoors", "greeting elders"). Do not name the group.
- "values": one or two values the norm expresses, copied exactly from this list:
{value_list}

If the passage is purely descriptive (for example an object's material, a place, a
date, or a definition) and states or implies no social norm or expectation, do not
invent one: return {{"norm": "none", "norm_type": "none", "values": []}}.

Return exactly: {{"norm": "...", "norm_type": "...", "values": ["..."]}}"""
# === END KAZCULTURE FRAMES V3K ================================================

