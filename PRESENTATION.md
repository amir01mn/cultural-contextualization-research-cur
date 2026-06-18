---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Cultural KG MVP'
footer: 'Layered Cultural Knowledge Graphs for Local-LLM Reasoning'
style: |
  section { font-size: 22px; line-height: 1.4; }
  h1 { color: #1f3a5f; }
  h2 { color: #3A7CA5; border-bottom: 2px solid #e3e8ef; padding-bottom: .2em; }
  code { background: #f4f6fa; }
  table { font-size: 18px; }
  .layer1 { color: #4C9F70; font-weight: 700; }
  .layer2 { color: #3A7CA5; font-weight: 700; }
  .layer3 { color: #A24C9F; font-weight: 700; }
  blockquote { border-left: 4px solid #3A7CA5; color: #444; }
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Cultural KG MVP

### Does a layered cultural knowledge graph help a local LLM reason about culture more respectfully and practically — and does critique-refining the graph help even more?

A fully **local, open-source, reproducible** research pipeline
built on **Ollama** (`llama3.1:8b`) — no paid APIs, ever.

---

## The experiment in one picture

We run the **same local LLM** under three conditions and compare scores:

| | System | What the LLM sees |
|---|---|---|
| **A** | `baseline_prompt_only` | Just the scenario question |
| **B** | `raw_kg_augmented` | Question **+ retrieved raw triples** |
| **C** | `critique_refined_kg_augmented` | Question **+ retrieved refined triples** |

> Hypothesis: **Baseline < Raw KG < Refined KG**

Because the model is held constant, any score difference is attributable to the **knowledge graph**, not the LLM.

---

## End-to-end pipeline (`scripts/run_mvp.py`)

```
01 Load + normalize records      → data/processed/culturebank_sample_N.jsonl
02 LLM extracts triples          → raw_triples_N.jsonl
03 Build RAW knowledge graph     → raw_cultural_kg.{graphml,json,html,csv}
04 LLM critique → refined triples → critiques_N.jsonl, refined_triples_N.jsonl
   Build REFINED knowledge graph → refined_cultural_kg.{...}
05 Generate scenario questions
   Answer with A / B / C systems
   Score all answers, summarize  → evaluation_results_N.{json,csv}, summary.md
```

Hard gate: if Ollama isn't running, the program **stops with setup instructions** — it never pretends LLM work happened (unless you pass `--allow_fallback`).

---

## Step 0 — Loading & normalizing data (`data_loader.py`)

- Tries to download **CultureBank** from Hugging Face (`SALT-NLP/CultureBank`).
- CultureBank has no `train` split — real splits are e.g. `tiktok`, `reddit`. If the requested split is missing, it reads **available splits in order** up to `--limit`.
- If download fails (no network / API change), it uses a **built-in 7-record fallback** (Japanese, Indian, Ethiopian, Brazilian, Saudi, Chinese, Māori) so the pipeline always runs.

**Field normalization** — every raw record is mapped to a fixed schema via `FIELD_CANDIDATES`, trying each candidate name in order:

```python
"cultural_group": ["cultural group", "cultural_group", "culture", "country", "region"],
"actor_behavior": ["actor_behavior", "actor behavior", "behavior", "action"],
"source_text":    ["source_text", "desc", "description", "text", "summary", "content"],
# ...goal, actor, recipient, relation, recipient_behavior, topic, agreement
```

Missing fields → `None`. If there's no explicit `source_text`, one is **synthesized** by joining `cultural_group · context · behaviors · topic`. The full original record is preserved under `raw`. Validated through the Pydantic `CulturalRecord` model.

---

## The cultural layers — the heart of the design

Every triple element (subject, predicate, object) is tagged with an **abstraction layer**:

| Layer | Meaning | Examples |
|---|---|---|
| <span class="layer1">LAYER_1_SURFACE</span> | Visible practices & objects | shoes, food, bowing, gifts, ceremonies |
| <span class="layer2">LAYER_2_SOCIAL</span> | Roles, norms, expectations | host/guest, hierarchy, etiquette, obligation |
| <span class="layer3">LAYER_3_DEEP_VALUE</span> | Values & beliefs | respect, harmony, hospitality, honor, dignity |
| `UNKNOWN` | Only when genuinely unclear | — |

Defined as a `LayerLabel` enum (`schemas.py`) with `coerce()` that maps shorthands (`L1`, `surface`, `value`…) to canonical labels, defaulting to `UNKNOWN`.

Why it matters: good cultural reasoning must connect a **surface practice** → a **social norm** → a **deep value**. The layers let us *measure* whether that bridge exists.

---

## Step 1 — How triples are generated (`triple_extractor.py`)

**Main method = the local LLM** (`extract_triples_llm`). Per record:

1. Build a user prompt from the normalized fields (`TRIPLE_EXTRACTION_USER`) + a strict system prompt (`TRIPLE_EXTRACTION_SYSTEM`).
2. Call `local_llm.generate_json(...)` → expects a JSON object `{ "triples": [...] }`.
3. For each returned triple:
   - Skip if `subject` or `object` is empty.
   - **Validate layers** (`validate_triple_layers`) — fix/guess any bad label.
   - Wrap into a `CulturalTriple` with a **stable, deterministic ID** and full **provenance**.

The system prompt instructs the model to: extract only culturally relevant triples, **not invent facts**, mark inferred bridges as `generated=true` with lower confidence, keep provenance, avoid stereotyping, and prefer **5–12 triples per record**.

---

## Step 1 — What a generated triple contains

Each `CulturalTriple` (Pydantic model) preserves everything needed for scoring & auditing:

```json
{
  "triple_id": "t_ab12cd34",
  "subject": "guest", "predicate": "removes", "object": "shoes at the entrance",
  "subject_layer": "LAYER_2_SOCIAL",
  "predicate_layer": "LAYER_1_SURFACE",
  "object_layer": "LAYER_1_SURFACE",
  "confidence": 0.9,
  "generated": false,                 // false = grounded in the record
  "evidence": "actor_behavior",       // which field/quote supports it
  "source_text": "...genkan...",
  "dataset_id": "culturebank", "instance_id": "fallback_001",
  "extraction_method": "llm_based",
  "raw_fields": { ... }, "notes": null
}
```

`confidence` is clamped to `[0,1]`; `triple_id` is a hash of `(instance_id, subject, predicate, object, method)` so re-runs are stable and dedupable.

**Modes:** `llm_based` (default) · `schema_fallback` (deterministic field rules) · `both` (LLM + schema, deduped). Fallback only fires when `--allow_fallback` is set.

---

## Step 1b — How layers are classified (`layer_classifier.py`)

The **LLM assigns layers during extraction**. This module only *guards* the result:

- `validate_layer()` — accepts a well-formed label, else collapses to `UNKNOWN`.
- `guess_layer()` — **keyword fallback** when a label is missing/UNKNOWN. It tokenizes the text and counts overlaps against three curated cue sets:

```python
_L1_CUES = {shoe, food, bow, gift, ceremony, greet, serve, wear, dance, ...}   # surface
_L2_CUES = {host, guest, elder, role, norm, polite, hierarchy, etiquette, ...} # social
_L3_CUES = {respect, harmony, hospitality, honor, dignity, value, sacred, ...} # deep value
```

- The layer with the most cue hits wins; **zero hits → UNKNOWN**.
- `validate_triple_layers()` runs this on all three elements of every triple.

So layers are **LLM-first, keyword-guarded** — never silently wrong, never blank.

---

## Step 2 — How the knowledge graph is built (`graph_builder.py`)

The graph is a **`networkx.MultiDiGraph`** (directed, allows parallel edges):

- **Nodes = entities** (normalized subject/object labels). Node ID = hash of the normalized label, so identical entities **merge across records**.
- **Edges = triples.** The predicate becomes the edge; the `triple_id` is the edge key.

Each **node** accumulates:
- a **soft `layer_distribution`** `[L1, L2, L3]` — summed from every triple element that lands on it (priors in `config.LAYER_DISTRIBUTIONS`, e.g. surface = `[0.70, 0.15, 0.15]`),
- `frequency` (how many triples touched it), `sources` (which records), and `generated` (true only if *every* contributing triple was generated).

Each **edge** carries full provenance: `predicate`, `predicate_layer`, `confidence`, `generated`, `evidence`, `source_text`, `dataset_id`, `instance_id`, `extraction_method`.

---

## Step 2 — Graph layers & exports

**Dominant layer per node** = `argmax` of its normalized distribution → used for coloring:

```
LAYER_1_SURFACE   → #4C9F70 (green)
LAYER_2_SOCIAL    → #3A7CA5 (blue)
LAYER_3_DEEP_VALUE→ #A24C9F (purple)
UNKNOWN           → #999999 (grey)
```

`export_all()` writes the graph to **five formats**:

| Format | Purpose |
|---|---|
| `.graphml` | Gephi / yEd (distribution stringified — GraphML can't hold lists) |
| `.json` | `node_link_data` for programmatic use |
| `_nodes.csv` / `_edges.csv` | Tabular analysis (pandas) |
| `.html` | **Interactive pyvis** view; generated edges drawn lighter (`#ccc`) than grounded (`#555`) |

Both the **raw** and **refined** triple sets go through the exact same builder → `raw_cultural_kg.*` and `refined_cultural_kg.*`.

---

## Step 3 — How critique & refinement work (`critique_agent.py`)

**Main method = a second LLM pass** (`critique_record_llm`). For each record, it sends the original record + its compact triples to the model (`CRITIQUE_SYSTEM`/`CRITIQUE_USER`) and parses JSON with three parts:

1. **`issues`** — typed problems (`missing_layer_3_value`, `surface_without_bridge`, `stereotyping`, …) each with a severity and affected triple IDs.
2. **`suggested_triples`** — cautious *bridge* triples (e.g. surface practice → deep value). These are **forced** to `generated=true` and confidence **capped at ≤ 0.49** unless directly supported by source text. ID prefix `tc_`, method `critique_suggested`.
3. **`triples_to_flag`** — IDs of unsupported/weak triples + a reason.

The prompt explicitly forbids the critic from treating its own suggestions as truth: *"improve the graph while preserving uncertainty."*

---

## Step 3 — Building the refined triple set

`build_refined_triples()` merges originals with critique output **conservatively**:

1. **Keep all original triples.** If a triple was flagged, keep it but annotate its `notes` with `[FLAGGED: reason]` (transparency, not deletion).
2. **Add suggested triples**, but only if they don't collide with an existing triple — **grounded triples always win** over generated ones on a `(instance_id, subject, predicate, object)` key.

The refined set is therefore: *every grounded fact* + *cautious, clearly-marked inferred bridges*, deduped.

> Rule-based critique (`critique_record_rule_based`) exists only as a fallback: it checks for missing Layer 2 / Layer 3, surface-without-bridge, and missing cultural group, and can add one cautious value bridge from `topic`.

In this pilot run: **14 raw triples → 4 critique suggestions → 18 refined triples.**

---

## Step 4 — How retrieval works (`retriever.py`)

For each scenario question we fetch the **top-k most relevant triples** (default `k=8`).

**Indexing** — each triple becomes a rich searchable document:

```python
doc = subject + predicate + object + evidence + notes
    + source_text + cultural_group + topic
```

**Two backends, chosen automatically:**

| Backend | When | How |
|---|---|---|
| **sentence-transformers** | if `all-MiniLM-L6-v2` is installed | normalized embeddings, score = **dot product** (cosine) |
| **TF-IDF** (fallback) | otherwise | `TfidfVectorizer` (English stop-words, ≤8192 features), score = **cosine similarity** |

Both rank by similarity and return up to `top_k` `(triple, score)` pairs. The **raw** and **refined** sets get **separate retriever indexes**, so systems B and C each retrieve from their own graph. All retrieved triples per question are logged to `retrieval_debug_N.json`.

---

## Step 5 — Scenario generation (`evaluator.py`)

Each record becomes a **cultural scenario question** used to test the systems.

- **LLM mode** (`generate_scenario_llm`): the model returns a question plus `expected_keywords`, `avoid_keywords`, and `target_layers`.
- **Fallback** (`generate_scenario_fallback`): a deterministic template —
  *"You are in a {cultural_group} setting: {context}. What should you do, avoid, or understand to act respectfully?"* — with keywords mined from the record fields and a fixed `avoid_keywords` list (`weird`, `primitive`, `strange`, `backward`, …).

These per-example keywords and target layers are exactly what the **scoring** stage checks against.

---

## Step 5 — How answers are generated (the 3 systems)

All three share one cautious **answer system prompt** (`ANSWER_SYSTEM`): avoid stereotyping, never claim all members behave alike, use hedged phrasing ("In this context…", "Based on the provided record…"), be practical, admit uncertainty.

| System | Prompt | Evidence supplied |
|---|---|---|
| **A. Baseline** | `BASELINE_USER` | *none* — "you do not have access to a KG" |
| **B. Raw KG** | `RAW_KG_USER` | retrieved **raw** triples as JSON |
| **C. Refined KG** | `REFINED_KG_USER` | retrieved **refined** triples as JSON |

The refined prompt adds a crucial instruction: *"Use dataset-grounded triples first. Treat `generated=true` triples only as cautious interpretations, never as guaranteed fact."*

---

## Step 6 — How answers are scored (out of 100)

Fully **automatic & deterministic** (`score_answer`) — a transparent proxy, computed per answer:

| Component | Max | How it's measured |
|---|---:|---|
| Keyword coverage | 30 | fraction of `expected_keywords` present in the answer |
| Cultural group match | 15 | answer mentions the group or a token of it |
| Provenance use | 20 | overlap between answer and **grounded** retrieved triples' subject/object terms |
| Layer coverage | 20 | 10 for Layer-2 terms + 10 for Layer-3 terms appearing in the answer |
| Answer specificity | 10 | length (up to 120 words → 6) + cautious-framing hints (→ 4) |
| Avoids bad keywords | 5 | full 5 only if **no** `avoid_keywords` appear |
| **Generated-dominance penalty** | **−15** | `−15 × (generated triples / retrieved triples)` |

Total is clamped to `[0, 100]`. The penalty discourages leaning on inferred (generated) triples instead of grounded evidence.

---

## Step 6 — Aggregation & the verdict

`summarize_results()` averages each system and computes:

- `raw_kg_improvement_over_baseline = avg(Raw) − avg(Baseline)`
- `refined_kg_improvement_over_raw  = avg(Refined) − avg(Raw)`
- `generated_support_ratio` — avg share of generated triples used by system C
- `layer_2 / layer_3 coverage` — % of refined answers that surface those layers

**`compute_verdict()` thresholds:**

| Verdict | Condition |
|---|---|
| **PASS** | Raw − Baseline ≥ **5** **and** Refined − Raw ≥ **3** **and** generated ratio ≤ **25%** |
| **FAIL** | Raw ≤ Baseline **or** Refined < Raw |
| **PARTIAL PASS** | Raw beats baseline, but refined gain small or generated dominance too high |

---

## Results (pilot run — `llama3.1:8b`, 3 records / 3 scenarios)

| System | Average score (/100) |
|---|---:|
| A. Baseline (no KG) | **56.83** |
| B. Raw KG | **81.67** |
| C. Refined KG | **82.55** |

```
Baseline   ████████████████████████              56.83
Raw KG     ██████████████████████████████████    81.67
Refined KG ██████████████████████████████████▍   82.55
```

- **Raw KG − Baseline: +24.84**  ← the graph clearly helps
- **Refined KG − Raw KG: +0.88**  ← refinement helps only marginally here
- Generated support ratio **20.83%** · Layer 2 & 3 coverage **100%**

### → Verdict: **PARTIAL PASS**

---

## The local LLM adapter (`local_llm.py`)

Everything routes through **Ollama** via HTTP — robust by design:

- **Preflight:** `check_ollama_available()` probes `/api/tags`; `warmup_model()` pre-loads weights (first cold load can take minutes) with a generous 600s timeout and `keep_alive=30m`.
- **JSON robustness:** `generate_json()` requests `format=json`, then on failure **retries** with a nudge. `repair_json_if_needed()` strips ``` fences, extracts the outermost `{...}`/`[...]`, removes trailing commas, and fixes smart quotes — because 8B models emit imperfect JSON.
- **Honesty:** if the server is unreachable it raises `LLMUnavailableError`; the pipeline stops rather than fabricating results (unless `--allow_fallback`).

Config (`config.py`) is env-driven: model, temperature `0.2`, `num_predict 1536`, top-k, num scenarios, layer-distribution priors — all overridable via `.env`.

---

## Design principles & guarantees

- **Local-first & reproducible** — pinned open model, fixed seed, deterministic IDs & scoring. No paid API.
- **Provenance everywhere** — every triple/edge knows its dataset, instance, evidence, source text, method, and whether it was `generated`.
- **Uncertainty preserved** — inferred triples are capped in confidence, flagged `generated=true`, and penalized if over-used.
- **Anti-stereotyping** — baked into every prompt; the graph reflects *source records only*, never an entire culture.
- **Always runs** — HF→fallback dataset, embeddings→TF-IDF, LLM→(optional) schema rules.
- **Fully tested offline** — `pytest` suite mocks the LLM (schemas, extractor, graph, retriever, critique, evaluator).

---

## Limitations & where it goes next

**Limitations**
- This summary reflects a **tiny pilot** (3 records / 3 scenarios) — directional only.
- Automatic scoring is a **proxy**, not human judgment.
- CultureBank field names vary; some dataset-specific fields may be missed.
- 8B models produce imperfect JSON (mitigated by repair/retry).

**Next steps**
- Scale `--limit` and `--num_scenarios` toward the full dataset.
- Larger local models (`llama3.1:70b`) for higher-quality triples.
- Install `sentence-transformers` for stronger semantic retrieval.
- Add LLM-as-judge (`USE_LLM_EVALUATION=true`) alongside the rubric.

---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Thank you

**Cultural KG MVP** — testing whether *layered* and *critique-refined* cultural knowledge graphs make a local LLM reason about culture more respectfully and practically.

Run it yourself:

```bash
ollama serve && ollama pull llama3.1:8b
python scripts/run_mvp.py --dataset culturebank --limit 100 \
  --extraction_mode llm_based --critic llm_based \
  --llm_provider ollama --ollama_model llama3.1:8b
```
