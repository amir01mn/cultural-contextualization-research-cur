# cultural-kg-mvp

A Python MVP pipeline that tests whether a **layered cultural knowledge graph (KG)**
improves cultural scenario reasoning compared to baseline prompting, and whether a
**critique-refined KG** improves over a **raw KG**.

Everything runs on **local / open-source LLMs via [Ollama](https://ollama.com)**.
No OpenAI, Claude, Anthropic, or any paid API is required or used.

## Research question

> Does a layered cultural KG help a local LLM reason about cultural scenarios more
> respectfully and practically than baseline prompting — and does critique-refining
> the KG help even more?

We compare three systems on the same local LLM:

| System | Description |
|---|---|
| **A. Baseline** | Local LLM answers with only the scenario question (no KG). |
| **B. Raw KG** | Local LLM answers using retrieved **raw** LLM-extracted triples. |
| **C. Refined KG** | Local LLM answers using retrieved **critique-refined** triples. |

The hypothesis we test: **Baseline < Raw KG < Refined KG**.

## Why local LLMs?

- **Reproducible and free** — no API keys, no per-token cost.
- **Privacy** — cultural data and prompts stay on your machine.
- **Research control** — you pin the exact open model (`llama3.1:8b` by default).

Rule-based extraction is **not** the main method. The main pipeline uses the local
LLM to build triples, critique them, refine them, and generate answers. Rule/schema
extraction exists **only as a fallback** when the local LLM is unavailable, and only
when you explicitly pass `--allow_fallback`.

## Cultural layers

Each triple element is assigned an abstraction layer:

- `LAYER_1_SURFACE` — visible practices, objects, rituals, gestures, food, etc.
- `LAYER_2_SOCIAL` — roles, expectations, host/guest norms, hierarchy, etiquette.
- `LAYER_3_DEEP_VALUE` — values/beliefs: respect, harmony, hospitality, honor, etc.
- `UNKNOWN` — only when genuinely unclear.

## Project structure

```
cultural-kg-mvp/
  README.md
  requirements.txt
  .env.example
  data/
    raw/  processed/  outputs/
  src/
    config.py  data_loader.py  schemas.py  local_llm.py
    triple_extractor.py  layer_classifier.py  graph_builder.py
    retriever.py  critique_agent.py  evaluator.py  prompts.py  utils.py
  scripts/
    01_load_dataset.py  02_extract_triples.py  03_build_graph.py
    04_run_critique.py  05_evaluate.py  run_mvp.py
  tests/
    test_schemas.py  test_triple_extractor.py  test_graph_builder.py
    test_retriever.py  test_critique_agent.py  test_evaluator.py
```

## Requirements

- Python **3.11+**
- [Ollama](https://ollama.com) running locally with the `llama3.1:8b` model pulled.

## Installation

```bash
# 1. (optional) create a virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 2. install Python dependencies
pip install -r requirements.txt

# 3. (optional) copy env defaults
cp .env.example .env
```

`sentence-transformers` is optional. If installed, retrieval uses embeddings;
otherwise it falls back to TF-IDF automatically.

## Install and run Ollama

```bash
# Install Ollama: https://ollama.com/download
# Start the local server (if not already running):
ollama serve

# Pull the default open model:
ollama pull llama3.1:8b
```

Verify it is up: `curl http://localhost:11434/api/tags` should list your models.

## Run the 100-sample experiment

```bash
python scripts/run_mvp.py \
  --dataset culturebank --split train --limit 100 \
  --extraction_mode llm_based --critic llm_based \
  --llm_provider ollama --ollama_model llama3.1:8b
```

This will:

1. Load 100 CultureBank records (auto-falls back to a small built-in set if the
   Hugging Face download is unavailable, so the pipeline always runs).
2. Save normalized records to `data/processed/`.
3. Use the local LLM to extract triples → save raw triples.
4. Build and export the **raw** KG.
5. Run the local LLM **critique agent** → save critiques.
6. Build **refined** triples (`data/processed/refined_triples_100.jsonl`) and the
   **refined** KG.
7. Generate cultural scenario questions.
8. Generate baseline / raw-KG / refined-KG answers and score them.
9. Save all evaluation outputs and print a summary.

> If Ollama is not running (or the model is not pulled), the run **stops** with
> setup instructions — it never pretends LLM extraction happened. To run with the
> rule/schema-based fallback instead, add `--allow_fallback`. For a real research
> run, keep the default (`--allow_fallback` off).

### Run step-by-step (optional)

```bash
python scripts/01_load_dataset.py --limit 100
python scripts/02_extract_triples.py --extraction_mode llm_based
python scripts/03_build_graph.py
python scripts/04_run_critique.py --critic llm_based
python scripts/05_evaluate.py
```

## Outputs

All artifacts are written to `data/outputs/`:

- `raw_cultural_kg.{graphml,json,html}` + `_nodes.csv` / `_edges.csv`
- `refined_cultural_kg.{graphml,json,html}` + `_nodes.csv` / `_edges.csv`
- `evaluation_examples_100.json` — the generated scenario questions
- `evaluation_results_100.json` — per-example answers + scores + summary
- `evaluation_results_100.csv` — flat table (one row per example × system)
- `evaluation_summary_100.md` — human-readable summary
- `evaluation_average_scores_100.png` — bar chart of average scores
- `retrieval_debug_100.json` — which triples were retrieved per question

## How to read the evaluation summary

Open `data/outputs/evaluation_summary_100.md`. The key numbers:

- **Average scores** for Baseline, Raw KG, Refined KG (out of 100).
- **Raw KG − Baseline** and **Refined KG − Raw KG** improvements.
- **Generated support ratio** — how much the refined system leaned on generated
  (inferred) triples. Lower is better (we want grounded evidence).
- **Layer 2 / Layer 3 coverage** — how often answers explained social and deep-value
  layers.

### Interpreting PASS / PARTIAL PASS / FAIL

- **PASS**
  - Raw KG average ≥ Baseline + 5, **and**
  - Refined KG average ≥ Raw KG + 3, **and**
  - Refined generated support ratio ≤ 25%.
- **PARTIAL PASS**
  - Raw KG improves on Baseline, but refined improvement is small, **or**
  - Refined KG improves but generated dominance is too high.
- **FAIL**
  - Raw KG does not improve on Baseline, **or**
  - Refined KG performs worse than Raw KG.

## Scoring formula (automatic, out of 100)

| Component | Max points |
|---|---|
| Keyword coverage | 30 |
| Cultural group match | 15 |
| Provenance use | 20 |
| Layer coverage | 20 |
| Answer specificity | 10 |
| Avoids bad keywords | 5 |
| Generated dominance penalty | up to −15 |

An optional LLM-as-judge can be added later (`USE_LLM_EVALUATION`), but it is not
required.

## Testing

```bash
pytest -q
```

Tests run fully offline and do not require Ollama (LLM calls are mocked).

## Scaling later

- Increase `--limit` and `--num_scenarios` to use the full CultureBank dataset.
- Try larger local models (e.g. `--ollama_model llama3.1:70b`) for higher quality.
- Install `sentence-transformers` for stronger semantic retrieval.
- Add LLM-as-judge evaluation by setting `USE_LLM_EVALUATION=true` and extending
  `src/evaluator.py`.

## Limitations

- CultureBank field names vary; the loader maps common candidates and may miss
  dataset-specific fields (kept under `raw`).
- Automatic scoring is a proxy, not human judgment; treat it as a directional signal.
- Local 8B models can produce imperfect JSON; the adapter repairs/retries, but some
  records may yield fewer triples.
- The graph and triples reflect the source records only — they should not be read as
  generalizations about an entire culture.
- This is an MVP for 100 samples; validate carefully before running the full dataset.

## License / data note

CultureBank is provided by its original authors under their own terms. This project
is a research MVP; use cultural data responsibly and avoid stereotyping.
