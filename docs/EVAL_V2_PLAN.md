# Evaluation v2 — protocol and build contract

Written 2026-09-27, **before any v2 results existed** (pre-registration). Branch `eval-v2`.

## Why

CulturalBench mostly asks for facts CultureBank lacks: oracle retrieval picks the
gold option only 27.9% of the time vs 25% chance (`docs/culturebank-culturalbench-coverage.pdf`
on `origin/main`). v2 evaluates the KG where its knowledge is actually tested:

1. **CultureBank grounded QA** (in-corpus; headline). Every CultureBank row ships
   `eval_whole_desc`, `eval_scenario`, `eval_persona`, `eval_question` (Shi et al. 2024, §6).
2. **NormAd** (external; its Country / Value / Rule-of-Thumb context levels mirror L3/L2).
3. **Intrinsic KG quality** (fact retention, triple audit, pathway audit; LLM-judged).
4. CulturalBench kept as a secondary result, stratified by coverage.

## Pre-registered choices (do not change after results are seen)

- Answering model: `Qwen/Qwen2.5-7B-Instruct` (as all prior runs). Judge: `microsoft/phi-4`
  (different family, avoids self-preference). Human validation of the judge is deferred.
- KG artifacts: the August full build in `data/processed/` (read-only). No rebuild.
- Systems: S0, S1, S2, S3, S4, S4_triple_retrieval (A3), S4_no_shared_pivot (A4), plus
  **gated** variants `S1g`, `S3g`, `S4g` = same system with `RETRIEVAL_STRICT_CULTURE=True`
  and `RETRIEVAL_MIN_SCORE=τ`, τ chosen per system on the **dev** split only from
  {0.15, 0.25, 0.35, 0.45} by accuracy, then frozen for test.
- top_k = 8 (config default). Temperature = config default.
- Primary comparisons: each system vs S0; S4 vs S1; S4g vs S4. Tests: McNemar + paired
  bootstrap (`src/evaluation/mcq_evaluator.py`), Holm–Bonferroni over the primary family.

### Evaluation 1 — CultureBank grounded
- Filter rows with `agreement >= 0.6`. Stratified sample by cultural group (≤ 20 per
  canonical culture, spread over topics), seed 42: **dev = 300, test = 1,500**. dev ∩ test = ∅.
- **MCQ (Format A)**: stem = group + context + persona question; gold = that row's
  `actor_behavior`; 3 distractors = `actor_behavior` of *other* cultures, *same topic*,
  with max embedding similarity < 0.75 to every descriptor of the gold culture. All four
  options are paraphrased by the answering model in the same way (no style cue).
  3 option-order seeds (42, 43, 44).
- **Open-ended (Format B)**: 500 test questions; systems S0, S1, S3, S4, S4g; answer the
  persona question; phi-4 judges whether the answer is consistent with `eval_whole_desc`
  (knowledge-entailment protocol of CultureBank §6); score = P(yes) proxy (yes=1/no=0).
- **Retrieval metrics**: gold = the question's own `instance_id`. Evidence Recall@k,
  MRR, same-culture precision@k, evidence tokens/question.
- **Held-out**: 300 test questions; their gold `instance_id`s masked from all indexes
  (`near`), and additionally every descriptor with embedding similarity > 0.85 to a
  masked one (`far`). Limitation: entity resolution/pivots were built with them present.
- **Leakage guard**: `raw_fields` carries the `eval_*` columns. No evidence string may
  contain `eval_question` / `eval_whole_desc` text; the run asserts leak count == 0.

### Evaluation 2 — NormAd (`akhilayerukola/NormAd`, 2,620 stories, labels yes/no/neither)
- Conditions: C1 country only; C2 country + gold Value; C3 country + gold Rule-of-Thumb;
  systems = country + retrieved evidence (retrieval query = story + country).
- Coverage: best MiniLM similarity between gold Rule-of-Thumb and the country's CultureBank
  descriptors; **covered ⇔ ≥ 0.5**. Report all + covered subset.
- Metrics: accuracy, macro-F1, gap closed = (acc − acc_C1)/(acc_C3 − acc_C1).
- Never add Cultural Atlas to the KG (NormAd's source → leakage).

### Evaluation 3 — Intrinsic (phi-4 judge)
- 300 records stratified by culture (seed 42). Atomic facts from `eval_whole_desc`
  (3–6 each); retention = fact inferable from that record's raw triples / refined triples
  / pathways. Triple audit on 500 refined triples (faithful, layer correct, culture-specific,
  stereotyping). Pathway audit on 200 FULL pathways (coherent chain, culture-consistent).

### CulturalBench (secondary)
- Re-score the existing August per-item results by coverage strata fixed from the
  coverage analysis (descriptors per country: <20, 20–99, ≥100). No new GPU run.

## Build contract (for implementers)

- Layering (`tests/test_architecture.py`): `evaluation` may import only core/llm/retrieval;
  `loaders` only core. **Data loading happens in `scripts/`**; evaluation modules receive
  loaded lists. New modules go in subpackages.
- Prompts live in `src/llm/prompts.py` inside your marked `EVAL_V2` section only.
- Each evaluation exposes stage functions `run_<stage>(backend, ..., out_dir: Path, **cfg) -> dict`
  that write JSON/JSONL/MD under `data/outputs/eval_v2/<eval>/` and return a summary dict.
  The GPU driver `scripts/run_eval_v2.py` loads each model **once** and calls stages.
- CPU-only preparation (sampling, gold maps, distractors, masks, coverage) goes in a
  `scripts/prep_*_v2.py` script writing `data/eval_v2/<eval>/`, runnable on the login node.
- Never write to `data/processed/` or the existing files in `data/outputs/`.
- Backend API: `backend.generate(prompts: list[str], system_prompt=..., temperature=..., max_tokens=...) -> list[str]`.
  Pass explicit `max_tokens` (never rely on a global budget).
