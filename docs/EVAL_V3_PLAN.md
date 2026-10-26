# Evaluation v3: KG v3 (frame KG + routing) — protocol

Written 2026-10-10, **before any v3 answer on a test split existed** (pre-registration).
Branch `kg-v3`. Builds on eval v2 (`docs/EVAL_V2_PLAN.md`, results in
`docs/results/eval_v2/RESULTS.md`); same answering model, splits, prompts and statistics
unless stated here.

## Why

Eval v2 found that the layered triples help on in-corpus questions but the August
pathways (S4) do not, and the intrinsic audit located the cause in the pathway builder.
Measured on `data/processed/corpus_pathways.jsonl` (2026-10-10):

- only **8,739 of 22,990 records (38%)** are reachable through any pathway;
- **19,322 of 29,578 pathways (65%)** join triples from different records;
- 16 cultures hit the 500-pathways-per-culture cap (US, Japan, China, ...);
- pathways are indexed and shown as bare node labels, without source text.

Two further findings change how v2 must be read:

1. **Eval-text leakage in the August triples.** The extraction prompt serialized the whole
   CultureBank row (`triple_extractor.py`, `json.dumps(rec["raw"])`), including
   `eval_whole_desc`, which is the open-ended judge's reference. 30,273 of 186,691 refined
   triples (16%; 14,242 records) carry at least two content words that occur only in the eval
   columns (`src/core/leakage.py`, a lexical proxy). The v2 leak guard checked verbatim
   containment only. v2 numbers for S2/S3/A3/S4 on CultureBank, and the 93% triple fact
   retention, are therefore likely inflated (most for open-ended). S1 and NormAd are unaffected.
2. **S3 was refined triples, identical to A3.** Decision (2026-10-10): S3 uses the **raw**
   layered triples as `RESEARCH_DESIGN.md` defines, so S2 -> S3 isolates layer labels,
   S3 -> A3 isolates critique-refinement, and A3 -> S4 isolates the retrieval unit.

## KG v3

Built by `scripts/build_kg_v3.py` (`slurm/build_kg_v3.sh`) into `data/kg_v3/`.

**Frame KG** (`src/construction/frame_builder.py`): one frame per CultureBank descriptor
(22,990 rows; rows without a group or behaviour are skipped).

| Layer | Content | How |
|---|---|---|
| L3 value | 1-2 of Schwartz et al. (2012)'s 19 refined values | LLM picks from the fixed list |
| L2 norm type | culture-neutral norm type, shared across cultures | LLM names it; leader clustering of the names (MiniLM cosine >= 0.75), most frequent name labels the cluster |
| L2c norm | this group's norm, one sentence | LLM |
| L1 practice | the descriptor (context, actor, behaviour, recipient, relation, goal, other) | mapped directly |

- Labeler: Qwen2.5-7B-Instruct, temperature 0, `max_tokens` 200, one retry on invalid JSON;
  failures stay as `label_status=failed` (reachable for refinement, absent from abstraction).
- **No eval column is read** by frames or labeling prompts (asserted in tests).
- Cultures connect only through norm types and values. Culture similarity = 0.5 x cosine of
  agreement-weighted value profiles + 0.5 x cosine of norm-type usage.
- **Held-out KGs built without the masked records** (`heldout_near`, `heldout_far`, the v2
  masks): the masked records shape neither norm types nor profiles.

**Record-local pathways** (`src/construction/record_pathways.py`, step 2): over the August
refined triples minus `llm_failed` and eval-text triples, one value->norm->practice chain
per record (no cross-record joins, no cap), carrying the record's descriptor text.

## Systems

| Label | Evidence |
|---|---|
| S0, S1, S1g, S2, S3, S3g, A3 (`S4_triple_retrieval`), S4 | as eval v2, except S3 = raw triples (above) and **all triple systems drop eval-text triples** (`--drop_eval_leak_triples`). S4 is the August pathway index, unchanged (historical baseline; it was built with those triples). |
| **S4L** | record-local pathways with descriptor text, culture-aware reranking (step 2) |
| **S4F** | frame KG, top-k frames rendered value -> norm type -> norm -> practice, always injected |
| **S4R** (ours) | frame KG with routing: *refine* (same-culture frame >= tau_ref: that culture's frames as chains) / *abstract* (no such frame, culture in KG, norm type >= tau_abs: norm type + values, the culture's top values and related norms, up to 2 similar cultures' norms labelled indirect) / *none* (closed-book) |
| S4R_noabs | ablation: S4R without the abstract route (refine or none) |

Thresholds for S4R / S4R_noabs are tuned **per evaluation on its dev split**, seed 42, by
accuracy, over tau_ref in {0.3, 0.4, 0.5, 0.6} x tau_abs in {0.3, 0.4, 0.5}
(S4R_noabs: tau_ref only); ties go to the higher thresholds. Then frozen for test.
S1g/S3g keep the v2 rule.

## Evaluations (all Qwen2.5-7B-Instruct, top_k 8, temperature 0 unless noted)

1. **CultureBank grounded MCQ** (v2 items, v2 paraphrases, 3 option orders), open-ended
   (500 items, phi-4 judge), held-out (300 items; frame systems use the build-time held-out
   KG, others mask at query time), retrieval metrics. Output `data/outputs/eval_v3/culturebank/`.
2. **CultureBank hard MCQ** (new): same items, distractors = the 3 most similar to the gold
   behaviour within the v2 valid pool (`scripts/prep_culturebank_hard_v3.py`). Newly
   paraphrased. Output `data/outputs/eval_v3/culturebank_hard/`.
3. **NormAd** (v2 splits): C1, C2, C3, S1, S2, S3, S4, A3, S1g, S3g, S4L, S4F, S4R, S4R_noabs.
   Output `data/outputs/eval_v3/normad/`. Covered subset as v2 (MiniLM >= 0.5); a
   sensitivity analysis at >= 0.45 is secondary.
4. **Intrinsic (phi-4)**: pathway audit (coherent / culture-consistent / correct) on 200 frame
   chains and 200 record-local pathways (4 per culture, seed 42); compare with v2's 22.5%
   coherent August pathways. Open-ended judge for the v3 open answers.
5. **Second answering model** (secondary): allenai/OLMo-2-1124-7B-Instruct (open, non-Qwen; Llama-3.1 access was not granted) on CultureBank MCQ
   (S0, S1, S3g, S4, S4L, S4R, S4R_noabs) and NormAd (C1, C2, C3, S1g, S4R, S4R_noabs),
   thresholds re-tuned on dev for that model.

## Hypotheses and primary comparisons

Holm-Bonferroni within each evaluation over this family only (McNemar, seed 42 for MCQ;
paired bootstrap CIs):

| Comparison | Tests |
|---|---|
| S4R vs S0 (NormAd: vs C1) | the routed frame KG helps |
| S4R vs S1 | against plain passages (no direction claimed; S1 is the strongest v2 system in-corpus) |
| S4R vs S3g | against the best v2 KG system |
| S4R vs S4 | the redesign fixes S4 |
| S4R vs S4R_noabs | **the abstraction route adds value** (novelty claim) |
| S4F vs S4 | frames vs August pathways, same injection policy |
| S4L vs S4 | record-local pathways fix the builder on the same triples (step 2) |

Expected: S4R, S4F, S4L > S4 on CultureBank; S4R >= S3g; S4R vs S4R_noabs > 0 mainly where
specific knowledge is missing (NormAd, held-out). Any of these may fail; all are reported.

## Fixed before results

The frame schema, value list, labeling prompt, norm-type threshold (0.75), routing rules and
rendering, threshold grids, distractor rule, systems and comparison families above. Changes
after seeing test results are reported as post-hoc.

## Limitations carried over

The phi-4 judge is not human-validated (labeling sheets to be produced); one run per
NormAd condition; CultureBank cultures include non-national groups; the leakage detector is a
lexical proxy (misses paraphrases using descriptor words).
