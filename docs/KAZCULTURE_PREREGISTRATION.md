# KazCulture final experiment: pre-registered analysis plan

**Branch:** `kazculture-full-eval` (from `experiment/kazculture-qwen3-phase1` + `main`)
**Written:** 2026-11-05, before any structured-evidence (S2/S4R) test-set run.
**Status:** frozen once committed. Any later change must be appended in a dated
"Amendments" section with the reason, and must not be motivated by test outcomes.

## 1. Question

Does our layered cultural knowledge (source-grounded facts S2 and norm/value
pathways S4R) improve Qwen3-32B on KazCulture beyond raw passage retrieval (S1)?

## 2. Frozen components

- Answer model: `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137`, non-thinking, frozen V2 letter scorer.
- Retriever: `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181`.
- S1 retrieval config: `top_k=8`, `min_similarity=0.45`, evidence budget 5,000 chars
  (`data/kazculture_pipeline/s1_full_dev_thresholds_v3.json`, tuned on train only).
- Test set: 1,334 questions. Primary setting: **strict** (1,262 exact-overlap train passages removed).
- Baselines to reproduce in our space: S0 = 552/1334 (41.38%), strict S1 = 831/1334 (62.29%),
  full S1 = 1221/1334 (91.53%). Reproduction tolerance: exact match expected (deterministic);
  any difference is investigated and reported before structured runs.

## 3. Conditions (all under the identical scorer and prompt template)

Code review (2026-11-05) found the branch's V2.1/V2.2 "S4R" is not our method (no
per-record frames, norm-type clustering, culture profiles or refine/abstract routing).
The structured conditions therefore use a **faithful port of the CultureBank v3 frame KG**
(`src/construction/frame_builder.py`, `src/retrieval/frame_retriever.py`): one frame per
train passage (practice = passage text; norm, norm type, 1-2 Schwartz values labelled by
Qwen3-32B), norm types clustered, culture profiles, routing thresholds tuned on train only,
abstract-route analogs from CultureBank v3 frames (marked indirect). Retrieval embeddings: BGE-M3.

| ID | Evidence in prompt |
|---|---|
| S0 | none |
| S1 | retrieved passages (frozen config) |
| S4R | v3 frame retrieval with routing (refine / abstract / none) at the same top-k and budget |
| S4R-noabs | S4R with the abstract route disabled |
| S1+L | S1 with each passage's own norm/value labels attached (enrichment variant) |
| CTRL-LEN | S1 with the extra characters S4R adds filled by further S1 passages |
| CTRL-SHUF | S4R layout, but norm/value labels and analogs taken from non-matching passages |

Divnoor's V2.1/V2.2 triple KG (S2) is not used in the primary analysis.

## 4. Order of work and go/no-go gate

1. Reproduce S0, strict S1 and full S1 in our space.
2. Build v3 frames for all 6,749 train passages (new versioned directory; nothing overwritten).
3. Tune routing thresholds and run a **train-dev** experiment on train questions
   (sources excluded from the index for those questions). Train questions are free text, so the
   metric is the gold-answer log-likelihood under Qwen3-32B with S1 vs S4R evidence.
   **Gate:** proceed with S4R as primary if it improves mean gold log-likelihood over S1 on train-dev.
   If not, the test set is still run once to report a negative result, with no further iteration.
4. Test set run **once** per condition. No changes after seeing test results.

## 5. Primary hypothesis and statistics

- **Primary comparison:** strict S4R vs strict S1.
- **Secondary (Holm-corrected family):** strict S4R-noabs vs S1, strict S1+L vs S1, and the full-setting
  versions of these comparisons.
- **Tests:** exact two-sided McNemar on the 1,334 paired outcomes; paired bootstrap (10,000 resamples,
  seed 0) 95% CI on the accuracy difference.
- **Significance:** p < 0.05 and CI excluding 0.
- **Minimum meaningful effect:** +2.0 percentage points (≈27 net questions).

## 6. Criteria for calling the method "better and effective"

1. Primary comparison significant (§5) with effect ≥ +2.0 pp.
2. Primary condition beats **both** CTRL-LEN and CTRL-SHUF (exact McNemar, p < 0.05, Holm across the two).
3. Test set touched once; all tuning on train only (audit trail in git history).
4. Frame labels human-validated (blind 100-frame audit).
5. Gain not concentrated in one category (positive in ≥ 5 of the 7 categories with n ≥ 10), and the
   direction replicates with a second model (OLMo-2-1124-7B-Instruct) on strict S4R vs S1.

Each criterion is reported as met / not met; no criterion is dropped after the fact.

Also reported: a near-duplicate audit of the strict index against test passages (exact-match
removal does not remove paraphrases or supersets), and disclosure that the branch's earlier
17-question pilot used test questions.

## 7. Known limitation at registration time

No human reviewer is available. Criterion 4 will be reported as **pending**; the blind review
sheet of 100 frame labels and its scorer are kept ready.
AI screening is not used as a substitute.

## Amendments

### 2026-11-05: frame retrieval ran with the wrong embedder (protocol deviation, corrected rerun)

**What happened.** §3 specifies BGE-M3 retrieval embeddings. The routing tuner set
`EMBEDDING_BACKEND=bge-m3`, but the train-dev gate (job 8590655) and the first test jobs
(8590660 strict A, 8590661 strict B, 8590663 OLMo) did not. The default `auto` backend
loaded the English-only `all-MiniLM-L6-v2`, and `FrameRetriever` silently re-embedded all
frames in that space (logged as "Loading sentence-transformers model (all-MiniLM-L6-v2)").
S0, S1 and S1+L do not use frame retrieval and are unaffected.

**How it was found.** While diagnosing the negative S4R test result: S4R evidence overlapped
S1's passages by 0.7% (same top passage for 2/709 questions) and 49% of it came from the 6.2%
of passages without labels. Re-running retrieval only (no scoring) for 40 strict test
questions with BGE-M3 gave 52% overlap, 18/40 same top passage, 5.5% unlabelled.

**Correction.** The affected stages are re-run once with the preregistered embedder, with no
other change (same code, thresholds, prompts, budget): gate → `evaluation/traindev_gate_v1b`,
test → `evaluation/final_v1b` and `final_v1b_olmo` (S0/S1 predictions copied from v1, since
they are embedder-independent and reproduced exactly). `slurm/kazculture_final_common.sh` now
defaults the backend to BGE-M3, and the KazCulture adapter refuses a KG whose stored backend
differs from the active one.

**Disclosure.** The deviation was found because the result was negative, and the as-run v1
numbers (`final_v1`, `traindev_gate_v1`) are kept and reported next to the corrected ones.
v1b is the preregistered analysis; v1 is the record of the deviation. No threshold, prompt or
condition was changed in response to test outcomes.
