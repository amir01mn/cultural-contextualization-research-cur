# KazCulture frames v3k (S4R port)

This is the code for preregistration §3/§4 step 2 (`KAZCULTURE_PREREGISTRATION.md`).
It ports the CultureBank KG v3 frame method (`construction/frame_builder.py`,
`retrieval/frame_retriever.py`) to KazCulture. It replaces nothing: the earlier
`kazculture_kg_v2*` "S4R" is a different method and is not used.

## Method

| Layer | KazCulture frame |
|---|---|
| L1 practice | the full train passage text (`passages.jsonl`), never truncated |
| L2c norm | one English sentence (Qwen3-32B) |
| L2 norm type | the labeller's norm-type label, clustered at cosine 0.75 (`cluster_norm_types`) |
| L3 values | 1–2 Schwartz values (19-value taxonomy) |

- There is one frame per passage (6,749). `frame_id` is `passage_id`, culture is
  `kazakh`, and the topic is the passage's first category.
- Labelling uses `prompts.KAZCULTURE_FRAME_LABEL_USER` with v3's system prompt,
  retry suffix, JSON schema and 200-token cap, and is parsed by
  `frame_builder.parse_label`.
- v3 has no explicit "no norm" label: its parser rejects an empty field, and the
  frame stays reachable through refine only. The Kazakh prompt also lets a purely
  descriptive passage answer `"none"`. That answer is final (no retry) and is
  stored as `label_status="none"`. These frames have no norm and stay out of the
  norm types and profiles.
- Only the labelling prompt truncates the passage. The limit is
  `min(4096, max_model_len − prompt overhead − 200 − 64)` Qwen3 tokens, and the
  label row records it in `practice_truncated` / `practice_tokens`.
- The KazCulture frames are joined with the CultureBank v3 frames in `data/kg_v3/full/`.
  The 10 CultureBank Kazakh/Kazakhs/Kazakhstani frames are dropped, so all direct
  Kazakh evidence comes from KazCulture. The union then goes through
  `build_frame_kg` and `build_frame_index` unchanged. As a result, norm types are
  shared across the two sources. The culture profiles let `culture_similarity`
  find CultureBank groups similar to Kazakh, and their frames become abstract-route
  analogs, marked `indirect`.
- CultureBank norm types are re-clustered in the union with BGE-M3 (v3 used
  MiniLM). Their norms are still the Qwen2.5-7B v3 labels.
- `strict/` is built without the test-overlap passages, which are selected by
  `select_passages` from `scripts/build_kazculture_s1_index.py`. Those passages
  shape neither the clusters nor the profiles. `full/` has every passage.
- The embeddings are `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`
  (`EMBEDDING_BACKEND=bge-m3`), cached on disk by model, revision and text sha256.

## Outputs (`data/kazculture_pipeline/frames_v3k/`)

`labels/shard_*.jsonl` (checkpoints), `labels.jsonl`, `labels_stats.json`,
`frames.jsonl` (KazCulture frames with full-KG norm types),
`{full,strict}/{frames.jsonl,norm_types.jsonl,culture_profiles.json,kg_meta.json,frame_embeddings.npy,norm_type_embeddings.npy,index_config.json}`,
`manifest.json` (model, revision, prompt hash, counts, input hashes),
`routing_thresholds.json`, and `audit/frame_review_100{.csv,_key.json}`.
The KG stage refuses to overwrite an existing build. Use a new `--out` instead.

## Retrieval (`retrieval/kazculture_frame_retriever.py`)

```python
KazCultureFrameRetriever(frames_dir, *, settings=("strict","full"), tau_ref=None, tau_abs=None)
  .retrieve_s4r(question_text, options=None, setting="strict", top_k=8, budget_chars=5000,
                route_mode="full", *, shuffled=False, exclude_passage_ids=None,
                tau_ref=None, tau_abs=None) -> list[dict]
  .labels_for_passage(passage_id, *, setting="full", shuffled=False) -> dict | None
# module level, used by src/evaluation/kazculture_final.py:
configure(**kwargs); retrieve_s4r(question_text, options=None, setting="strict", top_k=8,
  budget_chars=5000, route_mode="full", *, exclude_passage_ids=None, shuffled=False)
labels_for_passage(passage_id, *, shuffled=False)
```

- `route_mode` can be `full` or `routed` (S4R), `noabs` (abstract route off), or
  `shuffled` (CTRL-SHUF).
- Each block is a dict `{text, source_id, layer, route, indirect, similarity}`.
  `layer` is one of `frame`, `norm_type`, `culture_profile`, `norm` or `analog`.
  The first block carries the route header.
- Joined with `"\n"`, the block texts equal `render_route(...)`.
- The budget works like `render_evidence_with_hits`: blocks are taken in order,
  separators are counted, packing stops at the first block that does not fit, and
  only rendered blocks are returned. The eval harness re-numbers the blocks and
  re-packs them under its own budget.
- The query is the question only, as in S1. `options` is ignored.
- Thresholds come from `routing_thresholds.json` unless they are passed in. If
  neither is available, the call raises.
- `labels_for_passage` returns the passage's own labels: the raw norm-type label,
  not the cluster name. The result does not depend on the setting, and no other
  passage influences it.
- CTRL-SHUF uses seed 0. Each Kazakh frame gets a deterministic donor from a
  different category, and its norm, norm type and values are swapped for the
  donor's. The practice text is kept. Norm types get donor norm types, and analogs
  get CultureBank donors from a different norm type. Donors are drawn only from
  the setting's own KG.

## Routing thresholds

`scripts/tune_kazculture_routing.py` tunes τ_ref ∈ {.3,.4,.5,.6} × τ_abs ∈ {.3,.4,.5}.

- It uses 400 train questions sampled with seed 0 from rows 400 onward. These are
  disjoint from the first 400 rows that `tune_kazculture_retrieval.py` reads.
- It runs on the full index, with each question's own passage masked at query time.
- The objective is the mean per-token gold log-likelihood
  (`HFQwenLetterScorerV2.score_gold_loglik`), using the gate's `frame_evidence`
  and `free_text_prompt`.
- Ties go to the higher thresholds.
- It writes `routing_tuning_train_row_ids.json`. Pass that file to
  `eval_kazculture_traindev_gate.py --exclude_ids_file`.
- Limitation: the masked passage still shaped the clusters and the profile.

## Running (do not run on the login node)

```bash
sbatch slurm/kazculture_frames.sh                        # label array 0-3 (2xA100 each)
sbatch --dependency=afterok:<array_jobid> slurm/kazculture_frames_index.sh   # merge+kg (1 GPU)
sbatch slurm/kazculture_frames_tune.sh                   # thresholds (2xA100)
python scripts/build_kazculture_frame_audit.py build     # CPU, seconds
python scripts/build_kazculture_frame_audit.py score --reviewed <filled.csv>
```

CPU smoke run (fake labels, TF-IDF):
`python scripts/build_kazculture_frames.py --stages label,merge,kg --fake_labeler --embedding_backend tfidf --limit 200 --out /tmp/<dir>`.

## GPU budget (estimate, not measured)

Qwen3-32B in bf16 is 65.5 GB of weights, about 32.8 GB per GPU at tensor parallel 2.
On A100-40GB with `gpu_memory_utilization=0.92`, that leaves about 4–5 GB per GPU
for KV cache. At 128 KB per token per GPU, that is roughly 30k tokens, or about
20 concurrent ~1.5k-token requests. `max_model_len=6144` fits that.

For 6,749 passages, estimated at ~1.2k prompt tokens and ≤200 reply tokens each:

- Prefill is ~8M tokens, about 30 minutes.
- Decode is ~0.5M tokens, about 15–25 minutes.
- Model load is about 20–30 minutes per job.

With 4 shards, each task should take about 45–60 minutes, well under the 3-hour
limit. Fall back to `GPU_MEM_UTIL=0.95` or `ENFORCE_EAGER=1` (`--enforce_eager`)
if vLLM reports no KV-cache room.
