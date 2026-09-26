# Run History, Artifact Lineage, and Branch Map

This page records what was run, which artifacts came from which run, where
unmerged work lives, and which issues are still open. Update it whenever a run
finishes or a branch changes state. The code cannot tell you any of this.

Last updated: 2026-09-25.

## 1. Runs, oldest first

All GPU runs used `Qwen/Qwen2.5-7B-Instruct` through vLLM on Narval, top-k 8,
unless noted.

| When | Run | Scale | Outcome | Where the evidence is |
|---|---|---|---|---|
| 2026-06 | MVP (legacy) | 100 CultureBank records, Ollama, open-ended heuristic scoring | pipeline works; heuristic scores not a research metric | `scripts/legacy/` |
| 2026-07-08 | 2,000-record pilot | 2,000 records | retrieval and ranking improvements; S3 switched to refined triples (`033425a`) | commit `033425a`, `data/processed/corpus_build_state.json` |
| 2026-07-10 | **"Broken" full run** | 22,990 records, 500 eval units | S0 76.6%, S4 73.2% (McNemar p=0.04). The build was silently broken (below). | `data/outputs/broken_run_baseline/` |
| 2026-08-03 → 05 | **Rebuilt full run (current checked-in result)** | 22,990 records, 153,882 raw / 186,691 refined triples, 29,578 pathways, 2,454 units | **negative**: S0 52.2% beats every retrieval system; S4 38.4% | `data/outputs/culturalbench_summary.md`, `mechanism_analysis.md`, `before_after_comparison.md`; checkpoint `corpus_build_state_full.json`, cache `llm_cache_full/` |
| 2026-07-30 → 09-04 | Controlled food/hospitality experiments | best: 77 audited records, 19 pathway cultures, 40 units | S4 75.0% vs S0 65.0%; S4 +4/−0 vs S3; **not significant** | branch `origin/fix-flexible-pathways`, tag `full77-baseline`, `docs/experiments/` on that branch |
| 2026-09-05 | USA-only "rebuild v2" pilot | USA records, 16,162 triples, 233 pathways, 40 units | S0 70.0%, S3 62.5%, S4 52.5%; unsatisfactory, parked | `stash@{0}`, `slurm/logs/*usa*` (its `data/rebuild_v2_usa/` outputs no longer exist) |

### Why the July full run was broken

One setting cascaded:

1. `LLM_MAX_TOKENS=512` applied to every stage, which truncated extraction and
   critique JSON.
2. `CORPUS_SCHEMA_FALLBACK_ON_EMPTY` then silently replaced **96%** of records
   with schema triples, whose layers are hardcoded.
3. That left only 1,170 L3 entities and 19 `L3→L2` edges.
4. The pathway builder walked down from L3, so it found 150 pathways and
   **zero** FULL ones.
5. S4 therefore injected vague, off-culture evidence.

Separately, CulturalBench-Hard was scored per True/False row rather than per
question. This inflated Hard accuracy, which is why the absolute numbers drop
between the two full runs.

Every one of these failures now has a guard; see
[`ARCHITECTURE.md` §7](ARCHITECTURE.md#7-guards-against-known-failure-modes).

### What the rebuilt full run says

- Pathway count is no longer the problem: 22,064 of the 29,578 pathways are FULL.
- Retrieved evidence recovers about 20% of S0's errors but corrupts about 11% of
  S0's correct answers, so every system has a net loss. A perfect per-question
  router between S0 and any system would reach about 82–83%.
- S4 is significantly worse than S3 and than refined-triple retrieval.
- In the mechanism table, 99% of S4 questions fall in the FULL bucket, so the
  completeness analysis cannot separate the H2 buckets.
- The next lever is precision: culture matching, relevance gating, and
  abstention. Adding more structure is not.

## 2. Artifact lineage warnings

- `data/processed/` and `data/outputs/` hold artifacts from **several runs at once**.
  The triple and pathway stores and the eval reports come from the August rebuild.
- `corpus_build_state.json` is the July 8 checkpoint. The August rebuild's
  checkpoint is `corpus_build_state_full.json`.
- `outputs/phase2_validation.md` does **not** describe the August corpus. It
  reports 502 pathways, 500 of them "united states", and it reads the July 8
  checkpoint. Before 2026-09-25, `validate_corpus.py` always read the default
  checkpoint whatever `RUN_TAG` the build used. That is now fixed with
  `--state_path`.
- `RUN_TAG` isolates only the cache and checkpoint. The triple, pathway, index,
  and report files always use their default names, so a new build in this
  checkout overwrites the August artifacts. Copy `data/` aside, or use a separate
  checkout, before rebuilding.
- `data/processed/pilot_run/` and `llm_cache_pilot/` come from an Aug 3 pilot;
  `llm_cache/` comes from July.

Before any formal claim: rebuild once, then regenerate validation, evaluation,
mechanism, and comparison reports from that single checkpoint.

## 3. Branches and parked work

| Ref | State vs `reorganize-architecture` | Contents |
|---|---|---|
| `reorganize-architecture` | current | this reorganization (subpackages, renamed scripts, docs) |
| `test`, `full-pathway-construction`, `full-pathway-construction-usa`, `origin/full-pathway-construction` | same commit `330b45d` (the base of this branch) | culture normalizer, pipeline-integrity tests, L2-hub pathways |
| `main` / `origin/main` | behind; `origin/main` has one extra commit | adds `docs/culturebank-culturalbench-coverage.pdf` |
| `origin/fix-flexible-pathways` = `origin/validate-flexible-pathways` (tag `full77-baseline`) | **37 commits not merged**; forked at `491d33c` | controlled-subset tooling (`scripts/08`–`12_*`), flexible layer-aware pathways, layer repair, prompt guardrails, food/hospitality experiment docs |
| `origin/meeting-fixes-s4-plan`, `origin/integrate-v43-extraction-schema` | old, July | early S4 wiring and an alternative extraction schema; likely superseded |
| `stash@{0}` | parked 2026-09-21, based on `330b45d` | "provenance-safe rebuild v2": `artifact_manifest`, `bm25_index`, `candidate_reranker`, `grounding_verifier`, extra S4 ablations (`S4_path_raw`, `S4_no_gate`, …), USA pilot SLURM scripts, `tests/test_provenance_safe_rebuild.py` |

### Merging older work into the new layout

Both `origin/fix-flexible-pathways` and the stash predate the `src/` subpackage
split and the script renames.

- Merge with `git merge -X find-renames` (on by default with `ort`). Edits to
  moved modules follow them to their new path because basenames were kept.
  Expect conflicts on import lines.
- New modules those branches add at `src/` top level must be moved into a
  subpackage. `tests/test_architecture.py` fails until they are.
- Scripts `08`–`12_*` on the branch do not collide with the new names. Move
  them to descriptive names while merging.
- Turn the stash into a branch before it can be lost:
  `git branch parked/rebuild-v2-usa 'stash@{0}'`. Or run
  `git stash branch rebuild-v2-usa 'stash@{0}'` in a clean tree to apply it on
  its original base, `330b45d`.

## 4. Open issues

1. **S3 does not match the design.** `RESEARCH_DESIGN.md` defines S3 as the raw
   layered KG, but the code has fed it refined triples since `033425a`.
   Consequences: S3 and the A3 ablation (`S4_triple_retrieval`) receive identical
   evidence, and the refinement ablation A2 (S3 vs S4) is confounded. Decide
   which one is intended, then fix either the code or the design.
2. **Prompt-wording noise is about 3%.** S3 and A3 differ only in a one-line
   evidence header, yet they disagreed on 70 of 2,454 units. Treat smaller
   differences between systems with suspicion.
3. **Artifact lineage.** See §2. One manifest-backed rebuild is needed. The
   stash's `artifact_manifest.py` was a start on this.
4. **Branch consolidation.** Choose which parts of `fix-flexible-pathways` and
   the stash become canonical.
5. **Tests need the cluster modules.** The venv has
   `include-system-site-packages = false`, and scipy, pandas, and pyarrow come
   from Narval modules. See `AGENTS.md`.
6. **Unused flags.** `USE_LLM_*` in `config.py` are read nowhere.
7. **Protocol not yet run:** a second base model, ≥3 seeds, top-k and
   generated-evidence ablations, Holm–Bonferroni correction, human audits, and
   stereotype and fairness analysis (see `RESEARCH_DESIGN.md` §8–11).
