# Handoff Summary Since `meeting-steps-1-5-current-main`

Last updated: 2026-08-05  
Current working branch: `fix-flexible-pathways`  
Audience: project partner who has not been involved since work began on `meeting-steps-1-5-current-main`

## 1. Executive summary

Since the `meeting-steps-1-5-current-main` branch was created, the work shifted from a broad CultureBank pipeline toward a controlled, inspectable experiment on food/hospitality records. The main goal was to debug whether our layered cultural KG can produce useful pathway evidence for S4 retrieval.

The biggest finding was that the original S4 pathway corpus was not failing because the data lacked cultural structure. It was failing because the pipeline was not correctly turning extracted layered triples into meaningful pathways. The most important bug was that pathway construction expected mostly `L3 -> L2 -> L1` edge direction, while the extracted graph often had useful directions such as `L2 -> L1`, `L2 -> L3`, or `L1 -> L3`. After fixing this, another issue appeared: many supposed L3 nodes were actually topic/category metadata or schema placeholders. We then fixed extraction, critique filtering, layer validation, and pathway construction.

The current full Narval keep-only run now produces non-empty and meaningful pathway evidence:

- Bad placeholder triples: `0`
- FULL pathways: `22`
- SOCIAL pathways: `64`
- SURFACE pathways: `48`

This means the pathway construction problem is now fixed enough to move to small S0/S4 evaluation. It does not yet prove that S4 improves accuracy. The next task is to test whether the corrected pathway index helps answer CulturalBench-style questions.

## 2. Branch history and what changed

### `meeting-steps-1-5-current-main`

This branch was created to implement the immediate post-meeting plan. Relative to `main`, it added the controlled-subset workflow, evaluation/error-analysis documentation, and initial results/diagnostics.

Major additions on this branch included:

- `docs/MEETING_ACTION_PLAN.md`
- `docs/STEPS_1_TO_7_SUMMARY.md`
- `docs/S4_EVALUATION_GUIDE.md`
- `docs/CONTROLLED_SUBSET_GUIDE.md`
- `docs/CULTURALBENCH_BASELINE_NOTES.md`
- `docs/ERROR_ANALYSIS_GUIDE.md`
- `docs/KEEP_ONLY_38_PATHWAY_DIAGNOSTIC.md`
- `scripts/08_error_analysis.py`
- `scripts/09_select_controlled_subset.py`
- updates to `scripts/06_build_retrieval_corpus.py`
- updates to `slurm/build_corpus.sh`
- controlled food/hospitality data files under `data/controlled/`

The main purpose of this branch was to make the experiment controlled and comparable rather than continuing with a broad, hard-to-debug corpus.

### `fix-flexible-pathways`

This branch was created from `meeting-steps-1-5-current-main` after the first keep-only Narval run showed that the pipeline produced many refined triples but zero pathways.

Relative to `meeting-steps-1-5-current-main`, this branch adds/fixes:

- `src/pathway_builder.py`
- `src/prompts.py`
- `src/triple_extractor.py`
- `src/critique_agent.py`
- `src/layer_classifier.py`
- `docs/experiments/food_hospitality_keep_only_flexible_full38.md`

The purpose of this branch is to make pathway construction flexible, layer-aware, and robust to extraction artifacts.

## 3. Controlled food/hospitality subset

We created a controlled subset because the supervisor's feedback was that broad random context was too hard to debug. The goal was to first make the pipeline work on one interpretable topic.

Selection settings:

- CultureBank scan limit: `2000`
- Max selected records: `100`
- Minimum record score: `3`
- Required food/hospitality/social term: `True`
- Selected CultureBank records: `100`
- Selected CulturalBench questions: `40`
- Cultural group distribution: `American: 100`

The selected records were then manually audited into:

- `keep`: clearly useful food/hospitality cultural record
- `maybe`: relevant but weaker or ambiguous
- `remove`: not useful enough for this controlled pathway experiment

Final manually reviewed set used for the main keep-only experiment:

- `38` keep-only records

Important caveat: this is an American-only food/hospitality subset. It is useful for extraction, layer repair, pathway construction, and S4 pathway retrieval debugging. It is not useful for testing cross-cultural alignment yet, because there is only one cultural group.

## 4. First controlled pipeline tests

### Local 3-record tests

We first tested the controlled subset locally with Ollama on 3 records.

Early results showed that extraction and graph construction worked, but pathway quality was inconsistent. Some runs produced only one partial pathway, such as a direct value-to-practice link. This was useful as a smoke test but not enough for S4.

### First full Narval keep-only run before fixes

A full keep-only run was then done on Narval using the 38 manually kept records.

Run setup:

- Input records: `data/controlled/food_hospitality_records_keep_only.jsonl`
- Actual records loaded: `38`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Alignment verification skipped
- Critique enabled

Result before fixes:

- Raw triples: `196`
- Refined triples: `196`
- Canonical entities: `163`
- Alignments: `0`
- Pathways: `0`

This was the key diagnostic failure. The graph had layered triples, but the pathway builder returned no pathways.

## 5. Why the first full run produced zero pathways

The first diagnostic showed that layered evidence existed, but the direction and labeling did not match the pathway builder's assumptions.

Important pattern from the diagnostic:

- Subject layers were mostly `LAYER_2_SOCIAL` and `LAYER_1_SURFACE`
- Object layers included many `LAYER_1_SURFACE` and `LAYER_3_DEEP_VALUE`
- Many useful edges were `L2 -> L1`, `L2 -> L3`, or `L1 -> L3`
- The old pathway builder mostly expected pathways to start from L3 and traverse downward

So the problem was not simply extraction failure. The graph had useful cultural evidence, but the builder was too rigid.

The initial conclusion was:

> Do not run S0-S4 evaluation while the pathway index is empty. Fix pathway construction first.

## 6. Pathway construction fixes

The first set of fixes changed `src/pathway_builder.py` so that pathways are not forced to be only `L3 -> L2 -> L1`.

The updated logic supports:

- FULL pathways when L1, L2, and L3 evidence occur in the same record/component
- SOCIAL pathways such as `L2 -> L1` or `L2 -> L3`
- SURFACE pathways such as `L1 -> L3` or other partial evidence
- record-level pathway construction when direction is not ideal
- filtering of weak metadata/topic nodes
- filtering of generic pathway nodes like `American` or `people`

Local rebuild from existing triples initially produced many pathways, but this exposed quality issues:

- some generic nodes like `American` and `people` appeared as pathway nodes
- after filtering, pathways dropped from `134` to `87`
- generic hits dropped to `0`
- however, there were still no FULL pathways from the old refined triples

This led to the next diagnosis: the extraction/critique stage was misusing dataset topics as L3 values.

## 7. Extraction and critique fixes

The prompt and critique logic were then updated because the model often created bad L3 metadata triples such as:

```text
American -> has cultural topic -> Social Norms and Etiquette
```

These looked like L3 values to the code but were actually dataset metadata. They are not real deep values.

Fixes added:

### `src/prompts.py`

The extraction prompt now says:

- Topic is metadata, not a deep value
- do not create `has cultural topic`, `has topic`, `belongs to topic`, or `has category` triples
- do not use topic/category labels like `Social Norms and Etiquette`, `Food and Dining`, or `Social Interactions` as L3 values
- use real L3 values when supported, such as hospitality, gratitude, reciprocity, politeness, respect, generosity, care, family connection, and cleanliness

### `src/critique_agent.py`

Critique now:

- flags topic/category triples miscast as values
- drops metadata-as-value triples from the refined KG
- rejects critique suggestions that turn the Topic field into an L3 value
- can suggest cautious real value bridges from record wording rather than topic metadata

After this, local extraction began producing real L3 labels such as:

- gratitude
- hospitality
- respect for guests
- host-guest reciprocity
- hospitality and appreciation
- reciprocity

## 8. Placeholder-label bug and fix

After the topic-metadata problem was fixed, a second extraction artifact appeared. The local model sometimes used the schema label itself as an entity label, for example:

```text
individuals, birthday person, and guests -> express gratitude -> LAYER_3_DEEP_VALUE
```

This created fake pathway nodes named `LAYER_3_DEEP_VALUE` or `LAYER_1_SURFACE`.

Fixes added:

### `src/triple_extractor.py`

The extractor now:

- drops triples where subject/object is `LAYER_1_SURFACE`, `LAYER_2_SOCIAL`, `LAYER_3_DEEP_VALUE`, or `UNKNOWN`
- repairs a placeholder object when the predicate clearly names the real value, for example `express gratitude` becomes object `gratitude`
- removes topic/category metadata triples before they enter `corpus_triples.jsonl`

### `src/prompts.py`

The prompt now explicitly says subject/object text must be actual concepts, actors, practices, roles, norms, or values, not schema labels.

### `src/critique_agent.py`

Critique now also flags/drops layer-placeholder triples from the refined KG.

The local smoke test after this fix showed:

- Bad placeholder triples: `0`
- Real L3 values present
- FULL pathways present, but some layer assignments were still off

## 9. Layer-repair fix

The next issue was that the model sometimes labeled social actors as L1 surface practices. For example:

- `hosts and guests` was sometimes L1, but should be L2
- `birthday person and guests` was sometimes L1, but should be L2
- `gifts` was sometimes L2, but should be L1
- `hospitality and appreciation` was sometimes L1/L2, but should be L3

Fix added in `src/layer_classifier.py`:

- conservative high-confidence layer repair after validation
- actor/social role phrases are repaired to L2
- concrete practices/objects are repaired to L1
- explicit value terms are repaired to L3

Examples of repaired layer behavior:

```text
hosts and guests -> LAYER_2_SOCIAL
gifts/drinks/meals/food -> LAYER_1_SURFACE
gratitude/hospitality/reciprocity/appreciation -> LAYER_3_DEEP_VALUE
```

After this fix, local 3-record test results improved to:

- Raw triples: `16`
- Pathways: `14`
- FULL: `3`
- SOCIAL: `10`
- SURFACE: `1`

Example local FULL pathways:

```text
hospitality -> hosts and guests -> guests often bring gifts or their own drinks
gratitude -> guests -> birthdays with meals, gifts, and paying for others
gratitude -> individuals, birthday person, and guests -> birthdays with meals, gifts, and paying for others
```

This was good enough to move back to Narval.

## 10. Narval validation after fixes

### 10-record Narval smoke test

A 10-record Narval smoke test was run with:

- Branch: `fix-flexible-pathways`
- Input: `data/controlled/food_hospitality_records_keep_only.jsonl`
- Limit: `10`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Alignment skipped
- Alignment verification skipped
- Critique enabled

Result:

- Bad placeholder triples: `0`
- FULL pathways: `10`
- SOCIAL pathways: `18`
- SURFACE pathways: `16`

Example FULL pathway:

```text
gratitude -> friends and family -> engage in social activities involving food, alcohol, and conversation
hospitality -> hosts and guests -> offer, provide, and consume drinks and food, with guests often bringing gifts or their own drinks
```

This confirmed that the fixes worked on Narval with the stronger model.

### Full 38-record Narval keep-only run

The full keep-only Narval experiment was then run on all 38 manually kept records.

Run setup:

- Branch: `fix-flexible-pathways`
- Environment: Narval
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Input records: `data/controlled/food_hospitality_records_keep_only.jsonl`
- Corpus limit: `100`
- Actual records loaded: `38`
- Alignment: skipped
- Alignment verification: skipped
- Critique: enabled

Final result:

- Bad placeholder triples: `0`
- FULL pathways: `22`
- SOCIAL pathways: `64`
- SURFACE pathways: `48`

Top L3 labels included:

- gratitude
- hospitality
- family connection
- reciprocity
- cleanliness
- flexibility in dining customs
- affordability

Example FULL pathways:

```text
gratitude -> friends and family -> engage in social activities involving food, alcohol, and conversation
hospitality -> hosts and guests -> offer, provide, and consume drinks and food, with guests often bringing gifts or their own drinks
```

The run was documented in:

```text
docs/experiments/food_hospitality_keep_only_flexible_full38.md
```

Generated experiment files were saved on Narval under:

```text
data/experiments/food_hospitality_keep_only_flexible_full38/
```

Those generated files were intentionally not pushed to GitHub.

## 11. What the results tell us

### What is now fixed

The main S4 blocker is fixed. The pipeline no longer produces an empty pathway index for the controlled keep-only subset.

Specifically:

- controlled food/hospitality subset can be processed end-to-end
- topic/category metadata is no longer treated as real L3 value evidence
- schema layer labels are no longer becoming graph nodes
- social actors/roles are mostly repaired to L2
- food/drink/gift/meal practices are mostly repaired to L1
- values like gratitude and hospitality are mostly repaired to L3
- pathway builder now supports flexible pathways, not only strict `L3 -> L2 -> L1`
- full L3/L2/L1 pathways are now produced

### What is not proven yet

This does not prove that S4 improves model accuracy. It only proves that S4 now has a meaningful non-empty pathway corpus to retrieve from.

We still need to run evaluation comparing:

- S0: no KG
- S1: raw text RAG
- S2: flat KG
- S3: layered KG
- S4: flexible pathway KG

### Current limitations

- The controlled subset is American-only, so it does not test cross-cultural alignment.
- Some L3 labels are still too long or action-like, for example labels that include a whole practice phrase instead of a compact value.
- Some pathways are better than others, so retrieval quality still needs to be inspected.
- The full generated experiment artifacts are on Narval, not GitHub.
- We should not merge to main until S4 retrieval/evaluation behavior is checked.

## 12. Current GitHub state

As of this handoff, the important branches are:

- `meeting-steps-1-5-current-main`: controlled subset and initial meeting-step implementation
- `fix-flexible-pathways`: current active branch with pathway/extraction/layer fixes
- `integrate-v43-extraction-schema`: separate branch, not merged here

Important current files to read:

```text
docs/MEETING_ACTION_PLAN.md
docs/STEPS_1_TO_7_SUMMARY.md
docs/CONTROLLED_SUBSET_GUIDE.md
docs/S4_EVALUATION_GUIDE.md
docs/KEEP_ONLY_38_PATHWAY_DIAGNOSTIC.md
docs/experiments/food_hospitality_keep_only_flexible_full38.md
src/pathway_builder.py
src/triple_extractor.py
src/critique_agent.py
src/layer_classifier.py
src/prompts.py
```

## 13. Recommended next steps

1. Do not keep patching extraction locally unless evaluation exposes a concrete failure.
2. Use the corrected full 38-record keep-only corpus on Narval as the current S4 corpus snapshot.
3. Run a small S0 vs S4 evaluation first, not the full system suite.
4. Inspect retrieved pathways for each evaluated question.
5. If S4 helps or fails, do error analysis before changing the schema again.
6. Only after the controlled subset behaves reasonably should we scale beyond food/hospitality.

Suggested immediate experiment:

```text
Evaluate S0_no_kg vs S4_abstraction_refinement_kg on a small food/hospitality CulturalBench subset using the corrected pathway index.
```

Success criteria for the next step:

- S4 retrieves relevant pathways
- S4 evidence does not override the question/options incorrectly
- errors can be explained by retrieval mismatch, extraction issue, or model reasoning issue
- S4 shows at least some cases where pathway context improves or clarifies the answer

## 14. One-sentence takeaway

We moved from an empty and directionally brittle S4 pathway corpus to a corrected flexible pathway pipeline that produces real L3/L2/L1 cultural evidence on the controlled food/hospitality keep-only subset, but we still need to prove that this evidence improves downstream CulturalBench performance.
