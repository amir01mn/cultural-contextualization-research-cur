# Food/Hospitality Selected CulturalBench Evaluation

## Setup

- Branch: fix-flexible-pathways
- Model: Qwen/Qwen2.5-7B-Instruct
- Evaluation set: data/controlled/food_hospitality_culturalbench_questions.jsonl
- Units: 40
- KG corpus: 38 keep-only food/hospitality records
- Retrieval top-k: 8

## Results

S0 No-KG: 65.0%
S1 Raw-text RAG: 70.0%
S2 Flat KG: 72.5%
S3 Layered KG: 72.5%
S4 Pathway KG: 72.5%
S4 Triple ablation: 72.5%
S4 No-shared ablation: 72.5%

## Interpretation

KG retrieval improved over the no-KG baseline, but S3, S4, and the S4 ablations were effectively identical. S3 and S4 made the same predictions on 39/40 units, and S4 and S4-triple also matched on 39/40 units.

This means the current run does not show a pathway-specific advantage. The evidence suggests that retrieved context helps, but not that L3 -> L2 -> L1 pathway structure helps beyond layered triples or refined triples.

## Manual retrieval audit

The audit showed that S4 often retrieved American or generic food/hospitality pathways for non-American questions. Examples included Bangladesh, Iran, South Korea, Zimbabwe, and the Netherlands. This suggests the main limitation is culture mismatch between the KG corpus and the selected CulturalBench questions.

## Next step

Build a multi-culture food/hospitality KG matched to the selected evaluation cultures, then rerun S0-S4. This will test whether culture-matched pathways help the model answer better.

