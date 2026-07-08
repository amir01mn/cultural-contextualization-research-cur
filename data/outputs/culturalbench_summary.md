# CulturalBench Evaluation Summary

- LLM provider: vllm
- Model: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k: 8
- Corpus triples: 7, passages: 20
- Units: 40 (easy=20, hard=20)

## Accuracy (95% paired-bootstrap CI)

| System | Combined | Easy | Hard |
|---|---|---|---|
| S0 No-KG | 72.5% [57.5, 85.0] | 75.0% [55.0, 90.0] | 70.0% [50.0, 90.0] |
| S1 Raw-text RAG | 75.0% [62.5, 87.5] | 75.0% [55.0, 95.0] | 75.0% [55.0, 95.0] |
| S2 Flat KG | 70.0% [55.0, 82.5] | 70.0% [50.0, 90.0] | 70.0% [50.0, 90.0] |
| S3 Layered KG | 72.5% [57.5, 85.0] | 70.0% [50.0, 90.0] | 75.0% [55.0, 95.0] |
| S4 Pathway KG | 72.5% [57.5, 85.0] | 75.0% [55.0, 90.0] | 70.0% [50.0, 90.0] |

## McNemar pairwise tests (combined units)

| Comparison | delta acc | discordant (b/c) | p-value | method |
|---|---|---|---|---|
| S0_no_kg vs S1_raw_text_rag | -2.5% | 3/4 | 1.0 | exact_binomial |
| S0_no_kg vs S2_flat_kg | +2.5% | 5/4 | 1.0 | exact_binomial |
| S0_no_kg vs S3_layered_kg | +0.0% | 4/4 | 1.0 | exact_binomial |
| S0_no_kg vs S4_abstraction_refinement_kg | +0.0% | 0/0 | 1.0 | no_discordant_pairs |
| S1_raw_text_rag vs S2_flat_kg | +5.0% | 3/1 | 0.625 | exact_binomial |
| S1_raw_text_rag vs S3_layered_kg | +2.5% | 2/1 | 1.0 | exact_binomial |
| S1_raw_text_rag vs S4_abstraction_refinement_kg | +2.5% | 4/3 | 1.0 | exact_binomial |
| S2_flat_kg vs S3_layered_kg | -2.5% | 0/1 | 1.0 | exact_binomial |
| S2_flat_kg vs S4_abstraction_refinement_kg | -2.5% | 4/5 | 1.0 | exact_binomial |
| S3_layered_kg vs S4_abstraction_refinement_kg | +0.0% | 4/4 | 1.0 | exact_binomial |

## Notes
- S0–S3 are Phase 1 baselines (raw triple corpus for S2/S3).
- S4 uses critique-refined pathways from the Phase 2 corpus build.
- See mechanism_analysis.md for H2 completeness-bucket breakdown.
- Interpretation per docs/RESEARCH_DESIGN.md.