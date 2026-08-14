# CulturalBench Evaluation Summary

- LLM provider: vllm
- Model: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k: 8
- Corpus triples: 153882, passages: 22990
- Units: 2454 (easy=1227, hard=1227)

## Accuracy (95% paired-bootstrap CI)

| System | Combined | Easy | Hard |
|---|---|---|---|
| S0 No-KG | 52.2% [50.2, 54.2] | 71.6% [69.1, 74.2] | 32.8% [30.1, 35.5] |
| S1 Raw-text RAG | 41.4% [39.5, 43.4] | 67.3% [64.6, 69.9] | 15.5% [13.5, 17.5] |
| S2 Flat KG | 42.1% [40.1, 44.1] | 67.0% [64.4, 69.6] | 17.3% [15.2, 19.4] |
| S3 Layered KG | 42.5% [40.5, 44.4] | 67.3% [64.7, 70.0] | 17.7% [15.7, 19.8] |
| S4 Pathway KG | 38.4% [36.5, 40.3] | 66.0% [63.4, 68.6] | 10.8% [9.1, 12.6] |
| S4 Triple ablation | 42.3% [40.4, 44.2] | 67.2% [64.5, 69.8] | 17.5% [15.4, 19.6] |
| S4 No-shared ablation | 38.2% [36.2, 40.1] | 66.3% [63.7, 69.0] | 10.1% [8.4, 11.8] |

## McNemar pairwise tests (combined units)

| Comparison | delta acc | discordant (b/c) | p-value | method |
|---|---|---|---|---|
| S0_no_kg vs S1_raw_text_rag | +10.8% | 399/134 | 0.0 | exact_binomial |
| S0_no_kg vs S2_flat_kg | +10.1% | 385/138 | 0.0 | exact_binomial |
| S0_no_kg vs S3_layered_kg | +9.7% | 373/135 | 0.0 | exact_binomial |
| S0_no_kg vs S4_abstraction_refinement_kg | +13.8% | 434/96 | 0.0 | exact_binomial |
| S0_no_kg vs S4_triple_retrieval | +9.9% | 372/130 | 0.0 | exact_binomial |
| S0_no_kg vs S4_no_shared_pivot | +14.0% | 443/100 | 0.0 | exact_binomial |
| S1_raw_text_rag vs S2_flat_kg | -0.7% | 159/177 | 0.353723 | exact_binomial |
| S1_raw_text_rag vs S3_layered_kg | -1.1% | 157/184 | 0.159037 | exact_binomial |
| S1_raw_text_rag vs S4_abstraction_refinement_kg | +3.0% | 220/147 | 0.000163 | exact_binomial |
| S1_raw_text_rag vs S4_triple_retrieval | -0.9% | 154/177 | 0.226521 | exact_binomial |
| S1_raw_text_rag vs S4_no_shared_pivot | +3.2% | 224/146 | 5.9e-05 | exact_binomial |
| S2_flat_kg vs S3_layered_kg | -0.4% | 89/98 | 0.558652 | exact_binomial |
| S2_flat_kg vs S4_abstraction_refinement_kg | +3.7% | 237/146 | 4e-06 | exact_binomial |
| S2_flat_kg vs S4_triple_retrieval | -0.2% | 94/99 | 0.773491 | exact_binomial |
| S2_flat_kg vs S4_no_shared_pivot | +3.9% | 238/142 | 1e-06 | exact_binomial |
| S3_layered_kg vs S4_abstraction_refinement_kg | +4.1% | 241/141 | 0.0 | exact_binomial |
| S3_layered_kg vs S4_triple_retrieval | +0.2% | 37/33 | 0.720203 | exact_binomial |
| S3_layered_kg vs S4_no_shared_pivot | +4.3% | 241/136 | 0.0 | exact_binomial |
| S4_abstraction_refinement_kg vs S4_triple_retrieval | -3.9% | 136/232 | 1e-06 | exact_binomial |
| S4_abstraction_refinement_kg vs S4_no_shared_pivot | +0.2% | 38/33 | 0.635308 | exact_binomial |
| S4_triple_retrieval vs S4_no_shared_pivot | +4.1% | 236/135 | 0.0 | exact_binomial |

## Notes
- S0–S3 are Phase 1 baselines (raw triple corpus for S2/S3).
- S4 uses critique-refined pathways from the Phase 2 corpus build.
- See mechanism_analysis.md for H2 completeness-bucket breakdown.
- Interpretation per docs/RESEARCH_DESIGN.md.