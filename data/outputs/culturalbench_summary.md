# CulturalBench Evaluation Summary

- LLM provider: vllm
- Model: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k: 8
- Corpus triples: 119711, passages: 22990
- Units: 500 (easy=250, hard=250)

## Accuracy (95% paired-bootstrap CI)

| System | Combined | Easy | Hard |
|---|---|---|---|
| S0 No-KG | 76.6% [72.8, 80.2] | 71.6% [66.0, 77.2] | 81.6% [76.8, 86.4] |
| S1 Raw-text RAG | 71.4% [67.4, 75.4] | 65.6% [59.6, 71.2] | 77.2% [72.0, 82.4] |
| S2 Flat KG | 72.4% [68.4, 76.4] | 67.2% [61.6, 72.8] | 77.6% [72.4, 82.4] |
| S3 Layered KG | 70.6% [66.6, 74.6] | 64.0% [58.0, 69.6] | 77.2% [72.0, 82.0] |
| S4 Pathway KG | 73.2% [69.2, 77.0] | 68.0% [62.0, 73.6] | 78.4% [73.2, 83.2] |

## McNemar pairwise tests (combined units)

| Comparison | delta acc | discordant (b/c) | p-value | method |
|---|---|---|---|---|
| S0_no_kg vs S1_raw_text_rag | +5.2% | 53/27 | 0.004868 | exact_binomial |
| S0_no_kg vs S2_flat_kg | +4.2% | 48/27 | 0.020298 | exact_binomial |
| S0_no_kg vs S3_layered_kg | +6.0% | 56/26 | 0.00122 | exact_binomial |
| S0_no_kg vs S4_abstraction_refinement_kg | +3.4% | 39/22 | 0.039617 | exact_binomial |
| S1_raw_text_rag vs S2_flat_kg | -1.0% | 23/28 | 0.575849 | exact_binomial |
| S1_raw_text_rag vs S3_layered_kg | +0.8% | 24/20 | 0.651588 | exact_binomial |
| S1_raw_text_rag vs S4_abstraction_refinement_kg | -1.8% | 27/36 | 0.313503 | exact_binomial |
| S2_flat_kg vs S3_layered_kg | +1.8% | 26/17 | 0.222053 | exact_binomial |
| S2_flat_kg vs S4_abstraction_refinement_kg | -0.8% | 34/38 | 0.723948 | exact_binomial |
| S3_layered_kg vs S4_abstraction_refinement_kg | -2.6% | 30/43 | 0.159712 | exact_binomial |

## Notes
- S0–S3 are Phase 1 baselines (raw triple corpus for S2/S3).
- S4 uses critique-refined pathways from the Phase 2 corpus build.
- See mechanism_analysis.md for H2 completeness-bucket breakdown.
- Interpretation per docs/RESEARCH_DESIGN.md.