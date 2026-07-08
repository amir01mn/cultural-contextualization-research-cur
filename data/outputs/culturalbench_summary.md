# CulturalBench Evaluation Summary

- LLM provider: vllm
- Model: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k: 8
- Corpus triples: 555, passages: 2000
- Units: 500 (easy=250, hard=250)

## Accuracy (95% paired-bootstrap CI)

| System | Combined | Easy | Hard |
|---|---|---|---|
| S0 No-KG | 76.6% [72.8, 80.2] | 71.6% [66.0, 77.2] | 81.6% [76.8, 86.4] |
| S1 Raw-text RAG | 71.2% [67.2, 75.2] | 68.8% [63.2, 74.4] | 73.6% [68.0, 78.8] |
| S2 Flat KG | 71.0% [67.0, 75.0] | 66.0% [60.0, 71.6] | 76.0% [70.8, 81.2] |
| S3 Layered KG | 70.2% [66.2, 74.2] | 66.4% [60.4, 72.0] | 74.0% [68.4, 79.2] |
| S4 Pathway KG | 71.2% [67.2, 75.2] | 66.0% [60.0, 71.6] | 76.4% [71.2, 81.6] |

## McNemar pairwise tests (combined units)

| Comparison | delta acc | discordant (b/c) | p-value | method |
|---|---|---|---|---|
| S0_no_kg vs S1_raw_text_rag | +5.4% | 55/28 | 0.004039 | exact_binomial |
| S0_no_kg vs S2_flat_kg | +5.6% | 56/28 | 0.002985 | exact_binomial |
| S0_no_kg vs S3_layered_kg | +6.4% | 56/24 | 0.000452 | exact_binomial |
| S0_no_kg vs S4_abstraction_refinement_kg | +5.4% | 49/22 | 0.00182 | exact_binomial |
| S1_raw_text_rag vs S2_flat_kg | +0.2% | 28/27 | 1.0 | exact_binomial |
| S1_raw_text_rag vs S3_layered_kg | +1.0% | 28/23 | 0.575849 | exact_binomial |
| S1_raw_text_rag vs S4_abstraction_refinement_kg | +0.0% | 29/29 | 1.0 | exact_binomial |
| S2_flat_kg vs S3_layered_kg | +0.8% | 19/15 | 0.607591 | exact_binomial |
| S2_flat_kg vs S4_abstraction_refinement_kg | -0.2% | 30/31 | 1.0 | exact_binomial |
| S3_layered_kg vs S4_abstraction_refinement_kg | -1.0% | 24/29 | 0.583132 | exact_binomial |

## Notes
- S0–S3 are Phase 1 baselines (raw triple corpus for S2/S3).
- S4 uses critique-refined pathways from the Phase 2 corpus build.
- See mechanism_analysis.md for H2 completeness-bucket breakdown.
- Interpretation per docs/RESEARCH_DESIGN.md.