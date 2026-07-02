# CulturalBench Evaluation Summary (Phase 1: baselines S0-S3)

- LLM provider: vllm
- Model: Qwen/Qwen2.5-7B-Instruct
- Retrieval top-k: 8
- Corpus triples: 4, passages: 20
- Units: 40 (easy=20, hard=20)

## Accuracy (95% paired-bootstrap CI)

| System | Combined | Easy | Hard |
|---|---|---|---|
| S0 No-KG | 72.5% [57.5, 85.0] | 75.0% [55.0, 90.0] | 70.0% [50.0, 90.0] |
| S1 Raw-text RAG | 70.0% [55.0, 82.5] | 65.0% [45.0, 85.0] | 75.0% [55.0, 95.0] |
| S2 Flat KG | 65.0% [50.0, 80.0] | 65.0% [45.0, 85.0] | 65.0% [45.0, 85.0] |
| S3 Layered KG | 67.5% [52.5, 82.5] | 70.0% [50.0, 90.0] | 65.0% [45.0, 85.0] |

## McNemar pairwise tests (combined units)

| Comparison | delta acc | discordant (b/c) | p-value | method |
|---|---|---|---|---|
| S0_no_kg vs S1_raw_text_rag | +2.5% | 4/3 | 1.0 | exact_binomial |
| S0_no_kg vs S2_flat_kg | +7.5% | 4/1 | 0.375 | exact_binomial |
| S0_no_kg vs S3_layered_kg | +5.0% | 3/1 | 0.625 | exact_binomial |
| S1_raw_text_rag vs S2_flat_kg | +5.0% | 4/2 | 0.6875 | exact_binomial |
| S1_raw_text_rag vs S3_layered_kg | +2.5% | 4/3 | 1.0 | exact_binomial |
| S2_flat_kg vs S3_layered_kg | -2.5% | 0/1 | 1.0 | exact_binomial |

## Notes
- This is Phase 1 (baselines only). S2/S3 use a raw KG without entity
  resolution; the S4 abstraction->refinement system is a later phase.
- Interpretation per docs/RESEARCH_DESIGN.md: we expect KG-augmented
  systems (S2/S3) to improve over S0/S1 if structure helps.