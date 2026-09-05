# Food/Hospitality Multiculture Keep-Only Full-77 Results

## Corpus build

Input subset: `food_hospitality_multiculture_records_keep_only.jsonl`

The full 77-record Narval build completed successfully.

- Passages: 77
- Raw triples: 431
- Refined triples: 432
- Pathways: 327
- Entities: 443
- Alignments: 1
- Cultures represented in pathways: 19

Pathway completeness:

- FULL: 52
- SOCIAL: 160
- SURFACE: 115

Manual strict L1 actor-only check found 3 suspicious L1 pathway nodes, which was considered low enough to proceed without rebuilding.

## Retrieval diagnostics

S4 retrieval diagnostics were run on the controlled food/hospitality CulturalBench subset using top-k = 8.

- Questions: 40
- match@1: 25/40 = 62.5%
- match@3: 25/40 = 62.5%
- match@8: 25/40 = 62.5%
- FULL@8: 30/40 = 75.0%

Cultures with usable corpus coverage generally retrieved matched evidence. Remaining retrieval failures were concentrated in low/no-coverage cultures: Iran, Bangladesh, Zimbabwe, South Africa, and New Zealand.

## CulturalBench evaluation

Model: Qwen/Qwen2.5-7B-Instruct through vLLM  
Retrieval top-k: 8  
Units: 40 total, with 27 Easy and 13 Hard units

| System | Combined | Easy | Hard |
|---|---:|---:|---:|
| S0 No-KG | 65.0% | 63.0% | 69.2% |
| S1 Raw-text RAG | 65.0% | 66.7% | 61.5% |
| S2 Flat KG | 67.5% | 74.1% | 53.8% |
| S3 Layered KG | 65.0% | 70.4% | 53.8% |
| S4 Pathway KG | 75.0% | 77.8% | 69.2% |
| S4 Triple ablation | 65.0% | 70.4% | 53.8% |
| S4 No-shared ablation | 72.5% | 77.8% | 61.5% |

## Pairwise gain/loss analysis

S4 compared with S3 Layered KG:

- Same: 36
- S4 gains: 4
- S4 losses: 0

S4 compared with S4 Triple ablation:

- Same: 36
- S4 gains: 4
- S4 losses: 0

S4 compared with S0 No-KG:

- Same: 34
- S4 gains: 5
- S4 losses: 1

## S4-over-S3 gain cases

S4 corrected S3 on four units:

- `easy_23`, Bangladesh: collective cultural identity through festival celebration
- `easy_220`, Russia: meeting people for the first time
- `hard_5`, South Korea: Korean dining etiquette around drinks/payment
- `hard_221`, Netherlands: expected behavior when late to a dinner party

Qualitative inspection showed mixed evidential strength. The Russia case was the strongest culture-matched pathway-grounded gain. South Korea and the Netherlands were culture-matched but partially indirect. Bangladesh was answered correctly despite lacking Bangladesh-specific retrieval, so it should not be treated as strong evidence of pathway-grounded reasoning.

## Interpretation

S4 Pathway KG achieved the highest accuracy on the controlled food/hospitality subset. The improvement over S3 Layered KG and the S4 Triple ablation suggests that pathway-structured evidence may provide value beyond flat or layered triple retrieval. However, McNemar tests were not statistically significant on the 40-unit controlled subset, so the result should be framed as promising preliminary evidence rather than a definitive statistical claim.

The main limitation is culture coverage. Retrieval worked well for cultures represented in the keep-only corpus, but failed for cultures with little or no retained CultureBank evidence, especially Iran, Bangladesh, Zimbabwe, South Africa, and New Zealand.
