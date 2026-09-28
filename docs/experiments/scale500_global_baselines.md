# Scale-500 global-retrieval baselines

## Status

Controlled development-scale baseline. These results are not the final large-scale evaluation.

Input corpus:

- 500 randomly sampled CultureBank records
- 250 TikTok + 250 Reddit
- seed: 42
- SHA-256: `48a36783c28c130a6804441184dd84362215c8f1e98c5ac973710f22b47999f5`
- 195 raw culture labels
- 34 CultureBank topics

Evaluation:

- Qwen/Qwen2.5-7B-Instruct
- 40 controlled CulturalBench scoring units
- 27 Easy + 13 Hard
- top-k 8

## Validated Scale-500

Method base: `validated77-v2` / `c844b37`

Build job: `4110905` (COMPLETED, 18:13)

- Raw triples: 2,267
- Refined triples: 2,321
- Canonical entities: 2,731
- Verified pairwise alignments: 140 from 200 candidates
- Unique shared pivots: 10
- Total pathways: 1,404
- FULL: 150
- SOCIAL: 679
- SURFACE: 575
- Shared-pivot pathways: 155
- Maximum pathways for one culture: 185, below the 500-per-culture cap

Evaluation job: `4117055` (COMPLETED, 13:14)

| System | Accuracy |
|---|---:|
| S0 No-KG | 65.0% |
| S1 Raw-text RAG | 57.5% |
| S2 Flat KG | 60.0% |
| S3 Layered KG | 55.0% |
| S4 Pathway KG | 60.0% |
| S4 Triple ablation | 55.0% |
| S4 No-shared ablation | 62.5% |

Mean generated support:

- S4 Pathway KG: 18.1%
- S4 Triple: 1.2%
- S4 No-shared: 1.4%

## Historical-method Scale-500 control

Method base: `full77-baseline` / `d18b2fb`

Build job: `4118167` (COMPLETED, 19:11)

- Raw triples: 2,265
- Refined triples: 2,322
- Canonical entities: 2,388
- Alignments: 6
- Shared abstraction nodes: 0
- Total pathways: 1,797
- FULL: 261
- SOCIAL: 977
- SURFACE: 559

Evaluation job: `4122097` (COMPLETED, 20:14)

| System | Accuracy |
|---|---:|
| S0 No-KG | 65.0% |
| S1 Raw-text RAG | 57.5% |
| S2 Flat KG | 60.0% |
| S3 Layered KG | 55.0% |
| S4 Pathway KG | 62.5% |
| S4 Triple ablation | 55.0% |
| S4 No-shared ablation | 62.5% |

## Interpretation

The historical and validated pipelines show the same Scale-500 accuracy for S0, S1, S2, S3, the S4 triple ablation, and the S4 no-shared ablation. Full S4 differs by one 40-unit question (62.5% historical versus 60.0% validated).

The common degradation relative to the 77-record food/hospitality development corpus therefore does not appear to be primarily caused by the validation hardening. The Scale-500 corpus is both larger and much more heterogeneous, so this experiment does not isolate corpus size from domain heterogeneity.

The next controlled experiment holds the validated 500-record corpus fixed and changes only the pathway candidate-selection mechanism from global semantic retrieval to topic-filtered semantic retrieval.
