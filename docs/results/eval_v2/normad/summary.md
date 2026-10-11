# NormAd (eval v2, Evaluation 2)

Test stories: 2315 (covered: 134). top_k=8, max_tokens=256, temperature=0.0.
Gated taus (tuned on dev, frozen): {'S1g': 0.35, 'S3g': 0.35, 'S4g': 0.25}

## all

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 2315 | 0.5745 | 0.4559 | 0.0000 | 0 | 0 |
| C2_value | 2315 | 0.6130 | 0.4997 | 0.2521 | 0 | 0 |
| C3_rule_of_thumb | 2315 | 0.7270 | 0.6467 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 2315 | 0.5538 | 0.4574 | -0.1360 | 0 | 2315 |
| S2_flat_kg | 2315 | 0.5823 | 0.4818 | 0.0510 | 0 | 2315 |
| S3_layered_kg | 2315 | 0.5693 | 0.4658 | -0.0340 | 0 | 2315 |
| S4_abstraction_refinement_kg | 2315 | 0.5546 | 0.4726 | -0.1303 | 0 | 2315 |
| S4_triple_retrieval | 2315 | 0.5702 | 0.4655 | -0.0283 | 0 | 2315 |
| S4_no_shared_pivot | 2315 | 0.5620 | 0.4655 | -0.0822 | 0 | 2315 |
| S1g | 2315 | 0.5732 | 0.4630 | -0.0085 | 0 | 1238 |
| S3g | 2315 | 0.5732 | 0.4640 | -0.0085 | 0 | 1139 |
| S4g | 2315 | 0.5732 | 0.4794 | -0.0085 | 0 | 2019 |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0207 | 99/147 | 0.002657 | 0.02923 | [-0.0341, -0.0073] |
| S2_flat_kg vs C1_country | +0.0078 | 133/115 | 0.2803 | 1 | [-0.0056, +0.0212] |
| S3_layered_kg vs C1_country | -0.0052 | 113/125 | 0.4759 | 1 | [-0.0181, +0.0082] |
| S4_abstraction_refinement_kg vs C1_country | -0.0199 | 112/158 | 0.006067 | 0.06067 | [-0.0337, -0.0060] |
| S4_triple_retrieval vs C1_country | -0.0043 | 117/127 | 0.5646 | 1 | [-0.0177, +0.0091] |
| S4_no_shared_pivot vs C1_country | -0.0125 | 99/128 | 0.06287 | 0.503 | [-0.0246, +0.0004] |
| S1g vs C1_country | -0.0013 | 47/50 | 0.8392 | 1 | [-0.0095, +0.0073] |
| S3g vs C1_country | -0.0013 | 58/61 | 0.8546 | 1 | [-0.0108, +0.0078] |
| S4g vs C1_country | -0.0013 | 94/97 | 0.885 | 1 | [-0.0130, +0.0104] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0009 | 154/152 | 0.9544 | 1 | [-0.0138, +0.0156] |
| S4g vs S4_abstraction_refinement_kg | +0.0186 | 146/103 | 0.007651 | 0.06886 | [+0.0052, +0.0324] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0384, McNemar p 0

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1525, McNemar p 0

## covered

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 134 | 0.4851 | 0.4112 | 0.0000 | 0 | 0 |
| C2_value | 134 | 0.5224 | 0.4487 | 0.2941 | 0 | 0 |
| C3_rule_of_thumb | 134 | 0.6119 | 0.5684 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 134 | 0.5299 | 0.4680 | 0.3529 | 0 | 134 |
| S2_flat_kg | 134 | 0.5075 | 0.4429 | 0.1765 | 0 | 134 |
| S3_layered_kg | 134 | 0.5075 | 0.4408 | 0.1765 | 0 | 134 |
| S4_abstraction_refinement_kg | 134 | 0.5597 | 0.5182 | 0.5882 | 0 | 134 |
| S4_triple_retrieval | 134 | 0.5075 | 0.4406 | 0.1765 | 0 | 134 |
| S4_no_shared_pivot | 134 | 0.5224 | 0.4638 | 0.2941 | 0 | 134 |
| S1g | 134 | 0.5299 | 0.4681 | 0.3529 | 0 | 123 |
| S3g | 134 | 0.5000 | 0.4307 | 0.1176 | 0 | 125 |
| S4g | 134 | 0.5224 | 0.4772 | 0.2941 | 0 | 120 |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | +0.0448 | 7/1 | 0.07031 | 0.6328 | [+0.0075, +0.0896] |
| S2_flat_kg vs C1_country | +0.0224 | 4/1 | 0.375 | 1 | [-0.0075, +0.0597] |
| S3_layered_kg vs C1_country | +0.0224 | 5/2 | 0.4531 | 1 | [-0.0149, +0.0597] |
| S4_abstraction_refinement_kg vs C1_country | +0.0746 | 11/1 | 0.006348 | 0.06983 | [+0.0299, +0.1269] |
| S4_triple_retrieval vs C1_country | +0.0224 | 5/2 | 0.4531 | 1 | [-0.0149, +0.0597] |
| S4_no_shared_pivot vs C1_country | +0.0373 | 7/2 | 0.1797 | 1 | [-0.0075, +0.0821] |
| S1g vs C1_country | +0.0448 | 6/0 | 0.03125 | 0.3125 | [+0.0149, +0.0821] |
| S3g vs C1_country | +0.0149 | 4/2 | 0.6875 | 1 | [-0.0224, +0.0522] |
| S4g vs C1_country | +0.0373 | 7/2 | 0.1797 | 1 | [+0.0000, +0.0821] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0299 | 8/4 | 0.3877 | 1 | [-0.0224, +0.0821] |
| S4g vs S4_abstraction_refinement_kg | -0.0373 | 4/9 | 0.2668 | 1 | [-0.0896, +0.0149] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0373, McNemar p 0.0625

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1269, McNemar p 1.5e-05

## uncovered

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 2181 | 0.5800 | 0.4587 | 0.0000 | 0 | 0 |
| C2_value | 2181 | 0.6185 | 0.5030 | 0.2500 | 0 | 0 |
| C3_rule_of_thumb | 2181 | 0.7341 | 0.6517 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 2181 | 0.5552 | 0.4566 | -0.1607 | 0 | 2181 |
| S2_flat_kg | 2181 | 0.5869 | 0.4842 | 0.0446 | 0 | 2181 |
| S3_layered_kg | 2181 | 0.5731 | 0.4673 | -0.0446 | 0 | 2181 |
| S4_abstraction_refinement_kg | 2181 | 0.5543 | 0.4692 | -0.1667 | 0 | 2181 |
| S4_triple_retrieval | 2181 | 0.5740 | 0.4670 | -0.0387 | 0 | 2181 |
| S4_no_shared_pivot | 2181 | 0.5644 | 0.4655 | -0.1012 | 0 | 2181 |
| S1g | 2181 | 0.5759 | 0.4621 | -0.0268 | 0 | 1115 |
| S3g | 2181 | 0.5777 | 0.4661 | -0.0149 | 0 | 1014 |
| S4g | 2181 | 0.5763 | 0.4787 | -0.0238 | 0 | 1899 |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0248 | 92/146 | 0.000563 | 0.006193 | [-0.0390, -0.0110] |
| S2_flat_kg vs C1_country | +0.0069 | 129/114 | 0.3692 | 1 | [-0.0073, +0.0211] |
| S3_layered_kg vs C1_country | -0.0069 | 108/123 | 0.357 | 1 | [-0.0206, +0.0069] |
| S4_abstraction_refinement_kg vs C1_country | -0.0257 | 101/157 | 0.00059 | 0.006193 | [-0.0403, -0.0115] |
| S4_triple_retrieval vs C1_country | -0.0060 | 112/125 | 0.4358 | 1 | [-0.0197, +0.0078] |
| S4_no_shared_pivot vs C1_country | -0.0156 | 92/126 | 0.02519 | 0.2015 | [-0.0289, -0.0023] |
| S1g vs C1_country | -0.0041 | 41/50 | 0.4018 | 1 | [-0.0128, +0.0046] |
| S3g vs C1_country | -0.0023 | 54/59 | 0.7069 | 1 | [-0.0119, +0.0073] |
| S4g vs C1_country | -0.0037 | 87/95 | 0.604 | 1 | [-0.0160, +0.0083] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | -0.0009 | 146/148 | 0.9535 | 1 | [-0.0160, +0.0142] |
| S4g vs S4_abstraction_refinement_kg | +0.0220 | 142/94 | 0.002151 | 0.01936 | [+0.0083, +0.0358] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0385, McNemar p 0

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1541, McNemar p 0

## descriptors <20

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 794 | 0.6322 | 0.4705 | 0.0000 | 0 | 0 |
| C2_value | 794 | 0.6839 | 0.5261 | 0.3015 | 0 | 0 |
| C3_rule_of_thumb | 794 | 0.8035 | 0.6990 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 794 | 0.5970 | 0.4725 | -0.2059 | 0 | 794 |
| S2_flat_kg | 794 | 0.6259 | 0.4903 | -0.0368 | 0 | 794 |
| S3_layered_kg | 794 | 0.6184 | 0.4821 | -0.0809 | 0 | 794 |
| S4_abstraction_refinement_kg | 794 | 0.5894 | 0.4875 | -0.2500 | 0 | 794 |
| S4_triple_retrieval | 794 | 0.6234 | 0.4904 | -0.0515 | 0 | 794 |
| S4_no_shared_pivot | 794 | 0.6033 | 0.4834 | -0.1691 | 0 | 794 |
| S1g | 794 | 0.6398 | 0.4827 | 0.0441 | 0 | 78 |
| S3g | 794 | 0.6360 | 0.4800 | 0.0221 | 0 | 70 |
| S4g | 794 | 0.6335 | 0.5167 | 0.0074 | 0 | 582 |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0353 | 41/69 | 0.009722 | 0.0875 | [-0.0617, -0.0101] |
| S2_flat_kg vs C1_country | -0.0063 | 49/54 | 0.6937 | 1 | [-0.0315, +0.0189] |
| S3_layered_kg vs C1_country | -0.0139 | 45/56 | 0.3197 | 1 | [-0.0390, +0.0101] |
| S4_abstraction_refinement_kg vs C1_country | -0.0428 | 44/78 | 0.002662 | 0.02662 | [-0.0705, -0.0164] |
| S4_triple_retrieval vs C1_country | -0.0088 | 50/57 | 0.5621 | 1 | [-0.0340, +0.0164] |
| S4_no_shared_pivot vs C1_country | -0.0290 | 39/62 | 0.02809 | 0.2247 | [-0.0542, -0.0050] |
| S1g vs C1_country | +0.0076 | 10/4 | 0.1796 | 1 | [-0.0013, +0.0176] |
| S3g vs C1_country | +0.0038 | 8/5 | 0.5811 | 1 | [-0.0050, +0.0126] |
| S4g vs C1_country | +0.0013 | 41/40 | 1 | 1 | [-0.0214, +0.0239] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | -0.0076 | 57/63 | 0.6483 | 1 | [-0.0353, +0.0202] |
| S4g vs S4_abstraction_refinement_kg | +0.0441 | 75/40 | 0.001413 | 0.01554 | [+0.0176, +0.0705] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0516, McNemar p 4e-06

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1713, McNemar p 0

## descriptors 20-99

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 686 | 0.5816 | 0.4593 | 0.0000 | 0 | 0 |
| C2_value | 686 | 0.6093 | 0.4952 | 0.1743 | 0 | 0 |
| C3_rule_of_thumb | 686 | 0.7405 | 0.6577 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 686 | 0.5700 | 0.4696 | -0.0734 | 0 | 686 |
| S2_flat_kg | 686 | 0.6020 | 0.4984 | 0.1284 | 0 | 686 |
| S3_layered_kg | 686 | 0.5773 | 0.4645 | -0.0275 | 0 | 686 |
| S4_abstraction_refinement_kg | 686 | 0.5700 | 0.4750 | -0.0734 | 0 | 686 |
| S4_triple_retrieval | 686 | 0.5758 | 0.4643 | -0.0367 | 0 | 686 |
| S4_no_shared_pivot | 686 | 0.5729 | 0.4694 | -0.0550 | 0 | 686 |
| S1g | 686 | 0.5685 | 0.4541 | -0.0826 | 0 | 428 |
| S3g | 686 | 0.5758 | 0.4700 | -0.0367 | 0 | 370 |
| S4g | 686 | 0.5743 | 0.4725 | -0.0459 | 0 | 653 |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0117 | 25/33 | 0.3581 | 1 | [-0.0335, +0.0102] |
| S2_flat_kg vs C1_country | +0.0204 | 46/32 | 0.1405 | 1 | [-0.0044, +0.0466] |
| S3_layered_kg vs C1_country | -0.0044 | 35/38 | 0.8151 | 1 | [-0.0292, +0.0204] |
| S4_abstraction_refinement_kg vs C1_country | -0.0117 | 28/36 | 0.3817 | 1 | [-0.0350, +0.0102] |
| S4_triple_retrieval vs C1_country | -0.0058 | 36/40 | 0.731 | 1 | [-0.0306, +0.0190] |
| S4_no_shared_pivot vs C1_country | -0.0087 | 22/28 | 0.4799 | 1 | [-0.0292, +0.0117] |
| S1g vs C1_country | -0.0131 | 12/21 | 0.1628 | 1 | [-0.0292, +0.0029] |
| S3g vs C1_country | -0.0058 | 20/24 | 0.6516 | 1 | [-0.0248, +0.0131] |
| S4g vs C1_country | -0.0073 | 24/29 | 0.5831 | 1 | [-0.0277, +0.0131] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0000 | 42/42 | 1 | 1 | [-0.0262, +0.0262] |
| S4g vs S4_abstraction_refinement_kg | +0.0044 | 31/28 | 0.7948 | 1 | [-0.0175, +0.0262] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0277, McNemar p 0.01266

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1589, McNemar p 0

## descriptors >=100

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 835 | 0.5138 | 0.4343 | 0.0000 | 0 | 0 |
| C2_value | 835 | 0.5485 | 0.4732 | 0.2685 | 0 | 0 |
| C3_rule_of_thumb | 835 | 0.6431 | 0.5916 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 835 | 0.4994 | 0.4328 | -0.1111 | 0 | 835 |
| S2_flat_kg | 835 | 0.5246 | 0.4583 | 0.0833 | 0 | 835 |
| S3_layered_kg | 835 | 0.5162 | 0.4485 | 0.0185 | 0 | 835 |
| S4_abstraction_refinement_kg | 835 | 0.5090 | 0.4564 | -0.0370 | 0 | 835 |
| S4_triple_retrieval | 835 | 0.5150 | 0.4430 | 0.0093 | 0 | 835 |
| S4_no_shared_pivot | 835 | 0.5138 | 0.4467 | 0.0000 | 0 | 835 |
| S1g | 835 | 0.5138 | 0.4440 | 0.0000 | 0 | 732 |
| S3g | 835 | 0.5114 | 0.4391 | -0.0185 | 0 | 699 |
| S4g | 835 | 0.5150 | 0.4505 | 0.0093 | 0 | 784 |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0144 | 33/45 | 0.2127 | 1 | [-0.0347, +0.0060] |
| S2_flat_kg vs C1_country | +0.0108 | 38/29 | 0.3284 | 1 | [-0.0084, +0.0299] |
| S3_layered_kg vs C1_country | +0.0024 | 33/31 | 0.9007 | 1 | [-0.0168, +0.0216] |
| S4_abstraction_refinement_kg vs C1_country | -0.0048 | 40/44 | 0.7436 | 1 | [-0.0263, +0.0168] |
| S4_triple_retrieval vs C1_country | +0.0012 | 31/30 | 1 | 1 | [-0.0180, +0.0192] |
| S4_no_shared_pivot vs C1_country | +0.0000 | 38/38 | 1 | 1 | [-0.0204, +0.0204] |
| S1g vs C1_country | +0.0000 | 25/25 | 1 | 1 | [-0.0168, +0.0168] |
| S3g vs C1_country | -0.0024 | 30/32 | 0.8991 | 1 | [-0.0216, +0.0156] |
| S4g vs C1_country | +0.0012 | 29/28 | 1 | 1 | [-0.0168, +0.0192] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0096 | 55/47 | 0.4884 | 1 | [-0.0144, +0.0335] |
| S4g vs S4_abstraction_refinement_kg | +0.0060 | 40/35 | 0.6445 | 1 | [-0.0144, +0.0263] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0347, McNemar p 0.000264

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1293, McNemar p 0

## Wall time

| stage | retrieval s | generate s | prompts |
|---|---|---|---|
| tau_tuning | 91.4 |  |  |
| C1_country | 0.01 | 25.98 | 2315 |
| C2_value | 0.01 | 28.64 | 2315 |
| C3_rule_of_thumb | 0.01 | 26.81 | 2315 |
| S1_raw_text_rag | 16.11 | 101.75 | 2315 |
| S2_flat_kg | 43.95 | 44.97 | 2315 |
| S3_layered_kg | 51.19 | 74.93 | 2315 |
| S4_abstraction_refinement_kg | 20.74 | 104.17 | 2315 |
| S4_triple_retrieval | 49.48 | 74.87 | 2315 |
| S4_no_shared_pivot | 20.82 | 103.21 | 2315 |
| S1g | 16.24 | 44.65 | 2315 |
| S3g | 48.75 | 42.16 | 2315 |
| S4g | 19.48 | 77.85 | 2315 |
