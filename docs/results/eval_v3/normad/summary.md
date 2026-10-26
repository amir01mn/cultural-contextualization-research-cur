# Eval v3 - NormAd (Qwen2.5-7B)

Test stories: 2315 (covered: 134). top_k=8, max_tokens=256, temperature=0.0.
Gated taus (tuned on dev, frozen): {'S1g': 0.35, 'S3g': 0.35, 'S4R': (0.6, 0.4), 'S4R_noabs': (0.6, 1.01)}

## all

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 2315 | 0.5719 | 0.4536 | 0.0000 | 0 | 0 |
| C2_value | 2315 | 0.6151 | 0.5014 | 0.2740 | 0 | 0 |
| C3_rule_of_thumb | 2315 | 0.7296 | 0.6526 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 2315 | 0.5529 | 0.4568 | -0.1205 | 0 | 2315 |
| S2_flat_kg | 2315 | 0.5685 | 0.4639 | -0.0219 | 0 | 2315 |
| S3_layered_kg | 2315 | 0.5641 | 0.4578 | -0.0493 | 0 | 2315 |
| S4_abstraction_refinement_kg | 2315 | 0.5611 | 0.4810 | -0.0685 | 0 | 2315 |
| S4_triple_retrieval | 2315 | 0.5741 | 0.4661 | 0.0137 | 0 | 2315 |
| S1g | 2315 | 0.5706 | 0.4596 | -0.0082 | 0 | 1238 |
| S3g | 2315 | 0.5732 | 0.4632 | 0.0082 | 0 | 1179 |
| S4L | 2315 | 0.5728 | 0.4762 | 0.0055 | 0 | 2315 |
| S4F | 2315 | 0.5508 | 0.4578 | -0.1342 | 0 | 2315 |
| S4R | 2315 | 0.5775 | 0.4668 | 0.0356 | 0 | 1626 |
| S4R_noabs | 2315 | 0.5749 | 0.4562 | 0.0192 | 0 | 6 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0056 | 75/62 | 0.3052 | 0.9157 | [-0.0043, +0.0156] |
| S4R vs S1_raw_text_rag | +0.0246 | 161/104 | 0.000557 | 0.003899 | [+0.0112, +0.0384] |
| S4R vs S3g | +0.0043 | 95/85 | 0.5024 | 1 | [-0.0073, +0.0156] |
| S4R vs S4_abstraction_refinement_kg | +0.0164 | 161/123 | 0.02795 | 0.1677 | [+0.0022, +0.0307] |
| S4R vs S4R_noabs | +0.0026 | 70/64 | 0.666 | 1 | [-0.0073, +0.0125] |
| S4F vs S4_abstraction_refinement_kg | -0.0104 | 148/172 | 0.1985 | 0.7938 | [-0.0255, +0.0048] |
| S4L vs S4_abstraction_refinement_kg | +0.0117 | 150/123 | 0.1154 | 0.5771 | [-0.0022, +0.0259] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0190 | 98/142 | 0.0054 | 0.0432 | [-0.0324, -0.0056] |
| S2_flat_kg vs C1_country | -0.0035 | 117/125 | 0.6528 | 1 | [-0.0168, +0.0099] |
| S3_layered_kg vs C1_country | -0.0078 | 112/130 | 0.2745 | 1 | [-0.0212, +0.0052] |
| S4_abstraction_refinement_kg vs C1_country | -0.0108 | 124/149 | 0.1462 | 1 | [-0.0246, +0.0030] |
| S4_triple_retrieval vs C1_country | +0.0022 | 110/105 | 0.7851 | 1 | [-0.0104, +0.0143] |
| S1g vs C1_country | -0.0013 | 46/49 | 0.8376 | 1 | [-0.0095, +0.0073] |
| S3g vs C1_country | +0.0013 | 56/53 | 0.8482 | 1 | [-0.0078, +0.0104] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0082 | 160/141 | 0.2995 | 1 | [-0.0065, +0.0229] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0432, McNemar p 0

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1577, McNemar p 0

## covered

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 134 | 0.4851 | 0.4111 | 0.0000 | 0 | 0 |
| C2_value | 134 | 0.5299 | 0.4557 | 0.2727 | 0 | 0 |
| C3_rule_of_thumb | 134 | 0.6493 | 0.6220 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 134 | 0.5448 | 0.4878 | 0.3636 | 0 | 134 |
| S2_flat_kg | 134 | 0.5075 | 0.4371 | 0.1364 | 0 | 134 |
| S3_layered_kg | 134 | 0.5224 | 0.4655 | 0.2273 | 0 | 134 |
| S4_abstraction_refinement_kg | 134 | 0.5746 | 0.5388 | 0.5455 | 0 | 134 |
| S4_triple_retrieval | 134 | 0.5075 | 0.4466 | 0.1364 | 0 | 134 |
| S1g | 134 | 0.5448 | 0.4920 | 0.3636 | 0 | 123 |
| S3g | 134 | 0.5149 | 0.4479 | 0.1818 | 0 | 128 |
| S4L | 134 | 0.5299 | 0.4674 | 0.2727 | 0 | 134 |
| S4F | 134 | 0.5000 | 0.4383 | 0.0909 | 0 | 134 |
| S4R | 134 | 0.5299 | 0.4676 | 0.2727 | 0 | 112 |
| S4R_noabs | 134 | 0.5000 | 0.4300 | 0.0909 | 0 | 3 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0448 | 6/0 | 0.03125 | 0.1875 | [+0.0149, +0.0821] |
| S4R vs S1_raw_text_rag | -0.0149 | 4/6 | 0.7539 | 1 | [-0.0597, +0.0299] |
| S4R vs S3g | +0.0149 | 6/4 | 0.7539 | 1 | [-0.0299, +0.0597] |
| S4R vs S4_abstraction_refinement_kg | -0.0448 | 5/11 | 0.2101 | 0.8405 | [-0.1045, +0.0149] |
| S4R vs S4R_noabs | +0.0299 | 5/1 | 0.2188 | 0.8405 | [+0.0000, +0.0672] |
| S4F vs S4_abstraction_refinement_kg | -0.0746 | 3/13 | 0.02127 | 0.1489 | [-0.1343, -0.0224] |
| S4L vs S4_abstraction_refinement_kg | -0.0448 | 3/9 | 0.146 | 0.73 | [-0.0970, +0.0075] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | +0.0597 | 9/1 | 0.02148 | 0.1289 | [+0.0149, +0.1045] |
| S2_flat_kg vs C1_country | +0.0224 | 4/1 | 0.375 | 1 | [-0.0075, +0.0597] |
| S3_layered_kg vs C1_country | +0.0373 | 7/2 | 0.1797 | 0.8984 | [-0.0075, +0.0821] |
| S4_abstraction_refinement_kg vs C1_country | +0.0896 | 13/1 | 0.001831 | 0.01465 | [+0.0373, +0.1418] |
| S4_triple_retrieval vs C1_country | +0.0224 | 5/2 | 0.4531 | 1 | [-0.0149, +0.0597] |
| S1g vs C1_country | +0.0597 | 8/0 | 0.007812 | 0.05468 | [+0.0224, +0.1045] |
| S3g vs C1_country | +0.0299 | 6/2 | 0.2891 | 1 | [-0.0075, +0.0746] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0299 | 11/7 | 0.4807 | 1 | [-0.0299, +0.0896] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0448, McNemar p 0.03125

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1642, McNemar p 0

## uncovered

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 2181 | 0.5773 | 0.4563 | 0.0000 | 0 | 0 |
| C2_value | 2181 | 0.6204 | 0.5045 | 0.2741 | 0 | 0 |
| C3_rule_of_thumb | 2181 | 0.7345 | 0.6536 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 2181 | 0.5534 | 0.4546 | -0.1516 | 0 | 2181 |
| S2_flat_kg | 2181 | 0.5722 | 0.4657 | -0.0321 | 0 | 2181 |
| S3_layered_kg | 2181 | 0.5667 | 0.4566 | -0.0671 | 0 | 2181 |
| S4_abstraction_refinement_kg | 2181 | 0.5603 | 0.4765 | -0.1079 | 0 | 2181 |
| S4_triple_retrieval | 2181 | 0.5782 | 0.4668 | 0.0058 | 0 | 2181 |
| S1g | 2181 | 0.5722 | 0.4564 | -0.0321 | 0 | 1115 |
| S3g | 2181 | 0.5768 | 0.4639 | -0.0029 | 0 | 1051 |
| S4L | 2181 | 0.5754 | 0.4768 | -0.0117 | 0 | 2181 |
| S4F | 2181 | 0.5539 | 0.4592 | -0.1487 | 0 | 2181 |
| S4R | 2181 | 0.5805 | 0.4662 | 0.0204 | 0 | 1514 |
| S4R_noabs | 2181 | 0.5796 | 0.4576 | 0.0146 | 0 | 3 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0032 | 69/62 | 0.6003 | 1 | [-0.0069, +0.0133] |
| S4R vs S1_raw_text_rag | +0.0271 | 157/98 | 0.000265 | 0.001855 | [+0.0128, +0.0413] |
| S4R vs S3g | +0.0037 | 89/81 | 0.5915 | 1 | [-0.0083, +0.0156] |
| S4R vs S4_abstraction_refinement_kg | +0.0202 | 156/112 | 0.008501 | 0.05101 | [+0.0055, +0.0348] |
| S4R vs S4R_noabs | +0.0009 | 65/63 | 0.9296 | 1 | [-0.0092, +0.0110] |
| S4F vs S4_abstraction_refinement_kg | -0.0064 | 145/159 | 0.456 | 1 | [-0.0220, +0.0092] |
| S4L vs S4_abstraction_refinement_kg | +0.0151 | 147/114 | 0.04741 | 0.2371 | [+0.0009, +0.0293] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0238 | 89/141 | 0.000736 | 0.005888 | [-0.0376, -0.0101] |
| S2_flat_kg vs C1_country | -0.0050 | 113/124 | 0.5161 | 1 | [-0.0188, +0.0087] |
| S3_layered_kg vs C1_country | -0.0105 | 105/128 | 0.1494 | 0.8961 | [-0.0248, +0.0032] |
| S4_abstraction_refinement_kg vs C1_country | -0.0170 | 111/148 | 0.0251 | 0.1757 | [-0.0316, -0.0028] |
| S4_triple_retrieval vs C1_country | +0.0009 | 105/103 | 0.9447 | 1 | [-0.0119, +0.0133] |
| S1g vs C1_country | -0.0050 | 38/49 | 0.2836 | 1 | [-0.0133, +0.0032] |
| S3g vs C1_country | -0.0005 | 50/51 | 1 | 1 | [-0.0092, +0.0087] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0069 | 149/134 | 0.4053 | 1 | [-0.0083, +0.0220] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0431, McNemar p 0

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1573, McNemar p 0

## descriptors <20

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 794 | 0.6310 | 0.4692 | 0.0000 | 0 | 0 |
| C2_value | 794 | 0.6864 | 0.5310 | 0.3212 | 0 | 0 |
| C3_rule_of_thumb | 794 | 0.8035 | 0.6989 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 794 | 0.6008 | 0.4793 | -0.1752 | 0 | 794 |
| S2_flat_kg | 794 | 0.6121 | 0.4743 | -0.1095 | 0 | 794 |
| S3_layered_kg | 794 | 0.6045 | 0.4616 | -0.1533 | 0 | 794 |
| S4_abstraction_refinement_kg | 794 | 0.6020 | 0.5018 | -0.1679 | 0 | 794 |
| S4_triple_retrieval | 794 | 0.6159 | 0.4702 | -0.0876 | 0 | 794 |
| S1g | 794 | 0.6360 | 0.4795 | 0.0292 | 0 | 78 |
| S3g | 794 | 0.6348 | 0.4781 | 0.0219 | 0 | 82 |
| S4L | 794 | 0.6146 | 0.4854 | -0.0949 | 0 | 794 |
| S4F | 794 | 0.6008 | 0.4754 | -0.1752 | 0 | 794 |
| S4R | 794 | 0.6348 | 0.4862 | 0.0219 | 0 | 366 |
| S4R_noabs | 794 | 0.6322 | 0.4703 | 0.0073 | 0 | 0 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0038 | 19/16 | 0.7359 | 1 | [-0.0101, +0.0189] |
| S4R vs S1_raw_text_rag | +0.0340 | 71/44 | 0.01496 | 0.1048 | [+0.0076, +0.0617] |
| S4R vs S3g | +0.0000 | 22/22 | 1 | 1 | [-0.0164, +0.0164] |
| S4R vs S4_abstraction_refinement_kg | +0.0327 | 71/45 | 0.01988 | 0.1193 | [+0.0063, +0.0592] |
| S4R vs S4R_noabs | +0.0025 | 19/17 | 0.8679 | 1 | [-0.0113, +0.0176] |
| S4F vs S4_abstraction_refinement_kg | -0.0013 | 61/62 | 1 | 1 | [-0.0290, +0.0264] |
| S4L vs S4_abstraction_refinement_kg | +0.0126 | 61/51 | 0.3952 | 1 | [-0.0139, +0.0390] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0302 | 42/66 | 0.02642 | 0.2114 | [-0.0567, -0.0050] |
| S2_flat_kg vs C1_country | -0.0189 | 42/57 | 0.1591 | 0.7954 | [-0.0441, +0.0050] |
| S3_layered_kg vs C1_country | -0.0264 | 37/58 | 0.03961 | 0.2773 | [-0.0504, -0.0025] |
| S4_abstraction_refinement_kg vs C1_country | -0.0290 | 47/70 | 0.0415 | 0.2773 | [-0.0554, -0.0025] |
| S4_triple_retrieval vs C1_country | -0.0151 | 36/48 | 0.2299 | 0.9195 | [-0.0378, +0.0076] |
| S1g vs C1_country | +0.0050 | 7/3 | 0.3438 | 1 | [-0.0025, +0.0139] |
| S3g vs C1_country | +0.0038 | 9/6 | 0.6072 | 1 | [-0.0050, +0.0126] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0013 | 57/56 | 1 | 1 | [-0.0252, +0.0277] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0554, McNemar p 1e-06

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1725, McNemar p 0

## descriptors 20-99

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 686 | 0.5743 | 0.4506 | 0.0000 | 0 | 0 |
| C2_value | 686 | 0.6137 | 0.5054 | 0.2348 | 0 | 0 |
| C3_rule_of_thumb | 686 | 0.7420 | 0.6638 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 686 | 0.5598 | 0.4550 | -0.0870 | 0 | 686 |
| S2_flat_kg | 686 | 0.5816 | 0.4677 | 0.0435 | 0 | 686 |
| S3_layered_kg | 686 | 0.5743 | 0.4600 | 0.0000 | 0 | 686 |
| S4_abstraction_refinement_kg | 686 | 0.5729 | 0.4820 | -0.0087 | 0 | 686 |
| S4_triple_retrieval | 686 | 0.5962 | 0.4859 | 0.1304 | 0 | 686 |
| S1g | 686 | 0.5671 | 0.4508 | -0.0435 | 0 | 428 |
| S3g | 686 | 0.5773 | 0.4687 | 0.0174 | 0 | 381 |
| S4L | 686 | 0.5758 | 0.4712 | 0.0087 | 0 | 686 |
| S4F | 686 | 0.5554 | 0.4634 | -0.1130 | 0 | 686 |
| S4R | 686 | 0.5860 | 0.4773 | 0.0696 | 0 | 561 |
| S4R_noabs | 686 | 0.5802 | 0.4580 | 0.0348 | 0 | 1 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0117 | 30/22 | 0.3317 | 1 | [-0.0087, +0.0321] |
| S4R vs S1_raw_text_rag | +0.0262 | 44/26 | 0.04139 | 0.2897 | [+0.0029, +0.0510] |
| S4R vs S3g | +0.0087 | 37/31 | 0.5446 | 1 | [-0.0146, +0.0321] |
| S4R vs S4_abstraction_refinement_kg | +0.0131 | 46/37 | 0.38 | 1 | [-0.0117, +0.0394] |
| S4R vs S4R_noabs | +0.0058 | 26/22 | 0.6655 | 1 | [-0.0146, +0.0248] |
| S4F vs S4_abstraction_refinement_kg | -0.0175 | 41/53 | 0.2564 | 1 | [-0.0452, +0.0102] |
| S4L vs S4_abstraction_refinement_kg | +0.0029 | 36/34 | 0.905 | 1 | [-0.0204, +0.0262] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0146 | 22/32 | 0.2203 | 1 | [-0.0364, +0.0058] |
| S2_flat_kg vs C1_country | +0.0073 | 38/33 | 0.6353 | 1 | [-0.0161, +0.0321] |
| S3_layered_kg vs C1_country | +0.0000 | 39/39 | 1 | 1 | [-0.0248, +0.0248] |
| S4_abstraction_refinement_kg vs C1_country | -0.0015 | 35/36 | 1 | 1 | [-0.0262, +0.0233] |
| S4_triple_retrieval vs C1_country | +0.0219 | 43/28 | 0.09592 | 0.7674 | [-0.0015, +0.0452] |
| S1g vs C1_country | -0.0073 | 13/18 | 0.4731 | 1 | [-0.0233, +0.0087] |
| S3g vs C1_country | +0.0029 | 23/21 | 0.8804 | 1 | [-0.0160, +0.0219] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0131 | 49/40 | 0.3966 | 1 | [-0.0131, +0.0394] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0394, McNemar p 0.000355

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1676, McNemar p 0

## descriptors >=100

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 835 | 0.5138 | 0.4356 | 0.0000 | 0 | 0 |
| C2_value | 835 | 0.5485 | 0.4684 | 0.2566 | 0 | 0 |
| C3_rule_of_thumb | 835 | 0.6491 | 0.6010 | 1.0000 | 0 | 0 |
| S1_raw_text_rag | 835 | 0.5018 | 0.4368 | -0.0885 | 0 | 835 |
| S2_flat_kg | 835 | 0.5162 | 0.4478 | 0.0177 | 0 | 835 |
| S3_layered_kg | 835 | 0.5174 | 0.4475 | 0.0265 | 0 | 835 |
| S4_abstraction_refinement_kg | 835 | 0.5126 | 0.4607 | -0.0088 | 0 | 835 |
| S4_triple_retrieval | 835 | 0.5162 | 0.4425 | 0.0177 | 0 | 835 |
| S1g | 835 | 0.5114 | 0.4411 | -0.0177 | 0 | 732 |
| S3g | 835 | 0.5114 | 0.4390 | -0.0177 | 0 | 716 |
| S4L | 835 | 0.5305 | 0.4673 | 0.1239 | 0 | 835 |
| S4F | 835 | 0.4994 | 0.4362 | -0.1062 | 0 | 835 |
| S4R | 835 | 0.5162 | 0.4382 | 0.0177 | 0 | 699 |
| S4R_noabs | 835 | 0.5162 | 0.4365 | 0.0177 | 0 | 5 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0024 | 26/24 | 0.8877 | 1 | [-0.0144, +0.0192] |
| S4R vs S1_raw_text_rag | +0.0144 | 46/34 | 0.2185 | 1 | [-0.0060, +0.0359] |
| S4R vs S3g | +0.0048 | 36/32 | 0.7163 | 1 | [-0.0144, +0.0240] |
| S4R vs S4_abstraction_refinement_kg | +0.0036 | 44/41 | 0.8284 | 1 | [-0.0180, +0.0251] |
| S4R vs S4R_noabs | +0.0000 | 25/25 | 1 | 1 | [-0.0156, +0.0168] |
| S4F vs S4_abstraction_refinement_kg | -0.0132 | 46/57 | 0.3245 | 1 | [-0.0371, +0.0108] |
| S4L vs S4_abstraction_refinement_kg | +0.0180 | 53/38 | 0.1418 | 0.9926 | [-0.0036, +0.0407] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1_raw_text_rag vs C1_country | -0.0120 | 34/44 | 0.3082 | 1 | [-0.0335, +0.0084] |
| S2_flat_kg vs C1_country | +0.0024 | 37/35 | 0.9063 | 1 | [-0.0180, +0.0228] |
| S3_layered_kg vs C1_country | +0.0036 | 36/33 | 0.8099 | 1 | [-0.0168, +0.0228] |
| S4_abstraction_refinement_kg vs C1_country | -0.0012 | 42/43 | 1 | 1 | [-0.0228, +0.0204] |
| S4_triple_retrieval vs C1_country | +0.0024 | 31/29 | 0.8974 | 1 | [-0.0156, +0.0204] |
| S1g vs C1_country | -0.0024 | 26/28 | 0.8919 | 1 | [-0.0192, +0.0144] |
| S3g vs C1_country | -0.0024 | 24/26 | 0.8877 | 1 | [-0.0192, +0.0144] |
| S4_abstraction_refinement_kg vs S1_raw_text_rag | +0.0108 | 54/45 | 0.4215 | 1 | [-0.0120, +0.0347] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0347, McNemar p 0.000422

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1353, McNemar p 0

## Wall time

| stage | retrieval s | generate s | prompts |
|---|---|---|---|
| tau_tuning | 517.35 |  |  |
| C1_country | 0.01 | 57.01 | 2315 |
| C2_value | 0.01 | 63.43 | 2315 |
| C3_rule_of_thumb | 0.0 | 59.69 | 2315 |
| S1_raw_text_rag | 16.85 | 240.06 | 2315 |
| S2_flat_kg | 42.06 | 102.13 | 2315 |
| S3_layered_kg | 41.91 | 172.46 | 2315 |
| S4_abstraction_refinement_kg | 21.5 | 248.95 | 2315 |
| S4_triple_retrieval | 46.72 | 173.97 | 2315 |
| S1g | 16.12 | 104.85 | 2315 |
| S3g | 41.09 | 98.39 | 2315 |
| S4L | 19.92 | 440.27 | 2315 |
| S4F | 16.64 | 399.3 | 2315 |
| S4R | 26.86 | 97.75 | 2315 |
| S4R_noabs | 15.42 | 57.44 | 2315 |
