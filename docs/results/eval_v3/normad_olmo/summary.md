# Eval v3 - NormAd (OLMo-2-7B, second model)

Test stories: 2315 (covered: 134). top_k=8, max_tokens=256, temperature=0.0.
Gated taus (tuned on dev, frozen): {'S1g': 0.15, 'S4R': (0.5, 0.3), 'S4R_noabs': (0.3, 1.01)}

## all

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 2315 | 0.4406 | 0.4009 | 0.0000 | 0 | 0 |
| C2_value | 2315 | 0.4937 | 0.4618 | 0.2978 | 0 | 0 |
| C3_rule_of_thumb | 2315 | 0.6190 | 0.5552 | 1.0000 | 0 | 0 |
| S1g | 2315 | 0.4881 | 0.4437 | 0.2663 | 0 | 1396 |
| S4R | 2315 | 0.5438 | 0.4839 | 0.5787 | 0 | 1960 |
| S4R_noabs | 2315 | 0.4747 | 0.4443 | 0.1913 | 0 | 1766 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.1032 | 466/227 | 0 | 0 | [+0.0816, +0.1253] |
| S4R vs S4R_noabs | +0.0691 | 297/137 | 0 | 0 | [+0.0514, +0.0864] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1g vs C1_country | +0.0475 | 279/169 | 0 | 0 | [+0.0302, +0.0657] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0531, McNemar p 0

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1784, McNemar p 0

## covered

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 134 | 0.5373 | 0.5015 | 0.0000 | 0 | 0 |
| C2_value | 134 | 0.5373 | 0.5075 | 0.0000 | 0 | 0 |
| C3_rule_of_thumb | 134 | 0.6045 | 0.5732 | 1.0000 | 0 | 0 |
| S1g | 134 | 0.5522 | 0.5087 | 0.2222 | 0 | 129 |
| S4R | 134 | 0.5373 | 0.4894 | 0.0000 | 0 | 134 |
| S4R_noabs | 134 | 0.5672 | 0.5416 | 0.4444 | 0 | 134 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0000 | 23/23 | 1 | 1 | [-0.1045, +0.0970] |
| S4R vs S4R_noabs | -0.0299 | 6/10 | 0.4545 | 0.909 | [-0.0896, +0.0299] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1g vs C1_country | +0.0149 | 23/21 | 0.8804 | 0.8804 | [-0.0821, +0.1119] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0000, McNemar p 1

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.0672, McNemar p 0.2221

## uncovered

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 2181 | 0.4347 | 0.3946 | 0.0000 | 0 | 0 |
| C2_value | 2181 | 0.4911 | 0.4587 | 0.3045 | 0 | 0 |
| C3_rule_of_thumb | 2181 | 0.6199 | 0.5537 | 1.0000 | 0 | 0 |
| S1g | 2181 | 0.4842 | 0.4391 | 0.2673 | 0 | 1267 |
| S4R | 2181 | 0.5442 | 0.4837 | 0.5916 | 0 | 1826 |
| S4R_noabs | 2181 | 0.4691 | 0.4379 | 0.1856 | 0 | 1632 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.1096 | 443/204 | 0 | 0 | [+0.0871, +0.1325] |
| S4R vs S4R_noabs | +0.0752 | 291/127 | 0 | 0 | [+0.0569, +0.0931] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1g vs C1_country | +0.0495 | 256/148 | 0 | 0 | [+0.0321, +0.0679] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0564, McNemar p 0

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1852, McNemar p 0

## descriptors <20

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 794 | 0.4320 | 0.3789 | 0.0000 | 0 | 0 |
| C2_value | 794 | 0.4962 | 0.4406 | 0.2525 | 0 | 0 |
| C3_rule_of_thumb | 794 | 0.6864 | 0.5837 | 1.0000 | 0 | 0 |
| S1g | 794 | 0.4622 | 0.4109 | 0.1188 | 0 | 125 |
| S4R | 794 | 0.5264 | 0.4708 | 0.3713 | 0 | 439 |
| S4R_noabs | 794 | 0.4635 | 0.4156 | 0.1238 | 0 | 308 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0945 | 113/38 | 0 | 0 | [+0.0655, +0.1234] |
| S4R vs S4R_noabs | +0.0630 | 82/32 | 3e-06 | 3e-06 | [+0.0365, +0.0894] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1g vs C1_country | +0.0302 | 37/13 | 0.000936 | 0.000936 | [+0.0126, +0.0479] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0642, McNemar p 1e-05

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.2544, McNemar p 0

## descriptors 20-99

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 686 | 0.4388 | 0.4002 | 0.0000 | 0 | 0 |
| C2_value | 686 | 0.4840 | 0.4480 | 0.2743 | 0 | 0 |
| C3_rule_of_thumb | 686 | 0.6035 | 0.5354 | 1.0000 | 0 | 0 |
| S1g | 686 | 0.5029 | 0.4498 | 0.3894 | 0 | 478 |
| S4R | 686 | 0.5685 | 0.4750 | 0.7876 | 0 | 686 |
| S4R_noabs | 686 | 0.4913 | 0.4488 | 0.3186 | 0 | 642 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.1297 | 163/74 | 0 | 0 | [+0.0875, +0.1720] |
| S4R vs S4R_noabs | +0.0773 | 103/50 | 2.2e-05 | 2.2e-05 | [+0.0423, +0.1122] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1g vs C1_country | +0.0641 | 87/43 | 0.000142 | 0.000142 | [+0.0321, +0.0962] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0452, McNemar p 0.005727

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1647, McNemar p 0

## descriptors >=100

| condition | n | accuracy | macro-F1 | gap closed | unparsed | evidence non-empty |
|---|---|---|---|---|---|---|
| C1_country | 835 | 0.4503 | 0.4175 | 0.0000 | 0 | 0 |
| C2_value | 835 | 0.4994 | 0.4870 | 0.4184 | 0 | 0 |
| C3_rule_of_thumb | 835 | 0.5677 | 0.5362 | 1.0000 | 0 | 0 |
| S1g | 835 | 0.5006 | 0.4533 | 0.4286 | 0 | 793 |
| S4R | 835 | 0.5401 | 0.4921 | 0.7653 | 0 | 835 |
| S4R_noabs | 835 | 0.4719 | 0.4560 | 0.1837 | 0 | 816 |

KG v3 primary family (pre-registered, Holm over these only):

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S4R vs C1_country | +0.0898 | 190/115 | 2.1e-05 | 2.4e-05 | [+0.0491, +0.1305] |
| S4R vs S4R_noabs | +0.0683 | 112/55 | 1.2e-05 | 2.4e-05 | [+0.0383, +0.0982] |

| comparison | delta | a-only/b-only | McNemar p | Holm p | bootstrap 95% CI |
|---|---|---|---|---|---|
| S1g vs C1_country | +0.0503 | 155/113 | 0.01212 | 0.01212 | [+0.0120, +0.0886] |

Reference (not in Holm family): C2_value vs C1_country: delta +0.0491, McNemar p 0.000983

Reference (not in Holm family): C3_rule_of_thumb vs C1_country: delta +0.1174, McNemar p 0

## Wall time

| stage | retrieval s | generate s | prompts |
|---|---|---|---|
| tau_tuning | 444.14 |  |  |
| C1_country | 0.01 | 40.32 | 2315 |
| C2_value | 0.01 | 44.4 | 2315 |
| C3_rule_of_thumb | 0.01 | 44.59 | 2315 |
| S1g | 16.4 | 101.44 | 2315 |
| S4R | 24.49 | 81.54 | 2315 |
| S4R_noabs | 15.21 | 314.79 | 2315 |
