# CulturalBench re-scored by CultureBank coverage (secondary; Eval v2)

Source: August per-item results `data/outputs/culturalbench_results.json` (Qwen2.5-7B-Instruct, top-k 8). No new GPU run. Strata = CultureBank descriptors per canonical country (`corpus_passages.jsonl`): <20, 20-99, >=100. Hard = official group metric (all True/False items correct). McNemar vs S0 is per stratum, unadjusted (secondary analysis).

## Strata

| Stratum | Countries | Easy units | Hard units |
|---|---|---|---|
| <20 | 6 | 140 | 140 |
| 20-99 | 17 | 415 | 415 |
| >=100 | 22 | 672 | 672 |

## Easy accuracy by stratum (Δ vs S0, McNemar p)

| System | <20 | 20-99 | >=100 | all |
|---|---|---|---|---|
| S0_no_kg | 80.0% | 69.2% | 71.4% | 71.6% |
| S1_raw_text_rag | 67.1% (-12.9%, p=0.000534) | 64.1% (-5.1%, p=0.0238) | 69.3% (-2.1%, p=0.206) | 67.3% (-4.3%, p=0.000322) |
| S2_flat_kg | 72.1% (-7.9%, p=0.0266) | 65.3% (-3.9%, p=0.0929) | 67.0% (-4.5%, p=0.00462) | 67.0% (-4.7%, p=9e-05) |
| S3_layered_kg | 70.7% (-9.3%, p=0.00443) | 65.3% (-3.9%, p=0.0681) | 67.9% (-3.6%, p=0.021) | 67.3% (-4.3%, p=0.00013) |
| S4_abstraction_refinement_kg | 67.9% (-12.1%, p=0.000911) | 61.7% (-7.5%, p=0.000194) | 68.3% (-3.1%, p=0.0375) | 66.0% (-5.6%, p=0) |
| S4_no_shared_pivot | 68.6% (-11.4%, p=0.000855) | 61.5% (-7.7%, p=0.000208) | 68.9% (-2.5%, p=0.1) | 66.3% (-5.3%, p=3e-06) |
| S4_triple_retrieval | 68.6% (-11.4%, p=0.000402) | 65.3% (-3.9%, p=0.0681) | 68.0% (-3.4%, p=0.0235) | 67.2% (-4.5%, p=5.8e-05) |

## Hard accuracy by stratum (Δ vs S0, McNemar p)

| System | <20 | 20-99 | >=100 | all |
|---|---|---|---|---|
| S0_no_kg | 30.7% | 28.4% | 35.9% | 32.8% |
| S1_raw_text_rag | 5.7% (-25.0%, p=0) | 10.4% (-18.1%, p=0) | 20.7% (-15.2%, p=0) | 15.5% (-17.3%, p=0) |
| S2_flat_kg | 12.1% (-18.6%, p=6e-06) | 13.7% (-14.7%, p=0) | 20.5% (-15.3%, p=0) | 17.3% (-15.5%, p=0) |
| S3_layered_kg | 10.0% (-20.7%, p=5e-06) | 13.7% (-14.7%, p=0) | 21.7% (-14.1%, p=0) | 17.7% (-15.1%, p=0) |
| S4_abstraction_refinement_kg | 5.7% (-25.0%, p=0) | 8.7% (-19.8%, p=0) | 13.2% (-22.6%, p=0) | 10.8% (-21.9%, p=0) |
| S4_no_shared_pivot | 5.0% (-25.7%, p=0) | 8.7% (-19.8%, p=0) | 12.0% (-23.8%, p=0) | 10.1% (-22.7%, p=0) |
| S4_triple_retrieval | 8.6% (-22.1%, p=0) | 14.0% (-14.5%, p=0) | 21.6% (-14.3%, p=0) | 17.5% (-15.2%, p=0) |

## Combined accuracy by stratum (Δ vs S0, McNemar p)

| System | <20 | 20-99 | >=100 | all |
|---|---|---|---|---|
| S0_no_kg | 55.4% | 48.8% | 53.6% | 52.2% |
| S1_raw_text_rag | 36.4% (-18.9%, p=0) | 37.2% (-11.6%, p=0) | 45.0% (-8.6%, p=0) | 41.4% (-10.8%, p=0) |
| S2_flat_kg | 42.1% (-13.2%, p=0) | 39.5% (-9.3%, p=0) | 43.8% (-9.9%, p=0) | 42.1% (-10.1%, p=0) |
| S3_layered_kg | 40.4% (-15.0%, p=0) | 39.5% (-9.3%, p=0) | 44.8% (-8.8%, p=0) | 42.5% (-9.7%, p=0) |
| S4_abstraction_refinement_kg | 36.8% (-18.6%, p=0) | 35.2% (-13.6%, p=0) | 40.8% (-12.9%, p=0) | 38.4% (-13.8%, p=0) |
| S4_no_shared_pivot | 36.8% (-18.6%, p=0) | 35.1% (-13.7%, p=0) | 40.5% (-13.2%, p=0) | 38.2% (-14.0%, p=0) |
| S4_triple_retrieval | 38.6% (-16.8%, p=0) | 39.6% (-9.2%, p=0) | 44.8% (-8.8%, p=0) | 42.3% (-9.9%, p=0) |

## Country → stratum

| Country | Canonical | CultureBank descriptors | Stratum |
|---|---|---|---|
| United States | united states | 6170 | >=100 |
| Germany | germany | 1017 | >=100 |
| United Kingdom | united kingdom | 1008 | >=100 |
| Australia | australia | 612 | >=100 |
| Italy | italy | 601 | >=100 |
| France | france | 589 | >=100 |
| South Korea | korea | 481 | >=100 |
| Canada | canada | 448 | >=100 |
| Netherlands | netherlands | 438 | >=100 |
| Japan | japan | 415 | >=100 |
| Spain | spain | 400 | >=100 |
| Philippines | philippines | 320 | >=100 |
| China | china | 308 | >=100 |
| Indonesia | indonesia | 249 | >=100 |
| Russia | russia | 219 | >=100 |
| India | india | 207 | >=100 |
| Mexico | mexico | 194 | >=100 |
| Brazil | brazil | 186 | >=100 |
| New Zealand | new zealand | 150 | >=100 |
| Malaysia | malaysia | 132 | >=100 |
| Poland | poland | 124 | >=100 |
| Turkey | turkey | 101 | >=100 |
| Singapore | singapore | 93 | 20-99 |
| Argentina | argentina | 80 | 20-99 |
| South Africa | south africa | 79 | 20-99 |
| Vietnam | vietnam | 69 | 20-99 |
| Nigeria | nigeria | 67 | 20-99 |
| Thailand | thailand | 64 | 20-99 |
| Egypt | egypt | 63 | 20-99 |
| Ukraine | ukraine | 58 | 20-99 |
| Romania | romania | 58 | 20-99 |
| Czech Republic | czech republic | 53 | 20-99 |
| Israel | israel | 44 | 20-99 |
| Pakistan | pakistan | 27 | 20-99 |
| Saudi Arabia | saudi arabia | 26 | 20-99 |
| Taiwan | taiwan | 25 | 20-99 |
| Chile | chile | 24 | 20-99 |
| Iran | iran | 24 | 20-99 |
| Morocco | morocco | 21 | 20-99 |
| Lebanon | lebanon | 16 | <20 |
| Peru | peru | 15 | <20 |
| Hong Kong | hong kong | 9 | <20 |
| Nepal | nepal | 8 | <20 |
| Zimbabwe | zimbabwe | 5 | <20 |
| Bangladesh | bangladesh | 4 | <20 |
