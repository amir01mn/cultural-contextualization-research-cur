# Food/Hospitality Keep-Only Flexible Pathway Run

Branch: fix-flexible-pathways  
Environment: Narval  
Model: Qwen/Qwen2.5-7B-Instruct  
Input records: data/controlled/food_hospitality_records_keep_only.jsonl  
Corpus limit: 100  
Actual records loaded: 38  
Alignment: skipped  
Alignment verification: skipped  

## Result

Bad placeholder triples: 0

Pathway completeness:
- FULL: 22
- SOCIAL: 64
- SURFACE: 48

## Interpretation

The previous issue where the pathway builder produced 0 FULL pathways is fixed. The corrected pipeline now produces real full pathways with:

- L3 deep values, such as gratitude and hospitality
- L2 social actors or relations, such as hosts and guests or friends and family
- L1 surface practices, such as offering drinks/food or birthday meals/gifts

Example full pathways:
- gratitude -> friends and family -> engage in social activities involving food, alcohol, and conversation
- hospitality -> hosts and guests -> offer, provide, and consume drinks and food, with guests often bringing gifts or their own drinks

## Remaining limitations

Some L3 labels are still too action-like, such as long phrases about tipping or eating practices. However, the main blocker is fixed: topic/category metadata and layer placeholder strings are no longer becoming fake L3 nodes.
