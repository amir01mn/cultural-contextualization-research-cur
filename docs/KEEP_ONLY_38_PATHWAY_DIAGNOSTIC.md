# Keep-Only Food/Hospitality Subset Diagnostic

## Purpose

This run tests the audited keep-only food/hospitality subset before broader scaling. The goal is to determine whether the controlled subset can produce useful layered triples, resolved entities, and abstraction-refinement pathways.

## Run setup

- Input records: data/controlled/food_hospitality_records_keep_only.jsonl
- Records loaded: 38
- Backend/model: Qwen/Qwen2.5-7B-Instruct with vLLM on Narval
- Batch size: 16
- Resume: disabled
- Alignment verification: skipped

## Main results

- Raw triples: 196
- Refined triples: 196
- Canonical entities: 163
- Cross-cultural alignments: 0
- Pathways: 0

## Layer diagnostic

Subject layers:

- LAYER_2_SOCIAL: 157
- LAYER_1_SURFACE: 39
- LAYER_3_DEEP_VALUE: 0

Object layers:

- LAYER_1_SURFACE: 133
- LAYER_3_DEEP_VALUE: 38
- LAYER_2_SOCIAL: 25

Subject to object layer pairs:

- LAYER_2_SOCIAL -> LAYER_1_SURFACE: 95
- LAYER_1_SURFACE -> LAYER_1_SURFACE: 38
- LAYER_2_SOCIAL -> LAYER_3_DEEP_VALUE: 37
- LAYER_2_SOCIAL -> LAYER_2_SOCIAL: 25
- LAYER_1_SURFACE -> LAYER_3_DEEP_VALUE: 1
- LAYER_3_DEEP_VALUE -> LAYER_2_SOCIAL: 0
- LAYER_3_DEEP_VALUE -> LAYER_1_SURFACE: 0

## Interpretation

The controlled subset successfully produced layered triples, so this is not an extraction failure. However, the pathway builder returned zero pathways because the current graph direction does not match the extracted triple direction.

The current pathway builder expects pathways that begin at an L3 deep-value node and follow outgoing edges downward into L2 and L1 nodes. In this run, L3 nodes appear as objects, not subjects. This means cultural values are being extracted as things that practices or social norms point toward, rather than as starting nodes.

## Main conclusion

The controlled subset exposed a pathway-construction problem. The next technical step should be to revise pathway construction so it is layer-aware rather than strictly dependent on raw edge direction.

Possible fixes:

1. Allow reverse traversal when an edge points from L1 or L2 toward L3.
2. Build record-level pathways by grouping L1, L2, and L3 nodes from the same CultureBank record.
3. Add a canonical pathway direction independent of the extracted triple direction.

## Next step

Before running S0-S4 evaluation, fix pathway construction. Running S4 now would be misleading because the pathway index is empty.
