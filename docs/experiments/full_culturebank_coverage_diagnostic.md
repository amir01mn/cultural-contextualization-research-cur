# Full CultureBank coverage diagnostic

## Motivation

The 77-record and Scale-500 development corpora both produced zero fully
COVERED items in the controlled 40-unit CulturalBench coverage audit. Before
changing retrieval again, test whether the complete CultureBank dataset contains
the benchmark evidence at all.

This diagnostic does not build triples, entities, alignments, or pathways.

## Step 1: deterministic full CultureBank export

Export the cached CultureBank TikTok and Reddit splits using the same
`normalize_record()` function as the corpus pipeline.

The exporter preserves source split provenance in `raw["_split"]`, checks for
duplicate normalized instance IDs, records split/culture/topic counts, and
reports a SHA-256 checksum.

Recommended destination:

`$SCRATCH/cultural_kg_inputs/culturebank_full_normalized.jsonl`

## Step 2: exhaustive coverage audit

The ordinary coverage audit can inspect all records for small corpora, but a
full CultureBank culture may have too many records for a single prompt.

The exhaustive audit therefore:

1. finds every record whose culture matches the benchmark target culture using
   the validated culture matcher;
2. partitions all target-culture records into bounded chunks;
3. audits every chunk independently with the same strict COVERED/PARTIAL/ABSENT
   definitions;
4. validates cited support IDs;
5. aggregates with priority COVERED > PARTIAL > ABSENT.

No target-culture record cap is applied.

COVERED and PARTIAL both require at least one cited target-culture record.
A claimed COVERED/PARTIAL result without a valid supporting record is downgraded
to ABSENT.

## Decision

If many controlled CulturalBench items become COVERED on full CultureBank, the
main issue is corpus selection/coverage. The next construction corpus should be
coverage-aware or stratified before category-indexed retrieval is evaluated.

If most items remain ABSENT even on full CultureBank, the controlled
CulturalBench subset is not well aligned with CultureBank and the evaluation
pairing should be redesigned before further retrieval optimization.

All automatically cited support records should be manually inspected before a
publication claim.
