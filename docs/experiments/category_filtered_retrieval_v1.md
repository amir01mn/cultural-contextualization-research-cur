# Category-filtered pathway retrieval v1

## Question

When the CultureBank corpus becomes heterogeneous, can restricting pathway retrieval to relevant CultureBank topic strata improve retrieval relative to global semantic search?

This is a controlled diagnostic before implementing an LLM category router.

## Fixed components

The experiment must keep constant:

- validated Scale-500 corpus
- pathway construction
- Qwen/Qwen2.5-7B-Instruct
- the same 40 controlled CulturalBench scoring units
- top-k 8
- S4 pathway formatting and reranking
- prompts and answer scoring

Only pathway candidate selection changes.

## Conditions

### S4 global

Search all eligible pathways semantically, then apply the validated culture-aware reranking.

### S4 category-filtered

1. Map configured CultureBank topic labels to source record IDs using `corpus_passages.jsonl`.
2. Keep only pathways supported by at least one source record in the configured topics.
3. Run the same semantic retrieval and culture-aware reranking inside that filtered pathway set.
4. Return the same top-k evidence budget.

The category filter is applied before semantic candidate selection.

## Controlled food/hospitality topic set

For the first diagnostic, use the four CultureBank topics occurring at least five times in the frozen 77-record food/hospitality development corpus:

- Cultural Traditions and Festivals (20/77)
- Cultural Exchange (16/77)
- Food and Dining (15/77)
- Social Norms and Etiquette (13/77)

Together they account for 64/77 records (83.1%) in that controlled development subset.

This topic set is intentionally a development-time oracle/controlled filter. It is not a held-out routing result and must not be presented as an automatically learned category router.

## Success criteria

Primary comparison:

- `S4_abstraction_refinement_kg`
- `S4_category_filtered`

Record:

- CulturalBench accuracy
- exact McNemar discordant units
- mean eligible pathways after the category filter
- generated-support ratio
- pathway completeness mix
- retrieved evidence for gains and losses

If the controlled category filter improves relevance or accuracy, the next step is an automatic question-to-topic router. If it does not, category routing should not be made more complex until the failure mode is understood.
