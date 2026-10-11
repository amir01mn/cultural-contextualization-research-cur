# Evaluation 3: intrinsic KG quality (judge: phi-4)

- judge_model: microsoft/phi-4
- provider: vllm
- data_dir: /lustre06/project/6113454/amir01mn/cultural-contextualization-research-cur/cultural-contextualization-research-cur/data/eval_v2/intrinsic
- limit: None
- smoke: False
- stage_seconds: {'facts': 17.7, 'retention': 43.2, 'triple_audit': 21.6, 'pathway_audit': 12.8, 'open_judge': 132.0}

## Atomic facts

300 records, 1441 facts (mean 4.803/record); parse failures 0, records with <3 facts 5.

## Fact retention (fact inferable from the source)

| Source | Fact-level rate | Record macro | Empty-evidence facts |
|---|---|---|---|
| raw_triples | 92.8% (n=1441, unparsed=0) | 93.4% | 0 |
| refined_triples | 92.7% (n=1441, unparsed=0) | 93.2% | 0 |
| pathways | 28.7% (n=1441, unparsed=0) | 29.3% | 748 |

Empty evidence (e.g. no pathway contains any of the record's triples) counts as not retained. Per-culture rates are in summary.json.

## Triple audit (500 refined triples)

| Criterion | All | Non-generated | Generated |
|---|---|---|---|
| faithful | 71.8% (n=500, unparsed=0) | 74.8% (n=341, unparsed=0) | 65.4% (n=159, unparsed=0) |
| layers_correct | 88.8% (n=500, unparsed=0) | 91.2% (n=341, unparsed=0) | 83.7% (n=159, unparsed=0) |
| culture_specific | 79.6% (n=500, unparsed=0) | 80.9% (n=341, unparsed=0) | 76.7% (n=159, unparsed=0) |
| stereotyping | 26.6% (n=500, unparsed=0) | 23.5% (n=341, unparsed=0) | 33.3% (n=159, unparsed=0) |

## Pathway audit (200 FULL pathways)

| Criterion | Rate |
|---|---|
| coherent | 22.5% (n=200, unparsed=0) |
| culture_consistent | 19.0% (n=200, unparsed=0) |
| correct | 17.0% (n=200, unparsed=0) |

## Open-ended judge (CultureBank Format B)

| System | Mean entailment | n |
|---|---|---|
| S0 | 78.8% | 500 |
| S1 | 89.4% | 500 |
| S3 | 86.2% | 500 |
| S4 | 82.8% | 500 |
| S4g | 82.6% | 500 |

Paired tests: `open_judge_summary.md` next to `open_answers.jsonl`.
