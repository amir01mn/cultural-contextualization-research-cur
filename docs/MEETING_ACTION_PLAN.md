# Meeting Action Plan: CulturalBench + Cultural KG Evaluation

## Purpose

This document locks the immediate experimental plan after the supervisor meeting, using the current state of `main` as the source of truth.

The current repo already includes S0-S4 systems, S4 ablations, pathway retrieval, refined triples, shared-pivot logic, and pathway-completeness metadata. This action plan therefore does not recreate the older S4 implementation. It documents how the current system should be compared and reported.

## Core experimental rule

Every system must receive the same base question/input.

The only thing that changes across systems is the extra evidence block.

This rule protects the evaluation from the "not comparable" criticism. KG/RAG context must augment the original CulturalBench question, not replace it.

## Systems to compare

| System | Name in code | Input condition |
|---|---|---|
| S0 | `S0_no_kg` | Question/options only |
| S1 | `S1_raw_text_rag` | Question/options + retrieved raw CultureBank passages |
| S2 | `S2_flat_kg` | Question/options + flat KG triples |
| S3 | `S3_layered_kg` | Question/options + layered KG triples |
| S4 | `S4_abstraction_refinement_kg` | Question/options + abstraction-refinement pathways |
| S4 ablation | `S4_triple_retrieval` | Question/options + refined triples without full pathway retrieval |
| S4 ablation | `S4_no_shared_pivot` | Question/options + pathways without cross-cultural shared pivots |

## Prompt rule

Retrieved evidence should be framed as additional context, not as the only source of truth.

Preferred wording:

> Use the retrieved evidence as additional context, not as a replacement for the question/options. If it is incomplete, irrelevant, or insufficient, do not force it; answer using the base question and the most relevant evidence.

This rule should apply to S1, S2, S3, S4, and S4 ablations. S0 should continue receiving no evidence block.

## How to report the current Phase 1 result

Do not report the small S0-S3 result as "KG failed."

Report it as a Phase 1 sanity check:

> We successfully connected the CulturalBench evaluation pipeline. The small S0-S3 run is a sanity check. It does not by itself test the full S4 abstraction-refinement pathway KG.

Current Phase 1 result discussed in the meeting:

| System | Combined | Easy | Hard |
|---|---:|---:|---:|
| S0 No-KG | 72.5% | 75.0% | 70.0% |
| S1 Raw-text RAG | 70.0% | 65.0% | 75.0% |
| S2 Flat KG | 65.0% | 65.0% | 65.0% |
| S3 Layered KG | 67.5% | 70.0% | 65.0% |

Interpretation:

- S0 > S1 > S3 > S2 in this small run.
- The result is not statistically decisive.
- The corpus contained only 4 triples and 20 passages, so this mainly confirms that the evaluation pipeline runs.
- It does not yet settle whether abstraction-refinement pathways help.

## Immediate next steps

1. Keep this branch as the step 1-5 branch based on current `main`.
2. Keep the baseline constant: same question/options for every system.
3. Treat KG/RAG/pathway evidence as augmentation, not replacement.
4. Run manual error analysis before making large-scale claims.
5. Use `docs/CULTURALBENCH_BASELINE_NOTES.md` to explain zero-shot vs retrieval-augmented evaluation.
6. Use error analysis to decide whether the issue is retrieval, triple quality, pathway construction, shared-pivot logic, or prompt wording.

## Error-analysis priorities

Before scaling or presenting strong claims, inspect disagreement cases:

| Disagreement pattern | What it may reveal |
|---|---|
| S0 correct, retrieval systems wrong | Evidence may distract the model, retrieval may be irrelevant, or prompt wording may over-weight evidence |
| S0 wrong, retrieval systems correct | Retrieved evidence may add useful cultural specificity |
| S1 correct, KG systems wrong | Raw text may preserve context that triples/pathways lost |
| S2/S3 correct, S4 wrong | Pathway construction or pathway ranking may be noisy |
| S4 correct, S2/S3 wrong | Abstraction-refinement pathways may be helping beyond isolated triples |
| S4 correct, S4 ablations wrong | Shared pivots or full pathway retrieval may be adding value |
| S4 ablations correct, S4 wrong | The full S4 pathway setting may be over-constrained or poorly ranked |

## Reporting language for the next meeting

Suggested wording:

> We rebuilt the meeting action plan on top of the current main branch. The current repo now includes S0-S4 and S4 ablations, so the immediate goal is not to reimplement S4, but to keep the comparison controlled, document the baseline method, and run targeted error analysis to understand when pathways help or hurt.
