---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Abstraction→Refinement Cultural KG Research'
footer: 'Current status: experimental validation and retrieval diagnosis'
style: |
  section { font-size: 23px; line-height: 1.4; }
  h1 { color: #1f3a5f; }
  h2 { color: #3A7CA5; border-bottom: 2px solid #e3e8ef; padding-bottom: .2em; }
  table { font-size: 18px; }
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Abstraction→Refinement Cultural Knowledge Graphs

### Can structured cultural pathways improve culturally grounded LLM reasoning?

Current stage: **full pipeline implemented; central result under diagnosis**

---

## Research question

Does a cultural KG organized as:

```text
deep value (L3) → social norm (L2) → observable practice (L1)
```

improve CulturalBench accuracy compared with closed-book answering, raw-text
RAG, flat triples, and layered triples?

CultureBank is used for construction. CulturalBench is held out for evaluation.

---

## Systems compared

| System | Evidence |
|---|---|
| S0 No-KG | question and options only |
| S1 Raw-text RAG | CultureBank passages |
| S2 Flat KG | triples without layer structure |
| S3 Layered KG | refined triples with layers |
| S4 Pathway KG | complete abstraction→refinement pathways |
| Ablations | triple retrieval; no shared pivots |

The same model, benchmark input, and decoding setup are used throughout.

---

## Implemented pipeline

```text
CultureBank records
  → LLM extraction + provenance
  → critique/refinement
  → entity resolution + culture normalization
  → cross-cultural alignment
  → L3→L2→L1 pathway construction
  → persistent pathway index
  → S0–S4 CulturalBench evaluation
  → CIs, McNemar tests, ablations, mechanism analysis
```

Runs locally with Ollama and at scale with vLLM/SLURM.

---

## Full-run scale

| Artifact | Count |
|---|---:|
| CultureBank passages | 22,990 |
| Raw triples | 153,882 |
| Refined triples | 186,691 |
| Canonical entities | 114,472 |
| Cross-cultural alignments | 998 |
| Pathways | 29,578 |
| CulturalBench units | 2,454 |

Model: Qwen2.5-7B-Instruct; retrieval top-k: 8.

---

## Full-run result

| System | Combined accuracy |
|---|---:|
| **S0 No-KG** | **52.2%** |
| S1 Raw-text RAG | 41.4% |
| S2 Flat KG | 42.1% |
| S3 Layered KG | 42.5% |
| **S4 Pathway KG** | **38.4%** |
| S4 Triple ablation | 42.3% |
| S4 No-shared ablation | 38.2% |

The central hypothesis is **not supported by this run**.

---

## What the result tells us

- The corpus contains many complete pathways; pathway count is not the blocker.
- Retrieved evidence often distracts the model or mismatches the target culture.
- S4 pathways currently perform worse than refined-triple retrieval.
- Retrieval can recover some closed-book errors, but corrupts more correct answers.
- Better relevance control, culture matching, and abstention/gating are the main
  engineering-research priorities.

---

## Controlled experiments

A manually audited, multi-culture food/hospitality experiment produced a
promising small-sample result:

- 77 CultureBank records across 19 pathway cultures
- 40 CulturalBench units
- S0: 65.0%
- S4: 75.0%
- S4 gained four cases over S3 and lost none

This was not statistically significant and does not replace the full result. It
suggests that culture-matched, audited corpora are worth pursuing.

---

## Current stage

**Engineering:** full S0–S4 research pipeline implemented.

**Empirical research:** retrieval-quality diagnosis and controlled validation.

**Not yet complete:**

- human validation of triples, layers, merges, pathways, and retrieval
- stereotype and representation audits
- multiple models and seeds
- top-k and generated-evidence ablations at final scale
- one frozen, internally consistent corpus/report release

---

## Immediate next steps

1. Consolidate the active branch, later experiment branch, and useful v2 work.
2. Freeze the environment and make the full test suite green.
3. Rebuild and validate a provenance-safe, culture-matched corpus.
4. Add retrieval relevance gating and explicit abstention.
5. Run controlled multi-culture experiments before another broad full run.
6. Complete the preregistered multi-model, multi-seed and human-audit protocol.

---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Takeaway

The project has matured from an MVP into a complete experimental platform.

The open problem is now scientific rather than architectural: determine when
structured cultural evidence helps, when it harms, and how to retrieve it safely.
