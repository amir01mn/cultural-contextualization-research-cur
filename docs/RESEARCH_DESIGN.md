# Research Design: Abstraction→Refinement Cultural Knowledge Graphs for Culturally Grounded Reasoning

> Status: implemented research design under empirical validation. The S0–S4
> pipeline, full-corpus construction, core ablations, and mechanism reports now
> exist. The checked-in full run does not support H1 or H2, so this document is
> the evaluation contract for the next frozen, provenance-safe run—not a claim
> that the hypotheses have been confirmed.

---

## 0. One-line pitch

We build a theory-grounded cultural knowledge graph organized as
**abstraction→refinement pathways** — rooted in shared human/universal concepts
and refined downward into culture-specific practices — and show that this
**pathway structure (not raw fact count)** improves an LLM's culturally grounded
reasoning, while controlling stereotyping. All construction uses local /
open-source LLMs (no paid API).

---

## 1. Motivation and background

Cultural reasoning failures in LLMs are usually not failures of fact recall but
failures of **connecting an observable practice to the underlying social norm and
value that explain it**, and of **transferring** that understanding across
cultures. A flat list of cultural facts ("In India, younger people touch elders'
feet") does not expose *why* (respect for elders) or *how it generalizes* (other
cultures express the same value differently, e.g., bowing in Japan).

We hypothesize that organizing cultural knowledge as **specialization pathways** —
from an abstract, often cross-culturally shared concept, down through social-
normative mediation, to a concrete culture-specific practice — gives the model a
traversable structure that (a) supplies the missing "why," and (b) enables
cross-cultural analogy through **shared abstraction nodes**.

### 1.1 Theoretical grounding

The three-layer ontology and the abstract→specific organization are grounded in
established cross-cultural theory (to be cited, not invented):

- **Schein's three levels of culture**: basic underlying assumptions → espoused
  values → artifacts. Defines our layer semantics.
- **Schwartz's Theory of Basic Values**: a set of (near-)universal value
  dimensions that cultures instantiate with different priorities and expressions.
  This is the theoretical basis for "shared abstract value → culture-specific
  refinement."
- **Brown's Human Universals**: supports a shared-human conceptual anchor at the
  top of the hierarchy.

**Design note (important):** A single universal root ("human") is only a
conceptual anchor and is *not* where the discriminative power lies. The useful
structure is at **mid-level shared abstraction nodes** (e.g., *respect*,
*hospitality*, *face*, *purity*, *modesty*) that two or more cultures refine
differently. Metrics target these mid-level shared nodes, not the root.

---

## 2. Layer model

We use three cultural abstraction layers (most abstract → most refined):

| Layer | Name | Abstraction | Content |
|---|---|---|---|
| L3 | `LAYER_3_DEEP_VALUE` | most abstract | values, beliefs, moral principles (respect, harmony, hospitality, honor, dignity) |
| L2 | `LAYER_2_SOCIAL` | intermediate | roles, norms, expectations, host/guest relations, hierarchy, etiquette |
| L1 | `LAYER_1_SURFACE` | most refined | observable practices, gestures, food, rituals, objects |

`UNKNOWN` is allowed only when a layer is genuinely unclear and is excluded from
pathway analysis.

An **abstraction→refinement** direction means traversal from L3 (abstract) →
L2 → L1 (refined). (Note: this is the inverse reading direction of "a practice
expresses a value"; both directions describe the same edges.)

---

## 3. Formal definitions

Let the KG be a directed graph `G = (V, E)`. Each node `v ∈ V` has a layer
`layer(v) ∈ {L1, L2, L3, UNKNOWN}` and a set of associated cultural groups
`cultures(v) ⊆ C`. Each edge carries provenance (dataset_id, instance_id,
source_text, confidence, generated flag, evidence) per the existing schema.

### 3.1 Abstraction→refinement pathway

An **abstraction→refinement pathway** is a directed path
`p = (v3 → v2 → v1)` such that:

- `layer(v3) = L3`, `layer(v2) = L2`, `layer(v1) = L1`, and
- consecutive nodes are connected by directed edges in `E`, and
- the nodes share at least one culture: `cultures(v3) ∩ cultures(v2) ∩ cultures(v1) ≠ ∅`
  (the pathway is *coherent* for some culture `c`).

A pathway is **complete** if it spans all three layers (L3→L2→L1). We also define
partial pathways: **2-layer** (L3→L2 or L2→L1) and **1-layer** (a single node).

> We allow multi-hop within a layer band (e.g., L2→L2→L1) as long as the layer
> index is non-increasing along the path; "complete" still requires at least one
> node from each of L3, L2, L1.

### 3.2 Shared abstraction node (cross-cultural pivot)

A node `v` is a **shared abstraction node** if:

- `layer(v) ∈ {L3, L2}` (it is abstract / mid-level), and
- there exist complete (or 2-layer) pathways through `v` into refinements
  belonging to **≥ 2 distinct cultures**: `|cultures_refined_through(v)| ≥ 2`.

Shared abstraction nodes are the pivots that enable cross-cultural transfer and
are central to **H2**.

### 3.3 Per-question pathway-completeness

For a CulturalBench question `q` about culture `c(q)`, let `G_q ⊆ G` be the
subgraph retrieved for `q` (top-k triples expanded into their connected pathways).
Define the **completeness level** of `q`:

```
completeness(q) =
    FULL      if G_q contains a complete L3→L2→L1 pathway coherent for c(q)
    SOCIAL    elif G_q contains an L2-bearing 2-layer pathway for c(q)   (L1↔L2)
    SURFACE   elif G_q contains only L1 (surface) content for c(q)
    NONE      else (no relevant retrieved content)
```

And a graded scalar (for regression):

```
completeness_score(q) = (# distinct layers in {L1,L2,L3} present in the
                         largest c(q)-coherent connected component of G_q) / 3
                         ∈ {0, 1/3, 2/3, 1}
```

### 3.4 Shared-node availability

```
shared_node_available(q) = 1  if some node in G_q is a shared abstraction node
                                whose pivot connects c(q) to ≥1 other culture
                           0  otherwise
```

Used to test whether cross-cultural pivots specifically help (H2 transfer claim).

### 3.5 Safety / quality metrics

- **Generated-support ratio** `gsr(q)`: fraction of retrieved triples used for `q`
  that have `generated = true`. Reported per system (lower is safer).
- **Stereotype rate**: fraction of (a) refinement-suggested triples and (b) model
  answers flagged as over-generalizing/stereotyping by a rubric (human-rated on a
  sample; optionally LLM-screened on the full set with human-validated screener).

---

## 4. Research question and hypotheses

**RQ.** Does organizing cultural knowledge as abstraction→refinement pathways
improve LLM answering of culturally grounded (CulturalBench) questions, compared
to no KG, raw-text RAG, and a flat (non-layered) KG?

- **H1 — Structure helps.** Abstraction→refinement KG > flat KG > raw-text RAG >
  no-KG in CulturalBench accuracy (statistically significant).
- **H2 — Pathways, not facts (primary mechanism).** The accuracy advantage of the
  abstraction→refinement KG over the flat KG **concentrates on questions with
  `completeness(q) = FULL`** and/or `shared_node_available(q) = 1`. I.e., the gain
  comes from traversable complete pathways and cross-cultural pivots, not from
  having more triples.
- **H3 — Refinement completes pathways safely.** Critique-refinement increases the
  proportion of questions with complete pathways (coverage↑) while keeping
  `gsr` ≤ a preregistered threshold (e.g., 25%) and **not** increasing stereotype
  rate relative to the raw KG.

H2 is the make-or-break, novelty-bearing result and is computed entirely within
CulturalBench (no second benchmark needed).

---

## 5. Method / system to build

1. **Layered LLM extraction** (implemented): triples + layer labels + full
   provenance, local LLM main, schema fallback only.
2. **Entity resolution & node canonicalization** (NEW, critical): merge
   surface/lexical variants; produce canonical nodes per layer.
3. **Cross-cultural abstraction alignment** (NEW, critical): align refinements
   from different cultures under the same abstraction node → creates shared
   abstraction nodes (enables H2). This is the highest-risk, highest-value piece.
4. **Abstraction→refinement pathway construction** (NEW): assemble L3→L2→L1
   pathways; mark completeness; index pathways for retrieval.
5. **Critique-refinement** (exists, repurposed): detect *incomplete* pathways and
   complete them (add missing L2/L3 bridges) under hallucination/stereotype
   control; measured by H3.
6. **Pathway retrieval + answering** (implemented): retrieve pathways rather
   than isolated triples and inject them into the MCQ answering prompt.

---

## 6. Data plan

| Role | Dataset | Notes |
|---|---|---|
| **KG construction** | **CultureBank** (full; HF `SALT-NLP/CultureBank`, splits `tiktok`+`reddit`) | rich descriptive records; already loaded by the pipeline |
| **Evaluation (sole metric)** | **CulturalBench** (HF `kellycyy/CulturalBench`; `CulturalBench-Easy` + `CulturalBench-Hard`) | MCQ with gold answers → objective accuracy; held out from construction |

**Held-out principle:** the KG is built *only* from CultureBank; evaluation is
*only* on CulturalBench. No overlap → defeats the circularity objection.

---

## 7. Systems compared (KG conditions)

| ID | System | Description |
|---|---|---|
| **S0** | No-KG (closed-book) | LLM answers the MCQ directly |
| **S1** | Raw-text RAG | retrieve raw CultureBank text snippets (no KG), inject into prompt |
| **S2** | Flat KG | retrieve triples as a bag; layer labels & hierarchy **stripped** |
| **S3** | Layered KG (raw) | triples with layers; pathways **not** completed by critique |
| **S4** | Abstraction→Refinement KG (ours, refined) | completed pathways + cross-cultural shared nodes; pathway retrieval |

All systems share the same base LLM, retriever budget (top-k), and prompt
scaffold except for the injected evidence. Accuracy is exact-match on the MCQ
gold answer.

---

## 8. Ablation / evaluation matrix

Primary table: **CulturalBench accuracy (Easy, Hard, Combined)** with 95% CIs, for
each system × base model.

| | Base model A | Base model B |
|---|---|---|
| S0 No-KG | acc ± CI | acc ± CI |
| S1 Raw-text RAG | acc ± CI | acc ± CI |
| S2 Flat KG | acc ± CI | acc ± CI |
| S3 Layered KG (raw) | acc ± CI | acc ± CI |
| **S4 Abstraction→Refinement (ours)** | **acc ± CI** | **acc ± CI** |

Targeted ablations (each isolates one cause):

| Ablation | Comparison | Tests |
|---|---|---|
| **A1 — Structure** | S2 vs S3 vs S4 | does layering/pathway beat a flat bag of the same triples? (H1) |
| **A2 — Refinement** | S3 vs S4 | does critique-completing pathways help? (H3) |
| **A3 — Retrieval unit** | triple-retrieval vs pathway-retrieval (within S4) | is retrieving whole pathways better than isolated triples? |
| **A4 — Cross-cultural alignment** | S4 with vs without shared-node alignment | do shared abstraction pivots matter? (H2 transfer) |
| **A5 — top-k sweep** | k ∈ {3,5,8,12} | retrieval-budget sensitivity |
| **A6 — Generated triples** | include vs exclude `generated=true` triples | how much of any gain is grounded vs inferred (safety) |

Mechanism analyses (the H2 "money figures"):

- **Accuracy by completeness bucket**: for S2/S3/S4, plot accuracy within
  `completeness(q) ∈ {NONE, SURFACE, SOCIAL, FULL}`. Claim supported if S4's
  advantage grows with completeness and is largest at FULL.
- **Accuracy by shared-node availability**: S4 vs S2 within
  `shared_node_available ∈ {0,1}`.
- **Regression**: `correct(q) ~ system + completeness_score(q) + shared_node + controls`.

---

## 9. Statistical protocol

- **Significance for paired MCQ accuracy**: McNemar's test (per item, paired
  across systems on the same base model); report effect size.
- **Confidence intervals**: paired bootstrap over questions (≥10k resamples).
- **Seeds/runs**: ≥3 seeds for any stochastic step (decoding temperature fixed
  low; report mean ± sd).
- **Multiple base models** (≥2 open models) to show the effect is not model-
  specific. Optionally one frontier model as a *ceiling reference* (eval only).
- **Multiple-comparison control** across the ablation table (Holm–Bonferroni).
- **Preregister** thresholds (H1 significance, H3 `gsr` ≤ 25%, stereotype non-
  increase) before the full run.

---

## 10. KG-quality validation (separate from task accuracy)

Reviewers grade construction quality independently of downstream accuracy:

- **Extraction quality**: human precision/recall on a stratified sample; report
  inter-annotator agreement (Cohen's/Fleiss' κ).
- **Layer-label validity**: human vs. LLM layer agreement (validates the
  ontology).
- **Pathway-edge precision**: are L3→L2→L1 links actually correct?
- **Cross-cultural alignment precision**: are nodes merged under a shared
  abstraction node genuinely the same concept?
- **Entity-resolution quality**: precision/recall of node merging on a labeled
  subset.

---

## 11. Ethics, fairness, and cultural safety (first-class)

- **Stereotyping measured, not disclaimed** (Section 3.5); reported as a result.
- **Representation analysis**: per-culture coverage in the KG and per-culture
  accuracy; test whether the KG disproportionately helps high-resource cultures
  (a fairness finding either way).
- **Provenance + uncertainty preserved end-to-end**; generated triples never
  presented as ground truth.
- **Cultural-insider annotators** for the human-eval sample where feasible.
- **Artifacts**: Datasheet/Data Statement, licensing review for CultureBank and
  CulturalBench, responsible-use framing ("cultural humility"; no claims that the
  system "knows" a culture).

---

## 12. Scalable engineering status

| Concern | Implemented now | Remaining improvement |
|---|---|---|
| LLM serving | Ollama locally; batched vLLM on Narval | freeze exact model/runtime versions |
| Orchestration | cache plus checkpoint/resume and SLURM jobs | one manifest-backed end-to-end run |
| Stores | JSONL plus Parquet mirrors | optional DuckDB analysis layer |
| Graph | entity-resolved canonical networkx graph | optional Neo4j/RDF export if analytically useful |
| Retrieval | persistent dense pathway index plus triple/text retrieval | stronger hybrid ranking, culture matching, and abstention |

Reproducibility: pinned configs/seeds/model versions, `make reproduce`, released
KG + all prompts (extraction/critique/answer), leaderboard-style tables. Selling
point: fully open-source/local, **no paid API required**.

---

## 13. Phase status and forward plan

1. **Design lock and benchmark wiring — complete.** CulturalBench loading,
   S0–S3, exact-match scoring, confidence intervals, and McNemar tests exist.
2. **Scalable construction — complete.** Batched inference, caching,
   checkpointing, Parquet, entity resolution, culture normalization,
   cross-cultural alignment, and pathway indexing exist.
3. **Full construction, S4, and first ablations — complete for one main model.**
   The checked-in run covers all 2,454 official scoring units. It is a negative
   result: S0 reaches 52.2%, S3 42.5%, and S4 38.4%.
4. **Controlled validation and recovery — current.** Consolidate branches and
   artifacts, freeze a provenance-safe corpus, complete human audits, and improve
   culture matching, relevance gating, and evidence abstention. Small controlled
   food/hospitality experiments are promising but not statistically decisive.
5. **Confirmatory evaluation — pending.** Run at least two base models and three
   seeds, top-k and generated-evidence ablations, multiple-comparison correction,
   per-culture analysis, and the full mechanism protocol.
6. **Safety and writing — pending.** Complete stereotype/representation audits,
   error analysis, artifact release, and paper framing based on the confirmed
   evidence rather than the intended hypothesis.

**Venue fit:** EMNLP/ACL/NAACL (NLP-methods framing) or FAccT/CSCW (cultural-
fairness/HCI). C3NLP / cross-cultural NLP workshop as a de-risking stepping stone.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Circularity** (build & eval on same data) | build on CultureBank, eval on CulturalBench (held out) |
| **Structure not proven causal** | A1 ablation (flat vs layered) + H2 completeness-bucket analysis |
| **Entity resolution / alignment fails** → singleton graph, H2 untestable | invest early; validate ER precision/recall before full run |
| **Refinement just adds hallucinated/stereotyped triples** | H3 with preregistered `gsr` ≤ 25% + stereotype audit |
| **Novelty challenge** (WordNet/ConceptNet/CANDLE/Step-Back Prompting) | differentiate: auto-construction + provenance + cross-cultural pivoting evaluated on reasoning |
| **Single-benchmark generalization question** | accepted trade-off; NormAd can be added later without redesign |

---

## 15. Decisions still needed

1. **Target venue + rough deadline** (sets scope and framing).
2. **Canonical branch and artifact lineage** for the next frozen run.
3. **Second base model** to pair with Qwen2.5-7B-Instruct.
4. **Preregistered thresholds** confirmation (H3 `gsr` ≤ 25%, significance level,
   #seeds).

---

## 16. Immediate execution step

Consolidate the active full-pathway branch, the later controlled-experiment
branch, and the provenance-safe rebuild work into one canonical branch. Rebuild
the corpus under a frozen manifest, complete the independent audit packets, and
regenerate validation and evaluation reports from that exact artifact set before
running further benchmark-driven tuning.

Last updated: 2026-09-22.
