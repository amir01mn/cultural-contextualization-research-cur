"""Abstraction→refinement cultural knowledge-graph research pipeline.

Builds a layered cultural KG (L3 value → L2 social norm → L1 practice) from
CultureBank and evaluates whether it helps an LLM answer the held-out
CulturalBench benchmark (systems S0-S4). See docs/ARCHITECTURE.md.

Subpackages, in dependency order (a package may import only from those above it):

    core          config, schemas, IO/storage, culture names, embeddings
    llm           LLM backends (Ollama / vLLM) and prompt templates
    loaders       CultureBank (construction) and CulturalBench (evaluation)
    construction  extraction → critique → entity resolution → alignment → pathways
    retrieval     passage / triple / pathway retrieval and reranking
    evaluation    S0-S4 systems, scoring, statistics, mechanism analysis
    legacy        original MVP open-ended evaluator (not a research metric)

All LLM work runs through local/open-source models. No paid APIs.
"""

__version__ = "0.2.0"
