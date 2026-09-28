"""KG construction, in pipeline order:

    triple_extractor → critique_agent (+ layer_classifier) → entity_resolver
    → cross_cultural_aligner → graph_builder → pathway_builder

Orchestrated by ``scripts/build_corpus.py``. Depends on ``src.core`` and ``src.llm``.
"""
