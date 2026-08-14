"""Regression cover for the failures that made S4 underperform the baseline.

Each test pins one link in the measured causal chain:

    512-token budget -> truncated JSON -> silent schema fallback -> hardcoded
    L2 layers -> no L3 structure -> pathway builder finds nothing -> S4 injects
    noise -> accuracy drops below closed-book.
"""
import networkx as nx
import pytest

from src import config, storage
from src.critique_agent import _is_content_issue, build_refined_triples
from src.graph_builder import build_graph_resolved
from src.pathway_builder import build_pathways
from src.retrieval_ranking import rerank_hits
from src.schemas import (
    CanonicalEntity,
    CorpusBuildState,
    CritiqueIssue,
    CritiqueResult,
    CrossCulturalAlignment,
    CulturalTriple,
    FlaggedTriple,
    LayerLabel,
    PathwayCompleteness,
)
from src.triple_extractor import extract_triples_schema


# --- Phase 1: token budgets and cache keying -------------------------------

def test_stage_budgets_exceed_the_broken_global_default():
    assert config.stage_max_tokens("extraction") > 512
    assert config.stage_max_tokens("critique") > 512
    assert config.stage_max_tokens("unknown_stage") == config.LLM_MAX_TOKENS


def test_cache_key_varies_with_model_and_token_budget():
    """A prompt-only cache key would replay truncated completions forever."""
    base = ("extraction", "sys", "user")
    k_small = storage._cache_key(*base, model="m", max_tokens=512)
    k_large = storage._cache_key(*base, model="m", max_tokens=2048)
    k_other = storage._cache_key(*base, model="other", max_tokens=512)
    assert k_small != k_large
    assert k_small != k_other


# --- Phase 2: honest fallback and checkpoint integrity ---------------------

def test_schema_fallback_is_tagged(sample_record):
    tagged = extract_triples_schema(sample_record, llm_failed=True)
    assert tagged and all(t.llm_failed for t in tagged)
    assert all(t.extraction_method == "schema_fallback" for t in tagged)
    untagged = extract_triples_schema(sample_record)
    assert untagged and not any(t.llm_failed for t in untagged)


def test_failed_records_are_not_marked_complete():
    state = CorpusBuildState()
    storage.mark_failed(state, "critique", ["a", "b"])
    assert storage.read_completed_ids(state, "critique") == set()
    assert storage.read_failed_ids(state, "critique") == {"a", "b"}
    # A later success clears the failure so it is not retried forever.
    storage.mark_completed(state, "critique", ["a"])
    assert storage.read_completed_ids(state, "critique") == {"a"}
    assert storage.read_failed_ids(state, "critique") == {"b"}


@pytest.mark.parametrize("issue_type,droppable", [
    ("unsupported_claim", True),
    ("stereotyping", True),
    ("hallucinated_triple", True),
    ("surface_without_bridge", False),
    ("missing_layer_2_social", False),
])
def test_only_content_issues_are_droppable(issue_type, droppable):
    assert _is_content_issue(issue_type) is droppable


def _triples(n, instance="r1"):
    return [
        CulturalTriple(triple_id=f"t{i}", subject=f"s{i}", predicate="p",
                       object=f"o{i}", instance_id=instance)
        for i in range(n)
    ]


def test_structural_issue_never_erases_a_record():
    """The rule-based critique raises structural issues at high severity while
    listing every triple id; dropping on severity alone would empty the record.
    """
    ts = _triples(3)
    critique = CritiqueResult(
        record_id="r1",
        original_triple_ids=[t.triple_id for t in ts],
        issues=[CritiqueIssue(issue_type="surface_without_bridge", severity="high",
                              description="d",
                              affected_triple_ids=["t0", "t1", "t2"])],
        triples_to_flag=[FlaggedTriple(triple_id=f"t{i}", reason="r") for i in range(3)],
    )
    assert len(build_refined_triples({"r1": {}}, {"r1": ts}, [critique])) == 3


def test_flagged_content_issue_drops_only_that_triple():
    ts = _triples(3)
    critique = CritiqueResult(
        record_id="r1",
        original_triple_ids=[t.triple_id for t in ts],
        issues=[CritiqueIssue(issue_type="unsupported_claim", severity="high",
                              description="d", affected_triple_ids=["t1"])],
        triples_to_flag=[FlaggedTriple(triple_id="t1", reason="no support")],
    )
    kept = build_refined_triples({"r1": {}}, {"r1": ts}, [critique])
    assert len(kept) == 2
    assert "t1" not in {t.triple_id for t in kept}


def test_unflagged_content_issue_keeps_the_triple():
    ts = _triples(3)
    critique = CritiqueResult(
        record_id="r1",
        original_triple_ids=[t.triple_id for t in ts],
        issues=[CritiqueIssue(issue_type="unsupported_claim", severity="high",
                              description="d", affected_triple_ids=["t1"])],
    )
    assert len(build_refined_triples({"r1": {}}, {"r1": ts}, [critique])) == 3


# --- Phase 4: pathway construction ----------------------------------------

def _hub_graph():
    """Graph shaped the way the corpus actually stores structure.

    The social norm is the SUBJECT of both statements, so edges run L2->L3 and
    L2->L1. A traversal starting at L3 and following successors finds nothing.
    """
    g = nx.MultiDiGraph()
    g.add_node("v3", label="respect for elders", layer="LAYER_3_DEEP_VALUE",
               dominant_layer="LAYER_3_DEEP_VALUE", cultures={"japan"})
    g.add_node("v2", label="deference to elders", layer="LAYER_2_SOCIAL",
               dominant_layer="LAYER_2_SOCIAL", cultures={"japan"})
    g.add_node("v1", label="bow deeply", layer="LAYER_1_SURFACE",
               dominant_layer="LAYER_1_SURFACE", cultures={"japan"})
    g.add_edge("v2", "v3", key="t_a", predicate="reflects value", confidence=0.9)
    g.add_edge("v2", "v1", key="t_b", predicate="enacted by", confidence=0.9)
    return g


def test_l2_hub_yields_a_full_pathway():
    pathways = build_pathways(_hub_graph(), [], [])
    assert pathways, "L2 hub with L3 and L1 neighbours must yield a pathway"
    full = [p for p in pathways if p.completeness == PathwayCompleteness.FULL]
    assert full, "value -> norm -> practice must classify as FULL"
    pw = full[0]
    assert [n.layer for n in pw.nodes] == [
        LayerLabel.LAYER_3_DEEP_VALUE,
        LayerLabel.LAYER_2_SOCIAL,
        LayerLabel.LAYER_1_SURFACE,
    ], "nodes must be ordered abstract -> concrete regardless of edge direction"
    assert pw.triple_ids, "pathways require provenance"
    assert pw.predicates and all(pw.predicates), "each hop needs a relation label"


def test_pathway_requires_provenance():
    g = _hub_graph()
    g.remove_edge("v2", "v3", key="t_a")
    g.remove_edge("v2", "v1", key="t_b")
    assert build_pathways(g, [], []) == []


def test_l3_to_l1_jump_is_not_full():
    """Skipping the social layer is exactly what the research question is about."""
    g = nx.MultiDiGraph()
    g.add_node("v3", label="value", layer="LAYER_3_DEEP_VALUE",
               dominant_layer="LAYER_3_DEEP_VALUE", cultures={"japan"})
    g.add_node("v1", label="practice", layer="LAYER_1_SURFACE",
               dominant_layer="LAYER_1_SURFACE", cultures={"japan"})
    g.add_edge("v3", "v1", key="t", predicate="p", confidence=0.9)
    assert not [p for p in build_pathways(g, [], [])
                if p.completeness == PathwayCompleteness.FULL]


def test_pathway_hops_must_share_a_culture():
    g = _hub_graph()
    g.nodes["v1"]["cultures"] = {"brazil"}
    for p in build_pathways(g, [], []):
        assert "bow deeply" not in [n.label for n in p.nodes]


def test_alignment_pivots_are_connected_and_traversable():
    """Pivots were added as nodes but never given edges, so has_shared_pivot
    was structurally always False despite 107 verified alignments.
    """
    triples = [
        CulturalTriple(triple_id="t1", subject="deference", predicate="reflects",
                       object="respect", subject_layer=LayerLabel.LAYER_2_SOCIAL,
                       object_layer=LayerLabel.LAYER_3_DEEP_VALUE,
                       instance_id="r1"),
    ]
    entities = [
        CanonicalEntity(entity_id="e_norm", canonical_label="deference",
                        layer=LayerLabel.LAYER_2_SOCIAL, cultures=["japan"]),
        CanonicalEntity(entity_id="e_val", canonical_label="respect",
                        layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["japan"]),
        CanonicalEntity(entity_id="p_shared", canonical_label="respect",
                        layer=LayerLabel.LAYER_3_DEEP_VALUE,
                        cultures=["japan", "india"], is_shared_abstraction=True),
    ]
    label_map = {
        ("deference", "LAYER_2_SOCIAL"): "e_norm",
        ("respect", "LAYER_3_DEEP_VALUE"): "e_val",
    }
    alignments = [CrossCulturalAlignment(
        alignment_id="a1", abstraction_node_id="p_shared",
        culture_a="japan", culture_b="india",
        entity_ids=["e_val"], entity_label_a="respect", entity_label_b="respect",
        similarity=0.95, verified=True,
    )]
    graph = build_graph_resolved(triples, entities, label_map, alignments)
    assert graph.degree("p_shared") > 0, "pivot must not be an isolated node"


# --- Phase 5: retrieval precision -----------------------------------------

class _Item:
    def __init__(self, culture):
        self.culture = culture


def test_relevance_floor_can_return_nothing():
    """With nothing relevant the caller emits no evidence and answers closed-book."""
    hits = [(_Item("japan"), 0.01), (_Item("japan"), 0.02)]
    assert rerank_hits(hits, top_k=5, question_culture="Japan",
                       get_culture=lambda x: x.culture, min_score=0.15) == []


def test_relevant_hits_survive_the_floor():
    hits = [(_Item("japan"), 0.9), (_Item("japan"), 0.01)]
    kept = rerank_hits(hits, top_k=5, question_culture="Japan",
                       get_culture=lambda x: x.culture, min_score=0.15)
    assert len(kept) == 1


def test_diversity_cap_limits_one_culture():
    hits = [(_Item("japan"), 0.9 - i * 0.01) for i in range(8)]
    kept = rerank_hits(hits, top_k=8, question_culture=None,
                       get_culture=lambda x: x.culture, min_score=0.0,
                       max_per_culture=3)
    # Capped entries backfill rather than truncating the result set.
    assert len(kept) == 8


def test_country_demonym_hits_are_treated_as_matching():
    hits = [(_Item("Japanese"), 0.5), (_Item("Brazilian"), 0.9)]
    kept = rerank_hits(hits, top_k=2, question_culture="Japan",
                       get_culture=lambda x: x.culture, min_score=0.0)
    assert kept[0][0].culture == "Japanese"
