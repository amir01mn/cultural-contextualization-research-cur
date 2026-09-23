from src.pathway_retriever import PathwayRetriever, build_pathway_index, pathway_document
from src.schemas import (
    CulturalPathway,
    LayerLabel,
    PathwayCompleteness,
    PathwayEdgeRef,
    PathwayNodeRef,
)


def test_pathway_index_roundtrip(tmp_path):
    pathways = [
        CulturalPathway(
            pathway_id="pw1",
            culture="Japan",
            nodes=[
                PathwayNodeRef(entity_id="l3", label="respect", layer=LayerLabel.LAYER_3_DEEP_VALUE),
                PathwayNodeRef(entity_id="l1", label="remove shoes", layer=LayerLabel.LAYER_1_SURFACE),
            ],
            completeness=PathwayCompleteness.SOCIAL,
        )
    ]
    build_pathway_index(pathways, tmp_path)
    retriever = PathwayRetriever(tmp_path)
    hits = retriever.retrieve("respect shoes Japan", top_k=3)
    assert hits
    assert hits[0][0].pathway_id == "pw1"


def test_pathway_document_includes_culture():
    pw = CulturalPathway(pathway_id="x", culture="India", completeness=PathwayCompleteness.FULL)
    doc = pathway_document(pw)
    assert "India" in doc



def test_pathway_retrieval_penalizes_fully_generated_support(tmp_path):
    grounded = CulturalPathway(
        pathway_id="grounded",
        culture="Japan",
        nodes=[
            PathwayNodeRef(
                entity_id="l2",
                label="guest etiquette",
                layer=LayerLabel.LAYER_2_SOCIAL,
            ),
            PathwayNodeRef(
                entity_id="l1",
                label="remove shoes",
                layer=LayerLabel.LAYER_1_SURFACE,
            ),
        ],
        edges=[
            PathwayEdgeRef(
                triple_id="t_grounded",
                source_entity_id="l2",
                target_entity_id="l1",
                source_label="guest etiquette",
                target_label="remove shoes",
                predicate="expressed as",
                source_layer=LayerLabel.LAYER_2_SOCIAL,
                target_layer=LayerLabel.LAYER_1_SURFACE,
                cultural_group="Japan",
                generated=False,
            ),
        ],
        triple_ids=["t_grounded"],
        completeness=PathwayCompleteness.SOCIAL,
    )

    generated = grounded.model_copy(
        update={
            "pathway_id": "generated",
            "edges": [
                grounded.edges[0].model_copy(
                    update={
                        "triple_id": "t_generated",
                        "generated": True,
                    }
                )
            ],
            "triple_ids": ["t_generated"],
        }
    )

    retriever = build_pathway_index([grounded, generated], tmp_path)
    hits = retriever.retrieve(
        "Japan guest etiquette remove shoes",
        top_k=2,
        question_culture="Japan",
    )

    scores = {pathway.pathway_id: score for pathway, score in hits}
    assert set(scores) == {"grounded", "generated"}

    # Generated bridge evidence is explicitly uncertain elsewhere in the
    # pipeline. Pathway retrieval should therefore prefer otherwise identical
    # grounded support rather than treating both pathways as equally reliable.
    assert scores["grounded"] > scores["generated"]


def test_pathway_retrieval_scales_mixed_generated_support(tmp_path):
    base_edges = [
        PathwayEdgeRef(
            triple_id="t1",
            source_entity_id="l3",
            target_entity_id="l2",
            source_label="respect",
            target_label="guest etiquette",
            predicate="supports",
            source_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            target_layer=LayerLabel.LAYER_2_SOCIAL,
            cultural_group="Japan",
            generated=False,
        ),
        PathwayEdgeRef(
            triple_id="t2",
            source_entity_id="l2",
            target_entity_id="l1",
            source_label="guest etiquette",
            target_label="remove shoes",
            predicate="expressed as",
            source_layer=LayerLabel.LAYER_2_SOCIAL,
            target_layer=LayerLabel.LAYER_1_SURFACE,
            cultural_group="Japan",
            generated=False,
        ),
    ]
    base = CulturalPathway(
        pathway_id="grounded",
        culture="Japan",
        nodes=[
            PathwayNodeRef(entity_id="l3", label="respect", layer=LayerLabel.LAYER_3_DEEP_VALUE),
            PathwayNodeRef(entity_id="l2", label="guest etiquette", layer=LayerLabel.LAYER_2_SOCIAL),
            PathwayNodeRef(entity_id="l1", label="remove shoes", layer=LayerLabel.LAYER_1_SURFACE),
        ],
        edges=base_edges,
        triple_ids=["t1", "t2"],
        completeness=PathwayCompleteness.FULL,
    )
    mixed = base.model_copy(
        update={
            "pathway_id": "mixed",
            "edges": [
                base_edges[0],
                base_edges[1].model_copy(update={"generated": True}),
            ],
        }
    )
    generated = base.model_copy(
        update={
            "pathway_id": "generated",
            "edges": [
                edge.model_copy(update={"generated": True})
                for edge in base_edges
            ],
        }
    )

    retriever = build_pathway_index([base, mixed, generated], tmp_path)
    hits = retriever.retrieve(
        "Japan respect guest etiquette remove shoes",
        top_k=3,
        question_culture="Japan",
    )
    scores = {pathway.pathway_id: score for pathway, score in hits}
    assert scores["grounded"] > scores["mixed"] > scores["generated"]


def test_pathway_retrieval_no_shared_backfills_to_top_k(tmp_path):
    shared = CulturalPathway(
        pathway_id="shared",
        culture="Japan",
        nodes=[
            PathwayNodeRef(entity_id="a", label="hospitality", layer=LayerLabel.LAYER_3_DEEP_VALUE),
            PathwayNodeRef(entity_id="b", label="serve tea", layer=LayerLabel.LAYER_1_SURFACE),
        ],
        completeness=PathwayCompleteness.SOCIAL,
        has_shared_pivot=True,
    )
    plain = shared.model_copy(
        update={
            "pathway_id": "plain",
            "has_shared_pivot": False,
            "shared_pivot_id": None,
        }
    )
    retriever = build_pathway_index([shared, plain], tmp_path)
    hits = retriever.retrieve(
        "Japan hospitality serve tea",
        top_k=1,
        question_culture="Japan",
        exclude_shared=True,
    )
    assert len(hits) == 1
    assert hits[0][0].pathway_id == "plain"
