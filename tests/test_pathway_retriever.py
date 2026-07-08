from src.pathway_retriever import PathwayRetriever, build_pathway_index, pathway_document
from src.schemas import CulturalPathway, LayerLabel, PathwayCompleteness, PathwayNodeRef


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
