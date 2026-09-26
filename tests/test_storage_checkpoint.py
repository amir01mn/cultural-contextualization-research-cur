from src.evaluation.mcq_systems import SYSTEM_S4, RetrievalContext, build_prompt
from src.retrieval.pathway_retriever import PathwayRetriever, build_pathway_index
from src.core.schemas import CulturalPathway, LayerLabel, PathwayCompleteness, PathwayNodeRef
from src.core.schemas import CorpusBuildState
from src.core.storage import prepare_stage_state, save_corpus_state


def test_prepare_stage_state_preserves_other_stage(tmp_path):
    state_path = tmp_path / "state.json"
    state = CorpusBuildState(
        extraction_completed=["a"],
        critique_completed=["b"],
    )
    save_corpus_state(state_path, state)
    cleared = prepare_stage_state(state_path, "critique", resume=False)
    assert cleared.extraction_completed == ["a"]
    assert cleared.critique_completed == []


def test_s4_pathway_prompt_includes_pathway_header(mcq_easy_question, tmp_path):
    pathways = [
        CulturalPathway(
            pathway_id="pw1",
            culture="Japan",
            nodes=[
                PathwayNodeRef(entity_id="l3", label="respect", layer=LayerLabel.LAYER_3_DEEP_VALUE),
                PathwayNodeRef(entity_id="l2", label="home etiquette", layer=LayerLabel.LAYER_2_SOCIAL),
                PathwayNodeRef(entity_id="l1", label="remove shoes", layer=LayerLabel.LAYER_1_SURFACE),
            ],
            completeness=PathwayCompleteness.FULL,
        )
    ]
    build_pathway_index(pathways, tmp_path)
    ctx = RetrievalContext(pathway_retriever=PathwayRetriever(tmp_path))
    _sp, up, ids, meta = build_prompt(SYSTEM_S4, mcq_easy_question, ctx, top_k=3)
    assert "abstraction" in up.lower() or "pathway" in up.lower()
    assert ids == ["pw1"]
    assert meta["completeness"] == PathwayCompleteness.FULL.value
