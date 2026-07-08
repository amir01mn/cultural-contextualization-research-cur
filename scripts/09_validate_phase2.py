"""Step 9: validate Phase 2 pipeline artifacts and smoke-test pathway retrieval."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.pathway_retriever import PathwayRetriever  # noqa: E402
from src.storage import load_corpus_state, read_jsonl_list  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOGGER = get_logger("09_validate_phase2")


def _generated_support_ratio(triples: list[dict]) -> float:
    if not triples:
        return 0.0
    return sum(1 for t in triples if t.get("generated")) / len(triples)


def build_validation_report(
    *,
    query: str = "respect for elders in Japan",
    top_k: int = 5,
) -> str:
    state = load_corpus_state(config.CORPUS_STATE_PATH)
    raw = read_jsonl_list(config.PROCESSED_DIR / "corpus_triples.jsonl")
    refined = read_jsonl_list(config.CORPUS_TRIPLES_REFINED_PATH)
    entities = read_jsonl_list(config.ENTITY_MAP_PATH)
    alignments = read_jsonl_list(config.ALIGNMENTS_PATH)
    pathways = read_jsonl_list(config.PATHWAYS_PATH)

    completeness = Counter(p.get("completeness", "NONE") for p in pathways)
    cultures = Counter(p.get("culture", "") for p in pathways)

    retriever = PathwayRetriever(config.PATHWAY_INDEX_DIR)
    hits = retriever.retrieve(query, top_k=top_k)

    lines = [
        "# Phase 2 Validation Report",
        "",
        "## Checkpoint state",
        f"- Extraction completed: {len(state.extraction_completed)} records",
        f"- Critique completed: {len(state.critique_completed)} records",
        f"- Last updated: {state.updated_at or 'n/a'}",
        "",
        "## Triple stores",
        f"- Raw triples: {len(raw)} (generated ratio: {_generated_support_ratio(raw):.1%})",
        f"- Refined triples: {len(refined)} (generated ratio: {_generated_support_ratio(refined):.1%})",
        "",
        "## Entity resolution",
        f"- Canonical entities: {len(entities)}",
        f"- Shared abstraction nodes: {sum(1 for e in entities if e.get('is_shared_abstraction'))}",
        "",
        "## Cross-cultural alignment",
        f"- Alignments: {len(alignments)}",
        f"- Verified: {sum(1 for a in alignments if a.get('verified'))}",
        "",
        "## Pathways",
        f"- Total pathways: {len(pathways)}",
        f"- By completeness: {dict(completeness)}",
        f"- Top cultures: {dict(cultures.most_common(10))}",
        "",
        f"## Sample retrieval (`{query}`)",
        "",
    ]
    if not hits:
        lines.append("- No pathway hits (index empty or query unmatched).")
    else:
        for pw, score in hits:
            node_labels = " -> ".join(n.label for n in pw.nodes)
            lines.append(
                f"- score={score:.3f} culture={pw.culture} "
                f"completeness={pw.completeness.value} path={node_labels}"
            )

    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validate Phase 2 artifacts.")
    p.add_argument("--query", default="respect for elders in Japan")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--output", default=str(config.PHASE2_VALIDATION_PATH))
    args = p.parse_args(argv)

    report = build_validation_report(query=args.query, top_k=args.top_k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    LOGGER.info("Wrote validation report -> %s", args.output)


if __name__ == "__main__":
    main()
