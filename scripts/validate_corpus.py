"""Validate corpus artifacts against quality gates (pipeline step 2 of 4).

With ``--strict`` this exits non-zero when a quality gate fails, so a SLURM job
aborts instead of producing another silently-broken corpus. The first full run
reported "complete" while 96% of records had fallen back to schema extraction,
98% of critiques had failed to parse, and zero complete pathways existed.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import config  # noqa: E402
from src.retrieval.pathway_retriever import PathwayRetriever  # noqa: E402
from src.core.storage import load_corpus_state, read_jsonl_list  # noqa: E402
from src.core.utils import get_logger  # noqa: E402

LOGGER = get_logger("validate_corpus")


# Gate thresholds. The "broken" comments record what the first full run
# produced, so a regression back toward those numbers fails loudly.
# Env-overridable so a small pilot can use size-appropriate absolute counts.
import os  # noqa: E402

MIN_EXTRACTION_SUCCESS = float(os.getenv("GATE_MIN_EXTRACTION_SUCCESS", "0.80"))  # broken: 0.04
MIN_CRITIQUE_SUCCESS = float(os.getenv("GATE_MIN_CRITIQUE_SUCCESS", "0.80"))      # broken: 0.016
MIN_L3_ENTITY_SHARE = float(os.getenv("GATE_MIN_L3_SHARE", "0.10"))               # broken: 0.017
MIN_PATHWAYS = int(os.getenv("GATE_MIN_PATHWAYS", "5000"))                        # broken: 150
MIN_FULL_PATHWAYS = int(os.getenv("GATE_MIN_FULL_PATHWAYS", "1"))                 # broken: 0
MIN_SHARED_PIVOT_PATHWAYS = int(os.getenv("GATE_MIN_SHARED_PIVOT", "1"))          # broken: 0


def _ratio(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def _generated_support_ratio(triples: list[dict]) -> float:
    if not triples:
        return 0.0
    return sum(1 for t in triples if t.get("generated")) / len(triples)


def collect_metrics() -> dict:
    """Gather the measurements the gates are evaluated against."""
    state = load_corpus_state(config.CORPUS_STATE_PATH)
    raw = read_jsonl_list(config.PROCESSED_DIR / "corpus_triples.jsonl")
    refined = read_jsonl_list(config.CORPUS_TRIPLES_REFINED_PATH)
    entities = read_jsonl_list(config.ENTITY_MAP_PATH)
    alignments = read_jsonl_list(config.ALIGNMENTS_PATH)
    pathways = read_jsonl_list(config.PATHWAYS_PATH)
    critiques = read_jsonl_list(config.PROCESSED_DIR / "corpus_critiques.jsonl")

    llm_triples = sum(1 for t in raw if t.get("extraction_method") == "llm_based")
    fallback_triples = sum(1 for t in raw if t.get("llm_failed"))
    extraction_done = len(state.extraction_completed)
    extraction_failed = len(state.extraction_failed)
    critique_done = len(state.critique_completed)
    critique_failed = len(state.critique_failed)

    parse_failures = sum(
        1 for c in critiques if (c.get("summary") or "").startswith("Critique parse failed")
    )

    # Derive success rates from the artifacts, not only from the checkpoint.
    # A checkpoint written by an older build marked every record complete
    # regardless of outcome, so trusting it alone yields a false PASS on exactly
    # the corpus these gates exist to reject.
    records_with_llm = {
        t.get("instance_id") for t in raw if t.get("extraction_method") == "llm_based"
    }
    records_with_triples = {t.get("instance_id") for t in raw}
    artifact_extraction_success = _ratio(len(records_with_llm), len(records_with_triples))
    artifact_critique_success = (
        _ratio(len(critiques) - parse_failures, len(critiques)) if critiques else 0.0
    )

    entity_layers = Counter(e.get("layer") for e in entities)
    completeness = Counter(p.get("completeness", "NONE") for p in pathways)

    return {
        "state": state,
        "raw": raw,
        "refined": refined,
        "entities": entities,
        "alignments": alignments,
        "pathways": pathways,
        "critiques": critiques,
        "llm_triples": llm_triples,
        "fallback_triples": fallback_triples,
        # Take the pessimistic view: a stage counts as healthy only if BOTH the
        # checkpoint and the artifacts agree that it is.
        "extraction_success": min(
            _ratio(extraction_done, extraction_done + extraction_failed),
            artifact_extraction_success,
        ),
        "critique_success": min(
            _ratio(critique_done, critique_done + critique_failed),
            artifact_critique_success,
        ),
        "artifact_extraction_success": artifact_extraction_success,
        "artifact_critique_success": artifact_critique_success,
        "critique_parse_failures": parse_failures,
        "entity_layers": entity_layers,
        "l3_share": _ratio(entity_layers.get("LAYER_3_DEEP_VALUE", 0), len(entities)),
        "completeness": completeness,
        "shared_pivot_pathways": sum(1 for p in pathways if p.get("has_shared_pivot")),
        "shared_abstraction_entities": sum(
            1 for e in entities if e.get("is_shared_abstraction")),
        "cultures": Counter(p.get("culture", "") for p in pathways),
    }


def evaluate_gates(m: dict) -> list[tuple[str, bool, str]]:
    """Return (name, passed, detail) for each quality gate."""
    return [
        (
            "extraction success rate",
            m["extraction_success"] >= MIN_EXTRACTION_SUCCESS,
            f"{m['extraction_success']:.1%} (need >= {MIN_EXTRACTION_SUCCESS:.0%}, broken run: 4%)",
        ),
        (
            "critique success rate",
            m["critique_success"] >= MIN_CRITIQUE_SUCCESS,
            f"{m['critique_success']:.1%} (need >= {MIN_CRITIQUE_SUCCESS:.0%}, broken run: 1.6%)",
        ),
        (
            "L3 entity share",
            m["l3_share"] >= MIN_L3_ENTITY_SHARE,
            f"{m['l3_share']:.1%} (need >= {MIN_L3_ENTITY_SHARE:.0%}, broken run: 1.7%)",
        ),
        (
            "total pathways",
            len(m["pathways"]) >= MIN_PATHWAYS,
            f"{len(m['pathways'])} (need >= {MIN_PATHWAYS}, broken run: 150)",
        ),
        (
            "FULL-completeness pathways",
            m["completeness"].get("FULL", 0) >= MIN_FULL_PATHWAYS,
            f"{m['completeness'].get('FULL', 0)} (need >= {MIN_FULL_PATHWAYS}, broken run: 0)",
        ),
        (
            "pathways with shared pivot",
            m["shared_pivot_pathways"] >= MIN_SHARED_PIVOT_PATHWAYS,
            f"{m['shared_pivot_pathways']} (need >= {MIN_SHARED_PIVOT_PATHWAYS}, broken run: 0)",
        ),
    ]


def build_validation_report(
    *,
    query: str = "respect for elders in Japan",
    top_k: int = 5,
) -> tuple[str, bool]:
    m = collect_metrics()
    gates = evaluate_gates(m)
    all_passed = all(passed for _n, passed, _d in gates)

    retriever = PathwayRetriever(config.PATHWAY_INDEX_DIR)
    hits = retriever.retrieve(query, top_k=top_k)

    state = m["state"]
    lines = [
        "# Phase 2 Validation Report",
        "",
        "## Quality gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for name, passed, detail in gates:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
    lines += [
        "",
        f"**Overall: {'PASS' if all_passed else 'FAIL'}**",
        "",
        "## Checkpoint state",
        f"- Extraction completed: {len(state.extraction_completed)} records "
        f"(failed: {len(state.extraction_failed)})",
        f"- Critique completed: {len(state.critique_completed)} records "
        f"(failed: {len(state.critique_failed)})",
        f"- Last updated: {state.updated_at or 'n/a'}",
        "",
        "## Triple stores",
        f"- Raw triples: {len(m['raw'])} "
        f"(generated ratio: {_generated_support_ratio(m['raw']):.1%})",
        f"- LLM-extracted triples: {m['llm_triples']}",
        f"- Schema-fallback triples (excluded from pathways): {m['fallback_triples']}",
        f"- Refined triples: {len(m['refined'])} "
        f"(generated ratio: {_generated_support_ratio(m['refined']):.1%})",
        f"- Critique rows: {len(m['critiques'])} "
        f"(parse failures: {m['critique_parse_failures']})",
        "",
        "## Entity resolution",
        f"- Canonical entities: {len(m['entities'])}",
        f"- By layer: {dict(m['entity_layers'])}",
        f"- Shared abstraction nodes: {m['shared_abstraction_entities']}",
        "",
        "## Cross-cultural alignment",
        f"- Alignments: {len(m['alignments'])}",
        f"- Verified: {sum(1 for a in m['alignments'] if a.get('verified'))}",
        "",
        "## Pathways",
        f"- Total pathways: {len(m['pathways'])}",
        f"- By completeness: {dict(m['completeness'])}",
        f"- With shared pivot: {m['shared_pivot_pathways']}",
        f"- Distinct cultures: {len(m['cultures'])}",
        f"- Top cultures: {dict(m['cultures'].most_common(10))}",
        "",
        f"## Sample retrieval (`{query}`)",
        "",
    ]
    if not hits:
        lines.append("- No pathway hits (index empty, or nothing cleared the relevance floor).")
    else:
        for pw, score in hits:
            node_labels = " -> ".join(n.label for n in pw.nodes)
            lines.append(
                f"- score={score:.3f} culture={pw.culture} "
                f"completeness={pw.completeness.value} path={node_labels}"
            )

    return "\n".join(lines) + "\n", all_passed


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validate Phase 2 artifacts.")
    p.add_argument("--query", default="respect for elders in Japan")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--output", default=str(config.PHASE2_VALIDATION_PATH))
    p.add_argument("--strict", action="store_true", default=False,
                   help="Exit non-zero if any quality gate fails.")
    args = p.parse_args(argv)

    report, passed = build_validation_report(query=args.query, top_k=args.top_k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    LOGGER.info("Wrote validation report -> %s", args.output)

    if not passed:
        LOGGER.error("One or more quality gates FAILED. See %s", args.output)
        if args.strict:
            sys.exit(1)
    else:
        LOGGER.info("All quality gates passed.")


if __name__ == "__main__":
    main()
