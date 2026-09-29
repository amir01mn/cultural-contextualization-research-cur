"""Build a grounded CulturalBench-Easy retrieval set and controlled CultureBank corpus.

This is a deterministic bridge between grounding review and retrieval evaluation.

Inputs:
  - normalized CulturalBench-Easy JSONL
  - normalized full CultureBank JSONL
  - reviewed (question_id, instance_id) pair decisions

Outputs:
  - one grounded question row per supported CulturalBench item
  - a fixed-size CultureBank retrieval corpus that always includes all approved
    support records plus hard same-culture distractors and global distractors
  - a selection manifest and provenance metadata

The bundled decision file is intentionally marked AI-reviewed/provisional. It
must not be described as human-verified until a human reviewer confirms it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval_ranking import culture_match_score  # noqa: E402
from src.utils import read_jsonl, write_jsonl  # noqa: E402


FACT_GROUP_OVERRIDES = {
    # Same underlying Japanese home shoe-removal practice/reason.
    "easy_1126": "fact_japan_home_shoe_removal",
    "easy_214": "fact_japan_home_shoe_removal",
    "easy_806": "fact_japan_home_shoe_removal",
    "easy_847": "fact_japan_home_shoe_removal",
    # Same underlying Dutch direct-communication convention.
    "easy_140": "fact_dutch_direct_communication",
    "easy_917": "fact_dutch_direct_communication",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rank(seed: int, namespace: str, instance_id: str) -> str:
    payload = f"{seed}|{namespace}|{instance_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fact_group_id(question_id: str) -> str:
    return FACT_GROUP_OVERRIDES.get(question_id, f"fact_{question_id}")


def load_kept_pairs(path: Path) -> tuple[dict[str, list[dict]], set[str]]:
    """Return kept decisions grouped by question plus all review status labels."""
    by_question: dict[str, list[dict]] = defaultdict(list)
    statuses: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = str(row.get("review_status") or row.get("manual_status") or "").strip()
            statuses.add(status)
            if not status.lower().startswith("keep"):
                continue
            qid = str(row.get("question_id") or "").strip()
            iid = str(row.get("instance_id") or "").strip()
            if not qid or not iid:
                raise ValueError(f"Kept row is missing question_id/instance_id: {row}")
            by_question[qid].append(row)
    return dict(by_question), statuses


def _culture_matches(record: dict, question: dict) -> bool:
    return (
        culture_match_score(
            str(record.get("cultural_group") or ""),
            str(question.get("cultural_group") or ""),
        )
        >= 0.9
    )


def _take_ranked(
    candidates: list[dict],
    *,
    n: int,
    seed: int,
    namespace: str,
    selected_ids: set[str],
) -> list[dict]:
    if n <= 0:
        return []
    eligible = [
        r
        for r in candidates
        if str(r.get("instance_id") or "") not in selected_ids
    ]
    eligible.sort(
        key=lambda r: stable_rank(
            seed, namespace, str(r.get("instance_id") or "")
        )
    )
    chosen = eligible[:n]
    selected_ids.update(str(r.get("instance_id") or "") for r in chosen)
    return chosen


def build_grounded_rows(
    questions: list[dict],
    records_by_id: dict[str, dict],
    kept_by_question: dict[str, list[dict]],
) -> list[dict]:
    questions_by_id = {
        str(q.get("question_id") or ""): q
        for q in questions
        if q.get("question_id")
    }
    rows: list[dict] = []

    for qid in sorted(kept_by_question):
        q = questions_by_id.get(qid)
        if q is None:
            raise ValueError(f"Reviewed question_id not found in questions file: {qid}")

        support_ids: list[str] = []
        support_records: list[dict] = []
        review_rows: list[dict] = []
        for decision in kept_by_question[qid]:
            iid = str(decision["instance_id"])
            rec = records_by_id.get(iid)
            if rec is None:
                raise ValueError(f"Reviewed instance_id not found in CultureBank: {iid}")
            if not _culture_matches(rec, q):
                raise ValueError(
                    f"Reviewed support pair no longer culture-matches: {qid} / {iid}"
                )
            if iid not in support_ids:
                support_ids.append(iid)
                support_records.append(
                    {
                        "instance_id": iid,
                        "cultural_group": rec.get("cultural_group"),
                        "topic": rec.get("topic"),
                        "source_text": rec.get("source_text"),
                    }
                )
            review_rows.append(
                {
                    "instance_id": iid,
                    "review_status": (
                        decision.get("review_status")
                        or decision.get("manual_status")
                        or ""
                    ),
                    "review_notes": (
                        decision.get("review_notes")
                        or decision.get("manual_notes")
                        or ""
                    ),
                }
            )

        out = dict(q)
        out["fact_group_id"] = fact_group_id(qid)
        out["support_instance_ids"] = support_ids
        out["support_records"] = support_records
        out["grounding_reviews"] = review_rows
        rows.append(out)

    return rows


def select_controlled_corpus(
    grounded_rows: list[dict],
    records: list[dict],
    *,
    corpus_size: int,
    seed: int,
    same_topic_per_question: int,
    same_culture_per_question: int,
) -> tuple[list[dict], list[dict]]:
    if corpus_size <= 0:
        raise ValueError("--corpus_size must be positive")

    records_by_id = {
        str(r.get("instance_id") or ""): r
        for r in records
        if r.get("instance_id")
    }

    support_ids = {
        iid
        for q in grounded_rows
        for iid in q.get("support_instance_ids", [])
    }
    missing = sorted(support_ids - records_by_id.keys())
    if missing:
        raise ValueError(f"Support IDs missing from CultureBank: {missing}")

    if len(support_ids) > corpus_size:
        raise ValueError(
            f"{len(support_ids)} mandatory support records exceed corpus_size={corpus_size}"
        )

    selected_ids = set(support_ids)
    roles: dict[str, set[str]] = defaultdict(set)
    for iid in support_ids:
        roles[iid].add("verified_support")

    # Add hard distractors per grounded question. We use the exact culture matcher
    # from the retrieval pipeline so aliases/demonyms are handled consistently.
    for q in grounded_rows:
        qid = str(q["question_id"])
        support_topics = {
            str(records_by_id[iid].get("topic") or "").strip().lower()
            for iid in q["support_instance_ids"]
            if records_by_id[iid].get("topic")
        }

        culture_pool = [r for r in records if _culture_matches(r, q)]
        same_topic = [
            r
            for r in culture_pool
            if str(r.get("topic") or "").strip().lower() in support_topics
        ]
        chosen = _take_ranked(
            same_topic,
            n=same_topic_per_question,
            seed=seed,
            namespace=f"{qid}:same_topic",
            selected_ids=selected_ids,
        )
        for r in chosen:
            roles[str(r["instance_id"])].add(f"same_culture_same_topic:{qid}")

        chosen = _take_ranked(
            culture_pool,
            n=same_culture_per_question,
            seed=seed,
            namespace=f"{qid}:same_culture",
            selected_ids=selected_ids,
        )
        for r in chosen:
            roles[str(r["instance_id"])].add(f"same_culture:{qid}")

    # Fill the remaining budget from the full corpus deterministically.
    remaining = corpus_size - len(selected_ids)
    if remaining > 0:
        chosen = _take_ranked(
            records,
            n=remaining,
            seed=seed,
            namespace="global_fill",
            selected_ids=selected_ids,
        )
        for r in chosen:
            roles[str(r["instance_id"])].add("global_fill")

    if len(selected_ids) != corpus_size:
        raise ValueError(
            f"Could only select {len(selected_ids)} unique records for "
            f"requested corpus_size={corpus_size}"
        )

    # Preserve original CultureBank order for stable downstream processing.
    selected_records = [
        r for r in records if str(r.get("instance_id") or "") in selected_ids
    ]
    manifest = [
        {
            "instance_id": str(r.get("instance_id") or ""),
            "cultural_group": r.get("cultural_group"),
            "topic": r.get("topic"),
            "selection_roles": sorted(roles[str(r.get("instance_id") or "")]),
        }
        for r in selected_records
    ]
    return selected_records, manifest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Build grounded CulturalBench-Easy set and controlled CultureBank corpus."
    )
    p.add_argument("--questions_jsonl", required=True)
    p.add_argument("--records_jsonl", required=True)
    p.add_argument(
        "--pair_decisions",
        default="data/controlled/culturalbench_easy_grounded_ai_review_v1.csv",
    )
    p.add_argument("--output_grounded", required=True)
    p.add_argument("--output_corpus", required=True)
    p.add_argument("--output_manifest", required=True)
    p.add_argument("--output_meta", required=True)
    p.add_argument("--corpus_size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--same_topic_per_question", type=int, default=10)
    p.add_argument("--same_culture_per_question", type=int, default=10)
    args = p.parse_args(argv)

    questions_path = Path(args.questions_jsonl)
    records_path = Path(args.records_jsonl)
    decisions_path = Path(args.pair_decisions)

    questions = list(read_jsonl(questions_path))
    records = list(read_jsonl(records_path))
    records_by_id = {
        str(r.get("instance_id") or ""): r
        for r in records
        if r.get("instance_id")
    }
    kept_by_question, review_statuses = load_kept_pairs(decisions_path)
    if not kept_by_question:
        raise ValueError(f"No kept review pairs found in {decisions_path}")

    grounded = build_grounded_rows(
        questions,
        records_by_id,
        kept_by_question,
    )
    controlled, manifest = select_controlled_corpus(
        grounded,
        records,
        corpus_size=args.corpus_size,
        seed=args.seed,
        same_topic_per_question=args.same_topic_per_question,
        same_culture_per_question=args.same_culture_per_question,
    )

    write_jsonl(args.output_grounded, grounded)
    write_jsonl(args.output_corpus, controlled)
    write_jsonl(args.output_manifest, manifest)

    unique_support_ids = sorted(
        {
            iid
            for q in grounded
            for iid in q.get("support_instance_ids", [])
        }
    )
    fact_groups = sorted({q["fact_group_id"] for q in grounded})
    meta = {
        "grounded_question_count": len(grounded),
        "grounded_fact_group_count": len(fact_groups),
        "approved_pair_count": sum(
            len(q.get("support_instance_ids", [])) for q in grounded
        ),
        "unique_support_record_count": len(unique_support_ids),
        "controlled_corpus_size": len(controlled),
        "selection_seed": args.seed,
        "same_topic_per_question": args.same_topic_per_question,
        "same_culture_per_question": args.same_culture_per_question,
        "review_statuses": sorted(review_statuses),
        "human_verified": bool(review_statuses)
        and all("human" in s.lower() for s in review_statuses if s),
        "questions_sha256": sha256_file(questions_path),
        "records_sha256": sha256_file(records_path),
        "pair_decisions_sha256": sha256_file(decisions_path),
        "grounded_output": str(Path(args.output_grounded)),
        "controlled_corpus_output": str(Path(args.output_corpus)),
        "manifest_output": str(Path(args.output_manifest)),
        "support_instance_ids": unique_support_ids,
        "fact_group_ids": fact_groups,
        "note": (
            "Evidence judgments are lower-bound qrels: unlabelled records may also "
            "support a question. The bundled decisions are AI-reviewed/provisional "
            "until a human reviewer confirms them."
        ),
    }
    meta_path = Path(args.output_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n===== GROUNDED EASY RETRIEVAL SET =====\n")
    print(f"Grounded questions: {len(grounded)}")
    print(f"Fact groups: {len(fact_groups)}")
    print(f"Approved support pairs: {meta['approved_pair_count']}")
    print(f"Unique support records: {len(unique_support_ids)}")
    print(f"Controlled CultureBank records: {len(controlled)}")
    print(f"Human verified: {meta['human_verified']}")
    print(f"Grounded JSONL: {args.output_grounded}")
    print(f"Corpus JSONL: {args.output_corpus}")
    print(f"Manifest JSONL: {args.output_manifest}")
    print(f"Metadata JSON: {args.output_meta}")


if __name__ == "__main__":
    main()
