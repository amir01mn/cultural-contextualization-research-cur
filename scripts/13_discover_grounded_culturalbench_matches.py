"""Discover strictly grounded CultureBank <-> CulturalBench evidence matches.

This diagnostic starts from the controlled CulturalBench items and exhaustively
checks every target-culture CultureBank record. A record is accepted only when:

1. a first-pass verifier says the record directly supports every essential
   element of the gold target;
2. every cited support quote is copied from the CultureBank source_text;
3. an independent stricter adjudication pass reaches the same conclusion.

The script deliberately has no PARTIAL category. Its goal is to construct a
high-precision grounded subset for retrieval evaluation, not to measure topical
relatedness.

Both verifier stages are resumable JSONL streams so the full-corpus run can be
continued after a scheduler timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.llm_backend import get_backend  # noqa: E402
from src.retrieval_ranking import culture_match_score  # noqa: E402
from src.schemas import MCQQuestion, MCQVariant  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    normalize_text,
    read_jsonl,
    stable_id,
    truncate,
    write_jsonl,
)

LOGGER = get_logger("13_discover_grounded_matches")

STAGE1_SYSTEM = """You are performing strict source-grounding verification.

You receive ONE CulturalBench gold target and ONE CultureBank source record.
Decide whether this single CultureBank record, by itself, directly supports the
entire gold target.

Do not use outside knowledge, stereotypes, definitions, common sense, or facts
not literally stated or clearly entailed by the record.

Before deciding:
1. Decompose the gold target into every essential factual element. Include
   qualifiers and causal/reason relations from BOTH the question and answer.
2. Check each element separately against the record.
3. MATCH is allowed only if EVERY essential element is supported.
4. A related culture, topic, value, event, behavior, or one half of a
   multi-part answer is NO_MATCH.
5. For a hard item whose gold judgment is FALSE, MATCH requires the record to
   directly contradict or refute the candidate statement. Mere absence is
   NO_MATCH.
6. For each supported element, quote exact text copied from SOURCE_TEXT.
7. If an element has no exact supporting quote, mark it unsupported.
8. Return JSON only.

Return exactly:
{
  "label": "MATCH | NO_MATCH",
  "element_checks": [
    {
      "element": "one essential factual element",
      "supported": true,
      "quote": "exact contiguous text copied from SOURCE_TEXT, or empty string"
    }
  ],
  "rationale": "brief source-grounded explanation",
  "confidence": 0.0
}
"""

STAGE2_SYSTEM = """You are the adversarial completeness auditor for a
source-grounded benchmark match.

Stage 1 already decomposed the CulturalBench gold target, checked each element
against SOURCE_TEXT, and supplied exact source quotes. Your job is NOT to redo
that support judgment. Instead, audit whether Stage 1 omitted any genuinely
MATERIAL factual requirement from the gold target.

Rules:
1. Read the PREVIOUS VERIFIER ELEMENT CHECKS exactly as given.
2. Identify any genuinely missing MATERIAL factual requirement from the gold
   target and list it in missing_material_elements.
3. Material omissions include facts/qualifiers that change the claim, such as:
   - "beneficial for health";
   - a causal/reason relation such as "to show respect";
   - one clause of a multi-clause answer such as "elders pay for the meal";
   - a named practice/event/ritual;
   - a definition required to connect the source phrase to the gold answer.
4. Do NOT list a requirement if it is already represented by a Stage-1 element,
   even when the wording differs.
5. Do NOT treat grammatical/question words such as "why", "what", or "how" as
   missing material elements.
6. Do NOT require literal "common", "customary", "typically", or "usually"
   wording when Stage 1 already represents the cultural behavior in context.
7. Do NOT create duplicate rephrasings of prior elements.
8. For hard FALSE items, Stage 1 already required direct contradiction/refutation;
   do not weaken that rule.
9. Do not use outside cultural knowledge or definitions.
10. You may optionally list a Stage-1 element index in
    unsupported_prior_element_indices if you think its semantic support deserves
    human review. These indices are DIAGNOSTIC ONLY and are not the completeness
    decision.
11. Return JSON only.

Return exactly:
{
  "unsupported_prior_element_indices": [],
  "missing_material_elements": [],
  "rationale": "brief explanation",
  "confidence": 0.0
}
"""


FORMAT_RETRY_SUFFIX = """

OUTPUT FORMAT RETRY:
Your previous response did not contain a complete parseable JSON object.
Return the final verdict again as ONE compact JSON object.
The FIRST character must be { and the LAST character must be }.
Do not include analysis, markdown, or prose outside the JSON.
Keep the rationale under 25 words.
"""

def load_questions(path: Path) -> list[MCQQuestion]:
    return [MCQQuestion(**row) for row in read_jsonl(path)]


def gold_target(q: MCQQuestion) -> str:
    if q.variant == MCQVariant.EASY:
        answer = q.options.get(q.answer_key or "", "")
        return (
            f"Question: {q.question}\n"
            f"Gold answer: {q.answer_key}. {answer}\n"
            "The source must support the answer as the answer to this exact "
            "question, including all qualifiers in the question."
        )

    judgment = "TRUE" if q.answer_bool else "FALSE"
    extra = (
        "The source must directly support this candidate statement."
        if q.answer_bool
        else "The source must directly contradict/refute this candidate statement; absence is insufficient."
    )
    return (
        f"Question: {q.question}\n"
        f"Candidate statement: {q.option_text or ''}\n"
        f"Gold judgment: {judgment}\n"
        f"{extra}"
    )


def record_source_text(rec: dict) -> str:
    return normalize_text(rec.get("source_text"))


def pair_prompt(
    q: MCQQuestion,
    rec: dict,
    *,
    stage2: bool = False,
    prior_verdict: Optional[dict] = None,
) -> str:
    source = record_source_text(rec)
    header = "FINAL ADJUDICATION" if stage2 else "FIRST-PASS VERIFICATION"
    prior_block = ""
    if stage2 and prior_verdict is not None:
        prior_block = (
            "\nPREVIOUS VERIFIER ELEMENT CHECKS:\n"
            + json.dumps(
                {
                    "label": prior_verdict.get("label"),
                    "element_checks": prior_verdict.get("element_checks") or [],
                    "rationale": prior_verdict.get("rationale") or "",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    return f"""{header}

TARGET CULTURE:
{q.cultural_group or 'unknown'}

GOLD TARGET:
{gold_target(q)}
{prior_block}
CULTUREBANK RECORD:
instance_id: {rec.get('instance_id', '')}
culture: {rec.get('cultural_group') or 'unknown'}
topic: {rec.get('topic') or 'unknown'}
context: {rec.get('context') or ''}
actor_behavior: {rec.get('actor_behavior') or ''}
recipient_behavior: {rec.get('recipient_behavior') or ''}

SOURCE_TEXT:
{source}

Does SOURCE_TEXT directly support every material factual element of the gold target?
"""


def parse_verdict(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    obj = json.loads(cleaned[start:end + 1])

    label = str(obj.get("label", "")).strip().upper()
    if label not in {"MATCH", "NO_MATCH"}:
        raise ValueError(f"Invalid label: {label!r}")

    checks = obj.get("element_checks")
    if not isinstance(checks, list) or not checks:
        checks = []

    normalized_checks = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        normalized_checks.append({
            "element": normalize_text(check.get("element")),
            "supported": bool(check.get("supported")),
            "quote": normalize_text(check.get("quote")),
        })

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "label": label,
        "element_checks": normalized_checks,
        "rationale": normalize_text(obj.get("rationale")),
        "confidence": confidence,
    }


def parse_stage2_review(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    obj = json.loads(cleaned[start:end + 1])

    raw_indices = obj.get("unsupported_prior_element_indices") or []
    if not isinstance(raw_indices, list):
        raw_indices = []
    indices = []
    for value in raw_indices:
        try:
            indices.append(int(value))
        except (TypeError, ValueError):
            continue
    indices = sorted(set(indices))

    missing = obj.get("missing_material_elements") or []
    if not isinstance(missing, list):
        missing = []
    missing = [normalize_text(x) for x in missing if normalize_text(x)]

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "unsupported_prior_element_indices": indices,
        "missing_material_elements": missing,
        "rationale": normalize_text(obj.get("rationale")),
        "confidence": confidence,
    }


def _is_non_material_stage2_missing(text: str) -> bool:
    """Ignore reviewer complaints that are only question grammar/norm wording."""
    normalized = normalize_text(text).casefold()
    tokens = {
        token.strip(".,:;!?()[]{}'\"")
        for token in normalized.split()
        if token.strip(".,:;!?()[]{}'\"")
    }
    non_material_tokens = {
        "why", "what", "how", "it", "is", "a", "an", "the",
        "common", "customary", "typically", "usually", "usual",
        "practice", "behavior", "behaviour",
    }
    return bool(tokens) and tokens.issubset(non_material_tokens)


def _informative_tokens(text: str) -> set[str]:
    normalized = normalize_text(text).casefold()
    stop = {
        "a", "an", "the", "to", "of", "for", "in", "on", "at", "by",
        "and", "or", "is", "are", "was", "were", "be", "being", "been",
        "it", "its", "their", "his", "her", "one", "ones",
    }
    out = set()
    for token in normalized.split():
        cleaned = token.strip(".,:;!?()[]{}'\"")
        if cleaned and cleaned not in stop:
            out.add(cleaned)
    return out


def _missing_already_covered_by_prior(text: str, prior_checks: list[dict]) -> bool:
    missing_tokens = _informative_tokens(text)
    if not missing_tokens:
        return True
    for check in prior_checks:
        prior_tokens = _informative_tokens(check.get("element", ""))
        if missing_tokens.issubset(prior_tokens):
            return True
    return False


def validate_stage2_review(review: dict, prior_verdict: Optional[dict]) -> dict:
    prior = prior_verdict or {}
    prior_checks = prior.get("element_checks") or []
    problems: list[str] = []

    if not prior.get("accepted"):
        problems.append("prior stage-1 verdict was not accepted")

    valid_indices = []
    for idx in review.get("unsupported_prior_element_indices") or []:
        if 0 <= idx < len(prior_checks):
            valid_indices.append(idx)
        else:
            problems.append(f"invalid prior element index {idx}")

    missing = [
        item
        for item in (review.get("missing_material_elements") or [])
        if not _is_non_material_stage2_missing(item)
        and not _missing_already_covered_by_prior(item, prior_checks)
    ]
    if missing:
        problems.append("missing material elements: " + "; ".join(missing))

    accepted = bool(prior.get("accepted")) and not missing and not problems

    return {
        "label": "MATCH" if accepted else "NO_MATCH",
        "element_checks": prior_checks,
        "accepted": accepted,
        "validation_problems": problems,
        "rationale": review.get("rationale", ""),
        "confidence": float(review.get("confidence", 0.0)),
        "stage2_review": {
            "diagnostic_unsupported_prior_element_indices": valid_indices,
            "missing_material_elements": missing,
        },
    }


def quote_is_in_source(quote: str, source_text: str) -> bool:
    q = normalize_text(quote).casefold()
    s = normalize_text(source_text).casefold()
    return bool(q) and q in s


def validate_verdict(verdict: dict, source_text: str) -> dict:
    """Enforce structural grounding constraints on an LLM verdict."""
    checks = verdict.get("element_checks") or []
    problems: list[str] = []

    if not checks:
        problems.append("no element checks")

    for i, check in enumerate(checks):
        if not check.get("supported"):
            problems.append(f"element {i + 1} unsupported")
            continue
        quote = check.get("quote", "")
        if not quote_is_in_source(quote, source_text):
            problems.append(f"element {i + 1} quote not found in source_text")

    accepted = verdict.get("label") == "MATCH" and not problems
    return {
        **verdict,
        "accepted": accepted,
        "validation_problems": problems,
    }


def load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for row in read_jsonl(path):
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            rows[pair_id] = row
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()


def culture_records_for_question(q: MCQQuestion, records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        if culture_match_score(
            str(rec.get("cultural_group") or ""),
            q.cultural_group,
        ) >= 0.9:
            out.append(rec)
    return out


def run_stage(
    *,
    stage: int,
    pairs: list[tuple[MCQQuestion, dict]],
    backend,
    state_path: Path,
    batch_size: int,
    prior_state: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    state = load_state(state_path)
    system_prompt = STAGE1_SYSTEM if stage == 1 else STAGE2_SYSTEM

    pending = []
    for q, rec in pairs:
        pair_id = stable_id(q.question_id, rec.get("instance_id"), prefix="gm_")
        if pair_id not in state:
            pending.append((pair_id, q, rec))

    LOGGER.info(
        "Stage %d: %d total pairs, %d already complete, %d pending.",
        stage, len(pairs), len(state), len(pending),
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompts = [
            pair_prompt(
                q,
                rec,
                stage2=(stage == 2),
                prior_verdict=(
                    (prior_state or {}).get(pair_id)
                    if stage == 2
                    else None
                ),
            )
            for pair_id, q, rec in batch
        ]
        outputs = backend.generate(
            prompts,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=512,
        )

        rows = []
        for (pair_id, q, rec), prompt, output in zip(batch, prompts, outputs):
            parse_error: Optional[str] = None
            retry_used = False
            initial_output = output

            prior_verdict = (
                (prior_state or {}).get(pair_id)
                if stage == 2
                else None
            )

            def _parse_current(raw: str) -> dict:
                return (
                    parse_stage2_review(raw)
                    if stage == 2
                    else parse_verdict(raw)
                )

            try:
                verdict = _parse_current(output)
            except Exception as first_exc:  # noqa: BLE001
                retry_used = True
                retry_prompt = prompt + FORMAT_RETRY_SUFFIX
                retry_output = backend.generate_one(
                    retry_prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=1024,
                )
                output = retry_output
                try:
                    verdict = _parse_current(output)
                except Exception as second_exc:  # noqa: BLE001
                    parse_error = (
                        f"initial parse failed: {first_exc}; "
                        f"retry parse failed: {second_exc}"
                    )
                    verdict = (
                        {
                            "unsupported_prior_element_indices": [],
                            "missing_material_elements": [
                                "stage-2 review remained unparseable"
                            ],
                            "rationale": (
                                "Stage-2 review remained unparseable after one format retry."
                            ),
                            "confidence": 0.0,
                        }
                        if stage == 2
                        else {
                            "label": "NO_MATCH",
                            "element_checks": [],
                            "rationale": (
                                "Verifier output remained unparseable after one format retry."
                            ),
                            "confidence": 0.0,
                        }
                    )

            validated = (
                validate_stage2_review(verdict, prior_verdict)
                if stage == 2
                else validate_verdict(verdict, record_source_text(rec))
            )
            row = {
                "pair_id": pair_id,
                "stage": stage,
                "question_id": q.question_id,
                "group_id": q.group_id,
                "variant": (
                    q.variant.value if hasattr(q.variant, "value") else str(q.variant)
                ),
                "cultural_group": q.cultural_group,
                "instance_id": rec.get("instance_id"),
                "record_cultural_group": rec.get("cultural_group"),
                "topic": rec.get("topic"),
                "accepted": validated["accepted"],
                "label": validated["label"],
                "element_checks": validated["element_checks"],
                "validation_problems": validated["validation_problems"],
                "rationale": validated["rationale"],
                "confidence": validated["confidence"],
                "stage2_review": validated.get("stage2_review"),
                "parse_error": parse_error,
                "retry_used": retry_used,
                "initial_raw_output": (
                    truncate(initial_output, 4000)
                    if retry_used
                    else None
                ),
                "raw_output": truncate(output, 4000),
            }
            rows.append(row)
            state[pair_id] = row

        append_jsonl(state_path, rows)
        LOGGER.info(
            "Stage %d progress: %d/%d pending pairs processed.",
            stage,
            min(start + len(batch), len(pending)),
            len(pending),
        )

    return state


def write_summary(
    path: Path,
    questions: list[MCQQuestion],
    records_by_id: dict[str, dict],
    accepted_by_question: dict[str, list[dict]],
    *,
    stage1_pairs: int,
    stage1_matches: int,
    stage2_matches: int,
    corpus_records: int,
) -> None:
    ensure_dir(path.parent)
    supported = [q for q in questions if accepted_by_question.get(q.question_id)]

    lines = [
        "# Strict CultureBank ↔ CulturalBench Grounded Matches",
        "",
        "A benchmark item is included only when one CultureBank record passes both "
        "the first-pass verifier and the adversarial adjudication pass, with "
        "exact support quotes validated against source_text.",
        "",
        f"- Full CultureBank records: {corpus_records}",
        f"- CulturalBench units checked: {len(questions)}",
        f"- Exhaustive target-culture record/question pairs: {stage1_pairs}",
        f"- Stage-1 accepted pairs: {stage1_matches}",
        f"- Final accepted pairs after adversarial review: {stage2_matches}",
        f"- Benchmark units with at least one final grounded record: {len(supported)}",
        "",
        "## Supported benchmark units",
        "",
        "| Question | Variant | Culture | Supporting CultureBank records |",
        "|---|---|---|---|",
    ]

    for q in questions:
        matches = accepted_by_question.get(q.question_id, [])
        if not matches:
            continue
        ids = ", ".join(m["instance_id"] for m in matches)
        variant = q.variant.value if hasattr(q.variant, "value") else str(q.variant)
        lines.append(
            f"| {q.question_id} | {variant} | {q.cultural_group or 'unknown'} | {ids} |"
        )

    lines += [
        "",
        "## Manual review details",
        "",
    ]

    for q in supported:
        lines += [
            f"### {q.question_id}",
            "",
            f"- Culture: {q.cultural_group or 'unknown'}",
            f"- Gold target: {gold_target(q).replace(chr(10), ' ')}",
            "",
        ]
        for match in accepted_by_question[q.question_id]:
            rec = records_by_id[match["instance_id"]]
            lines += [
                f"**{match['instance_id']}**",
                "",
                f"- Topic: {rec.get('topic') or 'unknown'}",
                f"- Source text: {record_source_text(rec)}",
                f"- Stage-2 rationale: {match.get('rationale') or ''}",
                "- Element checks:",
            ]
            for check in match.get("element_checks") or []:
                status = "supported" if check.get("supported") else "unsupported"
                lines.append(
                    f"  - {status}: {check.get('element') or ''} "
                    f"| quote: {check.get('quote') or ''}"
                )
            lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- This is a high-precision candidate discovery step, not final human ground truth.",
        "- Manually verify every accepted pair before using it as a grounded evaluation item.",
        "- Unsupported units should not be interpreted as culturally false; they only lack a verified CultureBank evidence match under this procedure.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Discover strict grounded CultureBank-CulturalBench matches."
    )
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument(
        "--questions_jsonl",
        default="data/controlled/food_hospitality_culturalbench_questions.jsonl",
    )
    p.add_argument("--records_jsonl", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--limit_questions", type=int, default=None)
    p.add_argument(
        "--stage1_state",
        default="data/outputs/grounded_match_stage1.jsonl",
    )
    p.add_argument(
        "--stage2_state",
        default="data/outputs/grounded_match_stage2.jsonl",
    )
    p.add_argument(
        "--output_matches",
        default="data/outputs/grounded_culturalbench_matches.jsonl",
    )
    p.add_argument(
        "--output_md",
        default="data/outputs/grounded_culturalbench_matches.md",
    )
    args = p.parse_args(argv)

    questions = load_questions(Path(args.questions_jsonl))
    if args.limit_questions is not None:
        questions = questions[: max(0, args.limit_questions)]
    records = list(read_jsonl(Path(args.records_jsonl)))
    records_by_id = {
        str(rec.get("instance_id")): rec
        for rec in records
        if rec.get("instance_id") is not None
    }

    pairs: list[tuple[MCQQuestion, dict]] = []
    per_question_candidate_counts: dict[str, int] = {}
    for q in questions:
        docs = culture_records_for_question(q, records)
        per_question_candidate_counts[q.question_id] = len(docs)
        pairs.extend((q, rec) for rec in docs)

    LOGGER.info(
        "Loaded %d questions and %d CultureBank records; %d exhaustive target-culture pairs.",
        len(questions), len(records), len(pairs),
    )

    backend = get_backend(args.provider, model=args.model)
    status = backend.status()
    if not status.get("available"):
        raise RuntimeError(f"Backend unavailable: {status.get('message')}")
    if hasattr(backend, "warmup"):
        backend.warmup()

    stage1 = run_stage(
        stage=1,
        pairs=pairs,
        backend=backend,
        state_path=Path(args.stage1_state),
        batch_size=args.batch_size,
    )

    provisional_pairs: list[tuple[MCQQuestion, dict]] = []
    q_by_id = {q.question_id: q for q in questions}
    for row in stage1.values():
        if not row.get("accepted"):
            continue
        q = q_by_id.get(row.get("question_id"))
        rec = records_by_id.get(str(row.get("instance_id")))
        if q is not None and rec is not None:
            provisional_pairs.append((q, rec))

    LOGGER.info(
        "Stage 1 accepted %d/%d record-question pairs.",
        len(provisional_pairs), len(pairs),
    )

    stage2 = run_stage(
        stage=2,
        pairs=provisional_pairs,
        backend=backend,
        state_path=Path(args.stage2_state),
        batch_size=args.batch_size,
        prior_state=stage1,
    )

    final_rows = [
        row for row in stage2.values()
        if row.get("accepted")
        and row.get("question_id") in q_by_id
        and str(row.get("instance_id")) in records_by_id
    ]
    final_rows.sort(key=lambda r: (r["question_id"], -float(r["confidence"]), r["instance_id"]))
    write_jsonl(args.output_matches, final_rows)

    accepted_by_question: dict[str, list[dict]] = defaultdict(list)
    for row in final_rows:
        accepted_by_question[row["question_id"]].append(row)

    write_summary(
        Path(args.output_md),
        questions,
        records_by_id,
        accepted_by_question,
        stage1_pairs=len(pairs),
        stage1_matches=len(provisional_pairs),
        stage2_matches=len(final_rows),
        corpus_records=len(records),
    )

    supported_units = len(accepted_by_question)
    print("\n===== STRICT GROUNDED MATCH DISCOVERY =====\n")
    print(f"CultureBank records: {len(records)}")
    print(f"CulturalBench units: {len(questions)}")
    print(f"Exhaustive target-culture pairs: {len(pairs)}")
    print(f"Stage-1 accepted pairs: {len(provisional_pairs)}")
    print(f"Final accepted pairs: {len(final_rows)}")
    print(f"Units with >=1 grounded record: {supported_units}")
    print("Target-culture candidates by question:")
    for q in questions:
        print(f"  {q.question_id}: {per_question_candidate_counts[q.question_id]}")
    print(f"Matches JSONL: {args.output_matches}")
    print(f"Manual-review Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
