"""Check local real-case grounded-match regression output.

This verifies the final accepted (question_id, instance_id) pairs against the
manually specified expectations in tests/fixtures/grounded_real_cases_expected.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Check grounded real-case regression output.")
    p.add_argument("--matches", required=True)
    p.add_argument(
        "--expected",
        default="tests/fixtures/grounded_real_cases_expected.json",
    )
    args = p.parse_args(argv)

    rows = load_jsonl(Path(args.matches))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))

    actual_pairs = sorted(
        (str(row.get("question_id")), str(row.get("instance_id")))
        for row in rows
    )
    expected_pairs = sorted(
        (str(row["question_id"]), str(row["instance_id"]))
        for row in expected["expected_pairs"]
    )

    actual_units = sorted({q for q, _iid in actual_pairs})
    expected_units = sorted(str(x) for x in expected["expected_supported_units"])
    rejected_units = sorted(str(x) for x in expected["expected_rejected_units"])

    problems = []
    if actual_pairs != expected_pairs:
        problems.append(
            f"Final pair mismatch. expected={expected_pairs}, actual={actual_pairs}"
        )
    if actual_units != expected_units:
        problems.append(
            f"Supported-unit mismatch. expected={expected_units}, actual={actual_units}"
        )

    unexpected_rejected_matches = sorted(set(actual_units) & set(rejected_units))
    if unexpected_rejected_matches:
        problems.append(
            "Units expected to be rejected were accepted: "
            + ", ".join(unexpected_rejected_matches)
        )

    print("===== GROUNDED REAL-CASE REGRESSION CHECK =====")
    print(f"Expected pairs: {expected_pairs}")
    print(f"Actual pairs:   {actual_pairs}")
    print(f"Expected supported units: {expected_units}")
    print(f"Actual supported units:   {actual_units}")

    if problems:
        print("\nFAILED")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("\nPASSED")


if __name__ == "__main__":
    main()
