# Food/Hospitality Multi-Culture Next-Step Summary

## Files created locally

- `food_hospitality_multiculture_records_keep_only.jsonl`: 77 audited keep records for KG construction.
- `food_hospitality_multiculture_records_keep_plus_maybe.jsonl`: 128 keep+maybe records for optional recall-oriented testing.
- `food_hospitality_multiculture_maybe_review.csv`: 51 maybe rows separated for manual review.
- `food_hospitality_multiculture_keep_only_counts.csv`: per-culture coverage counts after audit.
- `scripts/11_make_keep_only_multiculture_subset.py`: reproducible local script to regenerate these files inside the repo after the audit file has been filled.

## Audit status counts

| Status | Count |
|---|---:|
| keep | 77 |
| maybe | 51 |
| remove | 68 |
| total | 196 |

## Keep-only culture coverage

| Culture | Original candidates | Keep | Maybe | Keep + maybe | Recommended use |
|---|---:|---:|---:|---:|---|
| south korea | 15 | 8 | 7 | 15 | usable keep-only coverage |
| iran | 3 | 0 | 0 | 0 | no usable coverage from current run |
| bangladesh | 1 | 0 | 0 | 0 | no usable coverage from current run |
| spain | 23 | 4 | 0 | 4 | low keep-only coverage; usable with caution |
| indonesia | 23 | 5 | 11 | 16 | usable keep-only coverage |
| zimbabwe | 0 | 0 | 0 | 0 | no usable coverage from current run |
| netherlands | 30 | 15 | 9 | 24 | usable keep-only coverage |
| new zealand | 5 | 2 | 1 | 3 | low keep-only coverage; usable with caution |
| turkey | 16 | 5 | 3 | 8 | usable keep-only coverage |
| china | 15 | 7 | 4 | 11 | usable keep-only coverage |
| pakistan | 5 | 2 | 1 | 3 | low keep-only coverage; usable with caution |
| south africa | 1 | 0 | 1 | 1 | maybe-only coverage; manually inspect before use |
| japan | 33 | 13 | 4 | 17 | usable keep-only coverage |
| philippines | 14 | 7 | 3 | 10 | usable keep-only coverage |
| russia | 15 | 8 | 3 | 11 | usable keep-only coverage |
| brazil | 7 | 1 | 4 | 5 | low keep-only coverage; usable with caution |

## Recommended next move

Use `food_hospitality_multiculture_records_keep_only.jsonl` for the first local KG smoke test after regenerating it from the completed audit CSV. Do not include maybe rows in the main run yet. Keep the keep+maybe file only as an optional sensitivity check if culture coverage is too low.

The low/no-coverage cultures remain Iran, Bangladesh, Zimbabwe, South Africa, Pakistan, New Zealand, Brazil, and Spain. These should be handled by either expanding CultureBank selection, lowering thresholds cautiously, or excluding those CulturalBench questions from the next culture-matched evaluation until coverage improves.
