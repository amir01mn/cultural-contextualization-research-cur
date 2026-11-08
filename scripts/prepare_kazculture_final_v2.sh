#!/bin/bash
# Seed the v2 output dirs with the v1b S0/S1 predictions, S1 retrieval cache and
# reproduction records (deterministic, reproduced exactly in v1). Refuses to
# overwrite an existing seeded file.
set -euo pipefail
B=data/kazculture_pipeline/evaluation
seed() {  # src_dir dst_dir files...
  local src=$1 dst=$2; shift 2
  for f in "$@"; do
    [[ -e "$src/$f" ]] || { echo "missing $src/$f" >&2; exit 2; }
    [[ -e "$dst/$f" ]] && { echo "exists: $dst/$f (kept)"; continue; }
    mkdir -p "$(dirname "$dst/$f")"; cp -p "$src/$f" "$dst/$f"; echo "seeded $dst/$f"
  done
}
seed $B/final_v1b $B/final_v2 closed_book/S0.jsonl closed_book/S0_summary.json \
  strict/S1.jsonl strict/S1_summary.json full/S1.jsonl full/S1_summary.json \
  retrieval/strict_S1.jsonl retrieval/full_S1.jsonl \
  reproduction_config.json reproduction_closed_book_S0.json reproduction_strict_S1.json \
  reproduction_full_S1.json reproduction_index_strict.json reproduction_index_full.json
seed $B/final_v1b_olmo $B/final_v2_olmo closed_book/S0.jsonl closed_book/S0_summary.json \
  strict/S1.jsonl strict/S1_summary.json retrieval/strict_S1.jsonl \
  reproduction_closed_book_S0.json reproduction_strict_S1.json
