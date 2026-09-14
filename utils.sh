#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-results/best_params}"
TASKS="${2:-finetune_tasks.tsv}"

[[ -d "$OUTDIR" ]] || { echo "ERR: OUTDIR not found: $OUTDIR" >&2; exit 1; }
[[ -f "$TASKS" ]] || { echo "ERR: TASKS file not found: $TASKS" >&2; exit 1; }

total="$(wc -l < "$TASKS" | awk '{print $1}')"
echo "Total expected jobs (lines in $TASKS): $total"

missing_file="tasks_missing_by_files.tsv"
: > "$missing_file"   # truncate

have=0
miss=0

while IFS=$'\t' read -r DATASET METHOD MODEL OUTER; do
  # Expected file names:
  # 1) with train-<outer>
  cand1="${OUTDIR}/${DATASET}_${METHOD}_${MODEL}_train-${OUTER}_best_params.csv"
  # 2) without train-<outer>
  cand2="${OUTDIR}/${DATASET}_${METHOD}_${MODEL}_best_params.csv"

  if [[ -f "$cand1" || -f "$cand2" ]]; then
    ((have++))
  else
    ((miss++))
    echo -e "${DATASET}\t${METHOD}\t${MODEL}\t${OUTER}" >> "$missing_file"
  fi
done < "$TASKS"

echo "Jobs with an existing best_params file: $have"
echo "Jobs with NO best_params file: $miss"
echo "Missing tasks written to: $missing_file"
