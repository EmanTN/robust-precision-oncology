#!/usr/bin/env bash
set -euo pipefail

MAX_CONC=24                         # cluster limit
JOB_NAME="embed"                   # must match #SBATCH --job-name in sbatch file
# JOB_NAME="feed_${EXTERNAL_DATASET}"

# Accept either 3 or 5 arguments
if [[ $# -ne 3 && $# -ne 5 ]]; then
  echo "Usage:"
  echo "  $0 <TASKS.tsv> <sbatch_file> <pred_dir>"
  echo "  $0 <TASKS.tsv> <sbatch_file> <pred_dir> --external_test <external_dataset>"
  exit 1
fi

TASKS="$1"
SBATCH_FILE="$2"
PRED_DIR="$3"

EXTRA_ARGS=()
if [[ $# -eq 5 ]]; then
  # Expect: --external_test ValidTest
  if [[ "$4" != "--external_test" ]]; then
    echo "ERROR: 4th arg must be --external_test"
    exit 1
  fi
  EXTRA_ARGS=(--external_test --external_dataset "$5")
fi

total=$(wc -l < "$TASKS")
echo "Total tasks: $total ; max concurrent: $MAX_CONC"

i=0
while (( i < total )); do
  # how many of OUR jobs (by name) are queued or running?
# Retry squeue safely (avoid breaking when empty)
  active=$(squeue -h -u "$USER" -n "$JOB_NAME" -t PD,R,CF,CG 2>/dev/null | wc -l)
  active=${active:-0}
  echo "active = $active"

  if (( active < MAX_CONC )); then
    # submit exactly ONE task index i (array i-i)
    echo "Submitting task index $i"

    sbatch --array=${i}-${i} --export=ALL,TASKS="$TASKS",PRED_DIR="$PRED_DIR",EXTRA_ARGS="${EXTRA_ARGS[*]}" \
      "$SBATCH_FILE" || echo "⚠️ Submission failed for $i"
      ((i=i+1))
    sleep 2
  else
    # wait until some jobs finish
    echo "Reached max concurrent ($active). Waiting..."
    sleep 30
  fi
done

echo "All tasks submitted."
