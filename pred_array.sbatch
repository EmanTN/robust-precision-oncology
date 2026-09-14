#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --output=logs/pred_%A_%a.out
#SBATCH --error=logs/pred_%A_%a.err
## SBATCH --array=0-19
#SBATCH --array=0-663%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00
#SBATCH --partition=pi_mghassem

set -euo pipefail
mkdir -p logs

# DO NOT PROVIDE DEFAULTS
if [[ -z "${TASKS:-}" || -z "${PRED_DIR:-}" ]]; then
    echo "ERROR: TASKS and PRED_DIR MUST be exported via sbatch"
    exit 1
fi

# CONFIG="./config/config.yaml"
CONFIG="${CONFIG:-./config/config.yaml}"

# Activate conda directly and portably:
CONDA_BASE="/nfs/software001/home/software-r8-x86_64/spack-20230328/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/anaconda3-2022.05-auh4o3tsby7ze6q6v3stn2hhvvnpoy5f"
# Initialize conda for this non-interactive shell
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate phylo

# Read line (SLURM_ARRAY_TASK_ID is 0-based, sed is 1-based)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS" || true)
if [[ -z "$LINE" ]]; then
  echo "No task for index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

# Skip commented lines if you have any
if [[ "$LINE" =~ ^# ]]; then
  echo "Comment line at index $SLURM_ARRAY_TASK_ID – skipping"
  exit 0
fi

# TSV: DATA  METHOD  MODEL  OUTER
IFS=$'\t' read -r DATA METHOD MODEL OUTER <<<"$LINE"

echo "=== Running best_run for DATA=$DATA METHOD=$METHOD MODEL=$MODEL OUTER=$OUTER ==="

# For LOGO and phylo_style, require a real OUTER value
if [[ ( "$METHOD" == "LOGO" || "$METHOD" == "phylo_style" ) && ( -z "$OUTER" || "$OUTER" == "-" ) ]]; then
  echo ">>> Skipping $DATA $METHOD $MODEL: OUTER is empty or '-'."
  exit 0
fi

CMD=(python3 finetune.py
    --best_run
    --only "$DATA"
    --methods "$METHOD"
    --models "$MODEL"
    --config "$CONFIG"
    --emit_preds_dir "$PRED_DIR"
)

# Only add --outer if non-empty and not "-"
if [[ -n "$OUTER" && "$OUTER" != "-" ]]; then
  CMD+=(--outer "$OUTER")
fi

# Optional external test args passed from feeder
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=(${EXTRA_ARGS})
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"