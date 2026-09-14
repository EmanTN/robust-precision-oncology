#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --output=logs/ft_%A_%a.out
#SBATCH --error=logs/ft_%A_%a.err
## SBATCH --array=0-19
#SBATCH --array=0-591%20
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3-00:00
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
# TASKS="remain_fine.tsv"

# Activate conda directly and portably:
CONDA_BASE="/nfs/software001/home/software-r8-x86_64/spack-20230328/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/anaconda3-2022.05-auh4o3tsby7ze6q6v3stn2hhvvnpoy5f"
# Initialize conda for this non-interactive shell
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate phylo

# Read the line for this array index
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS" || true)
if [[ -z "$LINE" ]]; then
  echo "No task for index $SLURM_ARRAY_TASK_ID"; exit 0
fi

# Split on any whitespace (tabs or spaces)
read -r DATASET METHOD MODEL OUTER <<<"$LINE"

echo "Running dataset=$DATASET method=$METHOD model=$MODEL outer=$OUTER"

CMD=(python3 finetune.py --config "$CONFIG" --only "$DATASET" --methods "$METHOD" --models "$MODEL" --emit_preds_dir "$PRED_DIR")
# Only pass --outer when using phylo_style and OUTER is not a placeholder
if [[ ( "$METHOD" == "phylo_style" || "$METHOD" == "LOGO" )  && -n "$OUTER" && "$OUTER" != "-" ]]; then
  CMD+=(--outer "$OUTER")
fi

"${CMD[@]}"
