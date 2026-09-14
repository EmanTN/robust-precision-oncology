#!/usr/bin/env bash
set -euo pipefail

PRED_DIR="results/preds"
mkdir -p $PRED_DIR

while IFS=$'\t' read -r DATA METHOD MODEL OUTER; do
    echo "=== Running best_run for $DATA $METHOD $MODEL $OUTER ==="

    # For LOGO and phylo_style, we require a real OUTER value
    if [[ ( "$METHOD" == "LOGO" || "$METHOD" == "phylo_style" ) && ( -z "$OUTER" || "$OUTER" == "-" ) ]]; then
        echo ">>> Skipping $DATA $METHOD $MODEL: OUTER is empty or '-'."
        continue
    fi
    python3 finetune.py \
        --best_run \
        --only "$DATA" \
        --methods "$METHOD" \
        --models "$MODEL" \
        --outer "$OUTER" \
        --config ./config/config.yaml \
        --emit_preds_dir "$PRED_DIR"
done < tasks_0.tsv
#done < finetune_tasks.tsv