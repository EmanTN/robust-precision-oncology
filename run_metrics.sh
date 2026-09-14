#!/bin/bash

outdir=$1
input_pattern=$2
mode=${3:-baseline}      # baseline or phylo
logo_scope=${4:-outer}  # "outer" or "all"
mkdir -p "$outdir"

for f in $input_pattern; do
    echo "Processing $f"
    python3 measure_metrics.py \
        --file "$f" \
        --mode "$mode" \
        --outdir "$outdir" \
        --logo_eval_scope "$logo_scope"
done
