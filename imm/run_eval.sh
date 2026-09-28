#!/usr/bin/env bash
# Evaluate the IMM one-step checkpoint FID across 5 seeds.
#
# Usage (run from the repo root):
#   bash imm/run_eval.sh [NUM_GPUS] [extra imm/eval.py args...]
#
# Examples:
#   bash imm/run_eval.sh 8
#   bash imm/run_eval.sh 8 --checkpoint-path /path/to/imagenet256_ts_a2.pkl
#   bash imm/run_eval.sh 8 --cfg-scale 1.5 --num-steps 2

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_GPUS="${1:-1}"
shift 1 2>/dev/null || true

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$(shuf -i 10000-65000 -n 1)" \
    "$REPO_ROOT/imm/eval.py" \
    --checkpoint-key imm \
    --download-missing \
    --num-images 50000 \
    --seeds "0,1,2,3,4" \
    "$@"
