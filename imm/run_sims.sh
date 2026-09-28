#!/usr/bin/env bash
# SIMS-style guidance with a finetuned IMM model (theta_s from imm/finetune.sh), then FID.
#
# Usage (from the repo root):
#   bash imm/run_sims.sh <theta_s.pkl> <sims_w> <fid_out.txt> [SEED] [NUM_GPUS]
set -euo pipefail

AUX="$(realpath "$1")"
W="$2"
OUT="$(realpath -m "$3")"
SEED="${4:-0}"
NGPUS="${5:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/checkpoints/imagenet256_ts_a2.pkl}"
FID_STATS="${FID_STATS:-$REPO_ROOT/fid_stats/adm_in256_stats.npz}"
NEON_DIR="$REPO_ROOT/third_party/Neon"

if [[ ! -d "$NEON_DIR" ]]; then
    git clone https://github.com/VITA-Group/Neon.git "$NEON_DIR"
    git -C "$NEON_DIR" checkout 754f12ba7cf4f6a1e72827848e09cea9a0bd14ae
    git -C "$NEON_DIR" apply "$REPO_ROOT/imm/neon_imm.patch"
fi
cp "$REPO_ROOT/imm/sims.py" "$NEON_DIR/imm/sims.py"

cd "$NEON_DIR/imm"
mkdir -p "$(dirname "$OUT")"
torchrun --nproc_per_node="$NGPUS" sims.py \
    --config-name=im256_generate_images.yaml \
    eval.resume="$(realpath "$BASE_CKPT")" \
    eval.seed="$SEED" \
    +aux_resume="$AUX" +sims_w="$W" \
    +per_class_count=50 +cfg_scale=1.5 \
    +preset=1_steps_cfg1.5_pushforward_uniform \
    +fid_stats="$(realpath "$FID_STATS")" \
    +fid_out="$OUT"
