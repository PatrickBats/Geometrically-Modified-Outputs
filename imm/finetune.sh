#!/usr/bin/env bash
# Finetune IMM on GMO samples with the IMM training code from Neon
# (https://github.com/VITA-Group/Neon), then Neon-merge with the base model.
#
# Usage (from the repo root):
#   bash imm/finetune.sh <png_dir> <out_dir> [NUM_GPUS]
#
# <png_dir> comes from imm/lmdb_to_png.py. The base checkpoint is
# checkpoints/imagenet256_ts_a2.pkl (see IMM, https://github.com/lumaai/imm).
set -euo pipefail

PNG_DIR="$(realpath "$1")"
OUT_DIR="$(realpath -m "$2")"
NGPUS="${3:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/checkpoints/imagenet256_ts_a2.pkl}"
TOTAL_TICKS="${TOTAL_TICKS:-60}"
NEON_DIR="$REPO_ROOT/third_party/Neon"

if [[ ! -d "$NEON_DIR" ]]; then
    git clone https://github.com/VITA-Group/Neon.git "$NEON_DIR"
    git -C "$NEON_DIR" checkout 754f12ba7cf4f6a1e72827848e09cea9a0bd14ae
    git -C "$NEON_DIR" apply "$REPO_ROOT/imm/neon_imm.patch"
fi

cd "$NEON_DIR/imm"
mkdir -p "$OUT_DIR"
python create_labels.py --input_dir "$PNG_DIR" --output_file "$PNG_DIR/dataset.json"
python dataset_tool.py encode --source="$PNG_DIR" --dest="$OUT_DIR/dataset.zip"

torchrun --nproc_per_node="$NGPUS" train.py \
    --config-name=im256.yaml \
    dataset.path="$OUT_DIR/dataset.zip" \
    outputdir="$OUT_DIR" \
    training.transfer="$BASE_CKPT" \
    training.total_ticks="$TOTAL_TICKS" \
    training.snapshot_ticks=1
