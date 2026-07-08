#!/bin/bash
# -------------------------------------------------------------------------------------
# Train ONLY ControlNet (skip DreamBooth), reusing each fold's existing db_out.
# Then plot the loss curve and generate a few sample images for a quick quality check.
#
# Use this after you already have work/fold{k}/db_out from a previous run and only
# need to re-train ControlNet (e.g. after lowering the learning rate).
#
# Run (default fold 2 only):
#   bash train_controlnet_only.sh
# Pick folds / lr / steps via env:
#   FOLDS="0 1 2 3 4" CN_LR=1e-5 CN_STEPS=8000 bash train_controlnet_only.sh
# -------------------------------------------------------------------------------------
set -euo pipefail

# ---- HF: offline-friendly (model already cached/local) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# module load ... && source activate weldgen   # uncomment if needed

cd "$(dirname "$0")"                 # -> code/eval
CODE_DIR="../generation"
WORK="/root/autodl-tmp/work"
SYN_DIR="/root/autodl-tmp/data/synthetic_test"
# Local SD path (full model incl. model_index.json). Falls back to the hub id.
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/sd15}"
[ -d "$BASE_MODEL" ] || BASE_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"

FOLDS="${FOLDS:-2}"
CN_LR="${CN_LR:-1e-5}"
CN_STEPS="${CN_STEPS:-8000}"
N_PER_CLASS="${N_PER_CLASS:-3}"      # small: just for visual quality check

clean_ckpt() { find "$1" -name .ipynb_checkpoints -type d -prune -exec rm -rf {} + 2>/dev/null || true; }

for k in $FOLDS; do
    FD="$WORK/fold$k"
    echo "############## ControlNet-only | FOLD $k ##############"
    [ -d "$FD/db_out" ] || { echo "ERROR: $FD/db_out missing (train DreamBooth first)"; exit 1; }
    clean_ckpt "$FD"
    rm -rf "$FD/cn_out"              # fresh ControlNet run

    DREAMBOOTH_LORA="$FD/db_out" \
    accelerate launch --mixed_precision=bf16 "$CODE_DIR/train_controlnet_i2i_dream.py" \
        --pretrained_model_name_or_path="$BASE_MODEL" \
        --train_data_dir="$FD/cn_data" \
        --output_dir="$FD/cn_out" \
        --image_column=image --caption_column=caption \
        --conditioning_image_column=conditioning_image \
        --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=4 \
        --max_train_steps="$CN_STEPS" --learning_rate="$CN_LR" --seed=42 --allow_tf32 \
        --checkpointing_steps=2000 --checkpoints_total_limit=4

    # loss curve -> work/fold{k}/cn_out/loss_curve.png
    python plot_loss.py --csv "$FD/cn_out/loss_log.csv" || echo "[warn] plot skipped"

    # a few sample images for visual check -> SYN_DIR/fold{k}/
    python generate_fold.py --fold "$k" --base_model "$BASE_MODEL" \
        --lora "$FD/db_out" --controlnet "$FD/cn_out" \
        --basepool "$FD/basepool" --maskpool "$FD/maskpool" \
        --out_dir "$SYN_DIR" --n_per_class "$N_PER_CLASS"
done

echo "Done. Check loss_curve.png and samples under $SYN_DIR/"
