#!/bin/bash
#SBATCH --job-name=weld_gen
#SBATCH --output=logs/gen_fold_%A_%a.log
#SBATCH --error=logs/gen_fold_%A_%a.log
#SBATCH --partition=gpu          # <-- EDIT
#SBATCH --gres=gpu:1             # <-- EDIT: 1 GPU per fold
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00          # per-fold: DreamBooth + ControlNet + generation
#SBATCH --array=0-4              # one task per fold (5 folds run in parallel)
# -------------------------------------------------------------------------------------
# JOB ARRAY: each task k builds the leakage-free synthetic set for fold k.
#   Stage 1  DreamBooth + LoRA   on fold-k train masks
#   Stage 2  ControlNet          on fold-k train images+masks
#   Stage 3  generate synthetic       -> data/synthetic/fold{k}/
#
# Prerequisite: `python make_folds.py` already ran (creates work/fold{k}/).
# submit_all.sh handles that ordering for you.
# -------------------------------------------------------------------------------------
set -euo pipefail

# ---- HF download: mirror + disable xet (fixes autodl 401) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.hf_cache}"
# ---- cluster module loads (uncomment if needed) ----
# module load cuda/11.8 anaconda3 && source activate weldgen
# ----------------------------------------------------

cd "$(dirname "$0")"                 # -> code/eval
mkdir -p logs
CODE_DIR="../generation"
BASE_MODEL="${BASE_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"  # auto-downloads
SYN_DIR="/root/autodl-tmp/data/synthetic"
WORK="/root/autodl-tmp/work"
N_PER_CLASS="${N_PER_CLASS:-60}"

k="${SLURM_ARRAY_TASK_ID:-0}"
FD="$WORK/fold$k"
echo "############## FOLD $k on $(hostname) ##############"
[ -d "$FD" ] || { echo "ERROR: $FD missing. Run make_folds.py first."; exit 1; }

# Stage 1: DreamBooth + LoRA
accelerate launch --mixed_precision=bf16 "$CODE_DIR/train_dreambooth_lora.py" \
    --pretrained_model_name_or_path="$BASE_MODEL" \
    --instance_data_dir="$FD/db_instance" \
    --output_dir="$FD/db_out" \
    --instance_prompt="an image of w*" \
    --resolution=512 --train_batch_size=8 --num_train_epochs=50 \
    --learning_rate=1e-4 --lr_scheduler=constant --train_text_encoder \
    --seed=42 --allow_tf32

# Stage 2: ControlNet (loads this fold's DreamBooth LoRA via $DREAMBOOTH_LORA)
DREAMBOOTH_LORA="$FD/db_out" \
accelerate launch --mixed_precision=bf16 "$CODE_DIR/train_controlnet_i2i_dream.py" \
    --pretrained_model_name_or_path="$BASE_MODEL" \
    --train_data_dir="$FD/cn_data" \
    --output_dir="$FD/cn_out" \
    --image_column=image --caption_column=caption \
    --conditioning_image_column=conditioning_image \
    --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=4 \
    --max_train_steps=8000 --learning_rate="${CN_LR:-1e-5}" --seed=42 --allow_tf32 \
    --checkpointing_steps=1000 --checkpoints_total_limit=2

# Stage 3: generate synthetic for this fold
python generate_fold.py --fold "$k" \
    --base_model "$BASE_MODEL" \
    --lora "$FD/db_out" --controlnet "$FD/cn_out" \
    --basepool "$FD/basepool" --maskpool "$FD/maskpool" \
    --out_dir "$SYN_DIR" --n_per_class "$N_PER_CLASS"

echo "Fold $k done -> $SYN_DIR/fold$k/"
