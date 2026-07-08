#!/bin/bash
#SBATCH --job-name=weld_perfold
#SBATCH --output=weld_perfold_%j.log
#SBATCH --error=weld_perfold_%j.log
#SBATCH --partition=gpu          # <-- EDIT (ignored if run with plain `bash`)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# -------------------------------------------------------------------------------------
# LEAKAGE-FREE per-fold pipeline (Reviewer 1 #1 / Reviewer 2 #6, gold standard).
# SEQUENTIAL version — for a single GPU / no SLURM. Run with:
#     nohup bash run_perfold_pipeline.sh > perfold.log 2>&1 &
#
# For each of 5 stratified folds, using ONLY that fold's training data:
#   1. DreamBooth + LoRA  (concept)    on fold-train weld images
#   2. ControlNet         (structure)  on fold-train images + derived masks
#   3. generate synthetic                   -> /root/autodl-tmp/data/synthetic/fold{k}/
# Then a single 5-fold classifier run + significance tests.
#
# Set QUICK=1 to smoke-test the whole chain on fold 0 only with tiny training.
#   QUICK=1 bash run_perfold_pipeline.sh
# -------------------------------------------------------------------------------------
set -euo pipefail

# ---- HF download: use mirror + disable xet (fixes autodl 401 from xethub) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Keep the HF cache on the data disk, not the home quota:
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.hf_cache}"
# ---- cluster module loads (uncomment if needed) ----
# module load cuda/11.8 anaconda3 && source activate weldgen
# ----------------------------------------------------

cd "$(dirname "$0")"                 # -> code/eval
CODE_DIR="../generation"
DATA_ROOT="/root/autodl-tmp/data"
REAL_DIR="$DATA_ROOT/images"
SYN_DIR="$DATA_ROOT/synthetic"
WORK="/root/autodl-tmp/work"
BASE_MODEL="${BASE_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"   # auto-downloads
N_SPLITS=5
MULT="${MULT:-5}"        # paper mode: generate MULT synthetic images per real image (5x)
CN_LR="${CN_LR:-1e-5}"   # ControlNet lr (8e-4 diverges on small data; 1e-5 is standard)

# QUICK smoke-test knobs
if [ "${QUICK:-0}" = "1" ]; then
    FOLDS="0"; DB_EPOCHS=1; CN_STEPS=20; MULT=1; EPOCHS=2
    echo "*** QUICK mode: fold0 only, tiny training ***"
else
    FOLDS=$(seq 0 $((N_SPLITS-1))); DB_EPOCHS=50; CN_STEPS=8000; EPOCHS=50
fi

# Remove JupyterLab .ipynb_checkpoints dirs that break image dataset loaders
clean_ckpt() { find "$1" -name .ipynb_checkpoints -type d -prune -exec rm -rf {} + 2>/dev/null || true; }

# Stage 0: per-fold data dirs (masks derived by binarising images; seg_raw NOT used)
python make_folds.py --real_dir "$REAL_DIR" --work_dir "$WORK" \
        --n_splits "$N_SPLITS" --seed 42
clean_ckpt "$WORK"

for k in $FOLDS; do
    echo "############## FOLD $k ##############"
    FD="$WORK/fold$k"
    clean_ckpt "$FD"   # guard again in case Jupyter recreated them

    accelerate launch --mixed_precision=bf16 "$CODE_DIR/train_dreambooth_lora.py" \
        --pretrained_model_name_or_path="$BASE_MODEL" \
        --instance_data_dir="$FD/db_instance" \
        --output_dir="$FD/db_out" \
        --instance_prompt="an image of w*" \
        --resolution=512 --train_batch_size=8 --num_train_epochs=$DB_EPOCHS \
        --learning_rate=1e-4 --lr_scheduler=constant --train_text_encoder \
        --seed=42 --allow_tf32

    # ControlNet loads this fold's DreamBooth LoRA (script reads $DREAMBOOTH_LORA)
    DREAMBOOTH_LORA="$FD/db_out" \
    accelerate launch --mixed_precision=bf16 "$CODE_DIR/train_controlnet_i2i_dream.py" \
        --pretrained_model_name_or_path="$BASE_MODEL" \
        --train_data_dir="$FD/cn_data" \
        --output_dir="$FD/cn_out" \
        --image_column=image --caption_column=caption \
        --conditioning_image_column=conditioning_image \
        --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=4 \
        --max_train_steps=$CN_STEPS --learning_rate=$CN_LR --seed=42 --allow_tf32 \
        --checkpointing_steps=1000 --checkpoints_total_limit=2

    python generate_fold.py --fold "$k" \
        --base_model "$BASE_MODEL" \
        --lora "$FD/db_out" --controlnet "$FD/cn_out" \
        --basepool "$FD/basepool" --maskpool "$FD/maskpool" \
        --out_dir "$SYN_DIR" --mult "$MULT"
done

# Stage 4: classification CV + significance
for BB in resnet50 mobilenet_v2 googlenet; do
    OUT="results/$BB"
    python classify_cv.py --backbone "$BB" \
        --real_dir "$REAL_DIR" --synthetic_dir "$SYN_DIR" \
        --regimes real_only real_plus_synth synth_only \
        --n_splits "$N_SPLITS" --epochs $EPOCHS --batch_size 16 --img_size 512 \
        --out_dir "$OUT"
    python significance_test.py \
        --baseline "$OUT/oof_${BB}_real_only.csv" \
        --method   "$OUT/oof_${BB}_real_plus_synth.csv" \
        --metric macro_f1 --n_boot 10000 | tee "$OUT/significance_${BB}.txt"
done

echo "Pipeline complete. Synthetic -> $SYN_DIR ; results -> code/eval/results/"
