#!/bin/bash
# -------------------------------------------------------------------------------------
# Traditional-augmentation baselines (Reviewer 2 #8).
# Same protocol as the main table (paper + 5-fold + select=test) so numbers are directly
# comparable: does "real-only + strong augmentation" reach "real + synthetic (99.8)"?
#
# Run:  bash run_baselines.sh
# -------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

REAL=/root/autodl-tmp/data/images
PRE=/root/code1/train_vision
SYN=/root/autodl-tmp/data/synthetic/fold-1
MAN=$SYN/manifest.csv
COMMON="--protocol paper --n_splits 5 --backbone resnet50 --real_dir $REAL --pretrained_dir $PRE"

# --- reference rows (for the same table) ---
python classify_cv.py $COMMON --regimes real_only            --out_dir results_bl/real_only
python classify_cv.py $COMMON --regimes real_plus_synth \
       --synthetic_dir $SYN --synthetic_manifest $MAN        --out_dir results_bl/real_synth

# --- traditional-augmentation baselines (real_only + technique) ---
python classify_cv.py $COMMON --regimes real_only --aug randaugment --out_dir results_bl/randaugment
python classify_cv.py $COMMON --regimes real_only --aug mixup       --out_dir results_bl/mixup
python classify_cv.py $COMMON --regimes real_only --aug cutmix      --out_dir results_bl/cutmix
python classify_cv.py $COMMON --regimes real_only --sampler balanced --out_dir results_bl/balanced
python classify_cv.py $COMMON --regimes real_only --loss focal       --out_dir results_bl/focal

# --- strong combined augmentation baseline (real-only + strong aug) ---
python classify_cv.py $COMMON --regimes real_only \
       --aug randaugment --sampler balanced --loss focal   --out_dir results_bl/strong_aug

echo "Done. Summaries under results_bl/*/summary_paper_resnet50.json"
