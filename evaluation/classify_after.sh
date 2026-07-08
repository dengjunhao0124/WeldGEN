#!/bin/bash
#SBATCH --job-name=weld_cls
#SBATCH --output=logs/classify_%j.log
#SBATCH --error=logs/classify_%j.log
#SBATCH --partition=gpu          # <-- EDIT
#SBATCH --gres=gpu:1             # <-- EDIT
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# -------------------------------------------------------------------------------------
# Final stage: 5-fold classification CV (consuming per-fold synthetic sets) +
# significance tests, for all three backbones.
# Runs after gen_fold_array.sh finishes all folds (submit_all.sh wires the dependency).
# -------------------------------------------------------------------------------------
set -euo pipefail

# ---- cluster module loads (uncomment if needed) ----
# module load cuda/11.8 anaconda3 && source activate weldgen
# ----------------------------------------------------

cd "$(dirname "$0")"                 # -> code/eval
mkdir -p logs
REAL_DIR="/root/autodl-tmp/data/images"
SYN_DIR="/root/autodl-tmp/data/synthetic"

for BB in resnet50 mobilenet_v2 googlenet; do
    echo "==================== $BB ===================="
    OUT="results/$BB"
    python classify_cv.py --backbone "$BB" \
        --real_dir "$REAL_DIR" --synthetic_dir "$SYN_DIR" \
        --regimes real_only real_plus_synth synth_only \
        --n_splits 5 --epochs 50 --batch_size 16 --img_size 512 \
        --out_dir "$OUT"

    echo "-------- significance: $BB real_only vs real_plus_synth --------"
    python significance_test.py \
        --baseline "$OUT/oof_${BB}_real_only.csv" \
        --method   "$OUT/oof_${BB}_real_plus_synth.csv" \
        --metric macro_f1 --n_boot 10000 \
        | tee "$OUT/significance_${BB}.txt"
done

echo "All done. Results under code/eval/results/<backbone>/"
