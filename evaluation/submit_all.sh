#!/bin/bash
# -------------------------------------------------------------------------------------
# One-command submission of the full leakage-free per-fold pipeline as SLURM jobs.
#
#   1. make_folds.py            (fast, CPU-only: runs here on the login node)
#   2. gen_fold_array.sh        (job array 0-4: 5 folds generate IN PARALLEL)
#   3. classify_after.sh        (runs only after ALL folds finish: afterok dependency)
#
# Usage:
#   bash submit_all.sh
#
# Tip: set N_PER_CLASS / BASE_MODEL as env vars to override defaults, e.g.
#   N_PER_CLASS=80 bash submit_all.sh
# -------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

REAL_DIR="/root/autodl-tmp/data/images"
WORK="/root/autodl-tmp/work"

echo "[1/3] Building per-fold data dirs (make_folds.py; masks derived from images) ..."
python make_folds.py --real_dir "$REAL_DIR" --work_dir "$WORK" \
    --n_splits 5 --seed 42

echo "[2/3] Submitting generation job array (5 folds in parallel) ..."
ARRAY_ID=$(sbatch --parsable gen_fold_array.sh)
echo "      array job id: $ARRAY_ID"

echo "[3/3] Submitting classification job (after all folds succeed) ..."
CLS_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" classify_after.sh)
echo "      classify job id: $CLS_ID"

echo ""
echo "Submitted. Monitor with:  squeue -u \$USER"
echo "  generation : $ARRAY_ID  (tasks _0 .. _4)"
echo "  classify   : $CLS_ID    (starts after generation finishes)"
