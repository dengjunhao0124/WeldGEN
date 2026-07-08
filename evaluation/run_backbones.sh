#!/bin/bash
# -------------------------------------------------------------------------------------
# Run the remaining two backbones (MobileNetV2, GoogLeNet) to complete the multi-backbone
# table, under BOTH protocols, real_only vs real_plus_synth. (ResNet50 already running.)
#
# Run:  bash run_backbones.sh
# -------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

REAL=/root/autodl-tmp/data/images
PRE=/root/code1/train_vision
SYN=/root/autodl-tmp/data/synthetic/fold-1
MAN=$SYN/manifest.csv

for BB in mobilenet_v2 googlenet; do
    echo "############## $BB ##############"

    # --- main table: paper protocol + 5-fold (select=test) ---
    python classify_cv.py --protocol paper --n_splits 5 --backbone "$BB" \
        --real_dir "$REAL" --pretrained_dir "$PRE" \
        --synthetic_dir "$SYN" --synthetic_manifest "$MAN" \
        --regimes real_only real_plus_synth \
        --out_dir "results_paper5_$BB"

    # --- appendix: honest protocol (5-fold + validation-based selection) ---
    python classify_cv.py --protocol honest --select val --backbone "$BB" \
        --real_dir "$REAL" --pretrained_dir "$PRE" \
        --synthetic_dir "$SYN" --synthetic_manifest "$MAN" \
        --regimes real_only real_plus_synth \
        --out_dir "results_honest_$BB"
done

echo "Done. Summaries under results_paper5_<bb>/ and results_honest_<bb>/"
