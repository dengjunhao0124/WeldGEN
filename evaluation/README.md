# Evaluation & reproduction

Scripts to reproduce the classification, baseline, generative-quality, and memorization results.
Dependencies are in the top-level `requirements.txt`.

## Data generation
- `make_folds.py` — build the generation data directories from the real images, deriving binary
  weld masks on the fly. `--unified` builds one set from all images; otherwise it builds per-fold
  sets. Outputs DreamBooth instance data, a ControlNet `metadata.jsonl`, and per-class base/mask pools.
- `generate_fold.py` — load the trained LoRA + ControlNet and generate synthetic images, writing a
  `manifest.csv` that maps every synthetic image to the real base image it was generated from.
  `--mult 5` produces 5 images per real image (paper setting); `--fold -1` writes a flat set.

## Classification
- `classify_cv.py` — stratified k-fold classification (ResNet50 / MobileNetV2 / GoogLeNet). Reports
  weighted P/R/F1, macro-F1, balanced accuracy, and per-class recall (mean ± std), and saves the
  per-sample out-of-fold predictions (`oof_*.csv`) and split files. Two protocols:
  `--protocol paper` (matches the original code) and `--protocol honest` (validation-based epoch
  selection). Synthetic images are filtered by the manifest so none derived from a fold's test
  images enter its training set.
- `significance_test.py` — McNemar exact test + paired bootstrap from two `oof_*.csv` files.
- `run_backbones.sh` — run all three backbones (real vs real+synthetic).
- `run_baselines.sh` — standard-augmentation baselines (RandAugment / MixUp / CutMix /
  class-balanced sampling / focal loss) plus the real-only and real+synthetic references.

## Generative evaluation
- `fid_kid.py` — FID via [torch-fidelity] and KID via [clean-fid] (the same libraries as the
  paper), overall and per-class, for one or more synthetic sets (e.g. diffusion vs VAE).
- `nn_memorization.py` — nearest-neighbour analysis in ResNet-50 feature space to verify the
  synthetic images are novel rather than near-duplicates of the training data.
- `vae_baseline.py` — a conditional-VAE generator (`--mode train` / `--mode generate`) used as a
  generative baseline under the identical evaluation.

## Plotting
- `plot_loss.py` — ControlNet training loss curve. `replot_confusion.py` — confusion matrices.

## Optional: per-fold (leakage-free) SLURM pipeline
`run_perfold_pipeline.sh`, `gen_fold_array.sh`, `submit_all.sh`, `classify_after.sh`, and
`train_controlnet_only.sh` orchestrate the stricter per-fold protocol (a separate generator is
trained per fold on that fold's training images). This is the gold standard for leakage control
but 5× the generation cost; the manifest-filtered unified generation above is the default.

`RESULTS.md` records the reported numbers.
