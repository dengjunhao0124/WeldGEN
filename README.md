# WeldGen: Structure-Controlled Diffusion for Data-Efficient Weld Image Synthesis and Inspection

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19557253-blue)](https://doi.org/10.5281/zenodo.19557253)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-green)](https://drive.google.com/drive/folders/1olkdHOuQDYZwWHIHhifn_WLpUeKVYcXv?usp=drive_link)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

Official implementation of **"WeldGen: Structure-Controlled Diffusion for Data-Efficient Weld Image Synthesis and Inspection"** (*The Visual Computer*).

> WeldGen is a semantic–structural decoupled diffusion augmentation framework for low-resource
> industrial weld inspection. Stage 1 fine-tunes Stable Diffusion v1.5 with **DreamBooth + LoRA**
> to learn a weld concept token; Stage 2 uses **ControlNet** with **random intra-class mask
> sampling** to synthesize structurally diverse weld images (text prompts encode class identity,
> binary masks encode geometry). Adding the synthetic images improves downstream weld-condition
> classification under 5-fold cross-validation from **93.2% to 99.8%** weighted-F1.

---

## Repository structure

```
code/
├── generation/                 # generative pipeline (Stable Diffusion v1.5)
│   ├── train_dreambooth_lora.py    # Stage 1: DreamBooth + LoRA concept adaptation
│   ├── train_controlnet_i2i_dream.py # Stage 2: ControlNet structure conditioning
│   ├── run.py                      # convenience launcher for the two training stages
│   ├── infer.py                    # generate synthetic weld images
│   └── test_examples_utils.py
├── evaluation/                 # evaluation, baselines, and reproduction scripts
│   ├── make_folds.py               # build (per-fold or unified) generation data dirs + masks
│   ├── generate_fold.py            # generate synthetic images + manifest (leakage-traceable)
│   ├── classify_cv.py              # k-fold classification (paper / honest protocols)
│   ├── significance_test.py        # McNemar + paired bootstrap
│   ├── nn_memorization.py          # nearest-neighbour / memorization analysis
│   ├── fid_kid.py                  # FID (torch-fidelity) + KID (clean-fid), per-class
│   ├── vae_baseline.py             # conditional-VAE generative baseline
│   ├── plot_loss.py, replot_confusion.py
│   ├── run_backbones.sh, run_baselines.sh  # one-click multi-backbone / augmentation baselines
│   ├── run_perfold_pipeline.sh, ...        # optional per-fold (leakage-free) SLURM pipeline
│   ├── RESULTS.md                  # summary of reported numbers
│   └── README.md
├── requirements.txt
└── LICENSE
```

The dataset (images + masks + generated samples + train/test splits) is released separately on
[Google Drive](https://drive.google.com/drive/folders/1olkdHOuQDYZwWHIHhifn_WLpUeKVYcXv?usp=drive_link);
place it under `data/`.

---

## Installation

**Hardware:** a CUDA-capable GPU (tested on NVIDIA RTX A5500, 24 GB). **Software:** Python ≥ 3.10,
PyTorch with CUDA.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Download the Stable Diffusion v1.5 weights from
[Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

---

## Dataset — Busbar-Weld-8

Eight busbar laser-welding conditions:

| Folder | Condition | Folder | Condition |
|--------|-----------|--------|-----------|
| `baseline` | nominal factory settings | `n-de` | negative defocus |
| `low power` | laser power below nominal | `oil` | oil-contaminated surface |
| `low gap` | gap < 0.5 mm | `water` | water-contaminated surface |
| `p-de` | positive defocus | `cold weld` | failed fusion joint |

```
data/
├── images/          # background-removed weld images, one sub-folder per condition
└── masks/           # binary weld-region masks (same filename as the image)
```

---

## 1. Generation

Run from `generation/`.

**Stage 1 — DreamBooth + LoRA** (learn the weld concept token `w*`):

```bash
accelerate launch --mixed_precision=bf16 train_dreambooth_lora.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5 \
  --instance_data_dir=../data/images \
  --output_dir=run/sd15-dream_lora \
  --instance_prompt="an image of w*" \
  --resolution=512 --num_train_epochs=50 --train_batch_size=8 \
  --learning_rate=1e-4 --lr_scheduler=constant --train_text_encoder --seed=42 --allow_tf32
```

**Stage 2 — ControlNet** (structure conditioning with masks):

```bash
accelerate launch --mixed_precision=bf16 train_controlnet_i2i_dream.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5 \
  --train_data_dir=../data/cn_data \
  --output_dir=run/sd15-controlnet \
  --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=4 \
  --max_train_steps=8000 --learning_rate=1e-5 --seed=42 --allow_tf32
```

**Generate** synthetic images (`infer.py`, or `evaluation/generate_fold.py` for a manifest-traceable
5× expansion). Key inference hyperparameters: sampler Euler, 50 steps, image-to-image strength 0.6,
guidance scale 3.0, ControlNet conditioning scale 1.0.

---

## 2. Evaluation & reproduction

Run from `evaluation/`. All classification runs use stratified k-fold cross-validation; synthetic
images are traceable to their base real image through a released **manifest**, so no test-derived
synthetic image enters training (see the "Data-split protocol and leakage prevention" section of
the paper).

```bash
# main result: 5-fold CV, real vs real+synthetic, all three backbones
bash run_backbones.sh

# standard-augmentation baselines (RandAugment / MixUp / CutMix / balanced / focal) + references
bash run_baselines.sh

# statistical significance (McNemar + paired bootstrap)
python significance_test.py --baseline <oof_real_only.csv> --method <oof_real_plus_synth.csv>

# generation quality: FID (torch-fidelity) + KID (clean-fid), overall and per-class
python fid_kid.py --real_dir ../data/images --synthetic_dirs ../data/synthetic --names diffusion --per_class

# memorization check (nearest-neighbour in ResNet-50 feature space)
python nn_memorization.py --real_dir ../data/images --synthetic_dir ../data/synthetic --manifest ../data/synthetic/manifest.csv

# VAE generative baseline (train -> generate -> classify)
python vae_baseline.py --mode train --real_dir ../data/images --ckpt vae.pt
python vae_baseline.py --mode generate --real_dir ../data/images --ckpt vae.pt --out_dir ../data/synthetic_vae
```

See [`evaluation/README.md`](evaluation/README.md) and [`evaluation/RESULTS.md`](evaluation/RESULTS.md)
for details and the reported numbers.

---

## Results

Weld-condition classification, 5-fold stratified cross-validation, weighted-F1 (mean ± std):

| Backbone | Real only | Real + Synthetic |
|----------|-----------|------------------|
| ResNet50 | 93.2 ± 1.5 | **99.8 ± 0.3** |
| MobileNetV2 | 93.2 ± 2.8 | **99.3 ± 0.6** |
| GoogLeNet | 94.5 ± 1.6 | **99.7 ± 0.4** |

The improvement is statistically significant (McNemar p ≈ 2e-11; paired-bootstrap 95% CIs exclude
zero) and far exceeds standard augmentations (≤ 95.1% w-F1) and a conditional-VAE generator
(94.0% w-F1; FID 293 vs 31 for diffusion).

---

## Citation

```bibtex
@article{zhang2025weldgen,
  title   = {WeldGen: Structure-Controlled Diffusion for Data-Efficient Weld Image Synthesis and Inspection},
  author  = {Zhang, Qin and Deng, Junhao and Zhao, Zhongyou and Song, Zhelong and
             Wang, Zhenmin and Wang, Hui-ping and Wan, Zixuan and Arinez, Jorge and Li, Guangze},
  journal = {The Visual Computer},
  year    = {2025},
  doi     = {10.5281/zenodo.19557253}
}
```

## Acknowledgements

The generation scripts build upon the [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
DreamBooth and ControlNet examples. FID/KID use [torch-fidelity](https://github.com/toshas/torch-fidelity)
and [clean-fid](https://github.com/GaParmar/clean-fid). Supported by the National Natural Science
Foundation of China (U23A20625, U2141216, 52375334) and the Natural Science Foundation of Guangdong
Province (2023B1515250003).

## License

[Apache License 2.0](LICENSE).
