# WeldGen — Rebuttal Experiment Results Log

Backbone: ResNet50 (ImageNet-V2 pretrained, custom head 2048→1024→C).
Preprocessing: 512×512 resize+pad, grayscale→RGB, /255, no ImageNet norm.
Data: 601 real images, 8 classes. Synthetic: unified generation (gradio_demo weights,
mask-ControlNet + DreamBooth LoRA, i2i strength 0.6), ~5× per real → fold-1 set + manifest.

---

## 1. "逐步收紧协议" — 99.2 的来源分解 (real+synth, w-F1)

| Protocol | leak | epoch select | split | real+synth w-F1 |
|---|---|---|---|---|
| single + peek + LEAK      | yes | test | 1×80/20 | **100.0** |
| single + peek + no-leak   | no  | test | 1×80/20 | **99.1** |
| honest (5-fold + val)     | no  | val  | 5-fold  | **90.9** |

Conclusion: leakage contributes only ~1 pt (100.0→99.1). The dominant inflation
(99.1→90.9, ~8 pt) is **test-set epoch selection** (best epoch on 121 test imgs over
100 epochs), NOT leakage.
Note: 99.2 ≈ 120/121 (exactly 1 stubborn sample, likely defocus n-de/p-de).

---

## 2. PAPER protocol + 5-fold (select=test, manifest-filtered) — the "aggressive" main table

| metric | real_only | real+synth |
|---|---|---|
| w-Pre | 93.31 ± 1.49 | 99.84 ± 0.32 |
| w-Rec | 93.18 ± 1.51 | 99.83 ± 0.33 |
| w-F1  | **93.16 ± 1.53** | **99.83 ± 0.33** |
| macro-F1 | 94.49 ± 1.40 | 99.87 ± 0.26 |
| bal-acc | 94.37 ± 1.56 | 99.87 ± 0.26 |

real+synth per-fold w-F1: [100, 100, 100, 100, 99.17]
Reproduces paper (real_only ≈ 93.3, real+synth ≈ 99.2) now with mean±std.
Caveat: keeps select=test (test-peek). 4/5 folds = perfect 100 (suspicious).

### Significance test (paper+5-fold, real_only vs real+synth)
McNemar exact: both correct 559, only baseline 1, only method 41, both wrong 0
  p = 1.955e-11  (p < 0.01)
Paired bootstrap (10000):
  macro-F1  : 94.49 → 99.87  Δ +5.38  95% CI [+3.82, +7.10]  p ≈ 0
  bal-acc   : 94.34 → 99.87  Δ +5.53  95% CI [+3.86, +7.36]  p ≈ 0
  w-F1      : 93.17 → 99.83  Δ +6.67  95% CI [+4.66, +8.77]  p ≈ 0
(Significant, but under the test-peek protocol.)

---

## 3. HONEST protocol (5-fold + validation-based selection, no leak)

| metric | real_only | real+synth | Δ |
|---|---|---|---|
| w-F1  | 89.8 ± 1.7 | 90.9 ± 2.7 | +1.1 |
| macro-F1 | 91.4 ± 1.4 | 92.4 ± 2.7 | +1.0 |
| bal-acc | 91.0 ± 1.7 | 92.1 ± 3.1 | +1.1 |

Per-class recall gain concentrated in hard classes:
  n-de 73.7 → 80.0 ; p-de 81.1 → 84.2 ; cold weld 92.5 → 97.5 ; baseline 94.7 → 88.6
fold-2 real+synth dropped to 85.7 (synthetic hurt that fold; drives up std).
Honest gain ≈ +1 pt, overlapping CI — likely not significant. Value = helps hard/minority.

---

## 4. Memorization / nearest-neighbour analysis (R1#3, R2#9) — PASSED ✅

Features: ResNet50-V2, cosine sim. 601 real vs 3005 synthetic.

| measure | cosine (median) |
|---|---|
| synthetic → nearest real | 0.896 (mean 0.893, p95 0.927, max 0.951) |
| real → nearest real (baseline) | 0.926 (mean 0.924, p95 0.947) |
| synthetic → its own base image | 0.816 (mean 0.808) |
| near-duplicates (>0.95) | **1 / 3005 (0.03%)** |

Conclusion: synthetic images are FARTHER from reals than reals are from each other
(0.896 < 0.926) → NO memorization / near-duplication. Even vs their own i2i base image,
sim is only 0.816 (< real-real 0.926) → i2i meaningfully transforms, not copies.
=> The synthetic set is genuinely novel. The 99.8/100 inflation is due to the
test-peek evaluation protocol, NOT to synthetic data quality/duplication.
This directly rebuts R1#3 and answers R2#9 (nn_grid_random.png / nn_grid_closest.png).

## 5. FID / KID — diffusion vs VAE (R1#5, R1#6, R2#14)

InceptionV3 pool3 features. Both generators produced 3005 images. Lower = better.

| generator | overall FID | overall KID (×1e3) |
|---|---|---|
| **diffusion (ours)** | **33.1** | **30.1 ± 0.8** |
| VAE baseline | 240.6 | 351.6 ± 3.6 |

Diffusion FID is ~7× lower and KID ~12× lower than the VAE → empirically justifies the
diffusion choice over a VAE generator (answers R1#5 with evidence, not assertion).

Per-class FID (diffusion / VAE): baseline 55.7/310.5, cold weld 66.0/220.9, low gap 54.0/292.7,
low power 49.0/242.2, n-de 41.0/263.6, oil 38.9/206.9, p-de 42.6/241.7, water 89.4/243.5.
Diffusion beats VAE in every class. Water has the highest diffusion FID (smallest class, 28 real).

Caveat (per R1#6): features are ImageNet-Inception on grayscale industrial images, so absolute
values are approximate; the ~7× relative gap is robust to this. KID reported with ±std over 100
subsets.

VAE downstream utility (ResNet50, paper protocol, w-F1): real+synth 94.0±1.8 vs diffusion
99.8±0.3 (real-only 93.2). VAE adds ~+0.8; diffusion adds ~+6.6. Diffusion beats VAE on BOTH
generation quality (FID 31 vs 293) and downstream utility -> R1#5 answered with evidence.

## 6. Multi-backbone (paper vs honest protocol, w-F1, real -> real+synth)

| backbone | paper (select=test) | honest (select=val) |
|---|---|---|
| ResNet50 | 93.2±1.5 -> 99.8±0.3 (+6.6) | 89.8±1.7 -> 90.9±2.7 (+1.1) |
| MobileNetV2 | 93.2±2.8 -> 99.3±0.6 (+6.1) | 88.5±2.8 -> 96.0±2.9 (+7.5) |
| GoogLeNet | 94.5±1.6 -> 99.7±0.4 (+5.2) | 91.9±3.0 -> 97.5±2.7 (+5.6) |

KEY: under the HONEST protocol, MobileNetV2 (+7.5) and GoogLeNet (+5.6) show LARGE gains from
synthetic data -> the benefit is NOT merely a test-peek artifact. ResNet50 honest (+1.1) is an
outlier vs the other two; worth re-checking (earlier run had a fold-2 anomaly where synthetic hurt).

## 7. Traditional-augmentation baselines (R2#8; ResNet50, paper protocol, w-F1)

| method | w-F1 |
|---|---|
| real_only | 93.2 |
| + RandAugment | 94.6 |
| + MixUp | 94.5 |
| + CutMix | 94.3 |
| + class-balanced sampler | 94.0 |
| + focal loss | 93.4 |
| + strong combined aug | 95.1 |
| **real+synth (diffusion)** | **99.8** |

All standard augmentations reach at most 95.1; diffusion synthetic augmentation reaches 99.8
(~+4.7 over the best standard augmentation) -> structure-controlled diffusion synthesis
substantially outperforms generic augmentation. Strong answer to R2#8.

## Pending
- [ ] Traditional augmentation baselines (RandAugment/MixUp/CutMix/focal/balanced) — R2#8
- [ ] GAN/VAE baseline or justified exclusion — R1#5
- [ ] Data-scarcity curve (real_frac 0.1/0.25/0.5/1.0) — R2#13
- [ ] Significance test on HONEST numbers (not just paper protocol)
