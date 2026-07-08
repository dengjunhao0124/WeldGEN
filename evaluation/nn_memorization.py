#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nearest-neighbour / memorization analysis for the synthetic weld images.

Addresses Reviewer 1 #3 (synthetic-vs-synthetic = 100% -> possible near-duplicates)
and Reviewer 2 #9 (nearest-neighbour visualisation to detect memorization).

For every synthetic image we measure how close it is to the REAL images in a
pretrained-ResNet50 feature space (cosine similarity):
  * nearest real neighbour  (over ALL real images)
  * its own base real image (from the manifest; the i2i init image)
and compare against the REAL-vs-REAL nearest-neighbour baseline. If synthetic images
are systematically closer to reals than reals are to each other, that indicates
memorization / near-duplication rather than genuine novel synthesis.

Outputs:
  out_dir/nn_stats.json          summary statistics
  out_dir/nn_hist.png            similarity histograms (syn->real vs real->real)
  out_dir/nn_grid_random.png     random synthetic + nearest real + its base
  out_dir/nn_grid_closest.png    the MOST similar synthetic (worst case) + neighbours
  out_dir/nn_per_synth.csv       per-synthetic: nearest-real sim, base sim, class

Run:
  python nn_memorization.py \
      --real_dir /root/autodl-tmp/data/images \
      --synthetic_dir /root/autodl-tmp/data/synthetic/fold-1 \
      --manifest /root/autodl-tmp/data/synthetic/fold-1/manifest.csv \
      --out_dir results_memorization
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_class_dirs(root):
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


def scan(root, classes):
    root = Path(root)
    items = []
    for cls in classes:
        d = root / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append({"path": str(p), "cls": cls, "name": p.name})
    return items


def build_extractor(device):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()          # -> 2048-d global-pooled features
    return m.eval().to(device)


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract(model, items, device, bs=64):
    feats = []
    batch = []
    for i, it in enumerate(items):
        img = Image.open(it["path"]).convert("L").convert("RGB")
        batch.append(TF(img))
        if len(batch) == bs or i == len(items) - 1:
            x = torch.stack(batch).to(device)
            f = model(x).cpu().numpy()
            feats.append(f)
            batch = []
    feats = np.concatenate(feats, 0).astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)   # L2 norm -> cosine
    return feats


def load_rgb(path, size=224):
    return Image.open(path).convert("L").convert("RGB").resize((size, size))


def save_grid(rows, titles, out_path, col_titles):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        for c in range(3):
            axes[r, c].imshow(rows[r][c]); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11)
        axes[r, 0].set_ylabel(titles[r], fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/root/autodl-tmp/data/images")
    ap.add_argument("--synthetic_dir", default="/root/autodl-tmp/data/synthetic/fold-1")
    ap.add_argument("--manifest", default=None,
                    help="synthetic_filename,base_real_filename mapping (for base-image sim)")
    ap.add_argument("--out_dir", default="results_memorization")
    ap.add_argument("--n_viz", type=int, default=6)
    ap.add_argument("--dup_thresh", type=float, default=0.95,
                    help="cosine sim above which a synthetic counts as a near-duplicate")
    ap.add_argument("--max_synth", type=int, default=0,
                    help="subsample synthetic for speed (0 = use all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    classes = list_class_dirs(args.real_dir)

    real = scan(args.real_dir, classes)
    syn = scan(args.synthetic_dir, classes)
    if args.max_synth and len(syn) > args.max_synth:
        rng = np.random.default_rng(0)
        syn = [syn[i] for i in rng.choice(len(syn), args.max_synth, replace=False)]
    print(f"[data] real={len(real)} synthetic={len(syn)} classes={len(classes)}")

    manifest = None
    if args.manifest:
        mdf = pd.read_csv(args.manifest)
        cols = {c.lower(): c for c in mdf.columns}
        manifest = dict(zip(mdf[cols.get("synthetic_filename", mdf.columns[0])].astype(str),
                            mdf[cols.get("base_real_filename", mdf.columns[1])].astype(str)))

    model = build_extractor(device)
    print("[feat] extracting real ..."); real_f = extract(model, real, device)
    print("[feat] extracting synthetic ..."); syn_f = extract(model, syn, device)

    real_name_to_idx = {r["name"]: i for i, r in enumerate(real)}

    # synthetic -> nearest real (cosine); and -> its base real
    sims = syn_f @ real_f.T                      # (Nsyn, Nreal)
    nn_idx = sims.argmax(1)
    nn_sim = sims.max(1)
    base_sim = np.full(len(syn), np.nan)
    if manifest:
        for i, s in enumerate(syn):
            b = manifest.get(s["name"])
            j = real_name_to_idx.get(b) if b else None
            if j is not None:
                base_sim[i] = float(syn_f[i] @ real_f[j])

    # real -> nearest OTHER real (baseline distribution)
    rr = real_f @ real_f.T
    np.fill_diagonal(rr, -1.0)
    real_nn_sim = rr.max(1)

    n_dup = int((nn_sim > args.dup_thresh).sum())
    stats = {
        "n_real": len(real), "n_synthetic": len(syn),
        "syn_to_nearest_real": {
            "mean": float(nn_sim.mean()), "median": float(np.median(nn_sim)),
            "p95": float(np.percentile(nn_sim, 95)), "max": float(nn_sim.max())},
        "real_to_nearest_real": {
            "mean": float(real_nn_sim.mean()), "median": float(np.median(real_nn_sim)),
            "p95": float(np.percentile(real_nn_sim, 95))},
        "syn_to_own_base": ({
            "mean": float(np.nanmean(base_sim)), "median": float(np.nanmedian(base_sim))}
            if manifest else None),
        "near_duplicate_thresh": args.dup_thresh,
        "n_near_duplicates": n_dup,
        "frac_near_duplicates": float(n_dup / len(syn)),
    }
    with open(out_dir / "nn_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    pd.DataFrame({
        "synthetic": [s["name"] for s in syn],
        "class": [s["cls"] for s in syn],
        "nearest_real": [real[j]["name"] for j in nn_idx],
        "nearest_real_sim": nn_sim,
        "base_sim": base_sim,
    }).to_csv(out_dir / "nn_per_synth.csv", index=False)

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(real_nn_sim, bins=40, alpha=0.6, label="real -> nearest real", density=True)
    plt.hist(nn_sim, bins=40, alpha=0.6, label="synthetic -> nearest real", density=True)
    plt.axvline(args.dup_thresh, color="red", ls="--", label=f"dup thresh {args.dup_thresh}")
    plt.xlabel("cosine similarity (ResNet50 features)"); plt.ylabel("density")
    plt.legend(); plt.title("Nearest-neighbour similarity: memorization check")
    plt.tight_layout(); plt.savefig(out_dir / "nn_hist.png", dpi=130); plt.close()

    # viz grids: random samples and the worst-case (closest) samples
    def make_grid(order, tag):
        picks = order[:args.n_viz]
        rows, titles = [], []
        for i in picks:
            syn_img = load_rgb(syn[i]["path"])
            near_img = load_rgb(real[nn_idx[i]]["path"])
            base_name = manifest.get(syn[i]["name"]) if manifest else None
            bj = real_name_to_idx.get(base_name) if base_name else None
            base_img = load_rgb(real[bj]["path"]) if bj is not None else Image.new("RGB", (224, 224))
            rows.append([syn_img, near_img, base_img])
            titles.append(f"{syn[i]['cls']}\nsim={nn_sim[i]:.3f}")
        save_grid(rows, titles, out_dir / f"nn_grid_{tag}.png",
                  ["synthetic", "nearest real", "its base real"])

    make_grid(np.random.default_rng(0).permutation(len(syn)), "random")
    make_grid(np.argsort(-nn_sim), "closest")     # highest similarity = worst case

    # report
    print("\n" + "=" * 60)
    print("  Memorization / nearest-neighbour analysis")
    print("=" * 60)
    print(f"  synthetic -> nearest real : median cos {stats['syn_to_nearest_real']['median']:.3f}")
    print(f"  real      -> nearest real : median cos {stats['real_to_nearest_real']['median']:.3f}")
    if manifest:
        print(f"  synthetic -> its base img : median cos {stats['syn_to_own_base']['median']:.3f}")
    print(f"  near-duplicates (>{args.dup_thresh}): {n_dup}/{len(syn)} "
          f"({stats['frac_near_duplicates']*100:.1f}%)")
    print(f"\nSaved -> {out_dir}/  (nn_stats.json, nn_hist.png, nn_grid_*.png, nn_per_synth.csv)")


if __name__ == "__main__":
    main()
