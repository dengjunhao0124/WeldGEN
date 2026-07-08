#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Materialise per-fold data directories for the LEAKAGE-FREE per-fold pipeline.

Masks are derived ON THE FLY by binarising the (already background-removed) weld
images: foreground = non-black region. This recovers exactly the segmentation region
that was used to mask the images, so no external mask folder is needed.  (seg_raw is
NOT used.)

For each of the 5 stratified folds, using ONLY that fold's training images:

  work/fold{k}/
    db_instance/                 # fold-train weld images        -> DreamBooth instance data
    cn_data/
        <img>.png                # fold-train target weld images (flat)
        train/<img>.png          # fold-train derived masks (random-sampling pool)
        metadata.jsonl           # {file_name, caption, conditioning_image:[mask names]}
    maskpool/<class>/<img>.png   # fold-train derived masks per class -> generation control
    basepool/<class>/<img>.png   # fold-train base images per class   -> generation init

The fold split is IDENTICAL to classify_cv.py (same scan order + StratifiedKFold seed),
so the synthetic images line up with the test folds used for evaluation.

Run:
  python make_folds.py --real_dir /root/autodl-tmp/data/images \
                       --work_dir /root/autodl-tmp/work --n_splits 5 --seed 42
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# reuse the EXACT same data scan + class discovery as the classifier, so folds match
from classify_cv import list_class_dirs, scan_real

PROMPT = 'an image of w* with {cls} defect on white background'


def link(src: Path, dst: Path):
    """Symlink src->dst (fall back to copy if symlinks unsupported)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def save_mask(img_path: Path, dst: Path, thresh: int):
    """Binarise a background-removed weld image: non-black -> white foreground."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    g = np.asarray(Image.open(img_path).convert("L"))
    m = (g > thresh).astype(np.uint8) * 255
    Image.fromarray(m, mode="L").convert("RGB").save(dst)


def build_gen_dirs(records, out_dir, classes, mask_thresh):
    """Build DreamBooth/ControlNet/generation dirs from a list of image records."""
    db = out_dir / "db_instance"
    cn = out_dir / "cn_data"
    cn_train = cn / "train"
    for d in (db, cn, cn_train):
        d.mkdir(parents=True, exist_ok=True)

    cls_masks = {c: [] for c in classes}   # per-class mask stems (random pool)
    for r in records:
        base, cls = r["base"], r["cls"]
        img = Path(r["path"])
        link(img, db / base)                              # DreamBooth instance
        link(img, out_dir / "basepool" / cls / base)      # generation init pool
        link(img, cn / base)                              # ControlNet target
        save_mask(img, cn_train / base, mask_thresh)      # ControlNet mask pool
        save_mask(img, out_dir / "maskpool" / cls / base, mask_thresh)
        cls_masks[cls].append(Path(base).stem)

    meta_rows = []
    for r in records:
        pool = cls_masks[r["cls"]]
        if not pool:
            continue
        meta_rows.append({"file_name": r["base"],
                          "caption": PROMPT.format(cls=r["cls"]),
                          "conditioning_image": pool})
    with open(cn / "metadata.jsonl", "w") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(records), len(meta_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/root/autodl-tmp/data/images")
    ap.add_argument("--work_dir", default="/root/autodl-tmp/work")
    ap.add_argument("--unified", action="store_true",
                    help="build ONE dir set from ALL images (unified generation) instead "
                         "of per-fold dirs. Output -> work_dir/all/")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask_thresh", type=int, default=10,
                    help="grayscale threshold for foreground (background is pure black)")
    args = ap.parse_args()

    real_dir = Path(args.real_dir)
    work = Path(args.work_dir)
    classes = list_class_dirs(real_dir)
    real = scan_real(real_dir, classes)
    print(f"[data] {len(real)} images, classes={classes}")

    if args.unified:
        n, cn_n = build_gen_dirs(real, work / "all", classes, args.mask_thresh)
        print(f"[unified] built from ALL {n} images  cn_images={cn_n}")
        print(f"\nDone -> {work}/all/  (train DreamBooth+ControlNet once on this, then "
              f"generate flat synthetic + manifest)")
        return

    y = np.array([r["label"] for r in real])
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, _) in enumerate(skf.split(np.arange(len(real)), y)):
        n, cn_n = build_gen_dirs([real[i] for i in tr_idx], work / f"fold{fold}",
                                 classes, args.mask_thresh)
        print(f"[fold{fold}] train={n}  cn_images={cn_n}")
    print(f"\nDone -> {work}/fold0..{args.n_splits-1}/")
    print("Next: per fold  DreamBooth(db_instance) -> ControlNet(cn_data) -> generate_fold.py")


if __name__ == "__main__":
    main()
