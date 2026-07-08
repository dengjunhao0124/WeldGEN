#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weld-condition classification with TWO selectable protocols, so we can (a) reproduce
the paper's single-split number and (b) report an honest cross-validated number, using
the SAME model and preprocessing — isolating the effect of the methodology alone.

  --protocol paper   : single stratified hold-out (80/20) + "best epoch on the test
                       set" model selection  (== original train_cls.py; optimistic).
  --protocol honest  : 5-fold stratified CV + last-epoch reporting (no test peeking),
                       with mean +/- std, macro-F1, balanced accuracy, per-class recall,
                       and out-of-fold predictions for McNemar / bootstrap tests.

Model & preprocessing (identical in both protocols, matching the original code):
  * 512x512, aspect-preserving resize + zero pad (resize_and_pad)
  * grayscale -> RGB, scaled to [0,1], NO ImageNet normalisation (--norm none)
  * ResNet50 (ImageNet-pretrained) with the original custom head 2048->1024->C
  * Adam (lr 5e-4), 3-epoch warmup + cosine annealing, CrossEntropy, no augmentation

Leakage handling for synthetic data is unchanged: per-fold synthetic dirs
(synthetic_dir/foldK/<class>) or a flat set filtered by a manifest.

Examples:
  # reproduce the paper number (single split, best-epoch):
  python classify_cv.py --protocol paper --backbone resnet50 \
      --synthetic_dir /root/autodl-tmp/data/synthetic --regimes real_only real_plus_synth

  # honest cross-validated number:
  python classify_cv.py --protocol honest --backbone resnet50 \
      --synthetic_dir /root/autodl-tmp/data/synthetic --regimes real_only real_plus_synth
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def list_class_dirs(root):
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


def scan_real(real_dir, classes):
    real_dir = Path(real_dir)
    items = []
    for label, cls in enumerate(classes):
        cdir = real_dir / cls
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append({"path": str(p), "label": label, "cls": cls, "base": p.name})
    return items


def scan_synthetic(syn_dir, classes, fold, manifest_df):
    syn_dir = Path(syn_dir)
    perfold = syn_dir / f"fold{fold}"
    src = perfold if perfold.is_dir() else syn_dir
    items = []
    for label, cls in enumerate(classes):
        cdir = src / cls
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            base_real = manifest_df.get(p.name) if manifest_df else None
            items.append({"path": str(p), "label": label, "cls": cls, "base_real": base_real})
    return items, (src == perfold)


def load_manifest(path):
    if path is None:
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    syn_c = cols.get("synthetic_filename") or df.columns[0]
    base_c = cols.get("base_real_filename") or df.columns[1]
    return dict(zip(df[syn_c].astype(str), df[base_c].astype(str)))


# --------------------------------------------------------------------------------------
# Preprocessing: aspect-preserving resize + zero pad (matches original dataset.py)
# --------------------------------------------------------------------------------------
def resize_and_pad(img_pil, size):
    w, h = img_pil.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img_pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


class WeldDataset(Dataset):
    def __init__(self, records, size, normalize, cache, augment=None):
        self.records = records
        self.size = size
        self.normalize = normalize
        self.augment = augment                       # None | torchvision transform
        self.cache = {} if (cache and augment is None) else None  # no cache when augmenting

    def __len__(self):
        return len(self.records)

    def _to_arr(self, img):
        arr = np.asarray(img, np.float32) / 255.0            # [0,1]
        if self.normalize:
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    def _load(self, path):
        img = Image.open(path).convert("L").convert("RGB")   # grayscale -> 3ch
        return resize_and_pad(img, self.size)

    def __getitem__(self, i):
        r = self.records[i]
        if self.cache is not None and r["path"] in self.cache:
            arr = self.cache[r["path"]]
        else:
            img = self._load(r["path"])
            if self.augment is not None:
                img = self.augment(img)                      # PIL-in, PIL-out
            arr = self._to_arr(img)
            if self.cache is not None:
                self.cache[r["path"]] = arr
        return torch.from_numpy(arr), r["label"]


# --------------------------------------------------------------------------------------
# Baseline augmentation helpers (Reviewer 2 #8: RandAugment / MixUp / CutMix / focal / balanced)
# --------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def mixup_cutmix(x, y, num_classes, mode, alpha=1.0):
    """Return mixed inputs and soft targets. mode in {'mixup','cutmix'}."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    y1 = nn.functional.one_hot(y, num_classes).float()
    y2 = y1[idx]
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[idx]
    else:  # cutmix
        H, W = x.shape[2:]
        rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1a, y1b = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1a, x1b = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x[:, :, y1a:y1b, x1a:x1b] = x[idx, :, y1a:y1b, x1a:x1b]
        lam = 1 - ((y1b - y1a) * (x1b - x1a) / (H * W))
    return x, lam * y1 + (1 - lam) * y2


def make_sampler(records, num_classes):
    """WeightedRandomSampler for class-balanced sampling."""
    counts = np.bincount([r["label"] for r in records], minlength=num_classes)
    w = 1.0 / np.maximum(counts, 1)
    weights = [w[r["label"]] for r in records]
    return torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)


# --------------------------------------------------------------------------------------
# Models with the ORIGINAL custom heads (model.py)
# --------------------------------------------------------------------------------------
class ResNet50Custom(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.fea_layer = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(0.3); self.relu = nn.ReLU()
        self.output_layer = nn.Linear(2048, 1024)
        self.proj = nn.Linear(1024, num_classes)
        self.dropout1 = nn.Dropout(0.3); self.relu1 = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]
        f = self.fea_layer(x)
        o = self.output_layer(self.relu(self.dropout(f.view(b, -1))))
        return self.proj(self.relu1(self.dropout1(o)))


class MobileNetV2Custom(nn.Module):
    def __init__(self, base, num_classes, size):
        super().__init__()
        self.fea_layer = base.features
        self.dropout = nn.Dropout(0.3); self.relu = nn.ReLU()
        feat_hw = size // 32
        self.output_layer = nn.Linear(1280 * feat_hw * feat_hw, num_classes)

    def forward(self, x):
        b = x.shape[0]
        f = self.fea_layer(x)
        return self.output_layer(self.relu(self.dropout(f.view(b, -1))))


def build_model(backbone, num_classes, size, pretrained_dir):
    backbone = backbone.lower()
    pdir = Path(pretrained_dir) if pretrained_dir else None

    def maybe_local(model_ctor, fname, weights):
        m = model_ctor()
        loaded = False
        if pdir and (pdir / fname).exists():
            m.load_state_dict(torch.load(pdir / fname, weights_only=True)); loaded = True
        if not loaded:
            m = model_ctor(weights=weights)   # torchvision auto-download/cache
        return m

    if backbone == "resnet50":
        # original used resnet50-11ad3fa6.pth == torchvision IMAGENET1K_V2
        base = maybe_local(models.resnet50, "resnet50-11ad3fa6.pth",
                           models.ResNet50_Weights.IMAGENET1K_V2)
        return ResNet50Custom(base, num_classes)
    if backbone in ("mobilenetv2", "mobilenet_v2"):
        base = maybe_local(models.mobilenet_v2, "mobilenet_v2-b0353104.pth",
                           models.MobileNet_V2_Weights.IMAGENET1K_V1)
        return MobileNetV2Custom(base, num_classes, size)
    if backbone == "googlenet":
        m = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        m.aux_logits = False; m.aux1 = None; m.aux2 = None
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(backbone)


def forward_logits(model, x):
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


# --------------------------------------------------------------------------------------
# Train one model; return per-epoch (y_true, y_pred) on the test set
# --------------------------------------------------------------------------------------
def warmup_cosine_lr(epoch, base_lr, epochs, warmup=3, eta_min=1e-6):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, epochs - warmup)
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        logits = forward_logits(model, x.to(device))
        preds.append(logits.argmax(1).cpu().numpy()); trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def train_eval(tr_records, te_records, val_records, num_classes, args, device):
    """Train; return per-epoch list of dicts {val_acc, test_acc, yt, yp}."""
    set_seed(args.seed)
    aug = transforms.RandAugment() if args.aug == "randaugment" else None
    tr_ds = WeldDataset(tr_records, args.img_size, args.normalize, args.cache, augment=aug)
    if args.sampler == "balanced":
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, drop_last=True,
                               sampler=make_sampler(tr_records, num_classes),
                               num_workers=args.num_workers)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, num_workers=args.num_workers)
    te_loader = DataLoader(WeldDataset(te_records, args.img_size, args.normalize, args.cache),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = None
    if val_records:
        val_loader = DataLoader(WeldDataset(val_records, args.img_size, args.normalize, args.cache),
                                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args.backbone, num_classes, args.img_size, args.pretrained_dir).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    crit = FocalLoss() if args.loss == "focal" else nn.CrossEntropyLoss()

    history = []
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = warmup_cosine_lr(ep, args.lr, args.epochs)
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if args.aug in ("mixup", "cutmix"):
                x, y_soft = mixup_cutmix(x, y, num_classes, args.aug)
                logits = forward_logits(model, x)
                loss = -(y_soft * nn.functional.log_softmax(logits, 1)).sum(1).mean()
            else:
                loss = crit(forward_logits(model, x), y)
            loss.backward(); opt.step()
        yt, yp = predict(model, te_loader, device)
        val_acc = None
        if val_loader is not None:
            vy, vp = predict(model, val_loader, device)
            val_acc = float(np.mean(vy == vp))
        history.append({"val_acc": val_acc, "test_acc": float(np.mean(yt == yp)),
                        "yt": yt, "yp": yp})
    return history


def metrics_from(y_true, y_pred, classes):
    return {
        "w_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "w_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "w_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) * 100,
        "per_class_recall": {classes[i]: v * 100 for i, v in enumerate(
            recall_score(y_true, y_pred, average=None,
                         labels=list(range(len(classes))), zero_division=0))},
    }


def select_epoch(history, how):
    """Pick the epoch to report test metrics from.
       val  -> argmax validation acc (correct, no test peeking)
       test -> argmax test acc (paper, optimistic)
       last -> final epoch."""
    if how == "val":
        idx = int(np.argmax([h["val_acc"] for h in history]))
    elif how == "test":
        idx = int(np.argmax([h["test_acc"] for h in history]))
    else:
        idx = len(history) - 1
    return history[idx]["yt"], history[idx]["yp"], idx


# --------------------------------------------------------------------------------------
def build_synth_train(syn_dir, classes, fold, manifest, keep_bases):
    """Keep only synthetic images whose base real image is in keep_bases (the real images
    actually used for training). This drops both test-derived synthetics and, in the
    data-scarcity setting, synthetics derived from real images we pretend not to have."""
    if syn_dir is None:
        return []
    syn_all, perfold = scan_synthetic(syn_dir, classes, fold, manifest)
    kept, dropped = [], 0
    for s in syn_all:
        if perfold:
            kept.append(s)                              # per-fold dir => already safe
        elif s["base_real"] is not None:
            if s["base_real"] in keep_bases:
                kept.append(s)
            else:
                dropped += 1                            # base not in training set -> drop
        else:
            kept.append(s)                             # no manifest => cannot filter
    tag = "per-fold" if perfold else ("manifest-filtered" if manifest else "UNFILTERED(leak!)")
    print(f"    synthetic: {len(syn_all)} found, {len(kept)} used, {dropped} dropped [{tag}]")
    return kept


def stratified_subsample(records, frac, seed):
    """Keep a stratified fraction per class (>=1 per class). Returns subset list."""
    if frac >= 1.0:
        return records
    rng = random.Random(seed)
    by_cls = {}
    for r in records:
        by_cls.setdefault(r["cls"], []).append(r)
    kept = []
    for cls, items in by_cls.items():
        k = max(1, round(frac * len(items)))
        idx = list(range(len(items)))
        rng.shuffle(idx)
        kept.extend(items[i] for i in idx[:k])
    return kept


def make_splits(real, n_splits, seed):
    """Yield (fold_idx, train_idx, test_idx)."""
    y = np.array([r["label"] for r in real])
    if n_splits <= 1:
        idx = np.arange(len(real))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        yield 0, tr, te
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(np.arange(len(real)), y)):
            yield k, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["paper", "honest"], default="honest")
    ap.add_argument("--real_dir", default="/root/autodl-tmp/data/images")
    ap.add_argument("--synthetic_dir", default=None)
    ap.add_argument("--synthetic_manifest", default=None)
    ap.add_argument("--pretrained_dir", default="../../train_vision",
                    help="dir with resnet50-11ad3fa6.pth etc. (else torchvision weights)")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "mobilenet_v2", "mobilenetv2", "googlenet"])
    ap.add_argument("--regimes", nargs="+", default=["real_only"],
                    choices=["real_only", "real_plus_synth", "synth_only"])
    ap.add_argument("--n_splits", type=int, default=None, help="override (paper=1, honest=5)")
    ap.add_argument("--select", choices=["test", "val", "last"], default=None,
                    help="epoch selection: test=peek test acc (paper, optimistic); "
                         "val=carve a validation set and pick best epoch on it (correct); "
                         "last=final epoch. Default: paper->test, honest->last.")
    ap.add_argument("--val_frac", type=float, default=0.125,
                    help="fraction of the TRAIN portion held out as validation when "
                         "--select val (0.125 of 80%% train ~= 10%% overall -> 70:10:20)")
    ap.add_argument("--real_frac", type=float, default=1.0,
                    help="fraction of real TRAIN data to keep (data-scarcity curve). "
                         "Test set stays full; synthetic is filtered to the kept reals.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--norm", choices=["none", "imagenet"], default="none")
    ap.add_argument("--aug", choices=["none", "randaugment", "mixup", "cutmix"],
                    default="none", help="augmentation baseline (Reviewer 2 #8)")
    ap.add_argument("--sampler", choices=["none", "balanced"], default="none",
                    help="class-balanced sampling (Reviewer 2 #8)")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="focal loss for imbalance (Reviewer 2 #8)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_cache", action="store_true", help="disable in-memory image cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    # protocol defaults
    if args.n_splits is None:
        args.n_splits = 1 if args.protocol == "paper" else 5
    if args.select is None:
        args.select = "test" if args.protocol == "paper" else "last"
    args.normalize = (args.norm == "imagenet")
    args.cache = not args.no_cache
    frac_tag = "" if args.real_frac >= 1.0 else f"_frac{args.real_frac}"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    classes = list_class_dirs(args.real_dir)
    real = scan_real(args.real_dir, classes)
    print(f"[protocol={args.protocol}] backbone={args.backbone} n_splits={args.n_splits} "
          f"select={args.select} epochs={args.epochs} img={args.img_size} norm={args.norm}")
    print(f"[data] {len(real)} real images, {len(classes)} classes")

    needs_syn = any(r != "real_only" for r in args.regimes)
    manifest = load_manifest(args.synthetic_manifest) if needs_syn else None
    if needs_syn and args.synthetic_dir is None:
        raise SystemExit("regime needs --synthetic_dir")

    fold_metrics = {rg: [] for rg in args.regimes}
    oof_rows = {rg: [] for rg in args.regimes}
    split_records = []

    for fold, tr_idx, te_idx in make_splits(real, args.n_splits, args.seed):
        print(f"\n===== fold {fold} | train={len(tr_idx)} test={len(te_idx)} =====")
        real_train = [real[i] for i in tr_idx]
        real_test = [real[i] for i in te_idx]

        # data-scarcity: keep only a stratified fraction of the real training data
        if args.real_frac < 1.0:
            n_before = len(real_train)
            real_train = stratified_subsample(real_train, args.real_frac, args.seed + fold)
            print(f"    real_frac={args.real_frac}: train {n_before} -> {len(real_train)}")

        # carve a stratified validation set from the real-train portion (model selection)
        real_val = []
        if args.select == "val" and args.val_frac > 0:
            y_tr = [r["label"] for r in real_train]
            tr_sub, val_sub = train_test_split(
                np.arange(len(real_train)), test_size=args.val_frac,
                stratify=y_tr, random_state=args.seed)
            real_val = [real_train[i] for i in val_sub]
            real_train = [real_train[i] for i in tr_sub]
            print(f"    val carved: train={len(real_train)} val={len(real_val)}")

        for i, r in [(i, real[i]) for i in te_idx]:
            split_records.append({"fold": fold, "split": "test", "path": r["path"]})
        for r in real_train:
            split_records.append({"fold": fold, "split": "train", "path": r["path"]})
        for r in real_val:
            split_records.append({"fold": fold, "split": "val", "path": r["path"]})

        # synthetic kept only if derived from a real image we actually train on
        keep_bases = {r["base"] for r in real_train}
        syn_train = build_synth_train(args.synthetic_dir if needs_syn else None,
                                      classes, fold, manifest, keep_bases)

        for rg in args.regimes:
            if rg == "real_only":      tr = real_train
            elif rg == "real_plus_synth": tr = real_train + syn_train
            else:                      tr = syn_train
            if not tr:
                print(f"  [skip] {rg}: empty train"); continue

            history = train_eval(tr, real_test, real_val, len(classes), args, device)
            yt, yp, ep = select_epoch(history, args.select)
            m = metrics_from(yt, yp, classes)
            fold_metrics[rg].append(m)
            print(f"  [{rg}] sel-epoch={ep} w-F1={m['w_f1']:.1f} macro-F1={m['macro_f1']:.1f} "
                  f"bal-acc={m['balanced_acc']:.1f}")
            for path_i, a, b in zip([r["path"] for r in real_test], yt, yp):
                oof_rows[rg].append({"fold": fold, "path": path_i,
                                     "y_true": int(a), "y_pred": int(b)})

    pd.DataFrame(split_records).to_csv(out_dir / f"splits_{args.protocol}.csv", index=False)

    # aggregate
    summary = {"protocol": args.protocol, "backbone": args.backbone,
               "n_splits": args.n_splits, "select": args.select,
               "classes": classes, "regimes": {}}
    keys = ["w_precision", "w_recall", "w_f1", "macro_f1", "balanced_acc"]
    for rg, folds in fold_metrics.items():
        if not folds: continue
        agg = {}
        for k in keys:
            vals = [f[k] for f in folds]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "per_fold": [round(v, 2) for v in vals]}
        agg["per_class_recall"] = {c: {"mean": float(np.mean([f["per_class_recall"][c] for f in folds])),
                                       "std": float(np.std([f["per_class_recall"][c] for f in folds]))}
                                   for c in classes}
        summary["regimes"][rg] = agg
        pd.DataFrame(oof_rows[rg]).to_csv(
            out_dir / f"oof_{args.protocol}_{args.backbone}_{rg}{frac_tag}.csv", index=False)

    summary["real_frac"] = args.real_frac
    with open(out_dir / f"summary_{args.protocol}_{args.backbone}{frac_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  {args.backbone} | protocol={args.protocol} ({args.n_splits} split, select={args.select})")
    print("=" * 78)
    print(f"{'regime':<18}{'w-Pre':>13}{'w-Rec':>13}{'w-F1':>13}{'macro-F1':>13}{'bal-acc':>13}")
    for rg, agg in summary["regimes"].items():
        def cell(k): return f"{agg[k]['mean']:>7.1f}±{agg[k]['std']:<4.1f}"
        print(f"{rg:<18}{cell('w_precision')}{cell('w_recall')}{cell('w_f1')}"
              f"{cell('macro_f1')}{cell('balanced_acc')}")
    print(f"\nSaved -> {out_dir}/summary_{args.protocol}_{args.backbone}.json")


if __name__ == "__main__":
    main()
