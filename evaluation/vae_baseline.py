#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class-conditional VAE baseline for weld-image synthesis (Reviewer 1 #5: compare against
a VAE generative baseline instead of only asserting it).

Design mirrors the diffusion i2i pipeline so the comparison is fair and plugs into the
SAME leakage-filtered evaluation:
  * train a conditional VAE on the real images (grayscale, resize+pad to img_size)
  * generate: encode a real BASE image -> perturb the latent -> decode a variation
    (mult variations per base image), writing a manifest (synthetic -> base_real)

Usage:
  # train
  python vae_baseline.py --mode train --real_dir /root/autodl-tmp/data/images \
      --ckpt vae.pt --epochs 200
  # generate 5x per real image
  python vae_baseline.py --mode generate --real_dir /root/autodl-tmp/data/images \
      --ckpt vae.pt --out_dir /root/autodl-tmp/data/synthetic_vae --mult 5

Then evaluate exactly like the diffusion synthetic:
  python classify_cv.py --protocol paper --n_splits 5 --backbone resnet50 \
      --real_dir ... --pretrained_dir ... \
      --synthetic_dir /root/autodl-tmp/data/synthetic_vae \
      --synthetic_manifest /root/autodl-tmp/data/synthetic_vae/manifest.csv \
      --regimes real_plus_synth --out_dir results_bl/vae
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_class_dirs(root):
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


def resize_and_pad(img, size):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def scan(root, classes):
    root = Path(root); items = []
    for lbl, cls in enumerate(classes):
        d = root / cls
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    items.append({"path": str(p), "label": lbl, "cls": cls, "base": p.name})
    return items


class WeldGray(Dataset):
    def __init__(self, items, size):
        self.items, self.size = items, size
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r = self.items[i]
        img = resize_and_pad(Image.open(r["path"]).convert("L"), self.size)
        x = np.asarray(img, np.float32)[None] / 255.0        # (1,H,W)
        return torch.from_numpy(x), r["label"]


# --------------------------------------------------------------------------------------
class CVAE(nn.Module):
    def __init__(self, num_classes, size=128, zdim=128, cemb=16):
        super().__init__()
        self.size, self.zdim = size, zdim
        self.cls_emb = nn.Embedding(num_classes, cemb)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),      # /2
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),     # /4
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),    # /8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),   # /16
        )
        self.fh = size // 16
        flat = 256 * self.fh * self.fh
        self.fc_mu = nn.Linear(flat, zdim)
        self.fc_lv = nn.Linear(flat, zdim)
        self.fc_dec = nn.Linear(zdim + cemb, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z, y):
        h = self.fc_dec(torch.cat([z, self.cls_emb(y)], 1))
        h = h.view(-1, 256, self.fh, self.fh)
        return self.dec(h)

    def forward(self, x, y):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z, y), mu, lv


def train(args, device):
    classes = list_class_dirs(args.real_dir)
    items = scan(args.real_dir, classes)
    dl = DataLoader(WeldGray(items, args.img_size), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers, drop_last=True)
    model = CVAE(len(classes), args.img_size, args.zdim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[vae] train on {len(items)} imgs, {len(classes)} classes")
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            xr, mu, lv = model(x, y)
            rec = F.mse_loss(xr, x, reduction="sum") / x.size(0)
            kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
            loss = rec + args.beta * kld
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs}  loss={tot/len(dl):.1f}")
    torch.save({"state": model.state_dict(), "classes": classes,
                "size": args.img_size, "zdim": args.zdim}, args.ckpt)
    print(f"[vae] saved -> {args.ckpt}")


@torch.no_grad()
def generate(args, device):
    ck = torch.load(args.ckpt, map_location=device)
    classes = ck["classes"]
    model = CVAE(len(classes), ck["size"], ck["zdim"]).to(device)
    model.load_state_dict(ck["state"]); model.eval()
    items = scan(args.real_dir, classes)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = []
    rng = torch.Generator(device=device).manual_seed(args.seed)
    for cls in classes:
        (out / cls).mkdir(parents=True, exist_ok=True)
    tag = {c: 0 for c in classes}
    for r in items:
        img = resize_and_pad(Image.open(r["path"]).convert("L"), ck["size"])
        x = torch.from_numpy(np.asarray(img, np.float32)[None, None] / 255.0).to(device)
        y = torch.tensor([r["label"]], device=device)
        mu, lv = model.encode(x)
        std = torch.exp(0.5 * lv)
        for _ in range(args.mult):
            z = mu + torch.randn(mu.shape, generator=rng, device=device) * std * args.perturb
            xr = model.decode(z, y)[0, 0].clamp(0, 1).cpu().numpy()
            arr = (xr * 255).astype(np.uint8)
            fname = f"vae_{r['cls'].replace(' ', '_')}_{tag[r['cls']]:05d}.png"
            Image.fromarray(arr, "L").convert("RGB").save(out / r["cls"] / fname)
            manifest.append([fname, r["base"], r["cls"]])
            tag[r["cls"]] += 1
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["synthetic_filename", "base_real_filename", "class"])
        w.writerows(manifest)
    print(f"[vae] generated {len(manifest)} images -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "generate"], required=True)
    ap.add_argument("--real_dir", default="/root/autodl-tmp/data/images")
    ap.add_argument("--ckpt", default="vae.pt")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/data/synthetic_vae")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mult", type=int, default=5)
    ap.add_argument("--perturb", type=float, default=1.0,
                    help="latent noise scale at generation (1.0 = full prior sampling)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (train if args.mode == "train" else generate)(args, device)


if __name__ == "__main__":
    main()
