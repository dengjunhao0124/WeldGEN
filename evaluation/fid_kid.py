#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FID / KID using the SAME libraries as the original paper (gradio_demo/cal_metrics.py):
  * FID via torch-fidelity   (torch_fidelity.calculate_metrics)
  * KID via clean-fid        (cleanfid.fid.compute_kid)
so the numbers are on the same scale as Table 5 (final method reported FID 69.38, KID 0.06).

Install:  pip install torch-fidelity clean-fid

Usage:
  python fid_kid.py --real_dir /root/autodl-tmp/data/images \
      --synthetic_dirs /root/autodl-tmp/data/synthetic/fold-1 /root/autodl-tmp/data/synthetic_vae \
      --names diffusion vae --per_class --out_dir results_fidkid
"""
import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_class_dirs(root):
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


def resize_flat(src_root, classes, dst, size):
    """Resize every class image to size×size RGB into a single flat dir
    (uniform size so torch-fidelity can batch them; matches the paper's 'resized_train')."""
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls in classes:
        d = Path(src_root) / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
                img.save(dst / f"{cls.replace(' ', '_')}__{p.name}")
                n += 1
    return n


def resize_class(src_cls_dir, dst, size):
    """Resize one class dir into dst (uniform size)."""
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(src_cls_dir).iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR).save(dst / p.name)
            n += 1
    return n


def compute_fid(syn_dir, real_dir):
    import torch_fidelity
    m = torch_fidelity.calculate_metrics(input1=str(syn_dir), input2=str(real_dir),
                                         cuda=True, fid=True, verbose=False)
    return float(m["frechet_inception_distance"])


def compute_kid(syn_dir, real_dir):
    from cleanfid import fid as cleanfid
    return float(cleanfid.compute_kid(str(real_dir), str(syn_dir), num_workers=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/root/autodl-tmp/data/images")
    ap.add_argument("--synthetic_dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--per_class", action="store_true")
    ap.add_argument("--size", type=int, default=512,
                    help="uniform resize size before FID/KID (paper used 'resized_train')")
    ap.add_argument("--out_dir", default="results_fidkid")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = args.names or [Path(d).name for d in args.synthetic_dirs]
    classes = list_class_dirs(args.real_dir)
    tmp = Path(tempfile.mkdtemp(prefix="fidkid_"))

    real_flat = tmp / "real_all"
    nr = resize_flat(args.real_dir, classes, real_flat, args.size)
    print(f"[resize] real overall = {nr} @ {args.size}px")

    results = {}
    for name, sdir in zip(names, args.synthetic_dirs):
        syn_flat = tmp / f"{name}_all"
        ns = resize_flat(sdir, classes, syn_flat, args.size)
        print(f"[{name}] overall = {ns}  computing FID/KID ...")
        entry = {"n_synthetic": ns,
                 "overall": {"fid": compute_fid(syn_flat, real_flat),
                             "kid": compute_kid(syn_flat, real_flat)}}
        print(f"  overall FID={entry['overall']['fid']:.2f}  KID={entry['overall']['kid']:.4f}")
        if args.per_class:
            entry["per_class"] = {}
            for cls in classes:
                rp = Path(args.real_dir) / cls
                sp = Path(sdir) / cls
                if not sp.is_dir() or not rp.is_dir():
                    continue
                try:
                    rr = tmp / f"real_{cls.replace(' ', '_')}"
                    ss = tmp / f"{name}_{cls.replace(' ', '_')}"
                    resize_class(rp, rr, args.size)
                    resize_class(sp, ss, args.size)
                    entry["per_class"][cls] = {"fid": compute_fid(ss, rr),
                                               "kid": compute_kid(ss, rr)}
                    print(f"  {cls:<12} FID={entry['per_class'][cls]['fid']:.2f}")
                except Exception as e:
                    print(f"  {cls:<12} skipped ({e})")
        results[name] = entry

    with open(out_dir / "fid_kid_paper.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_dir}/fid_kid_paper.json")
    print("\n  generator      FID        KID")
    for name, e in results.items():
        o = e["overall"]
        print(f"  {name:<12}{o['fid']:>8.2f}   {o['kid']:>8.4f}")


if __name__ == "__main__":
    main()
