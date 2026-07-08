#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic weld images for ONE fold, using only that fold's trained models
and that fold's training base-images / masks  -> leakage-free by construction.

Mirrors the paper's Method V (image-to-image + ControlNet) from infer.py:
  init image  = random fold-train base image of the class
  control img = random fold-train mask of the class  (random mask sampling)
  prompt      = "an image of w* with <class> defect on white background"
  LoRA        = this fold's DreamBooth output
  ControlNet  = this fold's ControlNet output

Outputs:
  out_dir/fold{k}/<class>/*.png
  out_dir/fold{k}/manifest.csv   (synthetic_filename, base_real_filename, mask, class)

Run:
  python generate_fold.py --fold 0 \
      --base_model ../stable-diffusion-v1-5 \
      --lora work/fold0/db_out --controlnet work/fold0/cn_out \
      --maskpool work/fold0/maskpool --basepool work/fold0/basepool \
      --out_dir ../data/synthetic --n_per_class 60
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}
# prompt used by the released gradio_demo (mask ControlNet + DreamBooth LoRA, i2i)
PROMPT = "a grayscale photo of two <w*> with {cls} defect"
# folder name -> defect string used at training/inference time (module.py DEFECTS)
CLASS_DISPLAY = {
    "baseline": "Baseline", "low power": "Low power", "low gap": "Low gap",
    "water": "Water", "oil": "Oil", "n-de": "Negative defocus",
    "p-de": "Positive defocus", "cold weld": "Cold weld",
}


def list_imgs(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1,
                    help="fold index -> output out_dir/fold{k}/. Use -1 for UNIFIED/flat "
                         "output (out_dir/<class>/ + out_dir/manifest.csv)")
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="HF hub id (auto-downloaded) or local path")
    ap.add_argument("--lora", required=True, help="fold DreamBooth+LoRA output dir")
    ap.add_argument("--controlnet", required=True, help="fold ControlNet output dir")
    ap.add_argument("--basepool", required=True, help="fold basepool/<class>/")
    ap.add_argument("--maskpool", required=True, help="fold maskpool/<class>/")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/data/synthetic")
    ap.add_argument("--mult", type=int, default=5,
                    help="paper mode: generate this many images PER real base image "
                         "(class total = num_real_in_class * mult, i.e. 5x expansion)")
    ap.add_argument("--n_per_class", type=int, default=0,
                    help="if >0, ignore --mult and generate a FIXED total per class "
                         "(random base+mask sampling)")
    ap.add_argument("--steps", type=int, default=50)          # gradio_demo default
    ap.add_argument("--strength", type=float, default=0.6)     # gradio_demo default
    ap.add_argument("--guidance_scale", type=float, default=3.0)  # gradio_demo default
    ap.add_argument("--prompt_template", default=PROMPT,
                    help="use {cls} placeholder; default matches gradio_demo mask method")
    ap.add_argument("--controlnet_scale", type=float, default=1.0,
                    help="ControlNet conditioning scale; set 0 to disable ControlNet "
                         "(diagnostic: isolates whether ControlNet diverged)")
    ap.add_argument("--lora_scale", type=float, default=1.0,
                    help="LoRA strength (diagnostic: lower if DreamBooth overfit)")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true",
                    help="use fp16 for speed; default is fp32 (fp16 can give black/NaN "
                         "images with heavily-trained weights)")
    args = ap.parse_args()

    from diffusers import (ControlNetModel, EulerAncestralDiscreteScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (args.fp16 and device == "cuda") else torch.float32

    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None,
    ).to(device)
    pipe.load_lora_weights(args.lora)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    # Even in fp16, decode the VAE in fp32 (classic SD1.5 fp16 black-image fix).
    if dtype == torch.float16:
        pipe.vae = pipe.vae.to(dtype=torch.float32)
        pipe.vae.config.force_upcast = True

    base_root = Path(args.basepool)
    mask_root = Path(args.maskpool)
    unified = args.fold < 0
    out_fold = Path(args.out_dir) if unified else Path(args.out_dir) / f"fold{args.fold}"
    out_fold.mkdir(parents=True, exist_ok=True)
    tag_fold = "all" if unified else f"fold{args.fold}"
    classes = sorted([d.name for d in base_root.iterdir() if d.is_dir()])

    manifest = []
    for cls in classes:
        bases = list_imgs(base_root / cls)
        masks = list_imgs(mask_root / cls)
        if not bases or not masks:
            print(f"[{tag_fold}] skip {cls}: bases={len(bases)} masks={len(masks)}")
            continue
        (out_fold / cls).mkdir(parents=True, exist_ok=True)
        prompt = args.prompt_template.format(cls=CLASS_DISPLAY.get(cls, cls))
        rng = random.Random(args.seed + hash(cls) % 10000)

        def gen_one(bpath, mpath, tag):
            init = Image.open(bpath).convert("RGB").resize((args.resolution, args.resolution))
            ctrl = Image.open(mpath).convert("RGB").resize((args.resolution, args.resolution))
            g = torch.Generator(device=device).manual_seed(args.seed * 100000 + tag)
            img = pipe(prompt, image=init, control_image=ctrl,
                       num_inference_steps=args.steps, strength=args.strength,
                       guidance_scale=args.guidance_scale,
                       controlnet_conditioning_scale=args.controlnet_scale,
                       cross_attention_kwargs={"scale": args.lora_scale},
                       generator=g).images[0]
            fname = f"syn_{cls.replace(' ', '_')}_{tag:05d}.png"
            img.save(out_fold / cls / fname)
            manifest.append([fname, bpath.name, mpath.name, cls])

        if args.n_per_class > 0:
            # fixed total per class (random base + random mask)
            for i in range(args.n_per_class):
                gen_one(rng.choice(bases), rng.choice(masks), i)
            print(f"[{tag_fold}] {cls}: generated {args.n_per_class} (fixed)")
        else:
            # paper mode: each real base image -> args.mult synthetic images
            tag = 0
            for bi, bpath in enumerate(bases):
                for j in range(args.mult):
                    gen_one(bpath, rng.choice(masks), tag); tag += 1
            print(f"[{tag_fold}] {cls}: generated {len(bases)}x{args.mult}={tag}")

    with open(out_fold / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["synthetic_filename", "base_real_filename", "mask_filename", "class"])
        w.writerows(manifest)
    print(f"[{tag_fold}] done -> {out_fold} ({len(manifest)} images)")


if __name__ == "__main__":
    main()
