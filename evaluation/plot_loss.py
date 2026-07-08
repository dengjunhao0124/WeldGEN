#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the ControlNet training loss curve from loss_log.csv (written by
train_controlnet_i2i_dream.py).

A healthy run: loss trends DOWN and flattens. A diverging run (e.g. lr too high):
loss is low early then climbs -> exactly the symptom that motivated lowering the lr.

Usage:
  python plot_loss.py --csv /root/autodl-tmp/work/fold2/cn_out/loss_log.csv
  python plot_loss.py --csv .../loss_log.csv --smooth 50 --out loss_curve.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to loss_log.csv")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside csv)")
    ap.add_argument("--smooth", type=int, default=50,
                    help="moving-average window for the smoothed curve")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = args.out or str(Path(args.csv).with_name("loss_curve.png"))

    step, loss = df["step"].values, df["loss"].values
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(step, loss, color="lightsteelblue", lw=0.7, label="loss (raw)")
    if len(loss) >= args.smooth > 1:
        k = args.smooth
        sm = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax1.plot(step[k - 1:], sm, color="navy", lw=1.8,
                 label=f"loss (MA-{k})")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("MSE loss")
    ax1.grid(alpha=0.3)

    if "lr" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(step, df["lr"].values, color="darkorange", lw=1.0,
                 alpha=0.6, label="lr")
        ax2.set_ylabel("learning rate")

    lines = ax1.get_lines() + (ax2.get_lines() if "lr" in df.columns else [])
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    plt.title(f"ControlNet training loss  ({Path(args.csv).parent.name})")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"saved -> {out}")
    print(f"first/last smoothed loss: {loss[:20].mean():.4f} -> {loss[-20:].mean():.4f}")


if __name__ == "__main__":
    main()
