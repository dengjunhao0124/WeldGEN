#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-plot the confusion-matrix figure (Fig. 7 / confusion1.pdf) as a 2x2 grid with
larger, legible fonts. NUMBERS ARE UNCHANGED — only layout and font size change.

>>> IMPORTANT <<<
The values below were read from the low-resolution confusion1.pdf and MAY CONTAIN
ERRORS, especially small off-diagonal cells. VERIFY every number against your original
high-resolution figure before using this output. Only the diagonal (large) values are
reliable; double-check the off-diagonal confusions.

Run:  python replot_confusion.py   ->  confusion_2x2.pdf / .png
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLASSES = ["baseline", "low power", "low gap", "p-de", "n-de", "water", "oil", "cold weld"]

# order of axes: rows/cols follow CLASSES above
# ---- VERIFY THESE AGAINST THE ORIGINAL FIGURE ----
M = {
    "Train: Real, Test: Real": [
        [22, 0, 0, 0, 0, 0, 0, 0],
        [ 2,19, 0, 0, 0, 0, 0, 0],
        [ 0, 0,20, 0, 0, 0, 0, 0],
        [ 0, 0, 0,15, 6, 0, 0, 0],
        [ 1, 0, 0, 1,17, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 4, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 8, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 8],
    ],
    "Train: Synthetic, Test: Synthetic": [
        [155,0,0,0,0,0,0,0],
        [0,168,0,0,0,0,0,0],
        [0,0,155,0,0,0,0,0],
        [0,0,0,152,0,0,0,0],
        [0,0,0,0,152,0,0,0],
        [0,0,0,0,0,45,0,0],
        [0,0,0,0,0,0,72,0],
        [0,0,0,0,0,0,0,63],
    ],
    "Train: Synthetic, Test: Real": [
        [21, 0, 0, 0, 1, 0, 0, 0],
        [ 0,21, 0, 0, 0, 0, 0, 0],
        [ 0, 0,20, 0, 0, 0, 0, 0],
        [ 0, 0, 0,17, 4, 0, 0, 0],
        [ 0, 0, 0, 0,19, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 4, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 8, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 8],
    ],
    "Train: Synthetic + Real, Test: Real": [
        [22, 0, 0, 0, 0, 0, 0, 0],
        [ 0,21, 0, 0, 0, 0, 0, 0],
        [ 0, 0,20, 0, 0, 0, 0, 0],
        [ 0, 0, 0,20, 1, 0, 0, 0],
        [ 0, 0, 0, 0,19, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 4, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 8, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 8],
    ],
}
# --------------------------------------------------

# font sizes (all enlarged vs the original 1x4 layout)
FS_ANNOT, FS_TICK, FS_TITLE, FS_LABEL = 15, 13, 16, 14

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
for ax, (title, mat) in zip(axes.flat, M.items()):
    sns.heatmap(np.array(mat), annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                annot_kws={"size": FS_ANNOT}, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=FS_LABEL)
    ax.set_ylabel("True Label", fontsize=FS_LABEL)
    ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=FS_TICK)
    ax.set_yticklabels(CLASSES, rotation=0, fontsize=FS_TICK)

plt.tight_layout(pad=3.0)
plt.savefig("confusion_2x2.pdf", bbox_inches="tight")
plt.savefig("confusion_2x2.png", dpi=200, bbox_inches="tight")
print("saved -> confusion_2x2.pdf / confusion_2x2.png")
