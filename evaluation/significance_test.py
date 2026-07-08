#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical significance of the improvement from adding synthetic data.

Addresses Reviewer 1 #2 / Reviewer 2 #7: the word "significant" must be backed by a
statistical test, not a single-split point estimate.

Two complementary tests on the pooled out-of-fold (OOF) predictions produced by
classify_cv.py (one prediction per real sample, across all CV folds):

  1. McNemar's exact test  -- are the two classifiers' error patterns different?
  2. Paired bootstrap       -- 95% CI and p-value for the macro-F1 (and balanced-acc)
                               improvement, resampling the test set with replacement.

Both are *paired*: baseline and method are compared on the SAME samples.

Usage:
  python significance_test.py \
      --baseline results/resnet50/oof_resnet50_real_only.csv \
      --method   results/resnet50/oof_resnet50_real_plus_synth.csv \
      --n_boot 10000 --metric macro_f1
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar


def load_paired(baseline_csv, method_csv):
    """Align baseline and method OOF predictions on sample path."""
    a = pd.read_csv(baseline_csv).set_index("path").sort_index()
    b = pd.read_csv(method_csv).set_index("path").sort_index()
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        print(f"[warn] aligning on {len(common)} common samples "
              f"(baseline={len(a)}, method={len(b)})")
    a, b = a.loc[common], b.loc[common]
    assert (a["y_true"].values == b["y_true"].values).all(), \
        "y_true mismatch after alignment"
    return a["y_true"].values, a["y_pred"].values, b["y_pred"].values


def metric_fn(name):
    if name == "macro_f1":
        return lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    if name == "w_f1":
        return lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)
    if name == "balanced_acc":
        return lambda yt, yp: balanced_accuracy_score(yt, yp)
    raise ValueError(name)


def mcnemar_test(y_true, pred_a, pred_b):
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n00 = int(np.sum(~a_correct & ~b_correct))  # both wrong
    n01 = int(np.sum(~a_correct & b_correct))   # only method correct
    n10 = int(np.sum(a_correct & ~b_correct))   # only baseline correct
    n11 = int(np.sum(a_correct & b_correct))    # both correct
    table = [[n11, n10], [n01, n00]]
    res = mcnemar(table, exact=True)
    return {
        "both_correct": n11, "only_baseline_correct": n10,
        "only_method_correct": n01, "both_wrong": n00,
        "statistic": float(res.statistic), "p_value": float(res.pvalue),
    }


def paired_bootstrap(y_true, pred_a, pred_b, metric, n_boot, seed):
    fn = metric_fn(metric)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = fn(y_true, pred_b) - fn(y_true, pred_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fn(y_true[idx], pred_b[idx]) - fn(y_true[idx], pred_a[idx])
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: fraction of resamples on the wrong side of zero, x2
    p = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "metric": metric,
        "baseline": float(fn(y_true, pred_a) * 100),
        "method": float(fn(y_true, pred_b) * 100),
        "observed_diff": float(obs * 100),
        "ci95_low": float(ci_low * 100),
        "ci95_high": float(ci_high * 100),
        "p_value": float(min(p, 1.0)),
        "n_boot": n_boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="OOF csv for baseline regime")
    ap.add_argument("--method", required=True, help="OOF csv for method regime")
    ap.add_argument("--metric", default="macro_f1",
                    choices=["macro_f1", "w_f1", "balanced_acc"])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y_true, pa, pb = load_paired(args.baseline, args.method)
    print(f"[data] {len(y_true)} paired samples\n")

    mc = mcnemar_test(y_true, pa, pb)
    print("McNemar exact test")
    print("------------------")
    print(f"  both correct        : {mc['both_correct']}")
    print(f"  only baseline right : {mc['only_baseline_correct']}")
    print(f"  only method right   : {mc['only_method_correct']}")
    print(f"  both wrong          : {mc['both_wrong']}")
    print(f"  p-value             : {mc['p_value']:.3e}\n")

    for met in dict.fromkeys([args.metric, "macro_f1", "balanced_acc", "w_f1"]):
        bs = paired_bootstrap(y_true, pa, pb, met, args.n_boot, args.seed)
        print(f"Paired bootstrap ({met}, {bs['n_boot']} resamples)")
        print("-" * 46)
        print(f"  baseline      : {bs['baseline']:.2f}")
        print(f"  method        : {bs['method']:.2f}")
        print(f"  improvement   : {bs['observed_diff']:+.2f} "
              f"(95% CI [{bs['ci95_low']:+.2f}, {bs['ci95_high']:+.2f}])")
        print(f"  p-value       : {bs['p_value']:.3e}\n")

    sig = "p < 0.01" if mc["p_value"] < 0.01 else (
        "p < 0.05" if mc["p_value"] < 0.05 else "n.s.")
    print(f"=> McNemar significance: {sig}")


if __name__ == "__main__":
    main()
