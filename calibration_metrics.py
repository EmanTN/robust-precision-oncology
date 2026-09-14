#!/usr/bin/env python3
"""
calibration_metrics.py
----------------------
Threshold-INDEPENDENT calibration + discrimination evidence for the
"calibration is the mechanism" story. Computes, per (dataset, method, model, outer):
    n, AUROC, ECE(15-bin), Brier, calibration-in-the-large (CITL),
    Cox calibration slope + intercept, WGA@0.5  (for the AUROC/ECE/WGA contrast table)
plus optional per-group ECE and reliability-curve data.

Runs on the EXISTING prediction files (results_v2/preds/*.csv). No model rerun.
Reads dataset/method/model/group/outer from INSIDE each file (no filename parsing).

CORRECTNESS GUARD: probability-calibration metrics (ECE/Brier/CITL/slope) are only
valid when y_score is a probability in [0,1]. LinearSVM uses decision_function ->
raw margin, NOT a probability. Those rows are flagged score_type='margin' and their
calibration metrics are set NaN (AUROC is still valid; it only needs monotone scores).

Usage:
    python calibration_metrics.py \
        --pred_glob "results_v2/preds/*.csv" \
        --out calibration_metrics.csv \
        --reliability_out reliability_curve_data.csv \
        --exclude_unknown_for_wga    # ValidTest WGA/EO excl-UNKNOWN, per locked decision

NOTE on ECE binning: equal-width 15-bin (the common definition). If measure_metrics.py
uses a different binning (e.g. equal-frequency), absolute ECE will differ slightly;
the cross-model CONTRAST (linear low vs nonlinear high) is unaffected. Set --n_bins to match.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

EPS = 1e-7

# Eval-set identity lives in the FILENAME prefix, not the internal `dataset` column
# (external pred files carry dataset=ValidTrain/TPM_Train/etc.). Longest first.
DATASET_PREFIXES = ["ComBat_Test", "TPM_Test", "ValidTest", "BRCA", "UCEC", "THCA"]

def eval_dataset_from_path(path):
    b = os.path.basename(path)
    for d in DATASET_PREFIXES:
        if b.startswith(d + "_") or b == d:
            return d
    return b.split("_")[0]  # fallback

def ece_equal_width(y_true, p, n_bins=15):
    """Expected Calibration Error, equal-width bins on [0,1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)  # 0..n_bins-1
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return ece

def reliability_bins(y_true, p, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(dict(bin=b, bin_lo=bins[b], bin_hi=bins[b+1],
                         mean_pred=float(p[m].mean()),
                         frac_pos=float(y_true[m].mean()),
                         count=int(m.sum())))
    return rows

def cox_slope_intercept(y_true, p):
    """
    Cox calibration regression: logit(y_true) ~ a + b*logit(p).
    slope b: 1=perfect, <1=overconfident. intercept a relates to calibration-in-the-large.
    Uses sklearn LogisticRegression with near-zero regularization.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return np.nan, np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    pc = np.clip(p, EPS, 1 - EPS)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
        lr.fit(z, y_true)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan

def safe_auroc(y_true, s):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, s))
    except Exception:
        return np.nan

def wga_at_05(df, exclude_unknown):
    """Worst-group accuracy at fixed 0.5 from y_score (recomputed, not from y_pred,
    so it's threshold-policy independent)."""
    d = df.copy()
    d["pred05"] = (d["y_score"].values >= 0.5).astype(int)
    if exclude_unknown:
        d = d[d["group"].astype(str).str.upper() != "UNKNOWN"]
    accs = []
    for g, sub in d.groupby("group"):
        if len(sub) == 0:
            continue
        accs.append((sub["y_true"].values == sub["pred05"].values).mean())
    return float(min(accs)) if accs else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_glob", required=True,
                    help='e.g. "results_v2/preds/*.csv"')
    ap.add_argument("--out", default="calibration_metrics.csv")
    ap.add_argument("--reliability_out", default="reliability_curve_data.csv")
    ap.add_argument("--pergroup_out", default="calibration_per_group.csv")
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--exclude_unknown_for_wga", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pred_glob))
    if not files:
        sys.exit(f"No files match {args.pred_glob}")
    print(f"[info] {len(files)} prediction files")

    # schema check on first file
    need = {"y_true", "y_score", "dataset", "method", "model", "group"}
    sample = pd.read_csv(files[0], nrows=5)
    missing = need - set(sample.columns)
    if missing:
        sys.exit(f"[schema] first file missing columns {missing}. "
                 f"Got: {list(sample.columns)}")
    print(f"[schema] OK. columns: {list(sample.columns)}")

    rows, rel_rows, pg_rows = [], [], []
    for f in files:
        df = pd.read_csv(f)
        if not need.issubset(df.columns):
            print(f"[skip] {os.path.basename(f)} missing cols")
            continue
        eval_ds = eval_dataset_from_path(f)   # reliable eval-set label
        # one file can contain multiple outer folds (LOGO); group by the unit
        keycols = ["method", "model"]
        if "outer" in df.columns:
            keycols.append("outer")
        for key, d in df.groupby(keycols, dropna=False):
            if isinstance(key, tuple):
                kd = dict(zip(keycols, key))
            else:
                kd = {keycols[0]: key}
            kd = {"dataset": eval_ds, **kd}
            y = d["y_true"].astype(int).values
            s = d["y_score"].astype(float).values
            ok = np.isfinite(s)
            y, s, dd = y[ok], s[ok], d[ok]
            if len(y) == 0:
                continue
            in_unit = float(np.nanmin(s)) >= -EPS and float(np.nanmax(s)) <= 1 + EPS
            score_type = "prob" if in_unit else "margin"
            auroc = safe_auroc(y, s)
            if score_type == "prob":
                p = np.clip(s, 0, 1)
                ece = ece_equal_width(y, p, args.n_bins)
                brier = float(np.mean((p - y) ** 2))
                citl = float(y.mean() - p.mean())  # calibration-in-the-large
                slope, icpt = cox_slope_intercept(y, p)
                for rb in reliability_bins(y, p, args.n_bins):
                    rb.update(kd); rel_rows.append(rb)
                # per-group ECE (calibration fairness angle)
                for g, sub in dd.groupby("group"):
                    pg = np.clip(sub["y_score"].astype(float).values, 0, 1)
                    yg = sub["y_true"].astype(int).values
                    if len(yg) >= 10 and np.isfinite(pg).all():
                        pg_rows.append(dict(**kd, group=str(g), n=len(yg),
                                            ece=ece_equal_width(yg, pg, args.n_bins),
                                            brier=float(np.mean((pg - yg) ** 2))))
            else:
                ece = brier = citl = slope = icpt = np.nan
            rows.append(dict(**kd, n=len(y), score_type=score_type,
                             auroc=auroc, ece=ece, brier=brier,
                             citl=citl, calib_slope=slope, calib_intercept=icpt,
                             wga05=wga_at_05(dd, args.exclude_unknown_for_wga)))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    pd.DataFrame(rel_rows).to_csv(args.reliability_out, index=False)
    pd.DataFrame(pg_rows).to_csv(args.pergroup_out, index=False)
    print(f"[done] {len(out)} units -> {args.out}")
    print(f"        reliability -> {args.reliability_out}")
    print(f"        per-group   -> {args.pergroup_out}")

    # quick contrast preview: real_dist, prob-score models, mean by linear/nonlinear x int/ext
    LIN = {"LR", "LR_L1", "LR_L2"}  # LinearSVM excluded: margin, no valid ECE
    EXT = {"ValidTest", "TPM_Test", "ComBat_Test", "TPM", "ComBat"}
    pv = out[(out.method == "real_dist") & (out.score_type == "prob")].copy()
    if len(pv):
        pv["fam"] = np.where(pv.model.isin(LIN), "linear", "nonlinear")
        pv["loc"] = np.where(pv.dataset.isin(EXT), "external", "internal")
        print("\n[contrast] real_dist, mean AUROC / ECE / WGA@0.5:")
        g = pv.groupby(["fam", "loc"])[["auroc", "ece", "wga05"]].mean().round(3)
        print(g.to_string())

if __name__ == "__main__":
    main()
