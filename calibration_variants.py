#!/usr/bin/env python3
"""
calibration_variants.py

Usage:
    python calibration_variants.py \
        --internal_glob "results_v2/preds/BRCA_*.csv" \
        --external_glob "results_v2/preds/{ValidTest,TPM_Test,ComBat_Test}_*.csv" \
        --source_task BRCA \
        --out calibration_variants.csv
(If your shell doesn't expand {..}, pass --external_glob "results_v2/preds/*.csv"
 and the script will keep only ValidTest/TPM_Test/ComBat_Test rows.)
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

EPS = 1e-7
EXT_DATASETS = {"ValidTest", "TPM_Test", "ComBat_Test"}
NEED = {"y_true", "y_score", "dataset", "method", "model", "group"}

# Eval-set identity is in the FILENAME, not the internal `dataset` column.
DATASET_PREFIXES = ["ComBat_Test", "TPM_Test", "ValidTest", "BRCA", "UCEC", "THCA"]

def eval_dataset_from_path(path):
    b = os.path.basename(path)
    for d in DATASET_PREFIXES:
        if b.startswith(d + "_") or b == d:
            return d
    return b.split("_")[0]

def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    e, n = 0.0, len(p)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.sum()/n) * abs(y[m].mean() - p[m].mean())
    return e

def wga(y, p, groups, thr=0.5, exclude_unknown=True):
    pred = (p >= thr).astype(int)
    accs = []
    for g in np.unique(groups):
        if exclude_unknown and str(g).upper() == "UNKNOWN":
            continue
        m = groups == g
        if m.any():
            accs.append((y[m] == pred[m]).mean())
    return float(min(accs)) if accs else np.nan

def fit_platt(scores, y):
    """Platt scaling: sigmoid(A*s + B) via 1-D logistic regression on the raw score."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(scores.reshape(-1, 1), y)
    return lr

def apply_platt(lr, scores):
    return lr.predict_proba(scores.reshape(-1, 1))[:, 1]

def oracle_oof(scores, y, n_splits=5):
    """Platt fit on external labels, cross_val_predict -> out-of-fold calibrated probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    if len(np.unique(y)) < 2 or np.bincount(y).min() < n_splits:
        n_splits = max(2, int(np.bincount(y).min()))
    if np.bincount(y).min() < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    return cross_val_predict(lr, scores.reshape(-1, 1), y, cv=skf,
                             method="predict_proba")[:, 1]

def load_units(files):
    """Return dict keyed (eval_dataset, method, model) -> concatenated df.
    eval_dataset comes from the FILENAME, not the internal `dataset` column."""
    out = {}
    for f in files:
        df = pd.read_csv(f)
        if not NEED.issubset(df.columns):
            continue
        eval_ds = eval_dataset_from_path(f)
        for key, d in df.groupby(["method", "model"], dropna=False):
            m, mo = key if isinstance(key, tuple) else (key, None)
            out.setdefault((eval_ds, m, mo), []).append(d)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal_glob", required=True)
    ap.add_argument("--external_glob", required=True)
    ap.add_argument("--source_task", default="BRCA")
    ap.add_argument("--out", default="calibration_variants.csv")
    args = ap.parse_args()

    ext_files = sorted(glob.glob(args.external_glob))
    int_files = sorted(glob.glob(args.internal_glob))
    if not ext_files or not int_files:
        sys.exit("No internal/external files found; check globs.")

    internal = load_units(int_files)   # keyed (dataset, method, model)
    external = load_units(ext_files)
    # source map: (method, model) -> internal source-task df
    src = {(m, mo): d for (ds, m, mo), d in internal.items() if ds == args.source_task}
    if not src:
        sys.exit(f"No internal source-task ({args.source_task}) units found.")

    rows = []
    for (ds, method, model), d in external.items():
        if ds not in EXT_DATASETS:
            continue
        skey = (method, model)
        if skey not in src:
            print(f"[warn] no source for {method}/{model}; skipping {ds}")
            continue
        sd = src[skey]
        ext_s = d["y_score"].astype(float).values
        ext_y = d["y_true"].astype(int).values
        ext_g = d["group"].astype(str).values
        ok = np.isfinite(ext_s)
        ext_s, ext_y, ext_g = ext_s[ok], ext_y[ok], ext_g[ok]
        s_s = sd["y_score"].astype(float).values
        s_y = sd["y_true"].astype(int).values
        sok = np.isfinite(s_s)
        s_s, s_y = s_s[sok], s_y[sok]

        rec = dict(dataset=ds, method=method, model=model, n_ext=len(ext_y))
        # (1) raw: clip to [0,1] only for ECE; WGA uses 0.5 on raw score
        praw = np.clip(ext_s, 0, 1)
        rec["wga_raw"] = wga(ext_y, ext_s, ext_g)
        rec["ece_raw"] = ece(ext_y, praw)
        # (2) source-cal
        try:
            lr = fit_platt(s_s, s_y)
            psrc = apply_platt(lr, ext_s)
            rec["wga_source"] = wga(ext_y, psrc, ext_g)
            rec["ece_source"] = ece(ext_y, psrc)
        except Exception as e:
            rec["wga_source"] = rec["ece_source"] = np.nan
            print(f"[warn] source-cal failed {method}/{model}/{ds}: {e}")
        # (3) oracle-cal (OOF within external)
        try:
            poff = oracle_oof(ext_s, ext_y)
            if poff is None:
                rec["wga_oracle"] = rec["ece_oracle"] = np.nan
            else:
                rec["wga_oracle"] = wga(ext_y, poff, ext_g)
                rec["ece_oracle"] = ece(ext_y, poff)
        except Exception as e:
            rec["wga_oracle"] = rec["ece_oracle"] = np.nan
            print(f"[warn] oracle-cal failed {method}/{model}/{ds}: {e}")
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["dataset", "method", "model"])
    out.to_csv(args.out, index=False)
    print(f"[done] {len(out)} rows -> {args.out}")

    # summary: real_dist, linear vs nonlinear, WGA across variants
    LIN = {"LR", "LR_L1", "LR_L2", "LinearSVM"}
    rd = out[out.method == "real_dist"].copy()
    if len(rd):
        rd["fam"] = np.where(rd.model.isin(LIN), "linear", "nonlinear")
        print("\n[summary] real_dist mean WGA@0.5  (raw -> source -> oracle):")
        g = rd.groupby("fam")[["wga_raw", "wga_source", "wga_oracle"]].mean().round(3)
        print(g.to_string())
        print("\n[summary] real_dist mean ECE  (raw -> source -> oracle):")
        g2 = rd.groupby("fam")[["ece_raw", "ece_source", "ece_oracle"]].mean().round(3)
        print(g2.to_string())

if __name__ == "__main__":
    main()
