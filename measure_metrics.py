#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)

from netcal.metrics import ECE

# ----------------------------------------------------------------------
#  Config: subtype / label mapping for PhyloFrame file
#  (for baseline finetune preds we already have y_true as 0/1)
# ----------------------------------------------------------------------
POSITIVE_CLASS_MAP = {
    "BRCA": {"Basal": 0, "Luminal": 1},
    "UCEC": {"Endometrioid": 0, "Serous": 1},
    "THCA": {"M0": 0, "MX": 1},
}

# ----------------------------------------------------------------------
#  Helper functions
# ----------------------------------------------------------------------
def safe_auc(y_true, y_score):
    """AUROC that returns NaN if only one class present."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def npv_score(y_true, y_pred):
    """Negative predictive value = TN / (TN + FN).
    Equivalent to precision of the negative class (pos_label=0)."""
    return precision_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)


def ece_score(y_true, y_score, bins=15):
    """Global ECE using netcal."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(ECE(bins=bins).measure(y_score, y_true))


def predictive_parity_difference(y_true, y_pred, sensitive_features):
    """PPV max-min across groups."""
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "group": np.asarray(sensitive_features),
        }
    )
    ppvs = []
    for g in df["group"].unique():
        sub = df[df["group"] == g]
        tp = ((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum()
        fp = ((sub["y_pred"] == 1) & (sub["y_true"] == 0)).sum()
        if tp + fp == 0:
            ppv = 0.0
        else:
            ppv = tp / (tp + fp)
        ppvs.append(ppv)
    return float(max(ppvs) - min(ppvs)) if ppvs else np.nan


def calibration_within_groups(y_true, y_prob, sensitive_features, n_bins=10):
    """Average calibration error magnitude across groups."""
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_prob": np.asarray(y_prob),
            "group": np.asarray(sensitive_features),
        }
    )
    diffs = []
    for g in df["group"].unique():
        sub = df[df["group"] == g]
        if sub["y_true"].nunique() < 2:
            continue
        try:
            prob, obs = calibration_curve(
                sub["y_true"], sub["y_prob"], n_bins=n_bins, strategy="quantile"
            )
            diffs.append(np.abs(np.asarray(prob) - np.asarray(obs)).mean())
        except Exception:
            continue
    return float(np.mean(diffs)) if diffs else np.nan


# ----------------------------------------------------------------------
#  Baseline metrics: for finetune.py prediction files
#  Columns expected:
#    sample_id,y_true,y_score,y_pred,dataset,method,model,outer,group,...
# ----------------------------------------------------------------------
def compute_baseline_performance(df, logo_eval_scope="outer"):
    """
    Performance:
    - For ANY method: one row per test_group (group)
    - For ANY method: one extra row with test_group='ALL'
    """
    df = df.copy()

    if "seed" in df.columns and df["seed"].nunique() > 1:
        group_cols = ["dataset", "method", "model", "outer", "test_group"]
        per_seed = []
        for s, sdf in df.groupby("seed"):
            result = compute_baseline_performance(
                sdf.drop(columns=["seed"]),
                logo_eval_scope=logo_eval_scope,
            )
            result["seed"] = s
            per_seed.append(result)
        combined = pd.concat(per_seed, ignore_index=True)
        numeric_cols = combined.select_dtypes(include="number").columns.difference(["seed"])
        mean_df = combined.groupby(group_cols, as_index=False)[list(numeric_cols)].mean()
        std_df = combined.groupby(group_cols, as_index=False)[list(numeric_cols)].std()
        std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
        result = mean_df.merge(
            std_df[group_cols + [f"{c}_std" for c in numeric_cols]],
            on=group_cols,
        )
        result["n_seeds"] = df["seed"].nunique()
        return result

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["y_score"] = df["y_score"].astype(float)

    dataset = df["dataset"].iloc[0]
    method = df["method"].iloc[0]
    model = df["model"].iloc[0]
    outer = df["outer"].iloc[0]

    # inside compute_baseline_performance / compute_baseline_fairness
    if method in ("phylo_style", "PhyloFrame", "PhyloBaseline") and "dataset_id" in df.columns:
        #group_col = "dataset_id"
        group_col = "group"

    else:
        group_col = "group"

    # 🔹 LOGO behavior depends on flag
    if method == "LOGO" and logo_eval_scope == "outer":
        # internal-LOGO case: only held-out group
        df = df[df["group"] == outer].copy()

    rows = []

    # 1) Per-group (or per-batch) rows
    for g in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == g]
        y_true = sub["y_true"]
        y_pred = sub["y_pred"]
        y_score = sub["y_score"]

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "model": model,
                "outer": outer,
                "test_group": g,
                "n_samples": len(sub),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "auc": safe_auc(y_true, y_score),
                "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) >= 2 else np.nan,
                "kappa": cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else np.nan,
                "precision": precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
                "npv": npv_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
                "f1": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
                "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "brier": brier_score_loss(y_true, y_score),
                "ece_15bins": ece_score(y_true, y_score, bins=15),
            }
        )

    # 2) ALL row (over TEST data only)
    y_true_all = df["y_true"]
    y_pred_all = df["y_pred"]
    y_score_all = df["y_score"]

    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "model": model,
            "outer": outer,
            "test_group": "ALL",
            "n_samples": len(df),
            "accuracy": accuracy_score(y_true_all, y_pred_all),
            "balanced_accuracy": balanced_accuracy_score(y_true_all, y_pred_all),
            "f1_weighted": f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
            "precision_weighted": precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
            "auc": safe_auc(y_true_all, y_score_all),
            "mcc": matthews_corrcoef(y_true_all, y_pred_all) if len(np.unique(y_true_all)) >= 2 else np.nan,
            "kappa": cohen_kappa_score(y_true_all, y_pred_all) if len(np.unique(y_true_all)) >= 2 else np.nan,
            "precision": precision_score(y_true_all, y_pred_all, average="binary", pos_label=1, zero_division=0),
            "npv": npv_score(y_true_all, y_pred_all),
            "recall": recall_score(y_true_all, y_pred_all, average="binary", pos_label=1, zero_division=0),
            "f1": f1_score(y_true_all, y_pred_all, average="binary", pos_label=1, zero_division=0),
            "recall_weighted": recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
            "brier": brier_score_loss(y_true_all, y_score_all),
            "ece_15bins": ece_score(y_true_all, y_score_all, bins=15),
        }
    )

    return pd.DataFrame(rows)


def compute_baseline_fairness(df, logo_eval_scope="outer"):
    """
    Fairness (one row per file):
    - For methods with >=2 groups in test (phylo_style, real_dist, equal_dist):
        compute DP / EO / EOpp / PP, worst_group_accuracy, etc.
    - For LOGO (usually single held-out group in test):
        disparity metrics become NaN (no comparison).
    """
    df = df.copy()

    if "seed" in df.columns and df["seed"].nunique() > 1:
        group_cols = ["dataset", "method", "model", "outer"]
        per_seed = []
        for s, sdf in df.groupby("seed"):
            result = compute_baseline_fairness(
                sdf.drop(columns=["seed"]),
                logo_eval_scope=logo_eval_scope,
            )
            result["seed"] = s
            per_seed.append(result)
        combined = pd.concat(per_seed, ignore_index=True)
        numeric_cols = combined.select_dtypes(include="number").columns.difference(["seed"])
        mean_df = combined.groupby(group_cols, as_index=False)[list(numeric_cols)].mean()
        std_df = combined.groupby(group_cols, as_index=False)[list(numeric_cols)].std()
        std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
        result = mean_df.merge(
            std_df[group_cols + [f"{c}_std" for c in numeric_cols]],
            on=group_cols,
        )
        result["n_seeds"] = df["seed"].nunique()
        return result

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["y_score"] = df["y_score"].astype(float)

    dataset = df["dataset"].iloc[0]
    method = df["method"].iloc[0]
    model = df["model"].iloc[0]
    outer = df["outer"].iloc[0]


    # inside compute_baseline_performance / compute_baseline_fairness
    if method in ("phylo_style", "PhyloFrame", "PhyloBaseline") and "dataset_id" in df.columns:
        # group_col = "dataset_id"
        group_col = "group"
    else:
        group_col = "group"

    # 🔹 LOGO behavior depends on flag
    if method == "LOGO" and logo_eval_scope == "outer":
        # internal-LOGO case: only held-out group
        df = df[df["group"] == outer].copy()

    y_true = df["y_true"]
    y_pred = df["y_pred"]
    y_score = df["y_score"]
    groups = df[group_col].astype(str)
    unique_groups = groups.unique()
    n_groups = len(unique_groups)

    # Defaults
    dp_diff = eo_diff = eopp_diff = pp_diff = calib_diff = worst_group_acc = np.nan

    # Only defined if >= 2 groups present
    if n_groups >= 2:
        dp_diff = demographic_parity_difference(
            y_true, y_pred, sensitive_features=groups
        )
        eo_diff = equalized_odds_difference(
            y_true, y_pred, sensitive_features=groups
        )
        eopp_diff = equal_opportunity_difference(
            y_true, y_pred, sensitive_features=groups
        )
        pp_diff = predictive_parity_difference(y_true, y_pred, groups)
        calib_diff = calibration_within_groups(y_true, y_score, groups)

        # Worst group accuracy
        accs = []
        for g in unique_groups:
            mask = groups == g
            if mask.sum() == 0:
                continue
            accs.append(accuracy_score(y_true[mask], y_pred[mask]))
        worst_group_acc = min(accs) if accs else np.nan

    fairness_row = {
        "dataset": dataset,
        "method": method,
        "model": model,
        "outer": outer,
        "n_samples": len(df),
        "n_groups": n_groups,
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "eopp_diff": eopp_diff,
        "pp_diff": pp_diff,
        "calib_within_groups": calib_diff,
        "ece_15bins": ece_score(y_true, y_score, bins=15),
        "avg_accuracy": accuracy_score(y_true, y_pred),
        "avg_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "worst_group_accuracy": worst_group_acc,
    }

    return pd.DataFrame([fairness_row])

# ----------------------------------------------------------------------
#  PhyloFrame / Benchmark metrics (their big CSV)
#  Expected columns (from your earlier description):
#   sample_id,model_train_data,model_num,test_data,test_batch_num,
#   subtype,disease,PF_.pred_class,PF_.pred_subtype1,...,
#   BM_.pred_class,BM_.pred_subtype1,...
# ----------------------------------------------------------------------
def prepare_phylo_labels(sub_df, cancer_upper, pred_col, prob_col):
    """
    Return y_true, y_pred, y_score, ancestry_group, dataset_id, used_df
    for a PhyloFrame-style subset.
    """
    mapping = POSITIVE_CLASS_MAP[cancer_upper]

    # True labels from string subtype
    y_true = sub_df["subtype"].map(mapping)
    mask = ~y_true.isna()
    sub_df = sub_df.loc[mask].copy()
    y_true = y_true.loc[mask].astype(int)

    # Predicted labels: string or numeric
    raw_pred = sub_df[pred_col]
    if raw_pred.dtype == object:
        y_pred = raw_pred.map(mapping).astype(int)
    else:
        y_pred = raw_pred.astype(int)

    y_score = sub_df[prob_col].astype(float)

    # Ancestry-level group (eur/afr/eas/admix)
    groups = sub_df["test_data"].astype(str)

    # Dataset-level identifier:
    #   - internal: "eur_batch14"
    #   - external (no test_batch_num): just "eur" / "afr" / ...
    if "test_batch_num" in sub_df.columns:
        dataset_ids = (
            sub_df["test_data"].astype(str) + "_" + sub_df["test_batch_num"].astype(str)
        )
    else:
        dataset_ids = sub_df["test_data"].astype(str)

    return y_true, y_pred, y_score, groups, dataset_ids, sub_df


def compute_phylo_metrics(df):
    """
    For PhyloFrame-style file (per-sample predictions):

    We compute metrics PER MODEL:

      model is defined by (disease, model_train_data, model_num, model_type)

      model_type ∈ {PhyloFrame, PhyloBaseline}
      model_train_data ∈ {admix, afr, eur, eas, ...}
      model_num ∈ {model_1, model_2, ...}

    For each such model, we:
      - build dataset_id = test_data + "_" + test_batch_num
      - performance: per dataset_id + ALL
      - fairness: across dataset_id (if ≥ 2 datasets)
    """
    perf_rows = []
    fair_rows = []

    # model_name, pred_class_col, prob_col
    model_specs = [
        ("PhyloFrame", "PF_.pred_class", "PF_.pred_subtype2"),
        ("PhyloBaseline", "BM_.pred_class", "BM_.pred_subtype2"),
    ]

    for cancer in df["disease"].unique():
        sub_cancer = df[df["disease"] == cancer].copy()
        cancer_upper = cancer.upper()

        for model_name, pred_col, prob_col in model_specs:
            if pred_col not in sub_cancer.columns:
                continue

            # 🔹 Loop over each trained model separately
            #    (train ancestry × model_num)
            for (train_data, model_id), sub_model in sub_cancer.groupby(
                ["model_train_data", "model_num"]
            ):
                (
                    y_true,
                    y_pred,
                    y_score,
                    groups,        # ancestry (eur/afr/eas/admix)
                    dataset_ids,   # dataset-level id (eur_batch14, ...)
                    used_df,
                ) = prepare_phylo_labels(sub_model, cancer_upper, pred_col, prob_col)

                tmp = pd.DataFrame(
                    {
                        "dataset": cancer_upper,
                        "method": model_name,       # "PhyloFrame" or "PhyloBaseline"
                        "model": model_name,
                        # we can use outer to encode training data if you like
                        "outer": str(train_data),
                        "y_true": y_true.values,
                        "y_pred": y_pred.values,
                        "y_score": y_score.values,
                        "group": groups.values,          # ancestry info
                        "dataset_id": dataset_ids.values  # batch-level grouping
                    }
                )

                perf_df = compute_baseline_performance(tmp)
                fair_df = compute_baseline_fairness(tmp)

                # 🔹 Attach which model this row belongs to
                perf_df["train_data"] = train_data
                perf_df["model_num"] = model_id

                fair_df["train_data"] = train_data
                fair_df["model_num"] = model_id

                perf_rows.append(perf_df)
                fair_rows.append(fair_df)

    perf_all = pd.concat(perf_rows, ignore_index=True) if perf_rows else pd.DataFrame()
    fair_all = pd.concat(fair_rows, ignore_index=True) if fair_rows else pd.DataFrame()

    return perf_all, fair_all
# ----------------------------------------------------------------------
#  Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to prediction CSV file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "phylo"],
        help="baseline: finetune.py preds; phylo: PhyloFrame+benchmark CSV",
    )
    parser.add_argument(
        "--outdir",
        default="measure_results",
        help="Directory to write performance/fairness CSVs",
    )

    # 🔹 NEW: control how LOGO is evaluated
    parser.add_argument(
        "--logo_eval_scope",
        choices=["outer", "all"],
        default="outer",
        help="For LOGO: 'outer' = only evaluate held-out group; 'all' = use all rows.",
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.file)
    if "group" in df.columns:
        df["group"] = df["group"].astype(str).str.strip().str.lower()
    if "outer" in df.columns:
        df["outer"] = df["outer"].astype(str).str.strip().str.lower()

    if args.mode == "baseline":
        perf_df = compute_baseline_performance(df, logo_eval_scope=args.logo_eval_scope)
        fair_df = compute_baseline_fairness(df, logo_eval_scope=args.logo_eval_scope)

        base = os.path.splitext(os.path.basename(args.file))[0]
        perf_path = os.path.join(args.outdir, f"{base}_performance.csv")
        fair_path = os.path.join(args.outdir, f"{base}_fairness.csv")

        perf_df.to_csv(perf_path, index=False)
        fair_df.to_csv(fair_path, index=False)
        print(f"✅ Baseline performance → {perf_path}")
        print(f"✅ Baseline fairness   → {fair_path}")

    elif args.mode == "phylo":
        perf_df, fair_df = compute_phylo_metrics(df)

        base = os.path.splitext(os.path.basename(args.file))[0]
        perf_path = os.path.join(args.outdir, f"{base}_performance.csv")
        fair_path = os.path.join(args.outdir, f"{base}_fairness.csv")

        perf_df.to_csv(perf_path, index=False)
        fair_df.to_csv(fair_path, index=False)
        print(f"✅ Phylo performance → {perf_path}")
        print(f"✅ Phylo fairness   → {fair_path}")


if __name__ == "__main__":
    main()