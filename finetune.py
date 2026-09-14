"""
finetune.py

Grid-search style hyperparameter tuning driven by a YAML config.
Writes per-(dataset, training_method, model[, outer_unit_value]) CSVs of best params.

Usage
-----
python finetune.py --config config.yaml
python finetune.py --config config.yaml --only BRCA,THCA --methods phylo_style,LOGO

Outputs
-------
{output.best_params_dir}/{dataset}_{method}_{model}_best_params.csv
For phylo_style, we also include train_batch (outer unit value):
{output.best_params_dir}/{dataset}_{method}_{model}_train-{outer}.csv
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
import numpy as np
import pandas as pd
import yaml
import ast
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_X_y
from models import build_model_and_grid

POSITIVE_CLASS_MAP = {
    "BRCA": {"Basal": 0, "Luminal": 1},
    "UCEC": {"Endometrioid": 0, "Serous": 1},
    "THCA": {"M0": 0, "MX": 1},
}

def split_for_method_outer(
    df,
    data_cfg,
    method_cfg,
    outer_name: str,
    seed: int,
):
    method_name = method_cfg["name"]
    label_col   = data_cfg.label_col
    group_col   = data_cfg.group_col
    batch_col   = data_cfg.batch_col  # needed for phylo_style

    # ⚠️ IMPORTANT: mimic tuning logic → use only gene features
    meta_cols   = getattr(data_cfg, "meta_cols", [])
    feat_cols   = _feature_cols(df, meta_cols)  # excludes [patient, group, dataset_id, subtype]

    X_full = df[feat_cols]
    y_full = df[label_col]
    idx    = df.index

   # ---------- phylo_style ----------
    if method_name == "phylo_style":
        if batch_col not in df.columns:
            raise KeyError(f"batch_col '{batch_col}' not found in dataframe")

        if outer_name == "":
            raise ValueError("outer_name (training batch) must be provided for phylo_style in best_run")

        batches = df[batch_col].astype(str)
        train_mask = (batches == str(outer_name))
        test_mask  = ~train_mask

        if not train_mask.any():
            raise ValueError(f"No samples found for training batch '{outer_name}' in phylo_style")

        if not test_mask.any():
            raise ValueError(f"No samples outside training batch '{outer_name}' to use as test in phylo_style")

        train_idx = idx[train_mask]
        test_idx  = idx[test_mask]

        X_tr, y_tr = X_full.loc[train_idx], y_full.loc[train_idx]
        X_te, y_te = X_full.loc[test_idx], y_full.loc[test_idx]
        return X_tr, X_te, y_tr, y_te, test_idx

    # ---------- real_dist ----------
    if method_name == "real_dist":
        # sample fraction pct from each group as train, rest = test
        pct = float(method_cfg.get("sample", {}).get("pct", 0.3))
        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' not found in dataframe")

        # Reuse the *same* helper as tuning so the split is identical
        train_idx, test_idx = _split_real_dist(df, group_col, pct, seed)

        X_tr, y_tr = X_full.loc[train_idx], y_full.loc[train_idx]
        X_te, y_te = X_full.loc[test_idx], y_full.loc[test_idx]
        return X_tr, X_te, y_tr, y_te, test_idx

    # ---------- LOGO ----------
    if method_name == "LOGO":
        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' not found in dataframe")

        if outer_name == "":
            raise ValueError("outer_name must be provided for LOGO in best_run")

        groups = df[group_col].astype(str)
        outer_name_str = str(outer_name)

        test_mask = (groups == outer_name_str)
        train_mask = ~test_mask

        if not test_mask.any():
            raise ValueError(f"No samples found for LOGO outer group '{outer_name_str}'")

        train_idx = idx[train_mask]
        test_idx = idx[test_mask]

        X_tr, y_tr = X_full.loc[train_idx], y_full.loc[train_idx]
        X_te, y_te = X_full.loc[test_idx], y_full.loc[test_idx]
        return X_tr, X_te, y_tr, y_te, test_idx
        
    # ---------- equal_dist ----------
    if method_name == "equal_dist":
        # Use the same dynamic equal-sampling strategy as in tuning
        frac = float(method_cfg.get("sample", {}).get("train_frac_of_smallest", 0.3))

        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' not found in dataframe")

        # Reuse the same helper as tuning; seed makes it deterministic
        train_idx, test_idx = _split_equal_dist_dynamic(df, group_col, frac, seed)

        X_tr, y_tr = X_full.loc[train_idx], y_full.loc[train_idx]
        X_te, y_te = X_full.loc[test_idx], y_full.loc[test_idx]
        return X_tr, X_te, y_tr, y_te, test_idx
# ------------------------------
# Dataclasses for parsed config
# ------------------------------
@dataclass
class DataCfg:
    label_col: str
    batch_col: str
    group_col: str
    meta_cols: List[str]
    fairness_groups: List[str]

# ------------------------------
# Helpers
# ------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def enforce_label_mapping(df, dataset_name, label_col):
    mapping = POSITIVE_CLASS_MAP.get(dataset_name)
    if mapping is None:
        return df  # fall back to your existing encoding
    # If labels are strings, map directly. If already ints, skip.
    if not np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].map(mapping)
    return df

def load_and_preprocess_dataset(dataset_name, data_cfg, datasets_cfg, missing_gene_thresh):
    """
    Load a dataset by name from `datasets_cfg` and apply the standard preprocessing:
      - drop duplicate label column "<label_col>.1" if present
      - encode the label column to numeric codes if it is not already numeric
      - drop columns with a fraction of missing values greater than `missing_gene_thresh`
      - apply any dataset-specific label mapping via `enforce_label_mapping`
    """
    d = next((d for d in datasets_cfg if d["name"] == dataset_name), None)
    if d is None:
        raise KeyError(f"Dataset '{dataset_name}' not found in config.yaml")

    path = d["path"]
    df = pd.read_csv(path)

    # Clean potential duplicate '<label>.1' column
    dup_col = f"{data_cfg.label_col}.1"
    if dup_col in df.columns:
        df = df.drop(columns=[dup_col]).copy()

    # Drop genes/columns with too much missingness
    df = df.loc[:, df.isna().mean() <= missing_gene_thresh]

    # Apply any dataset-specific label mapping
    df = enforce_label_mapping(df, dataset_name, data_cfg.label_col)

    # Ensure label is encoded to ints (e.g., for AUROC)
    if not np.issubdtype(df[data_cfg.label_col].dtype, np.number):
        df[data_cfg.label_col] = pd.Categorical(df[data_cfg.label_col]).codes


    return df

    
def stratified_kfold_from_cfg(inner_cfg: dict, seed: int) -> StratifiedKFold:
    n_splits = int(inner_cfg.get("n_splits", 5))
    shuffle = bool(inner_cfg.get("shuffle", True))
    random_state = int(inner_cfg.get("random_state", seed))
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def is_method_using_worst_group(method_name: str, tuning_cfg: dict) -> bool:
    lst = tuning_cfg.get("use_worst_group_for", []) or []
    return method_name in lst


def get_primary_metric_name(tuning_cfg: dict) -> str:
    return str(tuning_cfg.get("primary_metric", "auroc")).lower()


def _predict_proba_safe(clf, X) -> np.ndarray:
    # Works for sklearn, xgb, and our MLP wrapper
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # Fallback: decision_function -> convert with sigmoid-ish scaling
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        # Rank-based scaling to [0,1] if needed
        r = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return r
    raise RuntimeError("Classifier has neither predict_proba nor decision_function.")


def _auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Handle edge cases where a fold might contain a single class
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def _worst_group_auroc(y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray) -> float:
    vals = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            continue
        vals.append(_auroc(y_true[mask], y_prob[mask]))
    if not vals:
        return np.nan
    return float(np.nanmin(vals))


def _score_fold(y_true, y_prob, method_name: str, tuning_cfg: dict, val_groups: Optional[np.ndarray]) -> float:
    if is_method_using_worst_group(method_name, tuning_cfg) and val_groups is not None:
        return _worst_group_auroc(y_true, y_prob, val_groups)
    return _auroc(y_true, y_prob)


def _feature_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in meta_cols]


def _split_real_dist(df: pd.DataFrame, group_col: str, pct: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    for g, gdf in df.groupby(group_col):
        n = len(gdf)
        k = max(1, int(round(pct * n)))
        choice = rng.choice(gdf.index.values, size=min(k, n), replace=False)
        train_idx.extend(choice.tolist())
    train_idx = np.array(sorted(set(train_idx)))
    test_idx = df.index.difference(train_idx).to_numpy()
    return train_idx, test_idx


def _split_equal_dist_dynamic(df: pd.DataFrame, group_col: str, frac_smallest: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = df[group_col].value_counts()
    n_min = int(sizes.min())
    k = max(1, int(round(frac_smallest * n_min)))
    train_idx = []
    for g, gdf in df.groupby(group_col):
        m = min(k, len(gdf))
        choice = rng.choice(gdf.index.values, size=m, replace=False)
        train_idx.extend(choice.tolist())
    train_idx = np.array(sorted(set(train_idx)))
    test_idx = df.index.difference(train_idx).to_numpy()
    return train_idx, test_idx

# ------------------------------
# Core tuning routines
# ------------------------------

def tune_for_method(
    df: pd.DataFrame,
    dataset_name: str,
    method_cfg: dict,
    data_cfg: DataCfg,
    preprocessing_cfg: dict,
    models_cfg: List[dict],
    tuning_cfg: dict,
    seed: int,
    repeat_seeds: List[int],
    out_dir: str,
    outer_fixed: str = "",
    model_filter: Optional[Set[str]] = None,
) -> None:
    label_col = data_cfg.label_col
    group_col = data_cfg.group_col
    batch_col = data_cfg.batch_col
    meta_cols = data_cfg.meta_cols
    feat_cols = _feature_cols(df, meta_cols)

    primary_metric = get_primary_metric_name(tuning_cfg)

    method_name = method_cfg["name"]

    if method_name == "phylo_style":
        # Outer unit: each batch_col value can become its own tiny train set
        inner = stratified_kfold_from_cfg(method_cfg.get("inner_cv", {}), seed)

        batches_str = df[batch_col].astype(str)
        unique_batches = sorted(batches_str.unique())

        # If --outer is provided, use that single batch; otherwise loop over all batches
        if outer_fixed:
            if outer_fixed not in unique_batches:
                raise ValueError(
                    f"Requested outer batch '{outer_fixed}' for phylo_style "
                    f"not found in {batch_col} values: {unique_batches}"
                )
            allowed_outers = [outer_fixed]
        else:
            allowed_outers = unique_batches

        for outer_value in allowed_outers:
            sub = df[batches_str == str(outer_value)].copy()
            X = sub[feat_cols].values
            y = sub[label_col].values
            groups_val = sub[group_col].values  # for potential worst-group scoring (not used for phylo)
            # sanity
            check_X_y(X, y, accept_sparse=False)

            # Compute class imbalance ratio for XGB
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            imbalance_ratio = round(n_neg / max(n_pos, 1), 2)

            for m in models_cfg:
                if model_filter and m["name"] not in model_filter:
                    continue
                est, grid = build_model_and_grid(m, preprocessing_cfg, seed)
                # Inject dynamic scale_pos_weight for XGB
                if m.get("family", "").lower() == "xgb":
                    grid["scale_pos_weight"] = [1, imbalance_ratio]
                best_score = -np.inf
                best_params = None

                # Build CV splits
                for params in _grid_iter(grid):
                    scores = []
                    for train_idx, val_idx in inner.split(X, y):
                        X_tr, y_tr = X[train_idx], y[train_idx]
                        X_va, y_va = X[val_idx], y[val_idx]

                        model = _set_params(est, params)
                        model = _guard_calibrated_cv(model, y_tr)
                        model.fit(X_tr, y_tr)
                        y_proba = _predict_proba_safe(model, X_va)
                        sc = _score_fold(y_va, y_proba, method_name, tuning_cfg, groups_val[val_idx])
                        scores.append(sc)

                    mean_sc = np.nanmean(scores)
                    if mean_sc > best_score:
                        best_score = mean_sc
                        best_params = params

                _write_best_params(
                    out_dir,
                    dataset_name,
                    method_name,
                    m["name"],
                    best_params,
                    extra_suffix=f"train-{outer_value}",
                )
        return
    elif method_name == "LOGO":
        inner = stratified_kfold_from_cfg(method_cfg.get("inner_cv", {}), seed)

        # Use string labels so they match args.outer
        groups_str = df[group_col].astype(str)
        unique_groups = sorted(groups_str.unique())

        # If --outer is provided, only tune for that held-out group.
        # Otherwise, loop over all groups (full LOGO sweep).
        if outer_fixed:
            if outer_fixed not in unique_groups:
                raise ValueError(
                    f"Requested LOGO outer '{outer_fixed}' not found in {group_col} values: {unique_groups}"
                )
            logo_outers = [outer_fixed]
        else:
            logo_outers = unique_groups

        for g_left in logo_outers:
            # Train on ALL groups except the left-out one (LOGO outer fold)
            train_df = df[groups_str != g_left].copy()
            X = train_df[feat_cols].values
            y = train_df[label_col].values
            groups_val = train_df[group_col].astype(str).values

            # Sanity check
            check_X_y(X, y, accept_sparse=False)

            # Compute class imbalance ratio for XGB
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            imbalance_ratio = round(n_neg / max(n_pos, 1), 2)

            for m in models_cfg:
                if model_filter and m["name"] not in model_filter:
                    continue

                est, grid = build_model_and_grid(m, preprocessing_cfg, seed)
                # Inject dynamic scale_pos_weight for XGB
                if m.get("family", "").lower() == "xgb":
                    grid["scale_pos_weight"] = [1, imbalance_ratio]
                best_score = -np.inf
                best_params = None

                # Inner CV on the union of all training groups (for this g_left)
                for params in _grid_iter(grid):
                    scores = []

                    for tr_idx, va_idx in inner.split(X, y):
                        X_tr, y_tr = X[tr_idx], y[tr_idx]
                        X_va, y_va = X[va_idx], y[va_idx]

                        model = _set_params(est, params)
                        model = _guard_calibrated_cv(model, y_tr)
                        model.fit(X_tr, y_tr)
                        y_proba = _predict_proba_safe(model, X_va)

                        sc = _score_fold(
                            y_va,
                            y_proba,
                            method_name,
                            tuning_cfg,
                            groups_val[va_idx],
                        )
                        scores.append(sc)

                    mean_sc = np.nanmean(scores)
                    if mean_sc > best_score:
                        best_score = mean_sc
                        best_params = params.copy()

                # File name matches best_run candidate1 pattern
                _write_best_params(
                    out_dir,
                    dataset_name,
                    method_name,
                    m["name"],
                    best_params,
                    extra_suffix=f"train-{g_left}",
                )

    elif method_name == "real_dist":
        pct = float(method_cfg.get("sample", {}).get("pct", 0.3))
        inner = stratified_kfold_from_cfg(method_cfg.get("inner_cv", {}), seed)

        for m in models_cfg:
            if model_filter and m["name"] not in model_filter:
                continue
            est, grid = build_model_and_grid(m, preprocessing_cfg, seed)

            for s in repeat_seeds:
                train_idx, _ = _split_real_dist(df, group_col, pct, s)
                sub = df.loc[train_idx]
                X = sub[feat_cols].values
                y = sub[label_col].values
                groups_val = sub[group_col].values

                # Compute imbalance ratio from THIS seed's TRAINING split (no test leakage)
                if m.get("family", "").lower() == "xgb":
                    n_pos = (y == 1).sum()
                    n_neg = (y == 0).sum()
                    grid["scale_pos_weight"] = [1, round(n_neg / max(n_pos, 1), 2)]

                # Tune independently for THIS seed (no cross-seed pooling)
                param_scores = {}
                for params in _grid_iter(grid):
                    # key = tuple(sorted(params.items()))
                    key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)for k, v in params.items()))
                    scores = []
                    for tr_idx, va_idx in inner.split(X, y):
                        model = _set_params(est, params)
                        model = _guard_calibrated_cv(model, y[tr_idx])
                        model.fit(X[tr_idx], y[tr_idx])
                        y_proba = _predict_proba_safe(model, X[va_idx])
                        sc = _score_fold(y[va_idx], y_proba, method_name, tuning_cfg, groups_val[va_idx])
                        scores.append(sc)
                    param_scores[key] = np.nanmean(scores)

                best_key = max(param_scores, key=lambda k: param_scores[k])
                best_params = dict(best_key)
                _write_best_params(out_dir, dataset_name, method_name, m["name"], best_params, seed=s)

    elif method_name == "equal_dist":
        frac = float(method_cfg.get("sample", {}).get("train_frac_of_smallest", 0.3))
        inner = stratified_kfold_from_cfg(method_cfg.get("inner_cv", {}), seed)

        for m in models_cfg:
            if model_filter and m["name"] not in model_filter:
                continue
            est, grid = build_model_and_grid(m, preprocessing_cfg, seed)

            for s in repeat_seeds:
                train_idx, _ = _split_equal_dist_dynamic(df, group_col, frac, s)
                sub = df.loc[train_idx]
                X = sub[feat_cols].values
                y = sub[label_col].values
                groups_val = sub[group_col].values

                # Compute imbalance ratio from THIS seed's TRAINING split (no test leakage)
                if m.get("family", "").lower() == "xgb":
                    n_pos = (y == 1).sum()
                    n_neg = (y == 0).sum()
                    grid["scale_pos_weight"] = [1, round(n_neg / max(n_pos, 1), 2)]

                # Tune independently for THIS seed (no cross-seed pooling)
                param_scores = {}
                for params in _grid_iter(grid):
                    # key = tuple(sorted(params.items()))
                    key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)for k, v in params.items()))
                    scores = []
                    for tr_idx, va_idx in inner.split(X, y):
                        model = _set_params(est, params)
                        model = _guard_calibrated_cv(model, y[tr_idx])
                        model.fit(X[tr_idx], y[tr_idx])
                        y_proba = _predict_proba_safe(model, X[va_idx])
                        sc = _score_fold(y[va_idx], y_proba, method_name, tuning_cfg, groups_val[va_idx])
                        scores.append(sc)
                    param_scores[key] = np.nanmean(scores)

                best_key = max(param_scores, key=lambda k: param_scores[k])
                best_params = dict(best_key)
                _write_best_params(out_dir, dataset_name, method_name, m["name"], best_params, seed=s)

    else:
        raise ValueError(f"Unknown training method: {method_name}")


def _grid_iter(grid: Dict[str, List]) -> Iterable[Dict]:
    # Cartesian product over dict-of-lists
    from itertools import product
    keys = list(grid.keys())
    vals = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    for combo in product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def _set_params(est, params: Dict):
    # Clone-like behavior without importing sklearn.clone (works for pipelines too)
    import copy
    model = copy.deepcopy(est)
    model.set_params(**params)
    return model


def _find_best_threshold(model, X_tr, y_tr, cv=5, seed=42):
    """Threshold via OUT-OF-FOLD CV predictions on training data, Youden's J.

    Uses cross_val_predict so the threshold is chosen on held-out training
    folds (no in-sample overfit, no test leakage). Youden's J = sensitivity +
    specificity - 1 is prevalence-independent, so the operating point transfers
    across cohorts with different class balance better than an F1-tuned cutoff.
    Falls back to 0.5 if OOF scoring is not possible.
    """
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_curve
    import numpy as _np
    y_tr = _np.asarray(y_tr)
    if len(_np.unique(y_tr)) < 2:
        return 0.5
    try:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        if hasattr(model, "predict_proba"):
            oof = cross_val_predict(model, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
        else:
            oof = cross_val_predict(model, X_tr, y_tr, cv=skf, method="decision_function")
            oof = oof.ravel() if getattr(oof, "ndim", 1) > 1 else oof
    except Exception as e:
        print(f"[threshold] OOF failed ({e}); falling back to 0.5")
        return 0.5
    fpr, tpr, thr = roc_curve(y_tr, oof)
    j = tpr - fpr
    best = thr[int(_np.argmax(j))]
    # roc_curve can return inf as first threshold; clamp to [0,1]
    if not _np.isfinite(best):
        return 0.5
    return float(_np.clip(best, 0.01, 0.99))


def _guard_calibrated_cv(model, y_train):
    """Reduce CalibratedClassifierCV folds if a class is too small."""
    from sklearn.calibration import CalibratedClassifierCV
    from collections import Counter
    if isinstance(model, CalibratedClassifierCV):
        min_class = min(Counter(y_train).values())
        if min_class < 2:
            print(f"  [warn] min_class={min_class}, skipping calibration")
            return model.estimator
        if hasattr(model, "cv") and min_class < model.cv:
            model.cv = 2
    return model


def _write_best_params(out_dir: str, dataset: str, method: str, model_name: str, params: Dict | None, extra_suffix: str | None = None, seed: int | None = None) -> None:
    ensure_dir(out_dir)
    fn = f"{dataset}_{method}_{model_name}"
    if extra_suffix:
        fn += f"_{extra_suffix}"
    if seed is not None:
        fn += f"_seed{seed}"
    path = os.path.join(out_dir, f"{fn}_best_params.csv")
    if params is None:
        df = pd.DataFrame([{"note": "no valid params / all folds NaN"}])
    else:
        # Flatten potentially nested keys like 'clf__estimator__C' are kept as-is
        df = pd.DataFrame([params])
    df.to_csv(path, index=False)
    print(f"✅ Wrote {path}")


def _clean_params(best_params: Dict, model, family: str) -> Dict:
    """Keep only params the model understands; coerce torch MLP types."""
    valid_keys = set(model.get_params().keys())
    cleaned_best = {}
    for k, v in best_params.items():
        if pd.isna(v):
            continue
        if k not in valid_keys:
            print(f"[best_run] Ignoring best_param key '{k}' (not in model.get_params())")
            continue
        if family == "torch":
            if "hidden_layer_sizes" in k and isinstance(v, str):
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, int):
                        v = (parsed,)
                    elif isinstance(parsed, (list, tuple)):
                        v = tuple(int(x) for x in parsed)
                    else:
                        v = (128,)
                except Exception:
                    v = (128,)
            if any(k.endswith(suffix) for suffix in ("epochs", "batch_size", "input_dim")):
                try:
                    v = int(v)
                except Exception:
                    pass
        cleaned_best[k] = v
    if family == "xgb" and "scale_pos_weight" not in cleaned_best:
        cleaned_best["scale_pos_weight"] = 1
    return cleaned_best

# ------------------------------
# Entry point
# ------------------------------

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    repeat_seeds = list(cfg.get("repeat_seeds", [seed]))
    MISSING_GENE_THRESH = 0.50

    dataset_filter = set([s.strip() for s in args.only.split(",")]) if args.only else None
    method_filter = set([s.strip() for s in args.methods.split(",")]) if args.methods else None
    model_filter = set([s.strip() for s in args.models.split(",")]) if args.models else None

    outer_fixed = args.outer.strip() if getattr(args, "outer", "") else ""

    # Data-level cfg
    data_cfg = DataCfg(
        label_col=cfg["data"]["label_col"],
        batch_col=cfg["data"]["batch_col"],
        group_col=cfg["data"]["group_col"],
        meta_cols=list(cfg["data"].get("meta_cols", [])),
        fairness_groups=list(cfg["data"].get("fairness_groups", [])),
    )

    preprocessing_cfg = cfg.get("preprocessing", {})
    training_methods = cfg.get("training_methods", [])
    models_cfg = cfg.get("models", [])
    if args.models:
        keep = {m.strip() for m in args.models.split(",")}
        models_cfg = [m for m in models_cfg if m["name"] in keep]

    tuning_cfg = cfg.get("tuning", {})

    datasets_cfg = cfg["data"]["datasets"]
    
    out_dir = args.emit_preds_dir
    ensure_dir(out_dir)
    #############################################################################
    #############################################################################
    if args.best_run:
        assert args.only and args.methods and args.models and args.outer is not None, \
            "--best_run requires --only DATASET, --methods METHOD, --models MODEL, --outer OUTER"

        dataset_name = args.only
        method_name  = args.methods
        model_name   = args.models
        outer_name   = args.outer

        use_external_test = getattr(args, "external_test", False) and getattr(args, "external_dataset", None) is not None
        test_dataset = args.external_dataset if use_external_test else dataset_name

        # 1) load + preprocess dataset using the shared helper
        df = load_and_preprocess_dataset(
            dataset_name=dataset_name,
            data_cfg=data_cfg,
            datasets_cfg=datasets_cfg,
            missing_gene_thresh=MISSING_GENE_THRESH,
        )
        print(f"[best_run] Loaded {dataset_name} -> {df.shape}")

        if use_external_test:
            df_test = load_and_preprocess_dataset(
                dataset_name=test_dataset,
                data_cfg=data_cfg,
                datasets_cfg=datasets_cfg,
                missing_gene_thresh=MISSING_GENE_THRESH,
            )
            print(f"[best_run] Loaded EXTERNAL TEST dataset {test_dataset} -> {df_test.shape}")
        else:
            # internal eval: same dataset for train/test
            df_test = df

        root = args.emit_preds_dir  # e.g., "results" or "results_bf"
        best_dir = os.path.join(os.path.dirname(root), "best_params")

        # real_dist / equal_dist load per-seed params inside their seed loop below.
        # phylo_style / LOGO use a single params file per outer unit, loaded here.
        if method_name in ("real_dist", "equal_dist"):
            best_params = None
        else:
            candidate1 = os.path.join(best_dir, f"{dataset_name}_{method_name}_{model_name}_train-{outer_name}_best_params.csv")
            candidate2 = os.path.join(best_dir, f"{dataset_name}_{method_name}_{model_name}_best_params.csv")
            best_path = candidate1 if os.path.exists(candidate1) else (candidate2 if os.path.exists(candidate2) else None)
            if  best_path is None or not os.path.exists(best_path):
                raise FileNotFoundError(
                    f"No best_params CSV found for {dataset_name} {method_name} {model_name} {outer_name}"
                )
            best_params = pd.read_csv(best_path).iloc[0].to_dict()

        # 3) Build model via existing factory, using configs already loaded above
        method_cfg = next(m for m in training_methods if m["name"] == method_name)
        model_cfg  = next(m for m in models_cfg       if m["name"] == model_name)

        model, _grid = build_model_and_grid(model_cfg, preprocessing_cfg, seed)

        # Set the best params (keys like 'estimator__C' will match your pipeline)
        # Clean / coerce best_params before applying
        family = model_cfg.get("family", "").lower()

        # 4) Make the correct split for this method/outer (moved up to compute imbalance ratio)
        X_tr, X_te, y_tr, y_te, te_index = split_for_method_outer(
            df=df,
            data_cfg=data_cfg,
            method_cfg=method_cfg,
            outer_name=outer_name,
            seed=seed,
        )

        # Compute class imbalance ratio for XGB
        y_tr_arr_for_ratio = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
        n_pos = (y_tr_arr_for_ratio == 1).sum()
        n_neg = (y_tr_arr_for_ratio == 0).sum()
        imbalance_ratio = round(n_neg / max(n_pos, 1), 2)

        # 🔹 Only keep params that this model actually understands
        # (skipped for real_dist/equal_dist, which clean+apply per-seed below)
        cleaned_best = {}
        if best_params is not None:
            valid_keys = set(model.get_params().keys())

            for k, v in best_params.items():
                # skip NaNs
                if pd.isna(v):
                    continue

                # skip keys not in model.get_params() (e.g. stray 'estimator')
                if k not in valid_keys:
                    # optional: keep this print for debugging
                    print(f"[best_run] Ignoring best_param key '{k}' (not in model.get_params())")
                    continue

                # ---- Special handling for torch MLP ----
                if family == "torch":
                    # Fix hidden_layer_sizes read as string from CSV
                    if "hidden_layer_sizes" in k and isinstance(v, str):
                        try:
                            # e.g. "[128]" → [128], "(128, 64)" → (128, 64), "128" → 128
                            parsed = ast.literal_eval(v)
                            if isinstance(parsed, int):
                                v = (parsed,)
                            elif isinstance(parsed, (list, tuple)):
                                v = tuple(int(x) for x in parsed)
                            else:
                                print(f"[best_run] WARNING: unknown type for hidden_layer_sizes={type(parsed)}, using (128,)")
                                v = (128,)
                        except Exception as e:
                            print(f"[best_run] WARNING: could not parse hidden_layer_sizes={v!r}, using (128,). Error: {e}")
                            v = (128,)

                    # Ensure integer types for some params if needed
                    if any(k.endswith(suffix) for suffix in ("epochs", "batch_size", "input_dim")):
                        try:
                            v = int(v)
                        except Exception:
                            pass

                cleaned_best[k] = v

        # Inject dynamic scale_pos_weight for XGB (fallback for CSVs written before this change)
        if family == "xgb" and "scale_pos_weight" not in cleaned_best:
            cleaned_best["scale_pos_weight"] = 1

        # Now apply cleaned params
        if cleaned_best:
            model.set_params(**cleaned_best)

        # Use the same feature set as in tuning: drop meta_cols (patient, group, dataset_id, subtype)
        meta_cols = data_cfg.meta_cols
        feat_cols = _feature_cols(df, meta_cols)

        # Ensure X_tr / X_te only contain numeric gene columns
        X_tr = X_tr[feat_cols]
        X_te = X_te[feat_cols]

        # Build test set
        if use_external_test:
            # External evaluation: all rows of df_test
            # Make sure external dataset has the same feature columns
            missing_cols = [c for c in feat_cols if c not in df_test.columns]
            if missing_cols:
                raise ValueError(f"[best_run] External dataset {test_dataset} is missing columns: {missing_cols}")

            X_te = df_test[feat_cols]
            # assume label_col is defined in data_cfg for all datasets
            label_col = data_cfg.label_col
            y_te = np.asarray(df_test[label_col])
            te_index = df_test.index

        # 5–7) Fit + emit predictions
        if method_name == "LOGO":
            # We already computed the correct LOGO outer split above:
            # X_tr, X_te, y_tr, y_te, te_index = split_for_method_outer(...)
            # And we already loaded the correct best_params from train-<outer_name>

            group_col = data_cfg.group_col
            meta_cols = getattr(data_cfg, "meta_cols", [])
            feat_cols = _feature_cols(df, meta_cols)

            # Convert to numpy
            X_tr_np = X_tr.values
            X_te_np = X_te.values
            y_tr_np = np.asarray(y_tr)
            y_te_np = np.asarray(y_te)

            # Model with best params already applied
            mdl = model

            # Calibration fix (optional)
            mdl = _guard_calibrated_cv(mdl, y_tr_np)

            # Fit on all training groups except the outer group
            mdl.fit(X_tr_np, y_tr_np)

            # Threshold via out-of-fold CV (Youden's J) on training data
            best_thresh = _find_best_threshold(mdl, X_tr_np, y_tr_np, seed=seed)
            print(f"[best_run] OOF-Youden threshold: {best_thresh}")

            # Build predictions for the test dataset (internal or external)
            sample_ids = df_test.loc[te_index, "patient"].astype(str).values
            group_vals = df_test.loc[te_index, group_col].astype(str).values

            meta_data = {
                col: df_test.loc[te_index, col].values
                for col in meta_cols
                if col in df_test.columns
            }

            # Get prediction scores
            if hasattr(mdl, "predict_proba"):
                y_score = mdl.predict_proba(X_te_np)[:, 1]
            elif hasattr(mdl, "decision_function"):
                s = mdl.decision_function(X_te_np)
                y_score = s.ravel() if getattr(s, "ndim", 1) > 1 else s
            else:
                y_score = np.full_like(y_te_np, np.nan, dtype=float)

            y_pred = (y_score >= best_thresh).astype(int)

            fold_df = pd.DataFrame({
                "sample_id": sample_ids,
                "y_true": y_te_np.astype(int),
                "y_score": y_score,
                "y_pred": y_pred.astype(int),
                "threshold": best_thresh,
                "dataset": dataset_name,
                "method": method_name,
                "model": model_name,
                "outer": outer_name,   # ← IMPORTANT
                "group": group_vals,
            })

            for col, vals in meta_data.items():
                fold_df[col] = vals

            preds = fold_df

        elif method_name in ("real_dist", "equal_dist"):
            import copy as _copy
            all_seed_preds = []

            for s in repeat_seeds:
                X_tr_s, X_te_s, y_tr_s, y_te_s, te_index_s = split_for_method_outer(
                    df=df,
                    data_cfg=data_cfg,
                    method_cfg=method_cfg,
                    outer_name=outer_name,
                    seed=s,
                )
                X_tr_s = X_tr_s[feat_cols]
                X_te_s = X_te_s[feat_cols]

                if use_external_test:
                    X_te_s = df_test[feat_cols]
                    y_te_s = np.asarray(df_test[data_cfg.label_col])
                    te_index_s = df_test.index

                # Load THIS seed's tuned params and build a fresh estimator with seed=s
                best_path_s = os.path.join(
                    best_dir,
                    f"{dataset_name}_{method_name}_{model_name}_seed{s}_best_params.csv",
                )
                if not os.path.exists(best_path_s):
                    raise FileNotFoundError(
                        f"No per-seed best_params CSV found: {best_path_s}"
                    )
                best_params_s = pd.read_csv(best_path_s).iloc[0].to_dict()
                mdl_s, _grid_s = build_model_and_grid(model_cfg, preprocessing_cfg, s)
                cleaned_s = _clean_params(best_params_s, mdl_s, family)
                if cleaned_s:
                    mdl_s.set_params(**cleaned_s)
                y_tr_arr_s = y_tr_s.values if hasattr(y_tr_s, "values") else np.asarray(y_tr_s)
                mdl_s = _guard_calibrated_cv(mdl_s, y_tr_arr_s)
                X_tr_np_s = X_tr_s.values if hasattr(X_tr_s, "values") else X_tr_s
                mdl_s.fit(X_tr_np_s, y_tr_arr_s)

                best_thresh_s = _find_best_threshold(mdl_s, X_tr_np_s, y_tr_arr_s, seed=s)
                print(f"[best_run] seed={s} OOF-Youden threshold: {best_thresh_s}")

                X_te_np_s = X_te_s.values if hasattr(X_te_s, "values") else X_te_s
                y_score_s = _predict_proba_safe(mdl_s, X_te_np_s)
                y_pred_s = (y_score_s >= best_thresh_s).astype(int)

                sample_ids_s = df_test.loc[te_index_s, "patient"].astype(str).values
                group_vals_s = df_test.loc[te_index_s, data_cfg.group_col].astype(str).values

                fold_df_s = pd.DataFrame({
                    "sample_id": sample_ids_s,
                    "y_true": np.asarray(y_te_s, dtype=int),
                    "y_score": y_score_s,
                    "y_pred": y_pred_s.astype(int),
                    "threshold": best_thresh_s,
                    "dataset": dataset_name,
                    "method": method_name,
                    "model": model_name,
                    "outer": outer_name,
                    "group": group_vals_s,
                    "seed": s,
                })

                for col in meta_cols:
                    if col in df_test.columns:
                        fold_df_s[col] = df_test.loc[te_index_s, col].values

                all_seed_preds.append(fold_df_s)

            preds = pd.concat(all_seed_preds, ignore_index=True)

        else:
            # phylo_style: deterministic split, single seed
            # 🔹 Make sure y_tr is a plain 1D numpy array
            #    (important for the torch MLP wrapper, which chokes on a Series)
            if hasattr(y_tr, "values"):
                y_tr_arr = y_tr.values
            else:
                y_tr_arr = np.asarray(y_tr)

            # Tiny guard for CalibratedClassifierCV on very small folds
            model = _guard_calibrated_cv(model, y_tr_arr)

            # Fit
            # model.fit(X_tr, y_tr_arr)
            X_tr_np = X_tr.values if hasattr(X_tr, "values") else X_tr
            model.fit(X_tr_np, y_tr_arr)

            # Threshold via out-of-fold CV (Youden's J) on training data
            X_tr_arr = X_tr.values if hasattr(X_tr, "values") else X_tr
            best_thresh = _find_best_threshold(model, X_tr_arr, y_tr_arr, seed=seed)
            print(f"[best_run] OOF-Youden threshold: {best_thresh}")

            # --- USE REAL PATIENT IDS AND META DATA FROM TEST DATASET ---
            sample_ids = df_test.loc[te_index, "patient"].astype(str).values
            group_vals = df_test.loc[te_index, data_cfg.group_col].astype(str).values

            meta_data = {
                col: df_test.loc[te_index, col].values
                for col in meta_cols
                if col in df_test.columns
            }
            
            X_te_np = X_te.values if hasattr(X_te, "values") else X_te
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_te_np)[:, 1]
            elif hasattr(model, "decision_function"):
                s = model.decision_function(X_te_np)
                y_score = s.ravel() if getattr(s, "ndim", 1) > 1 else s
            else:
                y_score = np.full_like(y_te, np.nan, dtype=float)

            y_pred = (y_score >= best_thresh).astype(int)

            preds = pd.DataFrame({
                "sample_id": sample_ids,
                # y_te may be Series or ndarray → always coerce to int array
                "y_true": np.asarray(y_te, dtype=int),
                "y_score": y_score,
                "y_pred": y_pred.astype(int),
                "threshold": best_thresh,
                "dataset": dataset_name,
                "method": method_name,
                "model": model_name,
                "outer": outer_name,
                "group": group_vals,
            })

            for col, vals in meta_data.items():
                preds[col] = vals

        outdir = args.emit_preds_dir
        #outdir = os.path.join(args.emit_preds_dir, "preds")
        
        os.makedirs(outdir, exist_ok=True)
        if use_external_test:
            dataset = test_dataset
        else:
            dataset = dataset_name
        # For LOGO, this file contains predictions for the specified outer group only.
        if method_name == "LOGO":
            out_path = os.path.join(
                outdir,
                f"{dataset}_{method_name}_{model_name}_outer-{outer_name}.csv"
            )
        else:
            out_path = os.path.join(
                outdir,
                f"{dataset}_{method_name}_{model_name}_train-{outer_name}.csv"
            )

        preds.to_csv(out_path, index=False)
        print(f"✅ [best_run] Wrote predictions → {out_path}")
        return  # don't fall through to tuning
    #############################################################################
    #############################################################################
    for d in datasets_cfg:
        name = d["name"]
        if dataset_filter and name not in dataset_filter:
            continue

        df = load_and_preprocess_dataset(
            dataset_name=name,
            data_cfg=data_cfg,
            datasets_cfg=datasets_cfg,
            missing_gene_thresh=MISSING_GENE_THRESH,
        )
        print(f"Loaded {name} -> {df.shape}")

        for method in training_methods:
            if method_filter and method["name"] not in method_filter:
                continue
            print(f"\n=== Tuning {name} | method={method['name']} ===")
            method_name = method["name"]
            # Use --outer for phylo_style (training batch) and LOGO (held-out group)
            outer_for_method = outer_fixed if method_name in ("phylo_style", "LOGO") else ""

            tune_for_method(
                df=df,
                dataset_name=name,
                method_cfg=method,
                data_cfg=data_cfg,
                preprocessing_cfg=preprocessing_cfg,
                models_cfg=models_cfg,
                tuning_cfg=tuning_cfg,
                seed=seed,
                repeat_seeds=repeat_seeds,
                out_dir=out_dir,
                outer_fixed=outer_for_method,
                model_filter=model_filter,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--only", type=str, default="", help="Comma-separated dataset names to include (optional)")
    parser.add_argument("--methods", type=str, default="", help="Comma-separated training method names to include (optional)")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model names to run (optional)")
    parser.add_argument("--outer", type=str, default="", help="For phylo_style: specific outer unit value to tune on")
    parser.add_argument("--best_run", action="store_true", help="Run a single task using best params and emit predictions (no tuning).")
    parser.add_argument("--best_params", type=str, default=None, help="Optional explicit path to best_params CSV. If not given, will auto-resolve.")
    parser.add_argument("--emit_preds_dir", type=str, default="results/preds", help="Where to write prediction files.")
    parser.add_argument("--external_test", action="store_true", help="If set, train on --only using normal outer split, but test on ALL rows of --external_dataset.")
    parser.add_argument("--external_dataset", type=str, default=None, help="Optional external dataset to test on (train on --only, test on this dataset).")
    main(parser.parse_args())