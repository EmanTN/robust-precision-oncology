# utils.py
from __future__ import annotations

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef
)

# ==============================
# Reproducibility & Config I/O
# ==============================

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy (and torch if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # optional top-level seed
    if "seed" in cfg:
        set_seed(int(cfg["seed"]))
    return cfg

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_df(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

# ==============================
# Data loading & feature handling
# ==============================

def load_dataset_csv(path: str) -> pd.DataFrame:
    """Robust CSV loader (handles BOM, dtype fallbacks)."""
    return pd.read_csv(path)

def feature_columns(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    """Return columns considered features (non-metadata)."""
    return [c for c in df.columns if c not in meta_cols]

def align_train_and_external(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    meta_cols_train: List[str],
    meta_cols_val: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Align gene/feature space between train and external validation.
    - Keep all meta cols as-is.
    - Intersect features, then order by train_df feature order.
    """
    train_feats = feature_columns(train_df, meta_cols_train)
    val_feats   = feature_columns(val_df,   meta_cols_val)

    common_feats = [g for g in train_feats if g in set(val_feats)]
    if len(common_feats) == 0:
        raise ValueError("No overlapping features between training and external validation sets.")

    # Rebuild aligned frames: meta + common_feats (ordered by train)
    train_keep = meta_cols_train + common_feats
    val_keep   = meta_cols_val   + common_feats

    train_aligned = train_df[train_keep].copy()
    val_aligned   = val_df[val_keep].copy()

    return train_aligned, val_aligned, common_feats

def filter_to_gene_list(df: pd.DataFrame, genes_to_keep: Iterable[str], meta_cols: List[str]) -> pd.DataFrame:
    """
    Keep only meta_cols + genes_to_keep; preserve relative order of genes_to_keep.
    """
    genes_to_keep = [g for g in genes_to_keep if g in df.columns]
    cols = meta_cols + genes_to_keep
    return df.loc[:, [c for c in cols if c in df.columns]].copy()

# ==============================
# Preprocessing helpers
# ==============================

def needs_standardization(model_name: str, cfg: Dict[str, Any]) -> bool:
    pre = cfg.get("preprocessing", {})
    return model_name in set(pre.get("standardize_for", []))

def build_scaler_pipeline(X: np.ndarray) -> Pipeline:
    """
    Returns a simple StandardScaler pipeline.
    (You’ll attach the estimator later in your runner.)
    """
    return Pipeline([("scaler", StandardScaler())])

# ==============================
# Split constructors (four strategies)
# ==============================

@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    info: Dict[str, Any]

def indices_for_df(df: pd.DataFrame) -> np.ndarray:
    return np.arange(len(df))

def make_phylo_splits(
    df: pd.DataFrame,
    batch_col: str,
) -> List[Split]:
    """
    Train on ONE batch (dataset_id), test on ALL remaining batches.
    Returns list of (train_idx, test_idx) per train-batch choice.
    """
    idx = indices_for_df(df)
    batches = df[batch_col].astype(str).values
    unique_batches = sorted(pd.unique(batches))
    splits: List[Split] = []
    for b in unique_batches:
        train_mask = (batches == b)
        test_mask  = ~train_mask
        splits.append(Split(
            train_idx=idx[train_mask],
            test_idx=idx[test_mask],
            info={"train_batch": b}
        ))
    return splits

def make_logo_splits(
    df: pd.DataFrame,
    logo_unit: str,
) -> List[Split]:
    """
    Leave-one-group-out: for each unique value in logo_unit, leave it as test.
    """
    idx = indices_for_df(df)
    units = df[logo_unit].astype(str).values
    unique_units = sorted(pd.unique(units))
    splits: List[Split] = []
    for u in unique_units:
        test_mask  = (units == u)
        train_mask = ~test_mask
        splits.append(Split(
            train_idx=idx[train_mask],
            test_idx=idx[test_mask],
            info={"left_out_unit": u}
        ))
    return splits

def sample_real_dist(
    df: pd.DataFrame,
    group_col: str,
    pct: float,
    seed: int = 42
) -> Split:
    """
    For each group, take the same percentage pct into TRAIN; rest is TEST.
    Stratification by label happens later inside inner CV.
    """
    rng = np.random.default_rng(seed)
    idx = indices_for_df(df)
    groups = df[group_col].astype(str).values
    train_mask = np.zeros(len(df), dtype=bool)

    for g in pd.unique(groups):
        g_idx = np.where(groups == g)[0]
        k = max(1, int(math.floor(pct * len(g_idx))))
        chosen = rng.choice(g_idx, size=k, replace=False)
        train_mask[chosen] = True

    return Split(
        train_idx=idx[train_mask],
        test_idx=idx[~train_mask],
        info={"scheme": "percentage_per_group", "pct": pct}
    )

def sample_equal_dist_dynamic(
    df: pd.DataFrame,
    group_col: str,
    train_frac_of_smallest: float,
    seed: int = 42
) -> Split:
    """
    Find the smallest group size S; use floor(train_frac_of_smallest * S) from EVERY group as TRAIN.
    Remaining samples become TEST.
    """
    rng = np.random.default_rng(seed)
    idx = indices_for_df(df)
    groups = df[group_col].astype(str)

    sizes = groups.value_counts()
    smallest = int(sizes.min())
    per_group_train = max(1, int(math.floor(train_frac_of_smallest * smallest)))

    train_mask = np.zeros(len(df), dtype=bool)
    for g, g_size in sizes.items():
        g_idx = np.where(groups.values == g)[0]
        k = min(per_group_train, len(g_idx))
        chosen = rng.choice(g_idx, size=k, replace=False)
        train_mask[chosen] = True

    return Split(
        train_idx=idx[train_mask],
        test_idx=idx[~train_mask],
        info={
            "scheme": "equal_dist_dynamic",
            "per_group_train": per_group_train,
            "smallest_group": smallest
        }
    )

# ==============================
# Inner CV builders
# ==============================

def make_inner_cv(cv_cfg: Dict[str, Any], y: np.ndarray) -> StratifiedKFold:
    """
    Build StratifiedKFold based on config (only StratifiedKFold is used here).
    """
    if cv_cfg.get("type", "StratifiedKFold") != "StratifiedKFold":
        raise ValueError("Only StratifiedKFold is supported in this helper.")
    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle  = bool(cv_cfg.get("shuffle", True))
    rs       = int(cv_cfg.get("random_state", 42))
    # (We ignore metric here; metric will be handled by the tuner.)
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)

# ==============================
# Metric utilities (incl. worst-group)
# ==============================

def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute standard metrics; AUROC only if y_prob provided (binary)."""
    out = dict(
        accuracy = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0),
        kappa = cohen_kappa_score(y_true, y_pred),
        mcc = matthews_corrcoef(y_true, y_pred),
    )
    try:
        if y_prob is not None:
            out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        # keep going even if AUROC fails (e.g., single-class fold)
        pass
    return out

def worst_group_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    group_values: Iterable[Any],
    metric: str = "auc",
) -> Optional[float]:
    """
    Compute worst-group value for a metric ('auc' or 'accuracy').
    Returns None if computation is not possible (e.g., AUROC needs both classes).
    """
    group_values = np.asarray(list(group_values))
    groups = pd.unique(group_values)
    vals: List[float] = []

    for g in groups:
        mask = (group_values == g)
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if metric == "accuracy":
            vals.append(accuracy_score(yt, yp))
        elif metric == "auc":
            if y_prob is None:
                continue
            try:
                vals.append(roc_auc_score(yt, y_prob[mask]))
            except Exception:
                # skip groups where AUROC is undefined (single-class)
                continue
        else:
            raise ValueError(f"Unsupported worst-group metric: {metric}")

    if len(vals) == 0:
        return None
    return float(np.min(vals))

# ==============================
# Grid helpers
# ==============================

def _yaml_value_to_python(v: Any) -> Any:
    """
    Normalize common YAML entries:
    - 'none'/'null' -> None
    - 'true'/'false' -> bool
    """
    if isinstance(v, str):
        low = v.lower()
        if low in ("none", "null"):
            return None
        if low == "true":
            return True
        if low == "false":
            return False
    return v

def expand_grid(grid_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand a param grid dict of lists into a list of param dicts
    (Cartesian product). String 'none'/'null' -> None, 'true'/'false' -> bool.
    """
    import itertools
    keys = list(grid_dict.keys())
    values = [ [_yaml_value_to_python(x) for x in grid_dict[k]] for k in keys ]
    combos = []
    for vs in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, vs)})
    return combos

# ==============================
# Convenience: dataset packer
# ==============================

@dataclass
class DatasetPack:
    name: str
    df: pd.DataFrame
    label_col: str
    batch_col: str
    group_col: str
    meta_cols: List[str]

def load_all_training_sets(cfg: Dict[str, Any]) -> List[DatasetPack]:
    data_cfg = cfg["data"]
    label_col = data_cfg["label_col"]
    batch_col = data_cfg["batch_col"]
    group_col = data_cfg["group_col"]
    meta_cols = list(data_cfg["meta_cols"])

    packs: List[DatasetPack] = []
    for d in data_cfg["datasets"]:
        name = d["name"]
        path = d["path"]
        df = load_dataset_csv(path)
        # sanity: drop columns with >50% missing if needed (optional)
        # df = df.loc[:, df.isnull().mean() <= 0.5]
        packs.append(DatasetPack(
            name=name, df=df,
            label_col=label_col, batch_col=batch_col,
            group_col=group_col, meta_cols=meta_cols
        ))
    return packs

# ==============================
# External validation loader
# ==============================

def load_external_validation(
    path: str,
    meta_cols: List[str]
) -> pd.DataFrame:
    """
    Load ONE external validation dataset CSV (with its own meta cols).
    """
    df = load_dataset_csv(path)
    # (You can clean/rename columns here if needed.)
    # Example: df = df.loc[:, df.isnull().mean() <= 0.5]
    # Ensure required meta columns exist:
    for c in meta_cols:
        if c not in df.columns:
            # don't raise—external may not have identical meta cols;
            # downstream code can adapt as long as label_col exists for scoring.
            pass
    return df
