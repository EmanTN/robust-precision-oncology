"""
Model factory and grids
- Builds estimators (optionally wrapped in a StandardScaler pipeline)
- Supports sklearn, XGBoost, and a small PyTorch MLP classifier
- Handles Calibrated LinearSVM with prefixed grid keys (e.g., 'estimator__C')

Public API
----------
build_model_and_grid(model_cfg: dict, preprocessing_cfg: dict, seed: int)
    -> (estimator, param_grid)

normalize_grid(grid: dict) -> dict
    Map YAML-friendly tokens (none/null) to Python None, keep booleans as-is.

needs_standardize(model_name: str, preprocessing_cfg: dict) -> bool

Notes
-----
- The runner is expected to pass the *model-specific* grid returned here
  directly into a search procedure (e.g., GridSearchCV / custom search).
- If you add new models in the YAML, no changes are required here so long as
  you provide a valid 'class_path' (or 'wrapper' block for the Calibrated SVM)
  and the family is one of {'sklearn','xgb','torch'}.
"""
from __future__ import annotations

import numpy as np
from importlib import import_module
from typing import Any, Dict, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted

# Optional common sklearn classes used in defaults / type hints
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.svm import LinearSVC, SVC  # noqa: F401
from sklearn.ensemble import RandomForestClassifier  # noqa: F401

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # allows environments without xgboost

# --------------------------
# Torch MLP (sklearn-style)
# --------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None


class PytorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """A tiny, self-contained MLP with sklearn-like API.

    Parameters
    ----------
    input_dim : int
        Number of input features (set at fit time if -1)
    hidden_layer_sizes : tuple[int, ...]
    lr : float
    epochs : int
    batch_size : int (default 32)
    random_state : int | None
    device : str ("cpu" or "cuda")
    """

    def __init__(
        self,
        input_dim: int = -1,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int | None = None,
        device: str = "cpu",
        class_weight: str | None = None,
        dropout: float = 0.0,          # ← add
        weight_decay: float = 0.0,     # ← add
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for PytorchMLPClassifier.")
        self.input_dim = input_dim
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.class_weight = class_weight
        self.dropout = dropout
        self.weight_decay = weight_decay

    # --- helpers ---
    def _build_net(self, in_dim: int) -> nn.Module:
        layers = []
        last = in_dim
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        # layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def fit(self, X, y):  # type: ignore[override]
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        assert len(np.unique(y)) <= 2, f"MLP only supports binary classification, got {len(np.unique(y))} classes"
        
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        in_dim = X_t.shape[1] if self.input_dim == -1 else self.input_dim
        self.model_ = self._build_net(in_dim).to(self.device)

        # Class weighting
        if self.class_weight == "balanced":
            n_pos = y_t.sum().item()
            n_neg = len(y_t) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)
        else:
            pos_weight = None

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = self.model_(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):  # type: ignore[override]
        check_is_fitted(self, attributes=["model_"])
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t).detach().cpu()
            probs1 = torch.sigmoid(logits).numpy().ravel()
        probs0 = 1.0 - probs1
        return _stack_probs(probs0, probs1)

    def predict(self, X):  # type: ignore[override]
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def _stack_probs(p0, p1):
    return np.vstack([p0, p1]).T


# --------------------------
# Utilities
# --------------------------

def import_from_string(path: str) -> Any:
    """Import a class/function from a dotted path string."""
    module_name, attr = path.rsplit(".", 1)
    mod = import_module(module_name)
    return getattr(mod, attr)


def normalize_grid(grid: Dict[str, Any]) -> Dict[str, Any]:
    def _norm(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"none", "null"}:
                return None
            # try numeric coercion (ints, floats, scientific notation)
            try:
                if any(ch in s for ch in ".e"):   # float or sci
                    return float(v)
                return int(v)                      # plain int
            except Exception:
                return v                           # leave as string
        return v

    return {
        k: [_norm(x) for x in vals] if isinstance(vals, list) else _norm(vals)
        for k, vals in grid.items()
    }

def needs_standardize(model_name: str, preprocessing_cfg: Dict[str, Any]) -> bool:
    # return model_name in set(preprocessing_cfg.get("standardize_for", []))
    if preprocessing_cfg.get("disable_scaler", False):
        return False
    return model_name in set(preprocessing_cfg.get("standardize_for", []))


# --------------------------
# Factory
# --------------------------

def build_model_and_grid(
    model_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate estimator and return the matching parameter grid.

    Handles three families:
    - 'sklearn': class_path is a sklearn classifier
    - 'xgb': class_path is xgboost.XGBClassifier
    - 'torch': class_path 'pytorch_mlp' builds PytorchMLPClassifier

    For LinearSVM, use the YAML pattern:
    models:
      - name: LinearSVM
        class_path: sklearn.calibration.CalibratedClassifierCV
        wrapper: { base_class_path: sklearn.svm.LinearSVC, method: sigmoid, cv: 3 }
        grid:
          estimator__C: [...]
          estimator__loss: [...]
    """
    name = model_cfg["name"]
    family = model_cfg.get("family", "sklearn").lower()
    grid = normalize_grid(model_cfg.get("grid", {}))

    # --- Build base estimator ---
    if family == "torch":
        if model_cfg.get("class_path") != "pytorch_mlp":
            raise ValueError("Unsupported torch model; use class_path: pytorch_mlp")
        estimator: Any = PytorchMLPClassifier(random_state=seed)
    else:
        cls = import_from_string(model_cfg["class_path"])  # e.g., LogisticRegression or CalibratedClassifierCV
        if name.lower().startswith("linearSVM".lower()) or cls is CalibratedClassifierCV:
            # Special case: Calibrated wrapper with inner LinearSVC
            wrapper_cfg = model_cfg.get("wrapper", {})
            base_cls = import_from_string(wrapper_cfg.get("base_class_path", "sklearn.svm.LinearSVC"))
            base = base_cls()
            estimator = CalibratedClassifierCV(estimator=base,
                                               method=wrapper_cfg.get("method", "sigmoid"),
                                               cv=wrapper_cfg.get("cv", 3))
            
        else:
            # Try to pass random_state if available
            try:
                estimator = cls(random_state=seed)
            except TypeError:
                estimator = cls()

    # --- Wrap with StandardScaler pipeline if configured ---
    if needs_standardize(name, preprocessing_cfg):
        estimator = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", estimator),
        ])
        # If grid is for the bare estimator, prefix with 'clf__'
        # We infer whether the grid already contains prefixes like 'clf__' or 'estimator__'
        if not any(k.startswith("clf__") or k.startswith("scaler__") for k in grid.keys()):
            grid = {f"clf__{k}": v for k, v in grid.items()}

    return estimator, grid