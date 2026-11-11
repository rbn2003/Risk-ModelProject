"""
risk_core.py - the reusable logic behind the app (no user interface here).

Keeping the model logic separate from the UI means you can:
  - test it on its own,
  - reuse it in a script or notebook,
  - swap the interface later without touching the model.

Main functions:
  make_synthetic_data()   -> a DataFrame of simulated athlete-days
  train_and_evaluate()    -> fits a logistic regression on ANY dataset you give it
  metrics_at_threshold()  -> precision/recall/accuracy at a chosen cutoff
  predict_risk()          -> risk probability for one new day
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 30


# ------------------------------------------------------------------
# Synthetic data (same logic as risk_model_v2.py)
# ------------------------------------------------------------------
def make_synthetic_data(n_athletes: int = 40, n_days: int = 60, seed: int = 1) -> pd.DataFrame:
    """Simulate athlete-days. High load, low sleep, high soreness and sudden
    load spikes raise the chance of a high-risk day, with noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for a in range(n_athletes):
        base = rng.uniform(3, 6)  # this athlete's usual load
        load = np.clip(base + rng.normal(0, 1.5, n_days), 0, 10)
        sleep = np.clip(rng.normal(7, 1.2, n_days) - 0.2 * (load - base), 3, 10)
        sore = np.clip(2 + 0.5 * load + 0.4 * (7 - sleep) + rng.normal(0, 1, n_days), 1, 10)

        # today's load relative to the previous 7 days (acute:chronic style)
        recent = np.array([load[max(0, d - 7):d].mean() if d > 0 else load[0]
                           for d in range(n_days)])
        acwr = load / (recent + 1e-6)

        score = (0.5 * (load - 5) + 0.6 * (6.5 - sleep)
                 + 0.4 * (sore - 4) + 1.0 * (acwr - 1))
        prob = 1 / (1 + np.exp(-(score - 1.5 + rng.normal(0, 0.8, n_days))))
        risk = (rng.random(n_days) < prob).astype(int)

        frames.append(pd.DataFrame({
            "athlete_id": a,
            "day": np.arange(n_days),
            "training_load": load,
            "sleep_hours": sleep,
            "soreness": sore,
            "load_vs_recent_avg": acwr,
            "high_risk": risk,
        }))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# Label handling
# ------------------------------------------------------------------
def to_binary(series: pd.Series, positive_value=None) -> Tuple[pd.Series, object]:
    """Turn a two-valued column into 0/1. If positive_value is not given,
    the larger value (or True) counts as the 'high-risk' class."""
    values = list(series.dropna().unique())
    if len(values) != 2:
        raise ValueError(
            f"The label column must have exactly 2 distinct values, "
            f"but it has {len(values)}: {values[:6]}"
        )
    if positive_value is None:
        positive_value = sorted(values)[-1]
    return (series == positive_value).astype(int), positive_value


# ------------------------------------------------------------------
# Training + evaluation on any dataset
# ------------------------------------------------------------------
def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: Optional[str] = None,
    positive_value=None,
    test_size: float = 0.25,
    seed: int = 1,
) -> Dict:
    """Fit a scaled, class-weighted logistic regression and evaluate it.

    If group_col is given (e.g. athlete ID), the train/test split and the
    cross-validation keep each group entirely on one side, so the test
    athletes are truly unseen. Otherwise a stratified random split is used.
    """
    if not feature_cols:
        raise ValueError("Pick at least one feature column.")
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Feature '{c}' is not numeric. Choose numeric columns only.")
    if label_col in feature_cols:
        raise ValueError("The label column can't also be a feature.")

    cols = list(feature_cols) + [label_col] + ([group_col] if group_col else [])
    data = df[cols].dropna().reset_index(drop=True)
    dropped = len(df) - len(data)
    if len(data) < MIN_ROWS:
        raise ValueError(f"Only {len(data)} usable rows; need at least {MIN_ROWS}.")

    y_series, positive_value = to_binary(data[label_col], positive_value)
    y = y_series.to_numpy()
    X = data[feature_cols].to_numpy(dtype=float)
    groups = data[group_col].to_numpy() if group_col else None

    # ---- split ----
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < 3:
            raise ValueError("Need at least 3 distinct athletes/groups to split by group.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        cv = GroupKFold(n_splits=min(5, n_groups))
        split_type = "By group (test groups never appear in training)"
    else:
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        split_type = "Random stratified rows (rows from one athlete may be in both sets)"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Train or test set ended up with only one class. "
                         "Try a different test size or more data.")

    # ---- baseline + model ----
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # ---- cross-validation (may fail on very small / lopsided data) ----
    try:
        cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="roc_auc")
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    except ValueError:
        cv_mean, cv_std = None, None

    coefs = model.named_steps["logisticregression"].coef_[0]

    return {
        "model": model,
        "feature_cols": list(feature_cols),
        "label_col": label_col,
        "positive_value": positive_value,
        "y_test": y_test,
        "probs": probs,
        "auc": float(roc_auc_score(y_test, probs)),
        "baseline_acc": float(baseline_acc),
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "coefs": dict(zip(feature_cols, [float(c) for c in coefs])),
        "n_rows": len(data),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_dropped": dropped,
        "positive_rate": float(y.mean()),
        "split_type": split_type,
        "feature_medians": {c: float(np.median(data[c])) for c in feature_cols},
        "feature_ranges": {c: (float(data[c].min()), float(data[c].max())) for c in feature_cols},
    }


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Accuracy, precision, recall and confusion counts at a chosen cutoff."""
    preds = (probs >= threshold).astype(int)
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tp = int(((preds == 1) & (y_true == 1)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def predict_risk(result: Dict, values: Dict[str, float]) -> float:
    """Risk probability (0-1) for one new day, given a value for each feature."""
    row = np.array([[values[c] for c in result["feature_cols"]]], dtype=float)
    return float(result["model"].predict_proba(row)[0, 1])
