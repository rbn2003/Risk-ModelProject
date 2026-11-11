"""
Risk model v2: the same logistic regression idea as simple_risk_model.py,
built up step by step.

What's the same:  3 features (load, sleep, soreness), per-day high-risk label,
                  scikit-learn LogisticRegression.
What's new:       1. More data (synthetic athletes x days, with noise)
                  2. Split by ATHLETE so test athletes are truly unseen
                  3. Feature scaling (StandardScaler in a Pipeline)
                  4. One engineered feature: load vs. recent average (ACWR-style)
                  5. Better metrics: precision, recall, AUC, confusion matrix
                  6. A baseline to beat, and cross-validation
                  7. Standardized coefficients (now comparable across features)

NOTE: data is synthetic. Results show the pipeline works, not real-world accuracy.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------
# 1. Data: many athletes x many days, same logic as your 28-row table
#    (high load + low sleep + high soreness -> high risk), but noisy
#    so the labels are not a perfect copy of a rule.
# ------------------------------------------------------------------
def make_data(n_athletes=40, n_days=60):
    rows, labels, groups = [], [], []
    for a in range(n_athletes):
        base = rng.uniform(3, 6)                       # athlete's usual load
        load = np.clip(base + rng.normal(0, 1.5, n_days), 0, 10)
        sleep = np.clip(rng.normal(7, 1.2, n_days) - 0.2 * (load - base), 3, 10)
        sore = np.clip(2 + 0.5 * load + 0.4 * (7 - sleep) + rng.normal(0, 1, n_days), 1, 10)

        # engineered feature: today's load relative to the past 7 days
        recent = np.array([load[max(0, d - 7):d].mean() if d > 0 else load[0]
                           for d in range(n_days)])
        acwr = load / (recent + 1e-6)

        score = 0.5 * (load - 5) + 0.6 * (6.5 - sleep) + 0.4 * (sore - 4) + 1.0 * (acwr - 1)
        prob = 1 / (1 + np.exp(-(score - 1.5 + rng.normal(0, 0.8, n_days))))
        risk = (rng.random(n_days) < prob).astype(int)

        rows.append(np.column_stack([load, sleep, sore, acwr]))
        labels.append(risk)
        groups.append(np.full(n_days, a))
    return np.vstack(rows), np.concatenate(labels), np.concatenate(groups)


X, y, groups = make_data()
feature_names = ["training_load", "sleep_hours", "soreness", "load_vs_recent_avg"]
print(f"Dataset: {len(y)} athlete-days, {y.mean():.0%} high-risk, "
      f"{len(np.unique(groups))} athletes")

# ------------------------------------------------------------------
# 2. Split by athlete (not by random row) so no athlete is in both sets
# ------------------------------------------------------------------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(train_idx)} days | Test: {len(test_idx)} days (unseen athletes)\n")

# ------------------------------------------------------------------
# 3. Baseline: always predict the most common class. The model must beat this.
# ------------------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"Baseline (always majority class) accuracy: {dummy.score(X_test, y_test):.1%}")

# ------------------------------------------------------------------
# 4. Your model, now with scaling + class weighting in one Pipeline
#    (scaler is fit on training data only, so no leakage)
# ------------------------------------------------------------------
model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("\nLogistic regression, test set:")
print(f"  accuracy : {(preds == y_test).mean():.1%}")
print(f"  precision: {precision_score(y_test, preds):.3f}")
print(f"  recall   : {recall_score(y_test, preds):.3f}")
print(f"  AUC      : {roc_auc_score(y_test, probs):.3f}")
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(f"  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

# ------------------------------------------------------------------
# 5. Cross-validation by athlete: is the result stable, not a lucky split?
# ------------------------------------------------------------------
cv_auc = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=5),
                         scoring="roc_auc")
print(f"\n5-fold athlete-level CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

# ------------------------------------------------------------------
# 6. Which factors matter? Weights are now comparable because features
#    were standardized (same scale) before fitting.
# ------------------------------------------------------------------
coefs = model.named_steps["logisticregression"].coef_[0]
print("\nStandardized coefficients (bigger magnitude = stronger influence):")
for name, w in sorted(zip(feature_names, coefs), key=lambda t: -abs(t[1])):
    print(f"  {name:20s} {w:+.2f}")
