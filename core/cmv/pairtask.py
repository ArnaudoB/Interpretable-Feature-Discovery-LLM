"""Pairwise design, L1-logistic model selection, and single-shot heldout eval.

The classifier operates on pairs. Each pair's design row is the **difference**
between the positive and negative unit's feature vectors, with a **randomized
sign** (seeded) to avoid label leakage from ordering:

    x = sign * (f_pos - f_neg);  y = 1[sign == +1]

The signs are **dealt from a balanced deck within each split** (see :func:`assign_signs`),
not flipped as independent coins. Independent flips leave the label balance to luck — an
unlucky draw (e.g. 44/56 on 100 test pairs) hands a feature-blind classifier a
majority-class baseline *above* 50%, so "chance = 50%" silently stops being true and
accuracies get read against a phantom null. Dealing equal +1/-1 within train and within
test makes chance exactly 50% by construction. Which pair gets which sign is still random,
and accuracy is ~unaffected (flipping a pair's sign flips x and y together).

Model selection: L1 logistic regression, regularization chosen by 5-fold CV on
**train only**, folds **grouped by op_author**. Standardization and mean
imputation are fit inside the CV pipeline (train-fold only). Heldout is evaluated
**once**.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CS = np.logspace(-3, 3, 13)


def assign_signs(pairs, seed=42):
    """Balanced +1/-1 sign per pair, dealt separately within the train and test splits.

    Equal numbers of +1 and -1 are shuffled and dealt (a deck, not coin flips), so each split's
    label balance is 50/50 by construction and chance is exactly 50%. Uses the ``train`` column
    when present; otherwise deals one balanced deck over all pairs. With an odd split size the
    leftover goes to -1 (a <=0.5pp imbalance).

    Deterministic in ``(len(pairs), pairs['train'], seed)`` only — so every feature set gets the
    identical sign/label assignment, which is what makes the paired McNemar test valid.
    """
    rng = np.random.default_rng(seed)
    signs = np.empty(len(pairs), dtype=int)
    if "train" in getattr(pairs, "columns", []):
        tr = pairs["train"].to_numpy().astype(bool)
        blocks = [np.where(tr)[0], np.where(~tr)[0]]  # train first, then test (fixed order)
    else:
        blocks = [np.arange(len(pairs))]
    for idx in blocks:
        n = len(idx)
        if n == 0:
            continue
        v = np.array([1] * (n // 2) + [-1] * (n - n // 2))
        rng.shuffle(v)
        signs[idx] = v
    return signs


def build_design(pairs, pos_col, neg_col, extract, op_col="op_text", seed=42):
    """Return (X, y) for all pairs. ``extract(unit_text, op_text) -> vector``.

    The RNG is seeded, so calling with the same seed across feature sets yields an
    identical sign/label assignment per pair — required for paired McNemar tests.
    """
    f_pos = np.array([extract(t, o) for t, o in zip(pairs[pos_col], pairs[op_col])], dtype=float)
    f_neg = np.array([extract(t, o) for t, o in zip(pairs[neg_col], pairs[op_col])], dtype=float)
    signs = assign_signs(pairs, seed=seed)
    y = (signs == 1).astype(int)
    X = (f_pos - f_neg) * signs[:, None]
    return X, y


def _pipeline():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            # no centering: it would break the x -> -x / y -> 1-y antisymmetry that
            # fit_intercept=False relies on (scaling by std preserves it).
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="l1", solver="liblinear", max_iter=5000, fit_intercept=False
                ),
            ),
        ]
    )


def run_condition(pairs, pos_col, neg_col, extract, op_col="op_text", Cs=DEFAULT_CS, seed=42):
    """Fit + select on train, predict once on heldout. Returns a result dict."""
    X, y = build_design(pairs, pos_col, neg_col, extract, op_col=op_col, seed=seed)
    train = pairs["train"].to_numpy().astype(bool)
    groups = pairs["op_author"].to_numpy()

    Xtr, ytr, gtr = X[train], y[train], groups[train]
    Xte, yte = X[~train], y[~train]

    gs = GridSearchCV(
        _pipeline(),
        {"clf__C": list(Cs)},
        cv=GroupKFold(n_splits=5),
        scoring="accuracy",
        n_jobs=-1,
    )
    gs.fit(Xtr, ytr, groups=gtr)
    pred = gs.predict(Xte)
    return {
        "n_train": int(train.sum()),
        "n_heldout": int((~train).sum()),
        "n_features": X.shape[1],
        "best_C": gs.best_params_["clf__C"],
        "cv_acc": float(gs.best_score_),
        "heldout_acc": float((pred == yte).mean()),
        "label_balance": float(y.mean()),
        "pred": pred,
        "y_heldout": yte,
        "estimator": gs.best_estimator_,
    }


def mcnemar(y, pred_a, pred_b):
    """Exact-binomial McNemar test comparing model A vs baseline B on heldout.

    Returns (n_a_only_correct, n_b_only_correct, p_value). ``b > c`` means A wins.
    """
    ca = pred_a == y
    cb = pred_b == y
    b = int(np.sum(ca & ~cb))  # A right, B wrong
    c = int(np.sum(~ca & cb))  # B right, A wrong
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, float(p)
