"""Accuracy of two criterion panels at a matched number of criteria k (CMV paired task).

Estimand, per k: the heldout accuracy of the paired LR averaged over *all* k-subsets of panel S,
minus the same for panel D. It is a fixed number for a fixed test set; two sources of error sit
between it and what we compute, and they are handled differently:

* **Subsets** (Monte Carlo). When a panel has more than ``m`` k-subsets, ``m`` are drawn, which
  approximates the all-subsets average. This is numerical error, not sampling uncertainty: it
  vanishes as ``m`` grows. When ``C(K, k) <= m`` every subset is enumerated and the error is zero.
* **Test pairs** (sampling). The 800 heldout pairs are a sample; bootstrapping them is the
  interval proper.

Both are resampled jointly in each replicate: the pairs with one index matrix shared by both
panels and every k (so the curve's pointwise intervals and its simultaneous band come from the
same draws), and, independently per panel and k, the drawn subsets with replacement (enumerated
subsets are never resampled). Per-pair correctness is averaged over the (resampled) subsets
*before* the pair mean, so pairing across panels is kept.

The spread of accuracy across individual subsets is a different quantity -- how much the choice of
criteria matters -- and is reported descriptively, never as an interval on the estimand.

Intervals are conditional on the training pairs, the fitted models and the elicited scores.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def draw_subsets(features: list[str], k: int, m: int, rng: np.random.Generator):
    """``(subsets, exact)``: every k-subset if there are at most ``m``, else ``m`` distinct draws."""
    K = len(features)
    if comb(K, k) <= m:
        return [list(s) for s in combinations(features, k)], True
    seen, out = set(), []
    while len(out) < m:
        s = tuple(sorted(rng.choice(K, k, replace=False)))
        if s not in seen:
            seen.add(s)
            out.append([features[i] for i in s])
    return out, False


def _subset_weights(n_sub: int, exact: bool, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n_sub) weights summing to 1 per replicate: uniform if exact, else multinomial."""
    if exact:
        return np.full((n_boot, n_sub), 1.0 / n_sub)
    return rng.multinomial(n_sub, np.full(n_sub, 1.0 / n_sub), size=n_boot) / n_sub


def matched_k_bootstrap(
    corr: dict,
    exact: dict,
    arms: tuple[str, str],
    pair_idx: np.ndarray,
    seed: int = 0,
    level: float = 0.95,
):
    """Joint subset x pair bootstrap of the matched-k accuracy curves and their difference.

    ``corr[(arm, k)]``: (n_subsets, n_pairs) 0/1 heldout correctness, one row per subset.
    ``exact[(arm, k)]``: whether those subsets are all of them. ``arms = (a, b)``; the
    difference is a - b. ``pair_idx``: shared (n_boot, n_pairs) resample of pairs.

    Returns ``(curve, draws)``: per-k point estimates, pointwise and simultaneous (max-|t| over
    k) intervals, and the descriptive subset spread; ``draws`` holds the replicate differences
    (n_boot, n_k) for summaries over k.
    """
    a, b = arms
    ks = sorted({k for (_, k) in corr if (a, k) in corr and (b, k) in corr})
    n_boot = pair_idx.shape[0]
    rng = np.random.default_rng(seed)
    tail = 100 * (1 - level) / 2
    rows, D = [], np.empty((n_boot, len(ks)))
    for j, k in enumerate(ks):
        est, reps, subs = {}, {}, {}
        for arm in arms:
            C = corr[(arm, k)].astype(float)
            W = _subset_weights(len(C), exact[(arm, k)], n_boot, rng)
            cbar = W @ C  # (n_boot, n_pairs)
            reps[arm] = np.take_along_axis(cbar, pair_idx, axis=1).mean(axis=1)
            est[arm] = C.mean()
            subs[arm] = C.mean(axis=1)  # accuracy of each subset
        D[:, j] = reps[a] - reps[b]
        spread = np.subtract.outer(subs[a], subs[b]).ravel()  # random a-subset vs b-subset
        row = {
            "k": k,
            "delta": est[a] - est[b],
            "lo": np.percentile(D[:, j], tail),
            "hi": np.percentile(D[:, j], 100 - tail),
            "se": D[:, j].std(ddof=1),
            "spread_lo": np.percentile(spread, 10),
            "spread_hi": np.percentile(spread, 90),
            "p_subset_a_wins": float((spread > 0).mean() + 0.5 * (spread == 0).mean()),
        }
        for arm in arms:
            row |= {
                f"acc_{arm}": est[arm],
                f"acc_{arm}_lo": np.percentile(reps[arm], tail),
                f"acc_{arm}_hi": np.percentile(reps[arm], 100 - tail),
                f"n_subsets_{arm}": len(corr[(arm, k)]),
                f"exact_{arm}": exact[(arm, k)],
            }
        rows.append(row)
    curve = pd.DataFrame(rows)
    se = curve.se.to_numpy()
    q = float(np.percentile(np.max(np.abs(D - curve.delta.to_numpy()) / se, axis=1), 100 * level))
    curve["lo_simul"] = curve.delta - q * se
    curve["hi_simul"] = curve.delta + q * se
    curve.attrs["q_simul"] = q
    return curve, D


def summarize(curve: pd.DataFrame, D: np.ndarray, cols=None, level: float = 0.95) -> dict:
    """Mean of the difference over the k in ``cols`` (default all), with CI and bootstrap p.

    p is two-sided, from the replicate distribution re-centred at zero (the null of no gap).
    """
    cols = list(range(D.shape[1])) if cols is None else cols
    est = float(curve.delta.to_numpy()[cols].mean())
    reps = D[:, cols].mean(axis=1)
    tail = 100 * (1 - level) / 2
    return {
        "estimate": est,
        "lo": float(np.percentile(reps, tail)),
        "hi": float(np.percentile(reps, 100 - tail)),
        "se": float(reps.std(ddof=1)),
        "p": float(np.mean(np.abs(reps - est) >= abs(est))),
        "n_boot": len(reps),
    }


def marginal_curve(
    corr: dict, exact: dict, pair_idx: np.ndarray, seed: int = 0, level: float = 0.95
) -> pd.DataFrame:
    """One panel's subset-averaged accuracy per k, with the same joint pair x subset bootstrap.

    ``corr[k]``: (n_subsets, n_pairs) 0/1 correctness; ``exact[k]``: subsets enumerated.
    Pass the same ``pair_idx`` for every panel so their bands share resamples.
    """
    rng = np.random.default_rng(seed)
    tail = 100 * (1 - level) / 2
    rows = []
    for k in sorted(corr):
        C = corr[k].astype(float)
        W = _subset_weights(len(C), exact[k], pair_idx.shape[0], rng)
        reps = np.take_along_axis(W @ C, pair_idx, axis=1).mean(axis=1)
        rows.append(
            {
                "k": k,
                "acc": C.mean(),
                "lo": np.percentile(reps, tail),
                "hi": np.percentile(reps, 100 - tail),
                "n_subsets": len(C),
                "exact": exact[k],
            }
        )
    return pd.DataFrame(rows)
