"""Recovery metrics for a discovered feature set against the engineered dials.

Given a method's per-letter feature scores ``F`` (letters x K), the ground-truth dial
matrix ``G`` (letters x D), and a criteria-vs-dials cosine matrix ``cos`` (K x D) in
text-embedding-3-large space, this module implements the paper's five metrics:

  * SF  — statistical recovery: fraction of dials tracked by some feature (|r| > tau).
  * SM  — mean over dials of ``max_k |r_{k,d}|`` (misses included).
  * SR  — semantic recovery: a dial's Hungarian-matched feature both clears theta
          (aligned name) and tracks it (|r| > tau).      [eq. semrec]
  * AF  — attributable features: fraction of features that track a dial AND have that
          dial dominate their squared correlation (pi_k > 1/2).   [eq. attrib]
  * LK  — leakage (lower better): mean off-target |r| of the attributable features.  [eq. leak]

Semantic matching is a Hungarian assignment of dials to distinct features maximizing
total cosine. ``theta`` is calibrated as the 95th percentile of the alignment of the
pairs the assignment REJECTS (unrelated background), which the paper fixes at 0.406;
``tau`` is fixed a priori at 0.4. Both are held constant across bootstrap replicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

THETA = 0.406  # calibrated 95th-pct-of-rejected-pairs cosine acceptance (aligned name)
TAU = 0.4  # a-priori tracking threshold on |Pearson r| (feature "tracks" a dial)

METRIC_KEYS = ("SF", "SM", "SR", "AF", "LK")

#: The names the paper prints. The keys above are kept as internal identifiers: they are the
#: JSON keys of the shipped ``bootstrap_*.json`` files and the column headers of
#: ``tab_results.csv`` and ``tau_robustness.json``, so the display names are resolved here.
DISPLAY_NAME = {"SF": "Tracked", "SM": "Stat.R.", "SR": "Sem.R.", "AF": "Attrib.", "LK": "LK"}

#: Direction of improvement, used to pick the bold cell and to score "Ours better" shares.
HIGHER_BETTER = {"SF": True, "SM": True, "SR": True, "AF": True, "LK": False}

#: Metrics tab:main prints as percentages; SM and LK print as three decimals.
PERCENT = frozenset({"SF", "SR", "AF"})


def corr_matrix(F: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """Signed Pearson correlation of each feature column vs each dial column (K x D).

    Population std (ddof=0); a constant column under bootstrap resampling yields 0
    correlation (not nan)."""
    F = F.astype(float)
    G = G.astype(float)
    Fz = (F - F.mean()) / F.std(ddof=0).replace(0.0, np.nan)
    Gz = (G - G.mean()) / G.std(ddof=0).replace(0.0, np.nan)
    M = np.nan_to_num(Fz.values.T @ Gz.values) / len(F)
    return pd.DataFrame(M, index=F.columns, columns=G.columns)


def hungarian_match(cos: pd.DataFrame):
    """Assign each dial to a distinct feature maximizing total cosine (Hungarian).

    ``cos`` is (features x dials). Returns ``{dial: (feature, cosine)}`` for the
    assigned pairs (one per dial, assuming K >= D)."""
    feats = list(cos.index)
    dials = list(cos.columns)
    S = cos.values.T  # (D, K) dial x feature
    ri, cj = linear_sum_assignment(1 - S)  # maximize total cosine
    return {dials[i]: (feats[j], float(S[i, j])) for i, j in zip(ri, cj)}


def rejected_cosines(cos: pd.DataFrame) -> np.ndarray:
    """Cosines of every (dial, feature) pair the Hungarian assignment does NOT pick —
    the empirical background of unrelated pairs used to calibrate ``theta``."""
    match = hungarian_match(cos)
    assigned = {(d, f) for d, (f, _) in match.items()}
    vals = [cos.loc[f, d] for d in cos.columns for f in cos.index if (d, f) not in assigned]
    return np.asarray(vals, float)


def calibrate_theta(cos_matrices, q: float = 95.0) -> float:
    """95th percentile of rejected-pair cosines pooled over one or more cos matrices."""
    if isinstance(cos_matrices, pd.DataFrame):
        cos_matrices = [cos_matrices]
    pool = np.concatenate([rejected_cosines(c) for c in cos_matrices])
    return float(np.percentile(pool, q))


# --------------------------------------------------------------------------- #
# The five metrics. F: letters x K, G: letters x D, cos: K x D.
# --------------------------------------------------------------------------- #
def statistical_recovery(F, G, tau: float = TAU):
    """(SF, SM, per-dial max|r|). SF = frac dials with max_k|r|>tau; SM = mean of max_k|r|."""
    r = corr_matrix(F, G).abs()
    per_dial = r.max(axis=0)  # Series over dials
    sf = float((per_dial > tau).mean())
    sm = float(per_dial.mean())
    return sf, sm, per_dial


def semantic_recovery(F, G, cos, theta: float = THETA, tau: float = TAU) -> float:
    """SR (eq. semrec): dial recovered iff its Hungarian match clears theta AND tracks it."""
    r = corr_matrix(F, G).abs()
    match = hungarian_match(cos)
    D = G.shape[1]
    hit = 0
    for d in G.columns:
        feat, c = match.get(d, (None, -1.0))
        if feat is not None and c >= theta and r.loc[feat, d] > tau:
            hit += 1
    return hit / D


def _attributable_mask(r_abs: pd.DataFrame, tau: float):
    """Boolean Series over features: tracks some dial AND that dial dominates (pi_k>1/2)."""
    r2 = r_abs**2
    pi = r2.max(axis=1) / r2.sum(axis=1).replace(0.0, np.nan)
    tracks = r_abs.max(axis=1) > tau
    return (tracks & (pi > 0.5)).fillna(False)


def attributable_features(F, G, tau: float = TAU):
    """(AF, attributable-mask). AF = fraction of the K features that are attributable."""
    r = corr_matrix(F, G).abs()
    mask = _attributable_mask(r, tau)
    return float(mask.mean()), mask


def leakage(F, G, tau: float = TAU) -> float:
    """LK (eq. leak, lower better): mean off-target |r| of the attributable features.

    For each dial d, average |r_{k,d}| over the attributable features excluding d's own
    tracker (argmax_k |r_{k,d}|); then average over dials."""
    r = corr_matrix(F, G).abs()
    attr = _attributable_mask(r, tau)
    attr_feats = [k for k in r.index if attr[k]]
    if not attr_feats:
        return float("nan")
    per_dial = []
    for d in r.columns:
        tracker = r[d].idxmax()
        others = [k for k in attr_feats if k != tracker]
        if others:
            per_dial.append(r.loc[others, d].mean())
    return float(np.mean(per_dial)) if per_dial else float("nan")


def scorecard(F, G, cos, theta: float = THETA, tau: float = TAU) -> dict:
    """All five metrics plus K, as a flat dict (one method's row of Table `tab:main`)."""
    sf, sm, _ = statistical_recovery(F, G, tau)
    return {
        "K": int(F.shape[1]),
        "SF": sf,
        "SM": sm,
        "SR": semantic_recovery(F, G, cos, theta, tau),
        "AF": attributable_features(F, G, tau)[0],
        "LK": leakage(F, G, tau),
    }
