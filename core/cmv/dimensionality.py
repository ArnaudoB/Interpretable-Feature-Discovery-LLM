"""Effective dimensionality of a criterion panel, on RANK correlations, with a noise control.

Two problems with reading participation ratio off a covariance matrix of raw scores:

1. **Scale.** BT scores and 0-10 pointwise ratings live on different scales, and covariance PR
   weights a criterion by its variance, so the comparison partly measures units. Working from the
   Spearman correlation matrix removes scale and monotone reparameterisation alike.
2. **Noise.** PR counts independent directions of variation, and *independent measurement error is
   an independent direction*. A noisier panel therefore looks richer. A BT score aggregates many
   comparisons; a pointwise score is one coarse draw. Comparing them on raw PR is biased in favour
   of the noisier one.

:func:`cross_half_corr` is the control. Given two INDEPENDENT measurements A and B of the same
panel on the same items (for BT, fits on disjoint halves of the comparisons; for pointwise, two
scoring runs), the cross-half correlation ``C[i, j] = corr(A_i, B_j)`` estimates the correlation of
the *latent* criteria, because the two measurements' errors are independent and cancel in
expectation. Its diagonal ``C[i, i]`` is the criterion's reliability. Dividing through by
``sqrt(C_ii C_jj)`` (Spearman's classic disattenuation) gives a correlation matrix of the latent
scores, and PR on that is the noise-free estimate.

Caveats that travel with the disattenuated matrix: it is not guaranteed positive semi-definite, so
eigenvalues can come out slightly negative (see :func:`pr_from_corr`, which reports the clipped
mass); and a criterion with near-zero reliability divides by ~0, so it must be screened out rather
than trusted.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy.stats import rankdata


def spearman_corr(X: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix of the columns of ``X`` (ties averaged)."""
    R = np.apply_along_axis(rankdata, 0, np.asarray(X, dtype=float))
    return np.corrcoef(R, rowvar=False)


def cross_spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``C[i, j] = spearman(A[:, i], B[:, j])`` for two measurements of the same items."""
    Ra = np.apply_along_axis(rankdata, 0, np.asarray(A, dtype=float))
    Rb = np.apply_along_axis(rankdata, 0, np.asarray(B, dtype=float))
    Ra = (Ra - Ra.mean(0)) / Ra.std(0)
    Rb = (Rb - Rb.mean(0)) / Rb.std(0)
    return (Ra.T @ Rb) / len(Ra)


def cross_half_corr(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Disattenuated latent correlation matrix and per-criterion reliability.

    Returns ``(R_latent, reliability)``. ``R_latent[i, j] = S[i, j] / sqrt(S_ii S_jj)`` with
    ``S = (C + C.T) / 2`` the symmetrised cross-measurement correlation; the diagonal is 1 by
    construction. ``reliability[i] = C[i, i]``, the agreement of the two measurements on criterion
    ``i`` -- the ceiling any correlation involving it can reach.
    """
    C = cross_spearman(A, B)
    rel = np.diag(C).copy()
    S = 0.5 * (C + C.T)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    R = S / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R, rel


def pr_from_corr(R: np.ndarray) -> tuple[float, float]:
    """Participation ratio of a correlation matrix, and the negative-eigenvalue mass it clipped.

    ``PR = (sum lam)^2 / sum lam^2`` on the eigenvalues. A disattenuated matrix need not be PSD;
    negative eigenvalues are clipped to zero and their absolute mass is returned as a share of the
    total, so a reader can see whether the estimate rests on a well-formed matrix.
    """
    lam = np.linalg.eigvalsh(0.5 * (R + R.T))
    neg = float(-lam[lam < 0].sum() / np.abs(lam).sum()) if np.abs(lam).sum() > 0 else 0.0
    lam = np.clip(lam, 0.0, None)
    if lam.sum() <= 0:
        return 1.0, neg
    return float(lam.sum() ** 2 / (lam**2).sum()), neg


def effrank_from_corr(R: np.ndarray) -> float:
    """Roy & Vetterli (2007) entropy effective rank of a correlation matrix's eigenvalues."""
    lam = np.clip(np.linalg.eigvalsh(0.5 * (R + R.T)), 0.0, None)
    if lam.sum() <= 0:
        return 1.0
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def subset_curve(
    R: np.ndarray, *, metric: str = "pr", max_exact: int = 3000, n_sample: int = 3000, seed: int = 0
) -> np.ndarray:
    """Mean dimensionality over k-subsets of criteria, k = 1..K, from a correlation matrix.

    Holding k fixed is what keeps a larger panel from being credited with more dimensions merely
    for having more columns. Subsets are enumerated exactly while ``C(K, k) <= max_exact`` and
    sampled otherwise, so both ends of the curve are exact.
    """
    K = R.shape[0]
    rng = np.random.default_rng(seed)
    out = np.zeros(K)
    f = (lambda M: pr_from_corr(M)[0]) if metric == "pr" else effrank_from_corr
    for k in range(1, K + 1):
        subsets = (
            list(combinations(range(K), k))
            if comb(K, k) <= max_exact
            else [rng.choice(K, k, replace=False) for _ in range(n_sample)]
        )
        out[k - 1] = np.mean([f(R[np.ix_(list(s), list(s))]) for s in subsets])
    return out


def bootstrap_subset_curves(
    A: np.ndarray,
    B: np.ndarray,
    *,
    metric: str = "pr",
    n_boot: int = 200,
    n_sub: int = 200,
    seed: int = 0,
    pct: tuple[float, float] = (2.5, 97.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CI for the noise-corrected subset curve, resampling ITEMS.

    The uncertainty that matters for these curves is sampling of exchanges, not of criteria: the
    panel is fixed, the 1,600 test exchanges are the sample. Each replicate resamples exchanges with
    replacement, rebuilds the cross-measurement correlation from ``A`` and ``B`` on that resample
    (ranks recomputed inside the replicate, as they must be), and recomputes the k-subset curve.

    ``n_sub`` is deliberately smaller than the point estimate's subset budget: the subset average is
    an inner Monte Carlo whose error shrinks in the outer average over replicates, while the item
    resampling is what the band is about. Criterion subsets are drawn per replicate but with a
    replicate-independent stream, so bands are not smoothed by reusing one lucky draw.
    """
    K = A.shape[1]
    n = len(A)
    rng = np.random.default_rng(seed)
    f = (lambda M: pr_from_corr(M)[0]) if metric == "pr" else effrank_from_corr
    out = np.zeros((n_boot, K))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        R, _ = cross_half_corr(A[idx], B[idx])
        for k in range(1, K + 1):
            subsets = (
                list(combinations(range(K), k))
                if comb(K, k) <= n_sub
                else [rng.choice(K, k, replace=False) for _ in range(n_sub)]
            )
            out[b, k - 1] = np.mean([f(R[np.ix_(list(s), list(s))]) for s in subsets])
    return np.percentile(out, pct[0], axis=0), np.percentile(out, pct[1], axis=0)
