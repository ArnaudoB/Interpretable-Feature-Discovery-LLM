"""Design-tensor assembly + fit-diagnostics helpers for the comparative arm.

These sit between the taxonomy mapping (raw dimension phrase -> canonical criterion)
and :func:`core.model.gated.fit_model`:

  * :func:`build_X`   -- tally A/B citations per (pair, criterion) into the design
    tensor ``X in {-1, 0, +1}^(T x K)`` the model consumes.
  * :func:`eff_rank`  -- participation ratio of the score matrix's correlation
    eigenvalues (an over-splitting / halo diagnostic).
  * :func:`kappa_rho` -- per-criterion identification (condition number ``kappa``)
    and reliability (signal fraction ``rho``) used to drop poorly identified or
    unreliable criteria before the refit (see :mod:`core.model.diagnostics`).
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.model import diagnostics as gdiag


def build_X(phrases, cluster_of_norm, K, pair_to_idx):
    """Tally A/B per (pair, criterion) -> X in {-1, 0, +1}^(T x K)."""
    T = len(pair_to_idx)
    a = np.zeros((T, K))
    b = np.zeros((T, K))
    for r in phrases.itertuples(index=False):
        pi = pair_to_idx.get(str(r.pair_id))
        k = cluster_of_norm.get(r.phrase_norm)
        if pi is None or k is None:
            continue
        if r.winner_slot == "A":
            a[pi, k] += 1
        elif r.winner_slot == "B":
            b[pi, k] += 1
    X = np.zeros((T, K), dtype=np.int8)
    X[a > b] = 1
    X[b > a] = -1
    return X


def eff_rank(M):
    """Participation ratio of the score matrix's correlation eigenvalues."""
    ev = np.linalg.eigvalsh(np.corrcoef(StandardScaler().fit_transform(M).T))
    return float(ev.sum() ** 2 / (ev**2).sum())


def kappa_rho(fit, X, pairs, n, ridge):
    """Per-criterion identification kappa and reliability rho for a fitted model.

    kappa_k = lambda_max / lambda_2 of the unpenalized s-block Fisher Laplacian;
    rho_k = (var_k - noise_k) / var_k with var_k the (ddof=0) variance of the fitted scores
    and noise_k the mean sandwich variance of s_hat[:, k] (nan where var_k = 0).
    """
    fw = gdiag._fisher_weights(fit.s, fit.gamma, fit.beta, X, pairs)
    _, _, kappa = gdiag.per_criterion_kappa(fw.h_DD, pairs, n)
    noise = gdiag.block_sandwich_s_diag(fit.s, fit.gamma, fit.beta, X, pairs, ridge).mean(0)
    var = fit.s.var(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(var > 0, (var - noise) / var, np.nan)
    return kappa, rho
