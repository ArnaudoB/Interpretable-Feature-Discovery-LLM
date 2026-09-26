"""Cluster-robust (query-clustered) sandwich variance for the gated symmetric Bradley--Terry model.

Provides the full covariance (``sandwich_V``) and its (gamma, beta) block without the dense
inverse (``gamma_beta_V``); see Appendix ``app:uq`` (eq. ``eq:sandwich-clt``).

V = H^{-1} B H^{-1}, where:
  - H is the penalized expected (Fisher) Hessian of the NLL at the fit
    (``gated.expected_hessian_full``). Bordered-block-diagonal: the (s, gamma)
    part is block-diagonal by criterion; only beta couples globally (and, in the
    gated model, only to the s-block — gamma-beta coupling is 0).
  - B = sum_t U_t U_t^T, where U_t is the per-query score vector (length P),
    aggregated across criteria within a query.

Per-query score for the gated model: the s-entries use
``dL/dDelta = r_cite*tanh(Delta) + d_dir``; the gamma entries use ``r_cite``; and the
**beta** entry is ``sum_k d_dir`` (beta is absent from the citation gate, so the citation
residual does NOT reach beta).

Parameter layout matches ``gated.pack``: ``[s.ravel(C), gamma, beta]``, P = N*K+K+1.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from core.model.gated import (
    _predictors,
    expected_hessian_full,
    pack,
)


# ---------------------------------------------------------------------------
# Bread — penalized expected Hessian (dense P x P)
# ---------------------------------------------------------------------------


def bread(
    s: np.ndarray,
    gamma: np.ndarray,
    beta: float,
    X: np.ndarray,
    pairs: np.ndarray,
    ridge: float,
) -> np.ndarray:
    n, K = s.shape
    params = pack(s, gamma, beta)
    return expected_hessian_full(params, X, pairs, K, n, ridge=ridge)


# ---------------------------------------------------------------------------
# Meat: B = sum_t U_t U_t^T (size P x P, P = N*K + K + 1)
# ---------------------------------------------------------------------------


def per_query_scores(
    s: np.ndarray,
    gamma: np.ndarray,
    beta: float,
    X: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Return ``U`` of shape (T, P) where ``U[t]`` is the gradient of the
    log-likelihood (NOT NLL — sign flipped) for query t, summed across criteria.
    """
    n, K = s.shape
    T = pairs.shape[0]
    P = n * K + K + 1
    a = pairs[:, 0]
    b = pairs[:, 1]
    X_arr = np.asarray(X, dtype=np.float64)

    delta, z_cite, z_dir, _, _ = _predictors(s, gamma, beta, pairs)
    cited = (X_arr != 0).astype(np.float64)
    pos = (X_arr == 1).astype(np.float64)
    p_cite = expit(z_cite)
    p_dir = expit(z_dir)
    r_cite = cited - p_cite  # (T, K)
    d_dir = pos - cited * p_dir  # (T, K)
    tanh_d = np.tanh(delta)  # (T, K)
    dL_dDelta = r_cite * tanh_d + d_dir  # (T, K), d log_lik / d Delta

    # Per-query score vector U_t of length P:
    #   U_t[i*K + k] = +dL_dDelta[t, k]  if i = a_t ;  -dL_dDelta[t, k] if i = b_t
    #   U_t[N*K + k] = r_cite[t, k]
    #   U_t[N*K + K] = sum_k d_dir[t, k]    (beta component — d_dir only)
    U = np.zeros((T, P), dtype=np.float64)
    a_offset = a[:, None] * K + np.arange(K)[None, :]  # (T, K)
    b_offset = b[:, None] * K + np.arange(K)[None, :]  # (T, K)
    t_idx = np.arange(T)[:, None]
    U[t_idx, a_offset] += dL_dDelta
    U[t_idx, b_offset] -= dL_dDelta
    U[:, n * K : n * K + K] = r_cite
    U[:, n * K + K] = d_dir.sum(axis=1)
    return U


def meat_matrix(
    s: np.ndarray,
    gamma: np.ndarray,
    beta: float,
    X: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """B = U.T @ U, where U is per_query_scores. Size P x P, P = N*K + K + 1."""
    U = per_query_scores(s, gamma, beta, X, pairs)
    return U.T @ U


# ---------------------------------------------------------------------------
# Sandwich V = H^{-1} B H^{-1}
# ---------------------------------------------------------------------------


def sandwich_V(
    s: np.ndarray,
    gamma: np.ndarray,
    beta: float,
    X: np.ndarray,
    pairs: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Assemble the full P x P sandwich variance matrix."""
    H = bread(s, gamma, beta, X, pairs, ridge)
    H = 0.5 * (H + H.T)

    try:
        c, low = cho_factor(H, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        H = H + 1e-8 * np.eye(H.shape[0])
        c, low = cho_factor(H, lower=False, check_finite=False)
    H_inv = cho_solve((c, low), np.eye(H.shape[0]), check_finite=False)
    H_inv = 0.5 * (H_inv + H_inv.T)

    B = meat_matrix(s, gamma, beta, X, pairs)
    V = H_inv @ B @ H_inv
    return 0.5 * (V + V.T)


# ---------------------------------------------------------------------------
# The fast s-block sandwich diagonal (no dense H^{-1}, B, or V) used for rho is
# core.model.diagnostics.block_sandwich_s_diag.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Normal quantile for 95% CIs (model-agnostic given V)
# ---------------------------------------------------------------------------

_Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# (gamma, beta) sub-block without forming the dense P x P inverse / meat / V
# ---------------------------------------------------------------------------


def gamma_beta_V(
    s: np.ndarray,
    gamma: np.ndarray,
    beta: float,
    X: np.ndarray,
    pairs: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Sandwich covariance of ``[gamma_0..gamma_{K-1}, beta]``, shape (K+1, K+1).

    Mathematically identical to ``sandwich_V(...)[np.ix_(idx, idx)]`` for
    ``idx = [N*K .. N*K+K]``, but solves ``H Z = E`` for only those K+1 columns and
    contracts the meat as ``(U Z)^T (U Z)``. ``sandwich_V`` materializes four dense
    P x P matrices (~350 MB each at N=200, K=33), which does not fit comfortably in
    a few GB; this path needs only H plus the (T, P) score matrix.
    """
    n, K = s.shape
    P = n * K + K + 1
    H = bread(s, gamma, beta, X, pairs, ridge)
    H = 0.5 * (H + H.T)
    try:
        c, low = cho_factor(H, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        H = H + 1e-8 * np.eye(P)
        c, low = cho_factor(H, lower=False, check_finite=False)

    idx = np.concatenate([np.arange(n * K, n * K + K), [P - 1]])
    E = np.zeros((P, K + 1), dtype=np.float64)
    E[idx, np.arange(K + 1)] = 1.0
    Z = cho_solve((c, low), E, check_finite=False)  # H^{-1} E, (P, K+1)
    del H

    U = per_query_scores(s, gamma, beta, X, pairs)  # (T, P)
    W = U @ Z  # (T, K+1)
    V = W.T @ W
    return 0.5 * (V + V.T)
