"""Analytic Hessian of the holistic-verdict model NLL.

Block layout for params = [s.flatten(), gamma, beta] with include_gamma=True:

    [ H_ss   H_sg   H_sb ]      (n*K) x (n*K), (n*K) x K, (n*K) x 1
    [ H_sg^T H_gg   0    ]      K x K (diagonal), K x 1 (zero)
    [ H_sb^T 0      H_bb ]      1 x 1

Structural facts (see DERIVATION.md §5):
    * BT contribution: predictor uses row-sum of s, so H_ss BT part is
      (A^T diag(p(1-p)) A) tiled across all (k, k') blocks via Kronecker
      with J_K = ones((K, K)).
    * Mention contribution to H_ss: per criterion k, A^T diag(q_k(1-q_k)) A
      (note eta_t^2 = 1, so the eta sign drops out of the second derivative).
    * H_{s_{·,k}, gamma_k} = sum_t eta_t * q_{t,k}*(1-q_{t,k}) * A[t, :];
      diagonal in k (no cross-criterion coupling).
    * H_{s_{·,k}, beta}    = sum_t p_t*(1-p_t) * A[t, :];  independent of k.
    * H_{gamma_k, gamma_k} = sum_t q_{t,k}*(1-q_{t,k}); diagonal.
    * H_{gamma_k, beta}    = 0.
    * H_{beta, beta}       = sum_t p_t*(1-p_t).

Ridge adds + ridge * I to the s block only.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from core.model.holistic.likelihood import unpack, _predictors


def _build_A(pairs: np.ndarray, T: int, n: int) -> np.ndarray:
    """A[t, i] = +1 if a_t = i, -1 if b_t = i, 0 otherwise."""
    A = np.zeros((T, n), dtype=np.float64)
    A[np.arange(T), pairs[:, 0]] += 1.0
    A[np.arange(T), pairs[:, 1]] -= 1.0
    return A


def hessian(params, w, r, pairs, K, n, ridge=1e-3, include_gamma=True):
    s, gamma, beta = unpack(params, n, K, include_gamma)
    w_arr = np.asarray(w, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    pairs_arr = np.asarray(pairs, dtype=np.int64)
    T = pairs_arr.shape[0]

    z_bt, z_m, eta = _predictors(s, gamma, beta, w_arr, pairs_arr)
    p = expit(z_bt)
    s_bt = p * (1.0 - p)  # (T,)
    q = expit(z_m)  # (T, K)
    r_m = q * (1.0 - q)  # (T, K)

    A = _build_A(pairs_arr, T, n)  # (T, n)

    # ----- H_ss BT block: (A^T diag(s_bt) A) tiled across all K*K blocks. -----
    AtA_bt = A.T @ (s_bt[:, None] * A)  # (n, n)
    # Kronecker with J_K places a copy at every (k, k'). For (i, k), (i', k')
    # the entry is AtA_bt[i, i'] regardless of (k, k').
    H_ss = np.kron(AtA_bt, np.ones((K, K)))  # (nK, nK)

    # ----- H_ss mention block: per-k, A^T diag(r_m[:, k]) A on diagonal. -----
    # Place each AtA_k into rows/cols (i*K+k for all i), the within-(k=k') slice.
    for k in range(K):
        AtA_k = A.T @ (r_m[:, k : k + 1] * A)  # (n, n)
        idx = np.arange(n) * K + k
        H_ss[np.ix_(idx, idx)] += AtA_k

    if ridge > 0.0:
        H_ss[np.arange(n * K), np.arange(n * K)] += ridge

    # ----- BT/beta cross block: (A * s_bt).sum(axis=0), replicated across K. -----
    bt_cross_essay = (A * s_bt[:, None]).sum(axis=0)  # (n,)
    H_sb = np.repeat(bt_cross_essay, K)  # (nK,) — same value across k

    # ----- BT-only or full assembly. -----
    if not include_gamma:
        P = n * K + 1
        H = np.zeros((P, P), dtype=np.float64)
        H[: n * K, : n * K] = H_ss
        H[: n * K, n * K] = H_sb
        H[n * K, : n * K] = H_sb
        H[n * K, n * K] = float(s_bt.sum())
        return H

    # ----- Mention/gamma cross block: per criterion k. -----
    # H_{(i, k), gamma_k'} = delta(k=k') * sum_t eta_t * r_m[t, k] * A[t, i].
    # Shape (nK, K), zero off-diagonal in (k, k').
    eta_rm = eta[:, None] * r_m  # (T, K)
    cross_k = A.T @ eta_rm  # (n, K)  — col k holds
    # the per-essay vector for crit k.
    H_sg = np.zeros((n * K, K), dtype=np.float64)
    for k in range(K):
        idx = np.arange(n) * K + k
        H_sg[idx, k] = cross_k[:, k]

    # ----- gamma block: diagonal. -----
    H_gg_diag = r_m.sum(axis=0)  # (K,)

    # ----- beta block. -----
    H_bb = float(s_bt.sum())

    # ----- Assemble. -----
    P = n * K + K + 1
    H = np.zeros((P, P), dtype=np.float64)
    H[: n * K, : n * K] = H_ss
    H[: n * K, n * K : n * K + K] = H_sg
    H[n * K : n * K + K, : n * K] = H_sg.T
    H[n * K : n * K + K, n * K : n * K + K] = np.diag(H_gg_diag)
    # H_{gamma, beta} = 0, already zero.
    H[: n * K, n * K + K] = H_sb
    H[n * K + K, : n * K] = H_sb
    H[n * K + K, n * K + K] = H_bb
    return H
