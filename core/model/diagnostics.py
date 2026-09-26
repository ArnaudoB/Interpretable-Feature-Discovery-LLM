"""Per-criterion identification (condition number kappa) and reliability (rho) for the gated symmetric Bradley--Terry model.

The two screens of Appendix ``app:uq`` (eqs. ``eq:condition_number``, ``eq:noise``):

  * IDENTIFICATION  kappa_k = lambda_max / lambda_2 of the criterion's UNPENALIZED
    s-block Fisher information, the weighted graph Laplacian of the expected-Hessian
    weights h_DD[:,k] over the comparison graph; lambda_2 is its smallest eigenvalue on
    the sum-zero subspace. Poorly identified iff kappa_k > kappa_max (inf if disconnected).
  * RELIABILITY  rho_k = (var_k - noise_k) / var_k, where
        var_k   = Var_i(s_hat[:,k])                 (spread of the fitted scores)
        noise_k = mean_i Var(s_hat[i,k])            (s-block sandwich diagonal)
    i.e. the share of the observed score spread not attributable to estimation noise.

The Laplacian (``per_criterion_kappa`` / ``_laplacian``) and the rho formula are
model-agnostic; only the Fisher weights / per-query scores are model-specific. For the
gated model:
  * the per-query beta score is ``sum_k d_dir`` (citation residual does not reach beta);
  * h_bb (beta-beta bread) is ``sum cited*w_dir``, NOT sum h_DD;
  * the s-beta coupling column scatters ``cited*w_dir``, NOT h_DD;
  * the gamma-beta coupling is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from core.model.gated import _predictors

KAPPA_MAX = 1e6
RHO_THRESHOLD = 0.5


@dataclass
class FisherWeights:
    """Per-(pair, criterion) expected-Hessian weights + score residuals, all (T, K)/(T,)."""

    h_DD: np.ndarray  # score-score Fisher weight  w_cite*tanh^2 + cited*w_dir
    h_Dg: np.ndarray  # score-gamma cross weight    w_cite*tanh
    h_gg: np.ndarray  # gamma-gamma weight          w_cite
    h_Db: np.ndarray  # score-beta cross weight     cited*w_dir
    dL_dDelta: np.ndarray  # per-query score wrt Delta  r_cite*tanh + d_dir
    r_cite: np.ndarray  # citation residual           cited - p_cite
    d_dir: np.ndarray  # directional residual        pos - cited*p_dir


def _fisher_weights(s, gamma, beta, X, pairs) -> FisherWeights:
    delta, z_cite, z_dir, _, _ = _predictors(s, gamma, beta, pairs)
    Xf = np.asarray(X, float)
    cited = (Xf != 0).astype(float)
    pos = (Xf == 1).astype(float)
    p_cite, p_dir = expit(z_cite), expit(z_dir)
    w_cite, w_dir = p_cite * (1 - p_cite), p_dir * (1 - p_dir)
    tanh_d = np.tanh(delta)
    h_DD = w_cite * tanh_d**2 + cited * w_dir
    h_Dg = w_cite * tanh_d
    h_gg = w_cite
    h_Db = cited * w_dir
    r_cite = cited - p_cite
    d_dir = pos - cited * p_dir
    dL_dDelta = r_cite * tanh_d + d_dir
    return FisherWeights(h_DD, h_Dg, h_gg, h_Db, dL_dDelta, r_cite, d_dir)


def _laplacian(w, a, b, n):
    """Weighted graph Laplacian of edge weights ``w`` over pairs (a, b). Model-agnostic."""
    L = np.zeros((n, n))
    np.add.at(L, (a, a), w)
    np.add.at(L, (b, b), w)
    np.add.at(L, (a, b), -w)
    np.add.at(L, (b, a), -w)
    return L


def per_criterion_kappa(h_DD, pairs, n):
    """lambda_min (on sum-zero subspace), lambda_max, kappa of each criterion's UNPENALIZED
    s-block Fisher (weighted Laplacian of h_DD[:,k]). Returns arrays (K,). Model-agnostic."""
    a, b = pairs[:, 0], pairs[:, 1]
    K = h_DD.shape[1]
    lmin = np.zeros(K)
    lmax = np.zeros(K)
    kap = np.zeros(K)
    for k in range(K):
        L = _laplacian(h_DD[:, k], a, b, n)
        ev = np.linalg.eigvalsh(0.5 * (L + L.T))  # ascending; ev[0]~0 is the constant null
        lo = float(ev[1]) if n >= 2 else 0.0  # smallest eigenvalue ON the sum-zero subspace
        lo = max(lo, 0.0)
        hi = float(ev[-1])
        lmin[k], lmax[k] = lo, hi
        kap[k] = (hi / lo) if lo > 0 else np.inf
    return lmin, lmax, kap


def block_sandwich_s_diag(s, gamma, beta, X, pairs, ridge):
    """Var(s_hat[i,k]) as (n, K), per-criterion, without forming the dense P x P bread.

    Bordered-block-diagonal H (criterion blocks + beta border), meat B = U^T U;
    Var(s_hat[p]) = ||U H^{-1} e_p||^2 with the rank-1 beta Schur term. Gated-model
    beta border: u_beta = sum_k d_dir; h_bb = sum cited*w_dir; s-beta column scatters
    h_Db = cited*w_dir; gamma-beta = 0.
    """
    n, K = s.shape
    a, b = pairs[:, 0], pairs[:, 1]
    fw = _fisher_weights(s, gamma, beta, X, pairs)
    h_DD, h_Dg, h_gg, h_Db = fw.h_DD, fw.h_Dg, fw.h_gg, fw.h_Db
    dL, r_cite, d_dir = fw.dL_dDelta, fw.r_cite, fw.d_dir

    u_beta = d_dir.sum(axis=1)  # U[:, beta]  (T,)  -- d_dir only
    h_bb = float(h_Db.sum())  # beta-beta bread  -- cited*w_dir

    # Pass 1: per-criterion (n+1)x(n+1) inverse Dk^{-1}, beta-coupling ck, gk = Dk^{-1} ck.
    Dinv, gk_list = [], []
    ctDc = 0.0
    v = -u_beta.copy()  # v = U @ [g; -1]
    for k in range(K):
        Hk = np.zeros((n + 1, n + 1))
        Hk[:n, :n] = _laplacian(h_DD[:, k], a, b, n)
        Hk[np.arange(n), np.arange(n)] += ridge  # ridge on s diagonal
        col = np.zeros(n)  # s-gamma cross
        np.add.at(col, a, h_Dg[:, k])
        np.add.at(col, b, -h_Dg[:, k])
        Hk[:n, n] = col
        Hk[n, :n] = col
        Hk[n, n] = float(h_gg[:, k].sum())
        Dki = np.linalg.inv(Hk)
        col_sb = np.zeros(n)  # s-beta coupling: cited*w_dir
        np.add.at(col_sb, a, h_Db[:, k])
        np.add.at(col_sb, b, -h_Db[:, k])
        ck = np.concatenate([col_sb, [0.0]])  # gamma_k-beta = 0
        gk = Dki @ ck
        ctDc += float(ck @ gk)
        # U[:,idx] @ gk = dL*(gk[a]-gk[b]) + r_cite*gk[n]
        v += dL[:, k] * (gk[a] - gk[b]) + r_cite[:, k] * gk[n]
        Dinv.append(Dki)
        gk_list.append(gk)
    s_beta = 1.0 / (h_bb - ctDc)
    v_sq = float(v @ v)

    # Pass 2: per-criterion s-diagonal.
    diag = np.zeros((n, K))
    for k in range(K):
        Dki = Dinv[k]
        gk = gk_list[k]
        A = (
            dL[:, k][:, None] * (Dki[a, :n] - Dki[b, :n])
            + r_cite[:, k][:, None] * Dki[n, :n][None, :]
        )
        a_sq = np.einsum("ti,ti->i", A, A)
        gk_s = gk[:n]
        Atv = A.T @ v
        diag[:, k] = a_sq + 2.0 * s_beta * gk_s * Atv + (s_beta * gk_s) ** 2 * v_sq
    return diag
