"""Sandwich-CI inference for the holistic-verdict model.

Sandwich covariance (M-estimator CLT):

    Cov(θ̂) ≈ Ĥ^{-1}  Ĵ  Ĥ^{-1}

    Ĥ = sum_t ∇²ρ_t(θ̂) + (ridge term on s block)  — from ``hessian.hessian``
    Ĵ = sum_t  s_t  s_t^T                          — per-pair score outer products

The per-pair score s_t = ∇ρ_t(θ̂) is sparse: nonzero only on s_{a_t,·}, s_{b_t,·},
gamma (when included), and beta. We accumulate Ĵ by sparse rank-1 updates,
which avoids materializing a dense (T, P) score matrix.

Under correct specification A = B and Cov collapses to Ĥ^{-1}; under
misspecification (heteroscedasticity, model error) the sandwich is the
asymptotically correct covariance.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from core.model.holistic.hessian import hessian
from core.model.holistic.likelihood import _predictors, unpack


_JITTER = 1e-10


# ---------------------------------------------------------------------------
# Empirical Fisher Ĵ via sparse rank-1 updates.
# ---------------------------------------------------------------------------


def _empirical_fisher(fit_result, w, r, pairs) -> np.ndarray:
    """Ĵ = Σ_t score_t score_t^T at θ̂. Returns (P, P) numpy float64."""
    n = fit_result.n
    K = fit_result.K
    include_gamma = fit_result.include_gamma
    P = fit_result.params_flat.shape[0]

    w_arr = np.asarray(w, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    pairs_arr = np.asarray(pairs, dtype=np.int64)
    T = pairs_arr.shape[0]

    s_hat, gamma_hat, beta_hat = unpack(fit_result.params_flat, n, K, include_gamma)
    z_bt, z_m, eta = _predictors(s_hat, gamma_hat, beta_hat, w_arr, pairs_arr)
    u = expit(z_bt) - w_arr  # (T,)
    v = expit(z_m) - r_arr  # (T, K)

    nK = n * K
    gamma_offset = nK
    beta_offset = nK + (K if include_gamma else 0)

    J = np.zeros((P, P), dtype=np.float64)

    a = pairs_arr[:, 0]
    b = pairs_arr[:, 1]
    for t in range(T):
        at = int(a[t])
        bt = int(b[t])
        ut = float(u[t])
        eta_t = float(eta[t])
        vt = v[t]  # (K,)

        # Score support: 2K entries on s_{a,·} and s_{b,·}, K on gamma,
        # 1 on beta.  Build a dense score vector on this support and outer-it.
        score = np.zeros(P, dtype=np.float64)

        # s_{a, k} = ut + eta_t * v[t, k]   (BT row-sum part + mention part for a)
        a_idx = at * K + np.arange(K)
        score[a_idx] = ut + eta_t * vt
        # s_{b, k} = -ut - eta_t * v[t, k]
        b_idx = bt * K + np.arange(K)
        score[b_idx] = -ut - eta_t * vt
        # gamma_k = v[t, k]
        if include_gamma:
            score[gamma_offset : gamma_offset + K] = vt
        # beta = ut
        score[beta_offset] = ut

        # Accumulate rank-1 outer product on the score's support set.
        support = np.concatenate(
            [
                a_idx,
                b_idx,
                np.arange(gamma_offset, gamma_offset + K)
                if include_gamma
                else np.array([], dtype=np.int64),
                np.array([beta_offset], dtype=np.int64),
            ]
        )
        # If a_t == b_t (shouldn't happen but guard), merge.
        sub = score[support]
        # Use np.ix_ to slot the dense outer product into J.
        np.add.at(J, (support[:, None], support[None, :]), np.outer(sub, sub))

    return J


# ---------------------------------------------------------------------------
# Sandwich covariance
# ---------------------------------------------------------------------------


def sandwich_covariance(fit_result, w, r, pairs) -> np.ndarray:
    """Compute Cov(θ̂) ≈ Ĥ^{-1} Ĵ Ĥ^{-1}.

    A small jitter (``_JITTER``) is added to the diagonal of Ĥ for numerical
    invertibility. The ridge from ``fit_result.ridge`` is already inside Ĥ.
    """
    H = hessian(
        fit_result.params_flat,
        w,
        r,
        pairs,
        K=fit_result.K,
        n=fit_result.n,
        ridge=fit_result.ridge,
        include_gamma=fit_result.include_gamma,
    )
    H_reg = H + _JITTER * np.eye(H.shape[0])

    J = _empirical_fisher(fit_result, w, r, pairs)

    # Σ = H^{-1} J H^{-1}; compute via two solves.
    try:
        H_inv_J = np.linalg.solve(H_reg, J)
        cov = np.linalg.solve(H_reg, H_inv_J.T).T
    except np.linalg.LinAlgError:
        H_pinv = np.linalg.pinv(H_reg)
        cov = H_pinv @ J @ H_pinv

    # Symmetrize to scrub floating-point drift.
    cov = 0.5 * (cov + cov.T)
    return cov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def std_errors(fit_result, cov) -> dict:
    """Return dict with keys 's' (n,K), 'gamma' (K,) or None, 'beta' (scalar)."""
    n = fit_result.n
    K = fit_result.K
    include_gamma = fit_result.include_gamma

    diag = np.diag(cov).copy()
    diag[diag < 0] = 0.0  # clean tiny negatives from drift
    se = np.sqrt(diag)

    nK = n * K
    out = {"s": se[:nK].reshape(n, K)}
    if include_gamma:
        out["gamma"] = se[nK : nK + K]
        out["beta"] = float(se[nK + K])
    else:
        out["gamma"] = None
        out["beta"] = float(se[nK])
    return out


_PAT_S = re.compile(r"^s\[(\d+),\s*(\d+)\]$")
_PAT_GAMMA = re.compile(r"^gamma\[(\d+)\]$")


def _param_index(fit_result, name: str) -> int:
    """Map a param name like 's[5,3]' or 'gamma[2]' or 'beta' to its flat index."""
    n = fit_result.n
    K = fit_result.K
    include_gamma = fit_result.include_gamma
    nK = n * K

    if name == "beta":
        return nK + (K if include_gamma else 0)
    m = _PAT_GAMMA.match(name)
    if m:
        if not include_gamma:
            raise KeyError(f"gamma was not estimated; cannot index {name!r}")
        k = int(m.group(1))
        if not 0 <= k < K:
            raise IndexError(f"k={k} out of range [0, {K})")
        return nK + k
    m = _PAT_S.match(name)
    if m:
        i = int(m.group(1))
        k = int(m.group(2))
        if not 0 <= i < n:
            raise IndexError(f"i={i} out of range [0, {n})")
        if not 0 <= k < K:
            raise IndexError(f"k={k} out of range [0, {K})")
        return i * K + k
    raise ValueError(f"could not parse param name {name!r}")


def _param_value(fit_result, name: str) -> float:
    """Return the fitted value of a named parameter."""
    if name == "beta":
        return float(fit_result.beta)
    m = _PAT_GAMMA.match(name)
    if m:
        k = int(m.group(1))
        return float(fit_result.gamma[k])
    m = _PAT_S.match(name)
    if m:
        i = int(m.group(1))
        k = int(m.group(2))
        return float(fit_result.s[i, k])
    raise ValueError(f"could not parse param name {name!r}")


def confidence_interval(fit_result, cov, param_name: str, level: float = 0.95):
    """Wald CI for a named parameter.

    ``param_name``: 's[i,k]', 'gamma[k]', or 'beta'.
    """
    idx = _param_index(fit_result, param_name)
    se = float(np.sqrt(max(0.0, cov[idx, idx])))
    z = float(norm.ppf(0.5 + level / 2.0))
    val = _param_value(fit_result, param_name)
    return (val - z * se, val + z * se)


def per_criterion_se(fit_result, cov, criterion_index: int) -> dict:
    """SE for std(s_{·,k}) and gamma_k.

    delta method: with s_{·,k} centered (s̄_k = 0),
        ∂ std_k / ∂ s_{i,k} = s_{i,k} / (n * std_k)
        Var(std_k) ≈ g^T Σ g where g is the above (n,) gradient over the
        s_{·,k} sub-block of θ̂; off-criterion entries of g are zero.
    """
    n = fit_result.n
    K = fit_result.K
    k = int(criterion_index)
    if not 0 <= k < K:
        raise IndexError(f"criterion_index={k} out of [0, {K})")

    s_col = fit_result.s[:, k]
    std_k = float(np.std(s_col, ddof=0))

    if std_k > 0:
        # Indices of s_{·,k} in flat layout.
        idx = np.arange(n) * K + k
        g = s_col / (n * std_k)  # (n,)
        sub_cov = cov[np.ix_(idx, idx)]
        var_std = float(g @ sub_cov @ g)
        std_se = float(np.sqrt(max(0.0, var_std)))
    else:
        std_se = float("nan")

    out = {"std_s": std_k, "std_s_se": std_se}
    if fit_result.include_gamma:
        out["gamma"] = float(fit_result.gamma[k])
        gamma_idx = n * K + k
        out["gamma_se"] = float(np.sqrt(max(0.0, cov[gamma_idx, gamma_idx])))
    else:
        out["gamma"] = None
        out["gamma_se"] = None
    return out
