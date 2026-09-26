"""Gated symmetric Bradley--Terry model: likelihood, analytic gradient, expected Hessian (Fisher), fit.

The paper's main model (Section ``sec:method``, eq. ``eq:nll``; Appendix ``app:model``): a
citation gate with an even link of the score gap and a per-criterion intercept gamma, and a
direction channel with a global position term beta.

Parameters
----------
Flat layout ``params = [s.ravel(order='C'), gamma, beta]``, size ``n*K + K + 1``
(flat-index parameter packing over scores S, citation intercepts gamma, bias beta).

Model (per (t, k)), with X_{t,k} in {-1, 0, +1}, Delta = s[a,k] - s[b,k], and a
scalar position-bias parameter ``beta``:

    z^cite_{t,k} = log(2 cosh(Delta)) + gamma_k     # beta does NOT enter the gate
    p^cite       = sigma(z^cite)
    z^dir_{t,k}  = Delta + beta
    p^dir        = sigma(z^dir)                      # P(winner = A | cited)
    L_{t,k} = 1[X!=0] log p^cite + 1[X==0] log(1-p^cite)
            + 1[X==+1] log p^dir + 1[X==-1] log(1-p^dir)

NLL = - sum_{t,k} L_{t,k} + 0.5 * ridge * ||s||_F^2     (ridge applied to s only)

The citation gate "log(2 cosh Delta)" has positive curvature but enters through a
concave composition, so the NLL is *mildly* non-convex near Delta=0. Ridge>0, the
informative init, and the mean-zero s projection keep the fit well-behaved; the
*expected* Hessian used for the sandwich is PSD by construction.

See ``DERIVATION.md`` for the full derivation. ``python -m core.model.gated --gradcheck``
(from the repository root) checks the analytic gradient against finite differences.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param packing: flat vector = vec(S) ++ gamma ++ beta
# ---------------------------------------------------------------------------


def param_size(n: int, K: int) -> int:
    return n * K + K + 1


def pack(s: np.ndarray, gamma: np.ndarray, beta: float) -> np.ndarray:
    return np.concatenate(
        [
            s.ravel(order="C"),
            np.asarray(gamma, dtype=np.float64).ravel(),
            np.array([float(beta)], dtype=np.float64),
        ]
    )


def unpack(params: np.ndarray, n: int, K: int):
    nK = n * K
    s = params[:nK].reshape(n, K)
    gamma = params[nK : nK + K]
    beta = float(params[nK + K])
    return s, gamma, beta


# ---------------------------------------------------------------------------
# Numerically stable helper
# ---------------------------------------------------------------------------


def log2cosh(x: np.ndarray) -> np.ndarray:
    """Stable evaluator of ``log(2 cosh x)``, the even link of the citation gate.

    Identity: ``log(2 cosh u) = |u| + log(1 + exp(-2|u|))``.
    """
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax))


# ---------------------------------------------------------------------------
# NLL and gradient
# ---------------------------------------------------------------------------


def _predictors(s, gamma, beta, pairs):
    """Return (delta, z_cite, z_dir, a, b).

    ``z_cite = log(2 cosh(delta)) + gamma_k`` (beta-free gate);
    ``z_dir  = delta + beta``.
    """
    a = pairs[:, 0]
    b = pairs[:, 1]
    delta = s[a] - s[b]  # (T, K)
    z_cite = log2cosh(delta) + gamma[None, :]  # (T, K)
    z_dir = delta + float(beta)  # (T, K)
    return delta, z_cite, z_dir, a, b


def gate_probabilities(s, gamma, beta, pairs):
    """Per-(pair, criterion) probabilities of the model's two observable binary events.

    Returns ``(p_cite, p_dir)``, both (T, K):

      * ``p_cite = sigma(log(2 cosh(delta)) + gamma_k)`` -- criterion k is cited on pair p.
        Defined on every cell, and beta-free: which item is shown first cannot change
        *whether* a criterion is invoked.
      * ``p_dir = sigma(delta + beta)`` -- the first-shown item is the one favored, GIVEN
        that the criterion was cited. Only meaningful on cited cells.

    These are the two forecasts the calibration analysis scores (the paper's citation gate
    and direction gate). Named and shaped to match
    :func:`core.model.holistic.gate_probabilities`, so the two experiments' calibration
    drivers stay structurally the same.
    """
    _, z_cite, z_dir, _, _ = _predictors(
        np.asarray(s, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
        beta,
        np.asarray(pairs, dtype=np.int64),
    )
    return expit(z_cite), expit(z_dir)


def nll(params, X, pairs, K, n, ridge: float = 1e-3) -> float:
    s, gamma, beta = unpack(params, n, K)
    X = np.asarray(X, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)

    _, z_cite, z_dir, _, _ = _predictors(s, gamma, beta, pairs)

    cited = (X != 0).astype(np.float64)
    pos = (X == 1).astype(np.float64)
    neg = (X == -1).astype(np.float64)

    ll_cite = cited * log_expit(z_cite) + (1.0 - cited) * log_expit(-z_cite)
    ll_dir = pos * log_expit(z_dir) + neg * log_expit(-z_dir)

    pen = 0.5 * float(ridge) * float((s * s).sum()) if ridge > 0.0 else 0.0
    return float(-(ll_cite + ll_dir).sum() + pen)


def nll_and_grad(params, X, pairs, K, n, ridge: float = 1e-3):
    s, gamma, beta = unpack(params, n, K)
    X = np.asarray(X, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)

    delta, z_cite, z_dir, a, b = _predictors(s, gamma, beta, pairs)

    cited = (X != 0).astype(np.float64)
    pos = (X == 1).astype(np.float64)
    neg = (X == -1).astype(np.float64)

    # ---- value ----
    ll_cite = cited * log_expit(z_cite) + (1.0 - cited) * log_expit(-z_cite)
    ll_dir = pos * log_expit(z_dir) + neg * log_expit(-z_dir)
    pen = 0.5 * float(ridge) * float((s * s).sum()) if ridge > 0.0 else 0.0
    val = float(-(ll_cite + ll_dir).sum() + pen)

    # ---- gradient pieces (log-lik; flip sign at the end for NLL) ----
    p_cite = expit(z_cite)
    p_dir = expit(z_dir)
    r_cite = cited - p_cite  # citation residual (T, K)
    d_dir = pos - cited * p_dir  # directional residual (T, K)

    # dz_cite/dDelta = tanh(Delta);  dz_dir/dDelta = 1  ->  dL/dDelta = r_cite*tanh + d_dir.
    tanh_d = np.tanh(delta)  # (T, K)
    dL_dDelta = r_cite * tanh_d + d_dir  # (T, K)

    # NLL grad wrt s: chain through Delta. dDelta/ds[i,k] = 1[i=a_t] - 1[i=b_t].
    g_s = np.zeros_like(s)
    g_term = -dL_dDelta  # (T, K), NLL grad wrt Delta
    np.add.at(g_s, a, g_term)
    np.add.at(g_s, b, -g_term)
    if ridge > 0.0:
        g_s += ridge * s

    # NLL grad wrt gamma_k: dz_cite/dgamma_k = 1.
    g_gamma = -(r_cite.sum(axis=0))  # (K,)

    # NLL grad wrt beta: beta enters ONLY z_dir, so dL/dbeta = sum_{t,k} d_dir.
    g_beta = float(-(d_dir.sum()))

    grad = np.concatenate([g_s.ravel(order="C"), g_gamma, np.array([g_beta])])
    return val, grad


# ---------------------------------------------------------------------------
# Expected (Fisher) Hessian — dense P x P, including the beta border.
# ---------------------------------------------------------------------------


def expected_hessian_full(params, X, pairs, K, n, ridge: float = 1e-3) -> np.ndarray:
    """Return the dense ``(P, P)`` expected Hessian of the NLL at ``params``,
    where ``P = n*K + K + 1``.

    Layout of flat indices:
        s[i, k]   -> i * K + k         (0 .. n*K-1)
        gamma_k   -> n * K + k         (n*K .. n*K+K-1)
        beta      -> n * K + K         (last)

    "Expected" form: drops the ``r_cite * sech^2`` middle term (vanishes in
    expectation), leaving a PSD matrix.

    The beta border (beta enters only the direction logit):
      * s-beta coupling weight is ``cited * w_dir`` (NOT h_DD);
      * gamma-beta coupling is 0 (beta is absent from the gate);
      * beta-beta is ``sum cited * w_dir`` (NOT sum h_DD).
    """
    s, gamma, beta = unpack(params, n, K)
    X = np.asarray(X, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    a = pairs[:, 0]
    b = pairs[:, 1]

    delta, z_cite, z_dir, _, _ = _predictors(s, gamma, beta, pairs)
    cited = (X != 0).astype(np.float64)
    p_cite = expit(z_cite)  # (T, K)
    p_dir = expit(z_dir)
    w_cite = p_cite * (1.0 - p_cite)  # (T, K)
    w_dir = p_dir * (1.0 - p_dir)  # (T, K)
    tanh_d = np.tanh(delta)  # (T, K)

    # Expected -d2L/dDelta^2 = w_cite * tanh^2 + cited * w_dir.
    h_DD = w_cite * tanh_d**2 + cited * w_dir  # (T, K)
    # -d2L/dDelta dgamma = w_cite * tanh.
    h_Dg = w_cite * tanh_d  # (T, K)
    # -d2L/dgamma^2 = w_cite.
    h_gg = w_cite  # (T, K)
    # -d2L/dDelta dbeta = cited * w_dir  (beta only in the direction logit).
    h_Db = cited * w_dir  # (T, K)

    P = n * K + K + 1
    H = np.zeros((P, P), dtype=np.float64)

    s_idx_for_k = [np.array([i * K + k for i in range(n)], dtype=np.int64) for k in range(K)]
    gamma_flat = np.array([n * K + k for k in range(K)], dtype=np.int64)
    beta_idx = n * K + K

    for k in range(K):
        s_idx = s_idx_for_k[k]
        g_idx = int(gamma_flat[k])

        ht = h_DD[:, k]
        # Score-score block via scatter (weighted graph Laplacian of h_DD[:,k]).
        np.add.at(H, (s_idx[a], s_idx[a]), ht)
        np.add.at(H, (s_idx[b], s_idx[b]), ht)
        np.add.at(H, (s_idx[a], s_idx[b]), -ht)
        np.add.at(H, (s_idx[b], s_idx[a]), -ht)
        # s-gamma cross block.
        col = np.zeros(n, dtype=np.float64)
        np.add.at(col, a, h_Dg[:, k])
        np.add.at(col, b, -h_Dg[:, k])
        H[s_idx, g_idx] = col
        H[g_idx, s_idx] = col
        # gamma-gamma diagonal.
        H[g_idx, g_idx] = float(h_gg[:, k].sum())

        # beta border for criterion k.
        # s[i,k]-beta cross: sum_{t:a=i} h_Db[t,k] - sum_{t:b=i} h_Db[t,k].
        col_sb = np.zeros(n, dtype=np.float64)
        np.add.at(col_sb, a, h_Db[:, k])
        np.add.at(col_sb, b, -h_Db[:, k])
        H[s_idx, beta_idx] = col_sb
        H[beta_idx, s_idx] = col_sb
        # gamma_k-beta cross: 0 (beta absent from the gate). Left as zero.

    # beta-beta: sum over all (t, k) of cited * w_dir.
    H[beta_idx, beta_idx] = float(h_Db.sum())

    # Ridge on s diagonal only (s flat indices are exactly 0 .. n*K-1).
    if ridge > 0.0:
        di = np.arange(n * K)
        H[di, di] += ridge
    return H


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_params(X, K: int, n: int) -> np.ndarray:
    """Initialization at the symmetric point Delta=0.

    - s = 0
    - beta_init = logit(p_A_given_cited)            (z_dir = Delta + beta, slope 1)
    - gamma_k   = logit(p_cite_k) - log 2           (log(2 cosh 0) = log 2)

    Clip per-criterion citation rates and the conditional A-rate away from
    {0, 1} for numerical stability.
    """
    X = np.asarray(X)
    cited_mask = X != 0
    pos_mask = X == 1
    cited = cited_mask.astype(np.float64)
    pos = pos_mask.astype(np.float64)
    p_cite = cited.mean(axis=0)
    p_cite = np.clip(p_cite, 1e-6, 1.0 - 1e-6)

    total_cited = float(cited.sum())
    total_pos = float(pos.sum())
    p_A_cited = (total_pos / total_cited) if total_cited > 0 else 0.5
    p_A_cited = float(np.clip(p_A_cited, 1e-6, 1.0 - 1e-6))
    beta_init = float(np.log(p_A_cited / (1.0 - p_A_cited)))

    gamma0 = np.log(p_cite / (1.0 - p_cite)) - np.log(2.0)
    s0 = np.zeros((n, K), dtype=np.float64)
    return pack(s0, gamma0, beta_init)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    s: np.ndarray
    gamma: np.ndarray
    beta: float
    nll: float
    nll_unregularized: float
    n_iter: int
    converged: bool
    grad_norm: float
    params_flat: np.ndarray
    n: int
    K: int
    T: int
    ridge: float


def fit_model(
    X,
    pairs,
    K: int,
    n: int,
    ridge: float = 1e-3,
    tol: float = 1e-5,
    max_iter: int = 2000,
    init: np.ndarray | None = None,
    verbose: bool = False,
    pin_beta: float | None = None,
) -> FitResult:
    """Fit (s, gamma, beta) via L-BFGS-B; s is centred per criterion after the fit.

    ``pin_beta``: if not None, holds beta fixed at this value via bounds.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    pairs_arr = np.asarray(pairs, dtype=np.int64)
    T = pairs_arr.shape[0]

    p0 = init.copy() if init is not None else init_params(X_arr, K, n)
    if pin_beta is not None:
        p0[-1] = float(pin_beta)

    def fg(p):
        return nll_and_grad(p, X_arr, pairs_arr, K, n, ridge=ridge)

    bounds = None
    if pin_beta is not None:
        bounds = [(None, None)] * (n * K + K) + [(float(pin_beta), float(pin_beta))]

    res = minimize(
        fg,
        p0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": tol * 1e-2, "gtol": tol, "maxiter": max_iter},
    )

    p_hat = res.x
    s_hat, gamma_hat, beta_hat = unpack(p_hat, n, K)

    # Project s to mean-zero per criterion. Do NOT demean gamma or beta.
    s_hat = s_hat - s_hat.mean(axis=0, keepdims=True)
    params_flat = pack(s_hat, gamma_hat, beta_hat)

    L_centered = nll(params_flat, X_arr, pairs_arr, K, n, ridge=ridge)
    nll_unreg = L_centered - 0.5 * float(ridge) * float((s_hat * s_hat).sum())

    _, g_at_opt = nll_and_grad(params_flat, X_arr, pairs_arr, K, n, ridge=ridge)
    g_for_norm = g_at_opt[:-1] if pin_beta is not None else g_at_opt
    grad_norm = float(np.linalg.norm(g_for_norm, ord=np.inf))

    per_obs = grad_norm / max(T * K, 1)
    if per_obs > tol:
        warnings.warn(
            f"[gated fit] exit |grad|_inf={grad_norm:.3e} (per-obs {per_obs:.3e}) > tol {tol:.1e}",
            RuntimeWarning,
            stacklevel=2,
        )
    if verbose:
        log.info(
            "[gated fit] L=%.6f n_iter=%d grad_inf=%.3e beta=%.4f",
            L_centered,
            int(res.nit),
            grad_norm,
            float(beta_hat),
        )

    return FitResult(
        s=s_hat,
        gamma=np.asarray(gamma_hat, dtype=np.float64),
        beta=float(beta_hat),
        nll=float(L_centered),
        nll_unregularized=float(nll_unreg),
        n_iter=int(res.nit),
        converged=bool(res.success),
        grad_norm=grad_norm,
        params_flat=params_flat,
        n=int(n),
        K=int(K),
        T=int(T),
        ridge=float(ridge),
    )


# ---------------------------------------------------------------------------
# Gradient self-check (finite differences) — `python -m core.model.gated --gradcheck`
# ---------------------------------------------------------------------------


def _gradcheck(seed: int = 0) -> int:
    """Compare the analytic NLL gradient with central finite differences on a random
    small problem; print the max absolute error and return 0 if it is below 1e-6."""
    rng = np.random.RandomState(seed)
    n, K, T = 18, 4, 90
    s = rng.randn(n, K) * 0.5
    gamma = rng.randn(K) * 0.3
    beta = 0.25
    pairs = np.column_stack([rng.randint(0, n, T), rng.randint(0, n, T)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    X = rng.choice([-1, 0, 1], size=(len(pairs), K), p=[0.3, 0.4, 0.3]).astype(np.int8)
    ridge = 1e-3
    p0 = pack(s, gamma, beta)

    val, g = nll_and_grad(p0, X, pairs, K, n, ridge=ridge)
    eps = 1e-6
    gnum = np.zeros_like(p0)
    for i in range(p0.size):
        d = np.zeros_like(p0)
        d[i] = eps
        gnum[i] = (
            nll(p0 + d, X, pairs, K, n, ridge=ridge) - nll(p0 - d, X, pairs, K, n, ridge=ridge)
        ) / (2 * eps)
    err = float(np.max(np.abs(g - gnum)))
    print(f"gradcheck: max|analytic - finite-diff| = {err:.2e}  (nll={val:.4f})")
    ok = err < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.gradcheck:
        raise SystemExit(_gradcheck(args.seed))
    ap.error("nothing to do; pass --gradcheck")
