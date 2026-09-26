"""L-BFGS-B driver for the holistic-verdict model.

Fits (s, gamma, beta) by minimizing the ridge-regularized NLL from
``core.model.holistic.likelihood``. After convergence:

  * column-centers s (a no-op modulo float at the ridge optimum; see DERIVATION.md §7),
  * runs a perturbed-init "sanity" rerun for the convex-problem diagnostic
    ``inf_diff = |L_primary - L_sanity|``,
  * computes Pearson ``sign_corrs`` between each column of s and the per-essay
    won-cited counts (positive on well-behaved data; warning otherwise).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import pearsonr

from core.model.holistic.likelihood import nll_and_grad, param_size, unpack

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Result of the joint MLE fit.

    Field set mirrors the diagnostic conventions of the old codebase's
    ``joint_mle_meta.json`` so downstream cross-judge comparisons can share
    summary tables.
    """

    s: np.ndarray  # (n, K) per-essay-per-criterion scores
    gamma: np.ndarray | None  # (K,) mention intercepts, or None if not included
    beta: float  # scalar position bias
    nll: float  # final NLL value (with ridge)
    nll_unregularized: float  # NLL value without the ridge term
    converged: bool
    n_iter: int
    params_flat: np.ndarray  # flat parameter vector for use with hessian()
    include_gamma: bool  # whether gamma was estimated
    n: int
    K: int
    T: int
    ridge: float
    inf_diff: float  # |L - L_sanity| from a second L-BFGS-B run
    sigmoid_beta: float  # σ(beta)
    sign_corrs: np.ndarray  # (K,) Pearson(s[:, k], won_cited[:, k])
    beta_pinned: bool = False  # True iff β was held fixed during the fit


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_lbfgs(params0, w, r, pairs, K, n, ridge, include_gamma, tol, max_iter, pin_beta=None):
    """Minimize NLL from a given initialization. Returns (params, value, niter, converged).

    When ``pin_beta`` is set, the β coordinate is masked out of the L-BFGS-B
    vector: the optimizer never sees it, and ``nll_and_grad`` is called with the
    fixed β substituted at evaluation time.
    """

    if pin_beta is None:

        def fg(p):
            return nll_and_grad(
                p,
                w,
                r,
                pairs,
                K,
                n,
                ridge=ridge,
                include_gamma=include_gamma,
            )

        res = minimize(
            fg,
            params0,
            method="L-BFGS-B",
            jac=True,
            options={"ftol": tol, "gtol": tol, "maxiter": max_iter},
        )
        return res.x, float(res.fun), int(res.nit), bool(res.success)

    # β-pinned: optimize on the (s, γ) slice of params; β is appended at the end.
    P = params0.shape[0]
    beta_idx = P - 1
    reduced0 = np.concatenate([params0[:beta_idx]])

    def fg_reduced(p_reduced):
        p_full = np.concatenate([p_reduced, np.array([float(pin_beta)], dtype=np.float64)])
        val, g_full = nll_and_grad(
            p_full,
            w,
            r,
            pairs,
            K,
            n,
            ridge=ridge,
            include_gamma=include_gamma,
            pin_beta=float(pin_beta),
        )
        return val, g_full[:beta_idx]

    res = minimize(
        fg_reduced,
        reduced0,
        method="L-BFGS-B",
        jac=True,
        options={"ftol": tol, "gtol": tol, "maxiter": max_iter},
    )
    full = np.concatenate([res.x, np.array([float(pin_beta)], dtype=np.float64)])
    return full, float(res.fun), int(res.nit), bool(res.success)


def _won_cited_counts(w, r, pairs, n: int) -> np.ndarray:
    """``won_cited[i, k] = #{t : winner(t) = i AND r[t, k] = 1}``.

    Uses ``w == 1 -> winner = a_t``, else ``b_t``. Returns (n, K) int.
    """
    w_arr = np.asarray(w, dtype=np.int64)
    r_arr = np.asarray(r, dtype=np.float64)
    pairs_arr = np.asarray(pairs, dtype=np.int64)
    winner = np.where(w_arr == 1, pairs_arr[:, 0], pairs_arr[:, 1])
    out = np.zeros((n, r_arr.shape[1]), dtype=np.float64)
    np.add.at(out, winner, r_arr)
    return out


def _pearson_per_column(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    K = s.shape[1]
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if np.std(s[:, k]) == 0.0 or np.std(w[:, k]) == 0.0:
            out[k] = 0.0
        else:
            out[k] = float(pearsonr(s[:, k], w[:, k]).statistic)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit(
    w,
    r,
    pairs,
    K: int,
    n: int,
    ridge: float = 1e-3,
    include_gamma: bool = True,
    method: str = "L-BFGS-B",
    tol: float = 1e-8,
    max_iter: int = 1000,
    verbose: bool = False,
    seed: int = 42,
    pin_beta: float | None = None,
) -> FitResult:
    """Fit the joint MLE by minimizing regularized NLL.

    ``method`` is L-BFGS-B only (other optimizers explicitly out of scope).

    If ``pin_beta`` is provided, β is held fixed at that value and only
    ``(s, γ)`` are optimized. The returned :class:`FitResult` carries
    ``beta = pin_beta`` and ``beta_pinned = True``.
    """
    if method != "L-BFGS-B":
        raise ValueError(f"unsupported method {method!r}; only L-BFGS-B is supported")

    w_arr = np.asarray(w, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    pairs_arr = np.asarray(pairs, dtype=np.int64)
    T = pairs_arr.shape[0]

    P = param_size(n, K, include_gamma)
    rng = np.random.RandomState(int(seed))

    # Primary fit from zero init.
    p0_primary = np.zeros(P, dtype=np.float64)
    if pin_beta is not None:
        p0_primary[-1] = float(pin_beta)
    p_hat, L_primary, n_iter, converged = _run_lbfgs(
        p0_primary,
        w_arr,
        r_arr,
        pairs_arr,
        K,
        n,
        ridge,
        include_gamma,
        tol,
        max_iter,
        pin_beta=pin_beta,
    )
    if verbose:
        log.info("[fit] primary L=%.8f n_iter=%d converged=%s", L_primary, n_iter, converged)

    # Sanity rerun from a perturbed init.
    p0_sanity = rng.normal(scale=0.1, size=P)
    if pin_beta is not None:
        p0_sanity[-1] = float(pin_beta)
    _, L_sanity, _, _ = _run_lbfgs(
        p0_sanity,
        w_arr,
        r_arr,
        pairs_arr,
        K,
        n,
        ridge,
        include_gamma,
        tol,
        max_iter,
        pin_beta=pin_beta,
    )
    inf_diff = abs(L_primary - L_sanity)
    if verbose:
        log.info("[fit] sanity  L=%.8f  inf_diff=%.3e", L_sanity, inf_diff)
    if inf_diff > 1e-4:
        warnings.warn(
            f"convex sanity check: inf_diff={inf_diff:.3e} > 1e-4. "
            "Convex NLL should have a unique minimum to floating-point precision. "
            "Likely a gradient bug.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Post-hoc column-centering of s. The NLL is invariant to per-column shifts
    # of s (mention term: differences cancel; BT term: differences cancel
    # column-by-column; γ and β unaffected), so this leaves the loss unchanged.
    # At the ridge optimum, columns are already near zero-mean; this scrubs
    # floating-point residuals only.
    s_hat, gamma_hat, beta_hat = unpack(p_hat, n, K, include_gamma)
    if pin_beta is not None:
        beta_hat = float(pin_beta)
    s_hat = s_hat - s_hat.mean(axis=0, keepdims=True)

    # Recompute params_flat from the centered s so callers that pass it to
    # hessian()/inference get a consistent linearization point.
    if include_gamma:
        params_flat = np.concatenate([s_hat.ravel(order="C"), gamma_hat, [beta_hat]])
    else:
        params_flat = np.concatenate([s_hat.ravel(order="C"), [beta_hat]])

    # Recompute NLL after centering (should match L_primary to ~1e-10).
    from core.model.holistic.likelihood import nll as nll_fn

    L_centered = nll_fn(
        params_flat,
        w_arr,
        r_arr,
        pairs_arr,
        K,
        n,
        ridge=ridge,
        include_gamma=include_gamma,
        pin_beta=pin_beta,
    )
    nll_unreg = L_centered - 0.5 * ridge * float((s_hat * s_hat).sum())

    # Sign correlations (Pearson per column).
    won_cited = _won_cited_counts(w_arr, r_arr, pairs_arr, n)
    sign_corrs = _pearson_per_column(s_hat, won_cited)
    if (sign_corrs < 0).any():
        bad = np.where(sign_corrs < 0)[0]
        warnings.warn(
            f"negative sign correlations at criteria {bad.tolist()}: "
            f"{sign_corrs[bad].round(3).tolist()}",
            RuntimeWarning,
            stacklevel=2,
        )

    return FitResult(
        s=s_hat,
        gamma=gamma_hat if include_gamma else None,
        beta=float(beta_hat),
        nll=L_centered,
        nll_unregularized=float(nll_unreg),
        converged=converged,
        n_iter=n_iter,
        params_flat=params_flat,
        include_gamma=include_gamma,
        n=n,
        K=K,
        T=T,
        ridge=float(ridge),
        inf_diff=float(inf_diff),
        sigmoid_beta=float(expit(beta_hat)),
        sign_corrs=sign_corrs,
        beta_pinned=(pin_beta is not None),
    )
