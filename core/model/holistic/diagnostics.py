"""Per-criterion identification (kappa) and reliability (rho) for the holistic-verdict model.

The analogue of the gated model's diagnostics (``core/model/diagnostics.py``) for the model in
``core/model/holistic/``. Same two quantities, same conventions:

  * RELIABILITY   rho_k = (var_k - noise_k) / var_k, where
        var_k   = Var_i( s_hat[:, k] )                 spread of the fitted column
        noise_k = mean_i Var( s_hat[i, k] )            sandwich variance of its entries
    rho_k is the fraction of the column's spread that is signal rather than estimation
    noise. rho_k <= 0 means the column is pure noise.

  * IDENTIFICATION  kappa_k = lambda_max / lambda_min of criterion k's UNPENALIZED
    s-block expected Fisher, a weighted graph Laplacian over essays, with lambda_min
    taken on the sum-zero subspace (the constant vector is the model's null direction).
    Large kappa means the comparison graph, weighted by how informative each comparison
    is about criterion k, is badly conditioned -- the column is only weakly pinned down.

WHICH CHANNEL WEIGHTS KAPPA (the one real design decision). The holistic model has two
channels::

    z_bt_t     = sum_k Delta_{t,k} + beta          (winner)
    z_m_{t,k}  = eta_t * Delta_{t,k} + gamma_k     (mention), eta_t = 2 w_t - 1

Only the MENTION channel carries criterion-specific information: the winner channel sees
every criterion through the same ``sum_k``, so it constrains only the total ``sum_k s_k``
and contributes an identical term to every criterion. Including it would make every
column look well-conditioned regardless of how much the judge actually said about it.
So kappa is computed from the mention-channel Fisher weight

    h_k[t] = p_m(1 - p_m)      (eta_t^2 = 1, so eta drops out)

and ``kappa_bt`` -- the winner channel's own condition number, identical for all k -- is
returned alongside for reference.

Note this differs from the gated model in an important way: there the direction term is
gated on citation (``cited * w_dir``), whereas here EVERY (t, k) cell is an observation --
a non-citation is informative too. So even a rarely-cited criterion draws information from
all T comparisons, and its kappa is correspondingly better behaved than citation counts
alone would suggest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class CriterionDiagnostics:
    """All arrays are length-K unless noted."""

    rho: np.ndarray
    var: np.ndarray
    noise: np.ndarray
    kappa: np.ndarray
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    n_cited_pairs: np.ndarray
    kappa_bt: float  # winner-channel condition number (shared across criteria)


def _laplacian(weights: np.ndarray, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Weighted graph Laplacian of edge weights over pairs (a, b). Model-agnostic."""
    L = np.zeros((n, n), dtype=np.float64)
    np.add.at(L, (a, a), weights)
    np.add.at(L, (b, b), weights)
    np.add.at(L, (a, b), -weights)
    np.add.at(L, (b, a), -weights)
    return L


def _condition_number(L: np.ndarray, n: int) -> tuple[float, float, float]:
    """(lambda_min on the sum-zero subspace, lambda_max, kappa) of a Laplacian."""
    ev = np.linalg.eigvalsh(0.5 * (L + L.T))  # ascending; ev[0] ~ 0 is the constant null
    lo = max(float(ev[1]), 0.0) if n >= 2 else 0.0
    hi = float(ev[-1])
    return lo, hi, (hi / lo if lo > 0 else np.inf)


def mention_fisher_weights(fit_result, w: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Mention-channel expected-Fisher weight p(1-p) per (t, k). Shape (T, K)."""
    s = fit_result.s
    a, b = pairs[:, 0], pairs[:, 1]
    delta = s[a] - s[b]  # (T, K)
    eta = 2.0 * np.asarray(w, dtype=np.float64) - 1.0  # (T,)
    gamma = fit_result.gamma if fit_result.include_gamma else np.zeros(s.shape[1])
    p = expit(eta[:, None] * delta + gamma[None, :])
    return p * (1.0 - p)  # eta^2 = 1


def winner_fisher_weights(fit_result, pairs: np.ndarray) -> np.ndarray:
    """Winner-channel expected-Fisher weight p(1-p) per comparison. Shape (T,)."""
    s = fit_result.s
    a, b = pairs[:, 0], pairs[:, 1]
    p = expit((s[a] - s[b]).sum(axis=1) + fit_result.beta)
    return p * (1.0 - p)


def per_criterion_diagnostics(fit_result, w, r, pairs, se_s) -> CriterionDiagnostics:
    """Compute rho and kappa for every criterion.

    Args:
        fit_result: a ``core.model.holistic.FitResult`` (s already column-centred by ``fit``).
        w: (T,) winner indicators; r: (T, K) mention indicators; pairs: (T, 2) essay indices.
        se_s: (n, K) standard errors of ``s_hat`` -- i.e. ``std_errors(fit, cov)["s"]``.
    """
    s = fit_result.s
    n, K = s.shape
    a, b = pairs[:, 0], pairs[:, 1]
    se_s = np.asarray(se_s, dtype=np.float64)
    if se_s.shape != (n, K):
        raise ValueError(f"se_s must be {(n, K)}, got {se_s.shape}")

    var = s.var(axis=0)  # column spread (s is centred)
    noise = (se_s**2).mean(axis=0)  # mean sandwich variance of the entries
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(var > 0, (var - noise) / var, -np.inf)

    h_mention = mention_fisher_weights(fit_result, w, pairs)
    lmin = np.zeros(K)
    lmax = np.zeros(K)
    kap = np.zeros(K)
    for k in range(K):
        lmin[k], lmax[k], kap[k] = _condition_number(_laplacian(h_mention[:, k], a, b, n), n)

    _, _, kappa_bt = _condition_number(
        _laplacian(winner_fisher_weights(fit_result, pairs), a, b, n), n
    )

    return CriterionDiagnostics(
        rho=rho,
        var=var,
        noise=noise,
        kappa=kap,
        lambda_min=lmin,
        lambda_max=lmax,
        n_cited_pairs=np.asarray(r, dtype=np.int64).sum(axis=0),
        kappa_bt=float(kappa_bt),
    )


# --------------------------------------------------------------------------- #
# Score spread with sandwich uncertainty
# --------------------------------------------------------------------------- #
@dataclass
class ScoreSpread:
    """Per-criterion spread of the fitted score column. All arrays are length-K.

    ``std`` is the raw column spread; ``std_denoised`` removes the estimation-noise
    component the same way :func:`per_criterion_diagnostics` does when forming rho --
    ``var_denoised = var - noise`` -- so ``std_denoised == std * sqrt(rho)`` wherever
    ``rho > 0``. It answers "how much do essays *really* differ on this criterion",
    net of how imprecisely each essay's score is pinned down.
    """

    var: np.ndarray
    noise: np.ndarray
    rho: np.ndarray
    std: np.ndarray
    std_denoised: np.ndarray
    se_std: np.ndarray
    se_std_denoised: np.ndarray


def score_spread(fit_result, cov, se_s) -> ScoreSpread:
    """Column spreads of ``s_hat`` with delta-method standard errors.

    The spread of criterion k is ``sigma_k = sqrt(V_k)`` with
    ``V_k = (1/n) * s_k^T M s_k`` (M the centring matrix), matching the population
    variance ``s.var(axis=0)`` that :func:`per_criterion_diagnostics` uses for rho.

    Delta method: ``dV_k/ds_k = (2/n) M s_k``, so ``Var(V_k) ~ g^T Sigma_k g`` where
    ``Sigma_k`` is criterion k's (n, n) block of the sandwich covariance -- the entries
    ``s[i, k]`` live at flat indices ``i * K + k`` -- and
    ``SE(sigma_k) = sqrt(Var(V_k)) / (2 sigma_k)``.

    ``se_std_denoised`` is deliberately incomplete, in a known direction. The denoised
    spread is ``sqrt(V_k - N_k)`` where BOTH terms are estimated from the same data, so
    strictly ``Var(V_k - N_k) = Var(V_k) + Var(N_k) - 2 Cov(V_k, N_k)``. Only ``Var(V_k)``
    is propagated here; ``N_k`` is treated as a plug-in constant, because it averages
    diagonal entries of the sandwich covariance and its sampling variance is the variance
    of a variance estimator -- not available from ``cov`` alone, and needing a bootstrap
    over comparisons to obtain. So ``se_std_denoised`` is a LOWER BOUND on the true
    uncertainty of the denoised spread.

    Do not confuse that with the separate (and true) fact that ``se_std_denoised >=
    se_std`` whenever ``rho < 1``: that is mechanical, the same ``sqrt(Var(V_k))`` divided
    by a smaller spread. The denoised bar is wider than the raw bar, and simultaneously
    narrower than it should be.

    Args:
        fit_result: a ``core.model.holistic.FitResult``.
        cov: the (P, P) sandwich covariance from ``sandwich_covariance``.
        se_s: (n, K) entrywise standard errors, i.e. ``std_errors(fit, cov)["s"]``.
    """
    s = np.asarray(fit_result.s, dtype=np.float64)
    n, K = s.shape
    cov = np.asarray(cov, dtype=np.float64)
    se_s = np.asarray(se_s, dtype=np.float64)
    if se_s.shape != (n, K):
        raise ValueError(f"se_s must be {(n, K)}, got {se_s.shape}")
    if cov.shape[0] < n * K:
        raise ValueError(f"cov is {cov.shape}, too small for n*K={n * K}")

    var = s.var(axis=0)
    noise = (se_s**2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(var > 0, (var - noise) / var, -np.inf)

    std = np.sqrt(var)
    std_denoised = np.sqrt(np.clip(var - noise, 0.0, None))

    se_std = np.zeros(K)
    se_std_denoised = np.zeros(K)
    for k in range(K):
        idx = np.arange(n) * K + k  # s[:, k] in the flat parameter vector
        sigma_k = cov[np.ix_(idx, idx)]
        col = s[:, k] - s[:, k].mean()  # M s_k
        g = (2.0 / n) * col
        var_Vk = float(g @ sigma_k @ g)
        var_Vk = max(var_Vk, 0.0)
        se_std[k] = np.sqrt(var_Vk) / (2 * std[k]) if std[k] > 0 else np.inf
        se_std_denoised[k] = (
            np.sqrt(var_Vk) / (2 * std_denoised[k]) if std_denoised[k] > 0 else np.inf
        )

    return ScoreSpread(
        var=var,
        noise=noise,
        rho=rho,
        std=std,
        std_denoised=std_denoised,
        se_std=se_std,
        se_std_denoised=se_std_denoised,
    )
