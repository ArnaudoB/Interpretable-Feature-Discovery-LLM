"""Tests for core.model.holistic — the holistic-verdict model of app:verdict-extension.

Verifies analytic gradient and Hessian against finite differences, convexity
of the NLL, synthetic recovery, the no-gamma reduction, the convex sanity-check
diagnostic, sign-correlation positivity, and (slow) sandwich-SE calibration.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.model.holistic import (
    FitResult,
    fit,
    generate,
    grad,
    hessian,
    nll,
    sandwich_covariance,
    std_errors,
)
from core.model.holistic.likelihood import param_size


# ---------------------------------------------------------------------------
# Gradient / Hessian / convexity
# ---------------------------------------------------------------------------


def test_gradient_vs_finite_differences():
    """Analytic gradient matches central finite differences on a small problem."""
    data = generate(n=10, K=3, T=50, seed=0)
    rng = np.random.RandomState(1)
    params = rng.normal(size=param_size(10, 3, True)) * 0.1

    g_analytic = grad(params, data.w, data.r, data.pairs, K=3, n=10)

    eps = 1e-6
    g_fd = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        g_fd[i] = (
            nll(p_plus, data.w, data.r, data.pairs, 3, 10)
            - nll(p_minus, data.w, data.r, data.pairs, 3, 10)
        ) / (2 * eps)

    np.testing.assert_allclose(g_analytic, g_fd, atol=1e-5, rtol=1e-4)


def test_hessian_vs_finite_differences():
    """Analytic Hessian matches central finite differences of the gradient."""
    data = generate(n=10, K=3, T=50, seed=2)
    rng = np.random.RandomState(3)
    P = param_size(10, 3, True)
    params = rng.normal(size=P) * 0.1

    H_an = hessian(params, data.w, data.r, data.pairs, K=3, n=10)

    eps = 1e-6
    H_fd = np.zeros((P, P))
    for j in range(P):
        p_plus = params.copy()
        p_plus[j] += eps
        p_minus = params.copy()
        p_minus[j] -= eps
        H_fd[:, j] = (
            grad(p_plus, data.w, data.r, data.pairs, K=3, n=10)
            - grad(p_minus, data.w, data.r, data.pairs, K=3, n=10)
        ) / (2 * eps)

    np.testing.assert_allclose(H_an, H_fd, atol=1e-5, rtol=1e-4)
    # Symmetry sanity (analytic Hessian should be exactly symmetric).
    np.testing.assert_allclose(H_an, H_an.T, atol=1e-12)


def test_convexity_inequality():
    """NLL is convex: L(0.5 p1 + 0.5 p2) <= 0.5 (L(p1) + L(p2))."""
    data = generate(n=15, K=4, T=100, seed=4)
    rng = np.random.RandomState(5)
    P = param_size(15, 4, True)
    p1 = rng.normal(size=P) * 0.5
    p2 = rng.normal(size=P) * 0.5
    p_mid = 0.5 * (p1 + p2)

    L1 = nll(p1, data.w, data.r, data.pairs, 4, 15)
    L2 = nll(p2, data.w, data.r, data.pairs, 4, 15)
    Lmid = nll(p_mid, data.w, data.r, data.pairs, 4, 15)

    assert Lmid <= 0.5 * (L1 + L2) + 1e-8


# ---------------------------------------------------------------------------
# Recovery on synthetic data
# ---------------------------------------------------------------------------


def test_synthetic_recovery():
    """Fit recovers ground truth on a moderately-sized synthetic problem.

    Recovery thresholds set from per-entry MLE noise scale at T=2000, n=50, K=5
    (per-entry SE ~ 1/sqrt(8) ≈ 0.35 on the s block; max over 250 entries ~ 1.0,
    RMS ~ 0.3). β and γ are tighter because they're shared across all pairs.
    """
    data = generate(n=50, K=5, T=2000, seed=4)
    result = fit(data.w, data.r, data.pairs, K=5, n=50)

    # β recovery: identified globally, tight at T=2000.
    assert abs(result.beta - data.true_beta) < 0.15

    # γ recovery: per-criterion intercept, identified from ~T mention obs each.
    assert np.max(np.abs(result.gamma - data.true_gamma)) < 0.3

    # s recovery: both estimates are column-centered; compare directly.
    s_diff = result.s - data.true_s
    s_diff -= s_diff.mean(axis=0, keepdims=True)
    # Max-abs threshold reflects per-entry sampling noise (see docstring).
    assert np.max(np.abs(s_diff)) < 1.0
    # Tighter RMS check captures recovery quality without max-over-outliers.
    assert np.sqrt((s_diff**2).mean()) < 0.40


def test_no_gamma_reduction():
    """With include_gamma=False on data generated with true_gamma=0, fits match."""
    K = 3
    n = 30
    T = 1000
    data_no_gamma = generate(
        n=n,
        K=K,
        T=T,
        true_gamma=np.zeros(K),
        seed=5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fit_with = fit(
            data_no_gamma.w,
            data_no_gamma.r,
            data_no_gamma.pairs,
            K=K,
            n=n,
            include_gamma=True,
        )
        fit_without = fit(
            data_no_gamma.w,
            data_no_gamma.r,
            data_no_gamma.pairs,
            K=K,
            n=n,
            include_gamma=False,
        )

    # γ̂ near zero in the with-γ fit.
    assert np.max(np.abs(fit_with.gamma)) < 0.2
    # s estimates close between the two fits.
    assert np.max(np.abs(fit_with.s - fit_without.s)) < 0.1
    # β estimates close.
    assert abs(fit_with.beta - fit_without.beta) < 0.05


# ---------------------------------------------------------------------------
# Diagnostic fields
# ---------------------------------------------------------------------------


def test_inf_diff_is_small():
    """Convex sanity check: second L-BFGS run from a perturbed init gives same loss."""
    data = generate(n=20, K=4, T=500, seed=6)
    result = fit(data.w, data.r, data.pairs, K=4, n=20)
    assert result.inf_diff < 1e-3, f"Convex sanity check failed: inf_diff = {result.inf_diff:.3e}"


def test_sign_corrs_positive():
    """Empirical sign correlations are positive on well-behaved synthetic data."""
    data = generate(n=30, K=4, T=1500, seed=7)
    result = fit(data.w, data.r, data.pairs, K=4, n=30)
    assert (result.sign_corrs > 0).all(), f"Negative sign correlations: {result.sign_corrs}"


def test_fit_result_dataclass():
    """Spot-check that FitResult fields are populated and consistent."""
    data = generate(n=12, K=3, T=200, seed=8)
    result = fit(data.w, data.r, data.pairs, K=3, n=12)
    assert isinstance(result, FitResult)
    assert result.n == 12 and result.K == 3 and result.T == 200
    assert result.s.shape == (12, 3)
    assert result.gamma.shape == (3,)
    assert result.params_flat.shape == (12 * 3 + 3 + 1,)
    assert result.converged
    assert 0.0 < result.sigmoid_beta < 1.0
    # Column-centering invariant.
    np.testing.assert_allclose(result.s.mean(axis=0), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Sandwich CI calibration — slow.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sandwich_ci_calibration():
    """Predicted SE matches empirical SD across replications (within 20%)."""
    n, K, T = 20, 3, 500
    rng = np.random.RandomState(8)
    true_s = rng.normal(size=(n, K))
    true_s -= true_s.mean(axis=0)
    true_gamma = rng.normal(size=K)
    true_beta = 0.5

    beta_estimates = []
    beta_ses = []
    for rep in range(40):
        data = generate(
            n=n,
            K=K,
            T=T,
            true_s=true_s,
            true_gamma=true_gamma,
            true_beta=true_beta,
            seed=100 + rep,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = fit(data.w, data.r, data.pairs, K=K, n=n)
        cov = sandwich_covariance(result, data.w, data.r, data.pairs)
        ses = std_errors(result, cov)
        beta_estimates.append(result.beta)
        beta_ses.append(ses["beta"])

    empirical_sd = float(np.std(beta_estimates, ddof=1))
    mean_predicted_se = float(np.mean(beta_ses))
    ratio = mean_predicted_se / empirical_sd
    assert 0.8 < ratio < 1.2, (
        f"Sandwich SE / empirical SD = {ratio:.3f}, out of [0.8, 1.2] "
        f"(predicted={mean_predicted_se:.4f}, empirical={empirical_sd:.4f})"
    )
