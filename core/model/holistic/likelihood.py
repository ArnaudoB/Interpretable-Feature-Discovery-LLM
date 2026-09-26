"""NLL and analytic gradient for the holistic-verdict model.

Parameter layout (flat vector ``params``):

    include_gamma=True   ->  [s.flatten(order='C'), gamma, beta]   size n*K + K + 1
    include_gamma=False  ->  [s.flatten(order='C'), beta]          size n*K + 1

Model:
    z_bt[t]   = sum_k (s[a_t, k] - s[b_t, k]) + beta
    z_m[t,k]  = eta[t] * (s[a_t, k] - s[b_t, k]) + gamma[k]    eta = 2*w - 1
    NLL       = -sum_t [ w_t log sig(z_bt) + (1-w_t) log sig(-z_bt) ]
                -sum_{t,k} [ r_{t,k} log sig(z_m) + (1-r_{t,k}) log sig(-z_m) ]
                + 0.5 * ridge * ||s||_F^2     (ridge applied to s only)

The predictor is affine in (s, gamma, beta), so the NLL is convex.

All three entry points (``nll``, ``grad``, ``nll_and_grad``) accept an optional
``pin_beta: float | None``. When set, the β slot of ``params`` is overridden by
the pinned value and the corresponding gradient entry is forced to zero. This
allows refitting ``(s, γ)`` while holding the global position bias fixed, e.g.
at the unconstrained MLE β̂.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit


# ---------------------------------------------------------------------------
# Param packing
# ---------------------------------------------------------------------------


def unpack(params: np.ndarray, n: int, K: int, include_gamma: bool):
    """Return (s, gamma_or_None, beta)."""
    nK = n * K
    s = params[:nK].reshape(n, K)
    if include_gamma:
        gamma = params[nK : nK + K]
        beta = float(params[nK + K])
    else:
        gamma = None
        beta = float(params[nK])
    return s, gamma, beta


def pack(s: np.ndarray, gamma, beta: float, include_gamma: bool) -> np.ndarray:
    """Inverse of unpack."""
    parts = [s.ravel(order="C")]
    if include_gamma:
        if gamma is None:
            raise ValueError("include_gamma=True but gamma is None")
        parts.append(np.asarray(gamma, dtype=np.float64).ravel())
    parts.append(np.asarray([beta], dtype=np.float64))
    return np.concatenate(parts)


def param_size(n: int, K: int, include_gamma: bool) -> int:
    return n * K + (K + 1 if include_gamma else 1)


# ---------------------------------------------------------------------------
# NLL and gradient
# ---------------------------------------------------------------------------


def _predictors(s, gamma, beta, w, pairs):
    """Return (z_bt, z_m, eta) given current params and data.

    eta = 2*w - 1, used to fold the winner sign into the mention predictor.
    """
    a = pairs[:, 0]
    b = pairs[:, 1]
    ds = s[a] - s[b]  # (T, K)
    z_bt = ds.sum(axis=1) + beta  # (T,)
    eta = (2.0 * w - 1.0).astype(np.float64)  # (T,)
    if gamma is None:
        z_m = eta[:, None] * ds  # (T, K)
    else:
        z_m = eta[:, None] * ds + gamma[None, :]  # (T, K)
    return z_bt, z_m, eta


def nll(params, w, r, pairs, K, n, ridge=1e-3, include_gamma=True, pin_beta=None):
    s, gamma, beta = unpack(params, n, K, include_gamma)
    if pin_beta is not None:
        beta = float(pin_beta)
    w = np.asarray(w, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)

    z_bt, z_m, _ = _predictors(s, gamma, beta, w, pairs)

    # -[w log sig(z) + (1-w) log sig(-z)]  via log_expit
    nll_bt = -(w * log_expit(z_bt) + (1.0 - w) * log_expit(-z_bt)).sum()
    nll_m = -(r * log_expit(z_m) + (1.0 - r) * log_expit(-z_m)).sum()

    pen = 0.5 * float(ridge) * float((s * s).sum()) if ridge > 0.0 else 0.0
    return float(nll_bt + nll_m + pen)


def grad(params, w, r, pairs, K, n, ridge=1e-3, include_gamma=True, pin_beta=None):
    s, gamma, beta = unpack(params, n, K, include_gamma)
    if pin_beta is not None:
        beta = float(pin_beta)
    w = np.asarray(w, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)

    z_bt, z_m, eta = _predictors(s, gamma, beta, w, pairs)

    # Per-pair residuals.
    u = expit(z_bt) - w  # (T,)
    v = expit(z_m) - r  # (T, K)
    a = pairs[:, 0]
    b = pairs[:, 1]

    g_s = np.zeros_like(s)  # (n, K)

    # BT contribution to grad s[i, k]: same value for every k at row i, equal to
    #     sum_{t : a_t=i} u_t  -  sum_{t : b_t=i} u_t
    row_bt = np.zeros(n, dtype=np.float64)
    np.add.at(row_bt, a, u)
    np.add.at(row_bt, b, -u)
    g_s += row_bt[:, None]  # broadcast across K

    # Mention contribution to grad s[i, k]:
    #     sum_{t : a_t=i} eta_t * v_{t,k}  -  sum_{t : b_t=i} eta_t * v_{t,k}
    eta_v = eta[:, None] * v  # (T, K)
    np.add.at(g_s, a, eta_v)
    np.add.at(g_s, b, -eta_v)

    if ridge > 0.0:
        g_s += ridge * s

    g_beta = float(u.sum())

    if pin_beta is not None:
        g_beta = 0.0
    if include_gamma:
        g_gamma = v.sum(axis=0)  # (K,)
        return np.concatenate([g_s.ravel(order="C"), g_gamma, [g_beta]])
    return np.concatenate([g_s.ravel(order="C"), [g_beta]])


def nll_and_grad(params, w, r, pairs, K, n, ridge=1e-3, include_gamma=True, pin_beta=None):
    """Value + gradient computed in one pass (avoids duplicate expit calls).

    Used by ``fit.fit`` via ``scipy.optimize.minimize(jac=True)``.

    If ``pin_beta`` is not None, the unpacked β is overridden by that value and
    the β slot of the gradient is zeroed. The optimizer in ``fit.fit`` masks the
    β coordinate out of its L-BFGS-B vector when pinning, but the zero-gradient
    is the right belt-and-braces here in case a caller drives this directly.
    """
    s, gamma, beta = unpack(params, n, K, include_gamma)
    if pin_beta is not None:
        beta = float(pin_beta)
    w_arr = np.asarray(w, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    pairs_arr = np.asarray(pairs, dtype=np.int64)

    z_bt, z_m, eta = _predictors(s, gamma, beta, w_arr, pairs_arr)

    # Value.
    nll_bt = -(w_arr * log_expit(z_bt) + (1.0 - w_arr) * log_expit(-z_bt)).sum()
    nll_m = -(r_arr * log_expit(z_m) + (1.0 - r_arr) * log_expit(-z_m)).sum()
    pen = 0.5 * float(ridge) * float((s * s).sum()) if ridge > 0.0 else 0.0
    val = float(nll_bt + nll_m + pen)

    # Gradient.
    u = expit(z_bt) - w_arr
    v = expit(z_m) - r_arr
    a = pairs_arr[:, 0]
    b = pairs_arr[:, 1]

    g_s = np.zeros_like(s)
    row_bt = np.zeros(n, dtype=np.float64)
    np.add.at(row_bt, a, u)
    np.add.at(row_bt, b, -u)
    g_s += row_bt[:, None]

    eta_v = eta[:, None] * v
    np.add.at(g_s, a, eta_v)
    np.add.at(g_s, b, -eta_v)

    if ridge > 0.0:
        g_s += ridge * s

    g_beta = float(u.sum())

    if pin_beta is not None:
        g_beta = 0.0
    if include_gamma:
        g_gamma = v.sum(axis=0)
        g = np.concatenate([g_s.ravel(order="C"), g_gamma, [g_beta]])
    else:
        g = np.concatenate([g_s.ravel(order="C"), [g_beta]])
    return val, g
