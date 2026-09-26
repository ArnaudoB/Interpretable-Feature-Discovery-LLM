"""Per-observation gate probabilities of the holistic verdict model.

    p_win[t]     = sig( sum_k (s[a_t, k] - s[b_t, k]) + beta )
    p_cite[t, k] = sig( eta_t * (s[a_t, k] - s[b_t, k]) + gamma_k ),   eta_t = 2 w_t - 1

The citation probability conditions on the observed winner w_t, exactly as the
likelihood does, so these are the probabilities whose log-loss is the fitted NLL.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from core.model.holistic.likelihood import _predictors


def gate_probabilities(s, gamma, beta, w, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_win (T,), p_cite (T, K))`` for the given parameters and pairs."""
    z_bt, z_m, _ = _predictors(
        np.asarray(s, dtype=np.float64),
        None if gamma is None else np.asarray(gamma, dtype=np.float64),
        float(beta),
        np.asarray(w, dtype=np.float64),
        np.asarray(pairs, dtype=np.int64),
    )
    return expit(z_bt), expit(z_m)
