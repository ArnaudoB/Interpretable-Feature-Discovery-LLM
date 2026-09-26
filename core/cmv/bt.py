"""Plain per-feature Bradley-Terry (with positional bias) + frozen-anchor scoring.

For each feature we fit an independent Bradley-Terry model with a position term:

    Pr(shown_first wins) = sigmoid( s[first] - s[other] + beta )

using the repo's convex winner-only fit (`core.model.gated.fit_model`, ``include_gamma=False``, ``K=1``).
`fit_feature_bt` returns the per-item scores and beta for one feature.

`score_query_frozen` scores a NEW (test) item on one feature from its comparisons to a FIXED
(anchor) panel: with ``s_anchor`` and ``beta`` frozen, the query's scalar score is a 1-D convex
problem (the plain-BT analog of the gated model's frozen-score solve).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def fit_feature_bt(pairs, w, n, *, ridge=1e-3, tol=1e-6, max_iter=2000):
    """One feature's BT with position bias (convex, scipy L-BFGS).

    ``pairs`` (T,2) int = ``[first_idx, other_idx]``; ``w`` (T,) = 1 iff the first item won.
    Winner logit ``z = s[first] - s[other] + beta``. Returns ``(s, beta)`` with ``s`` shape (n,),
    projected mean-zero for identifiability (the position bias is carried by ``beta``).
    """
    pairs = np.asarray(pairs, dtype=int)
    w = np.asarray(w, dtype=float)
    a, b = pairs[:, 0], pairs[:, 1]

    def nll(x):
        s, beta = x[:n], x[n]
        z = s[a] - s[b] + beta
        return float(-(w * log_expit(z) + (1 - w) * log_expit(-z)).sum() + 0.5 * ridge * (s @ s))

    def grad(x):
        s, beta = x[:n], x[n]
        d = expit(s[a] - s[b] + beta) - w  # (T,)
        gs = np.zeros(n)
        np.add.at(gs, a, d)
        np.add.at(gs, b, -d)
        gs += ridge * s
        return np.concatenate([gs, [d.sum()]])

    res = minimize(
        nll,
        np.zeros(n + 1),
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol},
    )
    s = res.x[:n]
    return s - s.mean(), float(res.x[n])


def score_query_frozen(sign, s_anchor, w, beta, *, ridge=1e-3):
    """Frozen-anchor score of one query item on one feature (scalar, convex 1-D).

    Each comparison t is query-vs-anchor. ``sign[t]`` = +1 if the query was shown first (slot A),
    else −1; ``s_anchor[t]`` = the anchor's frozen score on this feature; ``w[t]`` = 1 iff the
    shown-first item won. The winner logit is ``z = sign*s_q + (beta - sign*s_anchor)``.
    """
    sign = np.asarray(sign, dtype=float)
    s_anchor = np.asarray(s_anchor, dtype=float)
    w = np.asarray(w, dtype=float)
    c = beta - sign * s_anchor

    def nll(x):
        z = sign * x[0] + c
        return float(-(w * log_expit(z) + (1 - w) * log_expit(-z)).sum() + 0.5 * ridge * x[0] ** 2)

    def grad(x):
        z = sign * x[0] + c
        return np.array([-(sign * (w - expit(z))).sum() + ridge * x[0]])

    res = minimize(nll, x0=[0.0], jac=grad, method="BFGS")
    return float(res.x[0])
