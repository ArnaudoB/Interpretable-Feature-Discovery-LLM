"""Synthetic data generator for the holistic-verdict model.

Draws ground-truth (s, gamma, beta), samples T unordered pairs uniformly from
C(n, 2), randomizes A/B presentation, samples winner labels and per-criterion
mention bits from the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from core.graph._unionfind import is_connected


@dataclass
class SyntheticData:
    w: np.ndarray  # (T,) int, in {0, 1}
    r: np.ndarray  # (T, K) int, in {0, 1}
    pairs: np.ndarray  # (T, 2) int — (a_t, b_t)
    true_s: np.ndarray  # (n, K) float — column-centered
    true_gamma: np.ndarray  # (K,) float
    true_beta: float


def _sample_pairs(n: int, T: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample T unordered pairs uniformly from C(n, 2). Without replacement
    when T <= C(n,2); with replacement above. Returns (T, 2) array with a_t
    and b_t in random A/B order."""
    n_pairs_total = n * (n - 1) // 2
    if n_pairs_total == 0:
        raise ValueError(f"n={n} has no valid pairs")
    replace = T > n_pairs_total
    flat = rng.choice(n_pairs_total, size=T, replace=replace)
    # Convert each flat index to (i, j) with i < j. Use the triangular bijection.
    # i is the row in a strict upper triangle: i < j; iterate via inversion.
    i = np.zeros(T, dtype=np.int64)
    j = np.zeros(T, dtype=np.int64)
    # Offset by row: row k has (n-1-k) entries. Cumulative offset = k*(2n-k-1)/2.
    cum = np.array([k * (2 * n - k - 1) // 2 for k in range(n)], dtype=np.int64)
    for t in range(T):
        f = int(flat[t])
        # Find largest k with cum[k] <= f.
        row = int(np.searchsorted(cum, f, side="right") - 1)
        col_offset = f - cum[row]
        i[t] = row
        j[t] = row + 1 + col_offset

    # Random A/B presentation order.
    flip = rng.rand(T) < 0.5
    a = np.where(flip, i, j)
    b = np.where(flip, j, i)
    return np.stack([a, b], axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    n: int,
    K: int,
    T: int,
    true_s: np.ndarray | None = None,
    true_gamma: np.ndarray | None = None,
    true_beta: float = 0.0,
    pair_sampling: str = "uniform",
    ensure_connected: bool = True,
    seed: int = 42,
) -> SyntheticData:
    """Generate synthetic data from the holistic-verdict model.

    If ``true_s`` or ``true_gamma`` is None, draws them from standard normals.
    ``true_s`` is always column-centered to mean zero so that the post-hoc
    identifiability convention matches.
    """
    if pair_sampling != "uniform":
        raise NotImplementedError(f"pair_sampling={pair_sampling!r} not supported")

    rng = np.random.RandomState(seed)

    if true_s is None:
        s = rng.normal(size=(n, K)).astype(np.float64)
    else:
        s = np.asarray(true_s, dtype=np.float64).copy()
    s -= s.mean(axis=0, keepdims=True)

    if true_gamma is None:
        gamma = rng.normal(size=K).astype(np.float64)
    else:
        gamma = np.asarray(true_gamma, dtype=np.float64).copy()

    beta = float(true_beta)

    # Sample pairs (with connectivity retry).
    seed_inc = 0
    while True:
        pair_rng = np.random.RandomState(seed + seed_inc)
        pairs = _sample_pairs(n, T, pair_rng)
        if not ensure_connected or is_connected(pairs, n):
            break
        seed_inc += 1
        if seed_inc > 50:
            raise RuntimeError("could not produce a connected comparison graph in 50 tries")

    # Sample winner labels.
    a = pairs[:, 0]
    b = pairs[:, 1]
    ds = s[a] - s[b]  # (T, K)
    z_bt = ds.sum(axis=1) + beta  # (T,)
    p_bt = expit(z_bt)
    label_rng = np.random.RandomState(seed + 1)
    w = (label_rng.rand(T) < p_bt).astype(np.int64)

    # Sample mention bits, conditional on winner.
    eta = 2.0 * w - 1.0
    z_m = eta[:, None] * ds + gamma[None, :]  # (T, K)
    p_m = expit(z_m)
    mention_rng = np.random.RandomState(seed + 2)
    r = (mention_rng.rand(T, K) < p_m).astype(np.int64)

    return SyntheticData(
        w=w,
        r=r,
        pairs=pairs,
        true_s=s,
        true_gamma=gamma,
        true_beta=beta,
    )
