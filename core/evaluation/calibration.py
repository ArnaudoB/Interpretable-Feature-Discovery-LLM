"""Calibration metrics (ECE, Brier, Brier skill) with a pair-level bootstrap.

A row of ``p`` / ``y`` is one comparison. Its K citation outcomes share the comparison's
essays and winner, so rows -- not (row, criterion) cells -- are the resampling unit.

Equal-width ECE is written as ``sum_b |sum_{i in b} p_i - sum_{i in b} y_i| / N``, so every
statistic is a function of per-row sums: a bootstrap replicate is one multinomial-weighted
sum over rows, which keeps thousands of replicates cheap.

An optional ``mask`` restricts every statistic to the cells where it is 1 -- e.g. a gate that
is only defined on some cells, such as a direction gate conditional on citation. Rows stay the
resampling unit, so a comparison's masked-in cells move together and the effective N varies
by replicate. With ``mask=None`` the arithmetic is bit-identical to the unmasked form.
"""

from __future__ import annotations

import warnings

import numpy as np

STAT_KEYS = ("ece", "brier", "brier_emp", "bss_chance", "bss_emp")


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)


def pair_bootstrap_calibration(
    p,
    y,
    p_ref,
    *,
    mask=None,
    n_bins: int = 10,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    chunk: int = 100,
) -> dict:
    """Point estimates and percentile CIs for calibration of ``p`` against ``y``.

    ``p``, ``y`` and ``p_ref`` are (T,) or (T, K); ``p_ref`` is the reference forecast for
    the empirical-rate skill score (chance is the fixed p = 1/2, Brier 0.25). ``mask`` (same
    shape as ``p``, 0/1) excludes cells from every statistic; ``None`` keeps all cells.

    Returns ``{stat, stat_lo, stat_hi}`` for ``STAT_KEYS`` plus ``bins``: per-bin
    ``bin_centers, n_per_bin, p_mean, y_mean, y_lo, y_hi`` (CIs on the observed rate).
    """
    p = np.asarray(p, dtype=np.float64)
    T = p.shape[0]
    p = p.reshape(T, -1)
    y = np.asarray(y, dtype=np.float64).reshape(T, -1)
    p_ref = np.broadcast_to(np.asarray(p_ref, dtype=np.float64).reshape(T, -1), p.shape)
    if mask is None:
        m = np.ones_like(p)
    else:
        m = np.asarray(mask, dtype=np.float64).reshape(T, -1)
        if m.shape != p.shape:
            raise ValueError(f"mask shape {m.shape} != p shape {p.shape}")

    cell = (np.arange(T)[:, None] * n_bins + _bin_index(p, n_bins)).ravel()

    def per_row(v):
        return np.bincount(cell, weights=v.ravel(), minlength=T * n_bins).reshape(T, n_bins)

    rows = np.hstack(
        [
            per_row(m),
            per_row(p * m),
            per_row(y * m),
            (m * (p - y) ** 2).sum(axis=1, keepdims=True),
            (m * (p_ref - y) ** 2).sum(axis=1, keepdims=True),
        ]
    )

    def stats(agg):
        cnt, sp, sy = (
            agg[..., :n_bins],
            agg[..., n_bins : 2 * n_bins],
            agg[..., 2 * n_bins : 3 * n_bins],
        )
        lm, lr = agg[..., -2], agg[..., -1]
        N = cnt.sum(axis=-1)  # == p.size when unmasked; varies by replicate when masked
        with np.errstate(invalid="ignore", divide="ignore"):
            return {
                "ece": np.abs(sp - sy).sum(axis=-1) / N,
                "brier": lm / N,
                "brier_emp": lr / N,
                "bss_chance": 1.0 - (lm / N) / 0.25,
                "bss_emp": 1.0 - lm / lr,
                "p_mean": sp / cnt,
                "y_mean": sy / cnt,
            }

    point = stats(rows.sum(axis=0))

    rng = np.random.default_rng(seed)
    reps = [
        rng.multinomial(T, np.full(T, 1.0 / T), size=min(chunk, n_boot - start)) @ rows
        for start in range(0, n_boot, chunk)
    ]
    boot = stats(np.vstack(reps))

    alpha = 100.0 * (1.0 - level) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty bins in some replicates
        q = {k: np.nanpercentile(v, [alpha, 100.0 - alpha], axis=0) for k, v in boot.items()}

    out = {}
    for k in STAT_KEYS:
        out[k], out[f"{k}_lo"], out[f"{k}_hi"] = float(point[k]), float(q[k][0]), float(q[k][1])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out["bins"] = {
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
        "n_per_bin": rows[:, :n_bins].sum(axis=0).astype(np.int64),
        "p_mean": point["p_mean"],
        "y_mean": point["y_mean"],
        "y_lo": q["y_mean"][0],
        "y_hi": q["y_mean"][1],
    }
    return out


def compute_ece(p, y, n_bins: int = 10) -> float:
    """Equal-width ECE on [0, 1] with ``n_bins`` bins, as an explicit loop over bins.

    An independent restatement of the ``ece`` that :func:`pair_bootstrap_calibration`
    computes from per-row sums; the drivers assert the two agree to 1e-10.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(p) == 0:
        return 0.0
    bin_idx = _bin_index(p, n_bins)
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        ece += (nb / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)
