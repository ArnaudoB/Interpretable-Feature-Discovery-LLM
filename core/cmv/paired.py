"""Shared-resample accuracy intervals for arms scored on the same heldout pairs (CMV).

Every arm of tab:cmv-predictive is evaluated on the same 800 heldout pairs, so one bootstrap
index matrix shared by all arms makes their intervals comparable: the same resample is applied to
every arm. Sampling unit is the heldout pair. Intervals are conditional on the fitted models and
the elicited scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def boot_indices(n: int, n_boot: int = 4000, seed: int = 0) -> np.ndarray:
    """One ``(n_boot, n)`` resample matrix, to be shared by every arm.

    Identical draws to :func:`core.cmv.designs.boot_ci` (``default_rng(0)``, 4000 x 800), so an
    arm's interval is the same whether it is computed alone or alongside the others.
    """
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def marginal_cis(C: pd.DataFrame, idx: np.ndarray, pct=(2.5, 97.5)) -> pd.DataFrame:
    """Per-arm accuracy and percentile CI from the shared indices. ``C``: pairs x arms in [0, 1]."""
    B = C.to_numpy(float)[idx].mean(axis=1)  # (n_boot, arms)
    lo, hi = np.percentile(B, pct, axis=0)
    return pd.DataFrame(
        {
            "arm": C.columns,
            "acc": C.mean().to_numpy(),
            "lo": lo,
            "hi": hi,
            "n_correct": C.sum().to_numpy(),
        }
    )


def average_runs(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """Score-level mean of two pointwise scoring runs; a cell missing in one run takes the other's.

    Run B is reindexed to run A's columns, so a run B that scored a superset of criteria (the
    29-criterion repeat, for the 16-criterion D_r panel) is restricted to A's.
    """
    B = B.reindex(index=A.index.union(B.index), columns=A.columns)
    A = A.reindex(B.index)
    return pd.DataFrame(
        np.nanmean(np.stack([A.to_numpy(float), B.to_numpy(float)]), axis=0),
        index=A.index,
        columns=A.columns,
    )
