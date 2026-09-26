"""Produces Figure `fig:cmv-diff-dr` (figures/cmv_diff_grid.pdf): matched-k accuracy difference
of the four D_r comparisons, 2 x 2.

    python scripts/10b_matched_k.py          # writes the four comparison curves
    python figures/make_cmv_diff_grid.py

Rows: comparator feature set (D, S); columns: D_r scoring (PW, BT). S has no BT arm, so the
bottom-right panel sets D_r + BT against S + PW. Each panel is the subset-averaged held-out
accuracy of step 9's paired L1-LR (PW sets on the mean of two scoring runs), reference minus
comparator, with its pointwise 95% interval: joint bootstrap over the 800 held-out pairs and the
drawn subsets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT / "figures"))
from paper_style import BLUE, FULL_W, PURPLE, RED, apply_style, save  # noqa: E402

SRC = EXPERIMENT / "results" / "matched_k"
OUT = EXPERIMENT / "figures"
AMBER = "#C4861C"
COLOUR = {
    "D + PW": BLUE,
    "D + BT": BLUE,
    "Dr + PW": AMBER,
    "Dr + BT": AMBER,
    "P + PW": RED,
    "P + BT": RED,
    "S + PW": PURPLE,
}
LABEL = {"Dr + PW": r"D$_{\mathrm{r}}$ + PW", "Dr + BT": r"D$_{\mathrm{r}}$ + BT"}


def slug(arm: str) -> str:
    return arm.replace(" + ", "").upper()


def lab(arm: str) -> str:
    return LABEL.get(arm, arm)


#: Rows = comparator feature set (D, S), columns = D_r scoring (PW, BT). S has no BT arm,
#: so the bottom-right panel sets D_r + BT against S + PW.
GRID = [
    [("Dr + PW", "D + PW"), ("Dr + BT", "D + BT")],
    [("Dr + PW", "S + PW"), ("Dr + BT", "S + PW")],
]


def _curve(ref, other):
    c = pd.read_parquet(SRC / f"{slug(ref)}_vs_{slug(other)}" / "matched_k_curve.parquet")
    return c.sort_values("k")


def grid() -> None:
    curves = [[_curve(r, o) for r, o in row] for row in GRID]
    allc = pd.concat([c for row in curves for c in row])
    lo = np.floor(100 * min(allc.lo.min(), 0) - 0.5)
    hi = np.ceil(100 * max(allc.hi.max(), 0) + 0.5)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.9), sharex=True, sharey=True)
    for i, row in enumerate(GRID):
        for j, (ref, other) in enumerate(row):
            ax, c = axes[i, j], curves[i][j]
            ax.fill_between(c.k, 100 * c.lo, 100 * c.hi, color=BLUE, alpha=0.25, lw=0, zorder=2)
            ax.plot(c.k, 100 * c.delta, color=BLUE, lw=1.4, zorder=3)
            ax.axhline(0, color="0.3", lw=0.6, zorder=1)
            ax.set_title(
                f"({'abcd'[2 * i + j]}) {lab(ref)} $-$ {lab(other)}",
                fontsize=7.5,
                loc="left",
                pad=3,
            )
            kmax = int(c.k.max())
            ax.set_xlim(1, kmax)
            ax.set_xticks(sorted({1, *range(4, kmax + 1, 4), kmax}))
            if i == 1:
                ax.set_xlabel("Number of features")
            if j == 0:
                ax.set_ylabel("Accuracy difference (points)")
            print(
                f"({'abcd'[2 * i + j]}) {ref} - {other}: k=1 {100 * c.delta.iloc[0]:+.2f}  "
                f"k={kmax} {100 * c.delta.iloc[-1]:+.2f}  "
                f"CI excludes 0 at k={c.loc[(c.lo > 0) | (c.hi < 0), 'k'].tolist()}"
            )
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 0].set_yticks(np.arange(lo + (lo % 2), hi + 1, 2))
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=1.0)
    save(fig, OUT, "cmv_diff_grid")


def main() -> int:
    apply_style()
    grid()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
