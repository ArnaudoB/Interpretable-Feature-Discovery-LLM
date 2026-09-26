"""NOISE-CORRECTED dimensionality vs number of criteria, seven arms.

    python scripts/10_dimensionality.py
    python figures/make_cmv_dimension_curve.py

Produces Figure `fig:cmv-dimension-curve-full` (figures/cmv_dimension_curve.pdf, appendix).

Two panels, participation ratio and entropy effective rank, both on the Spearman correlations with
the cross-measurement correction for measurement noise (BT: disjoint halves of the comparisons
refitted separately; PW: a second scoring run with the criteria reshuffled), averaged over
k-subsets so no panel is credited for having more columns, with 95% percentile-bootstrap bands over
the 1,600 test exchanges. The PR panel is the left panel of `fig:cmv-dimension-curve`.

  colour  blue = discovered (all 29), amber = D_r (16 kept by the kappa/rho gate),
          red = prior-elicited (16), purple = sample-informed (42)
  dash    solid = BT, dashed = PW
  shade   darker each time a panel runs out of criteria: from k=16 (D_r and P end), from k=29
          (D ends); dashed verticals mark those ends. Read comparisons at k <= 16.
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

RES = EXPERIMENT / "results"
OUT = EXPERIMENT / "figures"

AMBER = "#C4861C"  # D_r
PW_DASH = (0, (4, 1.6))
#: arm -> (colour, dash, label)
ARMS = {
    "D + BT": (BLUE, "-", "D + BT"),
    "D + PW": (BLUE, PW_DASH, "D + PW"),
    "Dr + BT": (AMBER, "-", r"D$_{\mathrm{r}}$ + BT"),
    "Dr + PW": (AMBER, PW_DASH, r"D$_{\mathrm{r}}$ + PW"),
    "P + BT": (RED, "-", "P + BT"),
    "P + PW": (RED, PW_DASH, "P + PW"),
    "S + PW": (PURPLE, PW_DASH, "S + PW"),
}
PANELS = [("pr", "Participation ratio", 2), ("effrank", "Effective rank", 4)]
K_READ = 16


def main() -> int:
    apply_style()
    df = pd.read_parquet(RES / "dimensionality" / "curves.parquet")
    missing = [a for a in ARMS if df[df.arm == a].empty]
    assert not missing, f"no noise-corrected curve for {missing}; run the dimensionality script"
    kmax = int(df.k.max())
    ends = sorted({int(df[df.arm == a].k.max()) for a in ARMS})

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for ax, (metric, ylab, step) in zip(axes, PANELS):
        dm = df[df.metric == metric]
        for arm, (colour, dash, label) in ARMS.items():
            d = dm[dm.arm == arm].sort_values("k")
            if d.ci_lo.notna().all():
                ax.fill_between(d.k, d.ci_lo, d.ci_hi, color=colour, alpha=0.13, lw=0, zorder=2)
            ax.plot(d.k, d.value, color=colour, ls=dash, lw=1.3, label=label, zorder=3)
            print(
                f"[{metric:7} {arm:>7}] k={int(d.k.max()):>2} {d.value.iloc[-1]:5.2f} "
                f"[{d.ci_lo.iloc[-1]:.2f}, {d.ci_hi.iloc[-1]:.2f}]   "
                f"k={K_READ} {d[d.k == K_READ].value.iloc[0]:5.2f}"
            )
        ax.plot(
            range(1, kmax + 1), range(1, kmax + 1), color="0.78", lw=0.6, ls=(0, (1, 2.5)), zorder=1
        )  # y = k (all criteria independent), above the shade
        # shade darkens each time a panel runs out of criteria
        edges = [b for b in ends if K_READ <= b < kmax] or [K_READ]
        for i, (lo, hi) in enumerate(zip(edges, edges[1:] + [kmax])):
            ax.axvspan(lo, hi, color=str(0.955 - 0.045 * i), zorder=0, lw=0)
            ax.axvline(lo, color="0.6", lw=0.5, ls=(0, (2, 2)), zorder=1)
        top = float(np.ceil(max(dm.value.max(), dm.ci_hi.max()) / step) * step) + step
        ax.set_ylim(0.5, top)
        ax.set_yticks(np.arange(step, top + 1, step))
        ax.set_xlim(1, kmax)
        ax.set_xticks(sorted({1, K_READ, *ends}))
        ax.set_xlabel("Number of features")
        ax.set_ylabel(ylab)
    axes[0].legend(
        frameon=False,
        loc="upper left",
        fontsize=6,
        handlelength=1.9,
        labelspacing=0.22,
        borderpad=0.15,
        borderaxespad=0.3,
    )
    fig.tight_layout(pad=0.25, w_pad=1.0)
    save(fig, OUT, "cmv_dimension_curve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
