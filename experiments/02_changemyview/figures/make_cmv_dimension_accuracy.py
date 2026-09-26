"""What the panels RESOLVE vs what they PREDICT: noise-corrected participation ratio (left) and
heldout accuracy (right) against the number of criteria, seven arms, one shared legend.

    python scripts/10_dimensionality.py     # left panel's input
    python scripts/10b_matched_k.py         # right panel's input
    python figures/make_cmv_dimension_accuracy.py

Produces Figure `fig:cmv-dimension-curve` (figures/cmv_dimension_accuracy.pdf, main text).

Left: PR of the Spearman correlation matrix with the cross-measurement noise correction, averaged
over k-subsets, 95% bootstrap bands over the 1,600 test exchanges; dotted y = k. Right: paired
L1-LR heldout accuracy averaged over k-subsets (PW panels: two scoring runs averaged), 95% joint
bootstrap over the 800 heldout pairs and the drawn subsets; dotted rule = #words (59.25%).
Both: shade darkens each time a panel runs out of criteria (k=16: D_r and P end; k=29: D ends).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

EXPERIMENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT / "figures"))
from paper_style import BLUE, FULL_W, PURPLE, RED, apply_style, save  # noqa: E402

RES = EXPERIMENT / "results"
OUT = EXPERIMENT / "figures"

AMBER = "#C4861C"  # D_r
PW_DASH = (0, (4, 1.6))
ARMS = {
    "D + BT": (BLUE, "-", "D + BT"),
    "D + PW": (BLUE, PW_DASH, "D + PW"),
    "Dr + BT": (AMBER, "-", r"D$_{\mathrm{r}}$ + BT"),
    "Dr + PW": (AMBER, PW_DASH, r"D$_{\mathrm{r}}$ + PW"),
    "P + BT": (RED, "-", "P + BT"),
    "P + PW": (RED, PW_DASH, "P + PW"),
    "S + PW": (PURPLE, PW_DASH, "S + PW"),
}
WORDS = 100 * float(
    pd.read_csv(RES / "accuracy.csv").set_index("arm").loc["#words", "acc"]
)  # 59.25
K_READ = 16


def _shade(ax, ends, kmax):
    """Darker each time a panel runs out of criteria; dashed verticals at those ends."""
    edges = [b for b in ends if K_READ <= b < kmax] or [K_READ]
    for i, (lo, hi) in enumerate(zip(edges, edges[1:] + [kmax])):
        ax.axvspan(lo, hi, color=str(0.955 - 0.045 * i), zorder=0, lw=0)
        ax.axvline(lo, color="0.6", lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(1, kmax)
    ax.set_xticks(sorted({1, K_READ, *ends}))
    ax.set_xlabel("Number of features")


def main() -> int:
    apply_style()
    pr = pd.read_parquet(RES / "dimensionality" / "curves.parquet").query("metric == 'pr'")
    acc = pd.read_parquet(RES / "matched_k" / "accuracy_curves.parquet")
    for name, d in (("PR", pr), ("accuracy", acc)):
        missing = [a for a in ARMS if d[d.arm == a].empty]
        assert not missing, f"no {name} curve for {missing}"
    ends = sorted({int(pr[pr.arm == a].k.max()) for a in ARMS})
    assert ends == sorted({int(acc[acc.arm == a].k.max()) for a in ARMS}), "k ranges differ"
    kmax = max(ends)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.45))
    for arm, (colour, dash, label) in ARMS.items():
        d = pr[pr.arm == arm].sort_values("k")
        ax1.fill_between(d.k, d.ci_lo, d.ci_hi, color=colour, alpha=0.13, lw=0, zorder=2)
        ax1.plot(d.k, d.value, color=colour, ls=dash, lw=1.3, label=label, zorder=3)
        d = acc[acc.arm == arm].sort_values("k")
        ax2.fill_between(d.k, 100 * d.lo, 100 * d.hi, color=colour, alpha=0.06, lw=0, zorder=2)
        ax2.plot(d.k, 100 * d.acc, color=colour, ls=dash, lw=1.3, zorder=3)

    # left: participation ratio
    ax1.plot(
        range(1, kmax + 1), range(1, kmax + 1), color="0.78", lw=0.6, ls=(0, (1, 2.5)), zorder=1
    )  # y = k: all criteria independent
    _shade(ax1, ends, kmax)
    top = float(np.ceil(max(pr.value.max(), pr.ci_hi.max()) / 2) * 2) + 2
    ax1.set_ylim(0.5, top)
    ax1.set_yticks(np.arange(2, top + 1, 2))
    ax1.set_ylabel("Participation ratio")

    # right: accuracy
    ax2.axhline(WORDS, color="0.6", lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax2.text(28.5, WORDS + 0.3, "#words", ha="right", va="bottom", fontsize=5.5, color="0.45")
    _shade(ax2, ends, kmax)
    ax2.set_ylim(np.floor(min(100 * acc.lo.min(), WORDS) - 0.5), np.ceil(100 * acc.hi.max() + 0.5))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.set_ylabel("Held-out accuracy (%)")

    fig.legend(
        *ax1.get_legend_handles_labels(),
        loc="upper center",
        ncol=len(ARMS),
        frameon=False,
        fontsize=6.5,
        handlelength=2.0,
        columnspacing=1.1,
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.25, w_pad=1.2, rect=(0, 0, 1, 0.92))
    save(fig, OUT, "cmv_dimension_accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
