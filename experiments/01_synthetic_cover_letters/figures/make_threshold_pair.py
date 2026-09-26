"""Produces Figure `fig:threshold_pair` (figures/threshold_pair.pdf; Remark rem:why-taxo):
the acceptance-threshold bind, in one figure.

    python figures/make_threshold_pair.py

Left  : P(dial surfaced in a single elicitation) per dial, over the acceptance
        threshold. Low threshold -> every dial "recovered".
Right : features surviving fusion vs pooling depth E, one curve per threshold on
        the SAME grid. Low threshold -> everything collapses into a handful of
        features; high threshold -> the rubric never converges.

Both panels are computed by sweep_common. The heatmap carries no per-cell numbers:
at half width there is no room for them at the 7pt floor, and the colorbar carries
the value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sw  # noqa: E402
from paper_style import FULL_W, GREY, SEQ_CMAP, apply_style, save  # noqa: E402

OUT = Path(__file__).resolve().parent  # figures land next to their scripts
E_MARK = 27


def main() -> int:
    # LEFT: fuse the E=27 elicitations at the x-axis threshold, THEN Hungarian-match
    # the dials to the fused set at the FIXED calibrated theta. Same knob as the
    # right panel -- both panels sweep the fusion acceptance threshold.
    THETA = 0.406
    E_MARK_ = 27
    P, dials = sw.dial_recovery_after_fusion(E=E_MARK_, theta=THETA)
    order = np.argsort(P.mean(axis=1))
    P = P[order]
    labels = [sw.DIAL_LABEL[dials[k]] for k in order]
    Es, mean, X = sw.pooling_curves()
    thresholds = sw.THRESHOLDS

    apply_style()
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(FULL_W, 3.05), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    # ---------------- left: dial surfacing over the threshold ----------------
    im = axL.imshow(P, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    show = [0, 2, 4, 6, 8, 10]  # 0.30, 0.40, ... 0.80
    axL.set_xticks(show)
    axL.set_xticklabels([f"{thresholds[j]:.2f}" for j in show])
    axL.set_yticks(range(len(labels)))
    axL.set_yticklabels(labels)
    axL.set_xlabel("fusion acceptance threshold")
    axL.tick_params(length=0)
    for sp in axL.spines.values():
        sp.set_visible(False)
    cbL = fig.colorbar(im, ax=axL, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
    cbL.set_label("P(dial recovered)", fontsize=8)
    cbL.ax.tick_params(labelsize=8)
    cbL.outline.set_linewidth(0.6)

    # ---------------- right: rubric size vs pooling depth --------------------
    norm = Normalize(vmin=min(thresholds), vmax=max(thresholds))
    for t in thresholds:
        axR.plot(Es, mean[t], color=SEQ_CMAP(0.18 + 0.82 * norm(t)), zorder=3)
    axR.axvline(E_MARK, ls="--", color=GREY, lw=0.9, zorder=1)
    axR.annotate(
        f"$E={E_MARK}$",
        xy=(E_MARK, axR.get_ylim()[1]),
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=GREY,
    )
    axR.set_xlabel("$E$ elicitations pooled")
    axR.set_ylabel("features after fusion")
    axR.set_xlim(1, X)
    axR.set_ylim(0, None)
    axR.yaxis.grid(True, alpha=0.25)
    axR.set_axisbelow(True)
    cbR = fig.colorbar(
        ScalarMappable(norm=norm, cmap=SEQ_CMAP),
        ax=axR,
        fraction=0.035,
        pad=0.02,
        ticks=[0.3, 0.5, 0.8],
    )
    cbR.set_label("fusion acceptance threshold", fontsize=8)
    cbR.ax.tick_params(labelsize=8)
    cbR.outline.set_linewidth(0.6)

    save(fig, OUT, "threshold_pair")
    print(
        f"[pair] left: fuse E={E_MARK_} then match at theta={THETA}; "
        f"{P.shape[0]} dials x {len(thresholds)} thresholds; right: E=1..{X}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
