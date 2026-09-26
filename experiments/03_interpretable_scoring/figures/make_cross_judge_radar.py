"""Two-panel radar: denoised per-criterion score spread sigma_k, four judges, ELLIPSE vs ASAP.

Produces Figure ``fig:cross_judge_radar`` (App. ``app:verdict-extension``). Axes are cross-judge constructs formed
by MEANING (GPT-5.4 high reasoning, step 7), one criterion per judge at most. The plotted
quantity is the noise-corrected spread ``sigma_k * sqrt(rho_k)`` on each judge's
RELIABLE criteria, with 95% delta-method intervals. A judge with no criterion in a cluster
sits at the radial floor with an open marker.

Usage
    python figures/make_cross_judge_radar.py

Output
    figures/cross_judge_denoised_sigma.{pdf,png}
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from interp_common import COLORS, CORPORA, ORDER, OUT, draw_radar, radar_axes
from paper_style import FULL_W, GREY, apply_style, save

#: Polar axes are circular, so at FULL_W two panels are height-limited to ~2.9in
#: circles that already overlap. Only constructs shared by >= MIN_JUDGES judges
#: fit legibly side by side; the per-corpus radars (make_radar_panels.py) show every construct.
MIN_JUDGES = 3

PANEL_TITLE = {"ellipse": "ELLIPSE", "asap2": "ASAP"}


def main() -> int:
    apply_style()
    fig_h = 3.25
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, fig_h), subplot_kw={"projection": "polar"})
    # Explicit layout for two reasons: constrained layout reserves no room for a figure
    # legend (it was clipped), and a polar axes draws a CIRCLE inscribed in its box -- so
    # a box taller than it is wide wastes the difference as dead space above and below.
    # Square boxes placed by hand remove that band entirely.
    fig.set_layout_engine("none")
    # Sized so the circle plus its radiating tick labels clears the figure edge, the
    # facing panel, and the legend band: labels overhang the circle by roughly 0.5in
    # sideways and 0.33in vertically.
    side = 1.75  # box edge, inches
    bw, bh = side / FULL_W, side / fig_h
    bottom = 0.78 / fig_h  # legend band + label overhang below
    for ax, cx in zip(axs, (0.26, 0.74)):  # panel centres, figure fractions
        ax.set_position([cx - bw / 2, bottom, bw, bh])
    for ax, corpus in zip(axs, CORPORA):
        name = PANEL_TITLE[corpus]
        axes, total = radar_axes(corpus, min_judges=MIN_JUDGES)
        draw_radar(ax, axes, ORDER, COLORS, panel_label=name, grey=GREY, wrap=14, label_size=7.5)
        print(
            f"{name}: {len(axes)} axes drawn of {total} clusters "
            f"({total - len(axes)} with < {MIN_JUDGES} judges omitted)"
        )

    handles, labels = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles = handles + [Line2D([], [], color=GREY, lw=0, marker="o", mfc="none", ms=4)]
    labels = labels + ["criterion absent for that judge"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
        title=r"denoised $\sigma_k$",
        title_fontsize=8.5,
        bbox_to_anchor=(0.5, -0.015),
    )
    save(fig, OUT, "cross_judge_denoised_sigma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
