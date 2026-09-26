"""Produces Figure `fig:paired` (figures/paired_{a,b}.pdf): the two panels of tab:paired.

    python figures/make_paired_panels.py

One horizontal dot-and-whisker (forest) plot per panel: the mean paired difference in
each recovery metric with its 95% percentile CI, over B=50 shared bootstrap draws.
Ours - Direct LLM (panel a) and Ours - Local LLMs (panel b, pooled over fusion and corpus
draws). Those are the arms coded 1a and 2a in the scripts.

Metric labels come from core.metrics.DISPLAY_NAME, so the panels, tab:main and tab:paired
cannot drift apart. The keys stay SF/SM/SR/AF/LK, which is what the shipped artifacts are
keyed on.

Values are read from results/tables/tab_paired.json, which scripts/10_paper_tables.py writes
from the bootstrap artifacts alongside tab:paired, rounded as the table prints them. Stars:

  --stars paper  (default) the star set of the figure as included in the paper
                 (``significant`` plus ``PAPER_STARS``).
  --stars ci     stars exactly the CIs that exclude zero (the ``significant`` flag written by
                 10_paper_tables.py).

Direction differs by metric: Tracked/Stat.R./Sem.R./Attrib. are higher-better (right favours
Ours), LK is lower-better (left favours Ours) -- marked with up/down arrows on the labels. dK
is on a different scale and carries no significance, so it is annotated as text, not plotted.

Writes paired_a.pdf and paired_b.pdf next to this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from paper_style import BLUE, GREY, HALF_W, apply_style, save  # noqa: E402
from core.metrics import DISPLAY_NAME  # noqa: E402

OUT = Path(__file__).resolve().parent  # figures land next to their scripts

RESULTS = OUT.parent / "results" / "tables"
TITLES = {"a": r"(a) Ours $-$ Direct LLM", "b": r"(b) Ours $-$ Local LLMs"}
#: (panel, metric) pairs additionally starred under ``--stars paper``
PAPER_STARS = {("a", "SR"), ("a", "LK"), ("b", "SF")}


def panels(stars: str) -> dict:
    """metric rows (key, mean, lo, hi, win%, significant, higher_is_better) per panel."""
    import json

    d = json.loads((RESULTS / "tab_paired.json").read_text())
    out = {}
    for key, spec in d.items():
        rows = [
            tuple(r[:5]) + ((r[5] or (stars == "paper" and (key, r[0]) in PAPER_STARS)), r[6])
            for r in spec["rows"]
        ]
        out[key] = {"title": TITLES[key], "rows": rows, "dK": tuple(spec["dK"])}
    return out


def _draw(key: str, spec: dict, xlim: tuple[float, float]) -> None:
    rows = spec["rows"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(HALF_W, 1.95))

    ys = list(range(n - 1, -1, -1))  # first row at the top
    ax.xaxis.grid(True, alpha=0.25, lw=0.5, zorder=0)  # light x-grid behind bars
    ax.set_axisbelow(True)
    ax.axvline(0.0, color=GREY, lw=0.8, zorder=1)  # baseline: no difference

    ylabels = []
    for y, (name, mean, lo, hi, win, sig, hib) in zip(ys, rows):
        color = BLUE if sig else GREY
        ax.barh(
            y,
            mean,
            height=0.62,
            color=color,
            alpha=(1.0 if sig else 0.45),
            edgecolor=color,
            lw=0.6,
            zorder=2,
        )
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            zorder=3,
        )
        # win-rate at the right margin
        ax.annotate(
            f"{win}%",
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-1, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
            color="#444444",
        )
        arrow = r"$\uparrow$" if hib else r"$\downarrow$"
        star = r"$^{\star}$" if sig else ""
        ylabels.append(f"{DISPLAY_NAME[name]}{star} {arrow}")

    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Mean paired difference (95% CI)")

    # Left-align the metric labels so every acronym starts at the same x regardless of
    # whether a significance star trails it -- this keeps the SF/SM/SR/... column aligned
    # both within a panel and across panels a and b (mathtext has no \phantom to pad with).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    max_w_pt = max(t.get_window_extent(rend).width for t in ax.get_yticklabels()) * 72.0 / fig.dpi
    ax.tick_params(axis="y", pad=max_w_pt + 3.0)
    for lbl in ax.get_yticklabels():
        lbl.set_ha("left")

    ax.set_xlim(*xlim)  # shared across both panels so bars, zero line and grid line up

    # panel identifier (each panel is its own PDF) + descriptive dK, both low-key
    ax.text(0.0, 1.02, spec["title"], transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
    ax.text(
        0.985,
        1.02,
        "Ours better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    dk = spec["dK"]
    ax.text(
        0.985,
        0.03,
        rf"$\Delta K = {dk[0]:.1f}$ [{dk[1]:.1f}, {dk[2]:.1f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )

    save(fig, OUT, f"paired_{key}")
    plt.close(fig)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stars", choices=["paper", "ci"], default="paper")
    PANELS = panels(ap.parse_args().stars)
    apply_style()
    # one shared x-range over both panels: headroom on the right for the win-rate column,
    # a touch on the left for the LK whisker.
    allrows = [r for spec in PANELS.values() for r in spec["rows"]]
    xlim = (min(r[2] for r in allrows) - 0.03, max(r[3] for r in allrows) + 0.12)
    for key, spec in PANELS.items():
        _draw(key, spec, xlim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
