"""fig:recovery-heatmap-comparative -- comparative criteria vs engineered dials.

    python figures/make_recovery_heatmap_comparative.py

Pearson r between each KEPT comparative criterion score and each engineered dial.
Reads results/scores.parquet, which is the refit on the criteria surviving the
paper operating point kappa <= config.model.kappa_max (10) and
rho > config.model.rho_threshold (0.75) -- K = 21. recovery_common asserts that
the fit on disk really was produced at those thresholds, so a figure can never
silently be drawn from a fit at some other operating point.

Columns use the canonical dial order shared with the pointwise heatmaps, so the
panels can be read side by side; rows are then ordered for maximum diagonality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_common as rc  # noqa: E402
from paper_style import DIV_CMAP, FULL_W, apply_style, save  # noqa: E402

OUT = Path(__file__).resolve().parent  # figures land next to their scripts


def main() -> int:
    dials, G = rc.ground_truth()
    dials = rc.canonical_dial_order(dials, G)  # shared across all panels
    r = rc.corr(rc.comparative_scores(), G, dials)
    r = rc.order_rows_for_diagonal(r)  # rows follow the fixed columns

    apply_style()
    # ~0.135 in per criterion row keeps 8pt labels legible without crowding.
    h = 0.90 + 0.135 * len(r)
    fig, ax = plt.subplots(figsize=(FULL_W, min(h, 4.5)))
    im = ax.imshow(
        r.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    )
    ax.set_xticks(range(len(dials)))
    ax.set_xticklabels([rc.DIAL_LABEL[d] for d in dials], rotation=45, ha="right")
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([c[:1].upper() + c[1:] for c in r.index])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson $r$")
    cb.outline.set_linewidth(0.6)

    save(fig, OUT, "recovery_heatmap_comparative")
    print(f"[comparative] K={len(r)} criteria x {len(dials)} dials (canonical order)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
