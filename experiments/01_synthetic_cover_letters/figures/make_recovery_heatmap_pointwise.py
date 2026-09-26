"""fig:recovery-heatmap-m1 -- the Direct LLM baseline's features vs engineered dials.

    python figures/make_recovery_heatmap_pointwise.py

Companion to make_recovery_heatmap_comparative.py: same dials, same seriation
rule, same diverging colormap, so the two heatmaps can be read side by side.

Arms are loaded EXACTLY as scripts/09_build_tables_figures.py::_load_arms builds
them for tab:main, so the figure shows the arm the table reports as Direct LLM (code M1):
one rep of pointwise/scores_long + rubric_stability/full_dataset_rubric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_common as rc  # noqa: E402
from paper_style import DIV_CMAP, FULL_W, apply_style, save  # noqa: E402

EXP = Path(__file__).resolve().parent.parent  # the experiment directory
OUT = Path(__file__).resolve().parent  # figures land next to their scripts


def load_arm(res: Path):
    """Mirror of 09_build_tables_figures.py::_load_arms for the Direct LLM (M1) arm."""
    full = json.loads((res / "rubric_stability/full_dataset_rubric.json").read_text())
    text = {f["name"]: f"{f['name']}. {f['description']}" for f in full["features"]}
    long = pd.read_parquet(res / "pointwise/scores_long.parquet")
    F = (
        long[long["rep"] == long["rep"].min()]
        .groupby(["letter_id", "dimension"])["score"]
        .mean()
        .unstack()
    )
    return F, {c: text[c] for c in F.columns if c in text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    dials, G = rc.ground_truth()
    # SAME dial order as the comparative panel, so the two line up column-for-column.
    dials_o = rc.canonical_dial_order(dials, G)

    F, ctext = load_arm(EXP / "results")
    F = F[[c for c in F.columns if c in ctext]].astype(float)
    r = rc.corr(F, G, dials_o)
    # rows ordered for maximum diagonality given those fixed columns
    r = rc.order_rows_for_diagonal(r)

    apply_style()
    h = 0.90 + 0.135 * len(r)
    fig, ax = plt.subplots(figsize=(FULL_W, min(h, 4.5)))
    im = ax.imshow(
        r.values, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    )
    ax.set_xticks(range(len(dials_o)))
    ax.set_xticklabels([rc.DIAL_LABEL[d] for d in dials_o], rotation=45, ha="right")
    ax.set_yticks(range(len(r)))
    # The elicited feature names are lowercase; sentence-case the first character
    # only, so internal capitals ("Python fluency", "AI deployment") survive.
    ax.set_yticklabels([c[:1].upper() + c[1:] for c in r.index])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson $r$")
    cb.outline.set_linewidth(0.6)

    name = "recovery_heatmap_m1"
    save(fig, OUT, name)
    peak = np.abs(r.values).max(axis=1)
    print(
        f"[M1] K={len(r)} features x {len(dials_o)} dials; "
        f"{(peak > 0.4).sum()} features with a |r|>0.4 dial, "
        f"{(peak <= 0.4).sum()} tracking nothing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
