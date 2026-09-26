"""Fitted total-score distributions per judge, with agreement against human scores.

Produces Figure ``fig:total_score_distribution`` (App. ``app:verdict-extension``).

For each judge the model's essay ranking is the row sum of the fitted score matrix,
``sum_k S_ik`` (the paper's notation) -- the quantity the winner channel compares, since
``P(w) = sigma(sum_k Delta_k + beta)``. This figure shows how that total is distributed
over the 100 essays for each judge, on both corpora, annotated with its rank agreement
against the human holistic score.

Agreement is Spearman's ``r_s`` (written r_s, NOT rho, because rho_k is already the
per-criterion signal share in this work). Rank correlation is the right measure: the total
is on an arbitrary logit scale, so only the induced ordering is comparable to a human
1-5 (ELLIPSE) or 1-6 (ASAP) holistic score.

Usage
    python figures/make_total_score_distributions.py

Output
    figures/total_score_distributions.{pdf,png}
    results/total_score_agreement.csv   the r_s printed on the figure, per (corpus, judge);
                                        read by scripts/09_paper_numbers.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from interp_common import COLORS, CORPORA, EXPERIMENT, ORDER, OUT, essays_in_fit_order, load_fit
from paper_style import FULL_W, GREY, apply_style, save

JUDGE_OF = dict(
    zip(ORDER, ["gpt_5_4_mini", "claude_haiku_4_5", "gemini_3_6_flash", "deepseek_v4_flash"])
)
PANEL_TITLE = {"ellipse": "ELLIPSE", "asap2": "ASAP"}


def load(judge: str, corpus: str) -> tuple[np.ndarray, float]:
    """(total score per essay, Spearman r_s against the human holistic score)."""
    fit = load_fit(JUDGE_OF[judge], corpus)
    # essay_index.json fixes the row order of s; the essays frame must be aligned to it.
    essays = essays_in_fit_order(JUDGE_OF[judge], corpus)
    total = fit.s.sum(axis=1)
    return total, float(spearmanr(total, essays["overall"].to_numpy()).statistic)


def main() -> int:
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.7), sharey=True)

    rows = []
    for ax, corpus in zip(axs, CORPORA):
        name = PANEL_TITLE[corpus]
        for row, judge in enumerate(ORDER):
            total, rs = load(judge, corpus)
            rows.append(
                {
                    "corpus": corpus,
                    "judge": judge,
                    "spearman_rs": rs,
                    "total_sd": float(total.std(ddof=1)),
                    "n_essays": int(len(total)),
                }
            )
            y = len(ORDER) - 1 - row
            parts = ax.violinplot(
                [total],
                positions=[y],
                vert=False,
                widths=0.82,
                showextrema=False,
                showmedians=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(COLORS[judge])
                body.set_alpha(0.32)
                body.set_edgecolor(COLORS[judge])
                body.set_linewidth(0.8)
            q1, med, q3 = np.percentile(total, [25, 50, 75])
            ax.plot([q1, q3], [y, y], color=COLORS[judge], lw=2.4, solid_capstyle="butt", zorder=3)
            ax.plot([med], [y], marker="|", ms=7, mew=1.4, color="white", zorder=4)
            ax.text(
                0.985,
                y + 0.30,
                f"$r_s$={rs:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7,
                color=COLORS[judge],
            )
            print(f"{name:8s} {judge:18s} sd={total.std(ddof=1):5.2f}  r_s={rs:.3f}")

        ax.axvline(0, color=GREY, lw=0.6, ls=":", zorder=1)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(list(reversed(ORDER)), fontsize=7.5)
        # Capital S and no "T_i =" prefix: matches the paper's notation.
        ax.set_xlabel(r"fitted total score  $\sum_k S_{ik}$")
        ax.set_title(name, fontsize=9, pad=6)
        ax.set_ylim(-0.65, len(ORDER) - 0.25)
        ax.margins(x=0.04)
        ax.grid(axis="x", color=GREY, lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)

    save(fig, OUT, "total_score_distributions")
    out = EXPERIMENT / "results" / "total_score_agreement.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
