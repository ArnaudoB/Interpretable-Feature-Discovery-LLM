"""The three calibration figures of App. ``sec:extension-appendix``, drawn from step 8's results.

Produces Figures ``fig:calibration_reliability_ellipse``, ``fig:calibration_reliability_asap``
and ``fig:calibration_skill``. No refitting here: every number comes from
``results/calibration/`` (``scripts/08_calibration.py``).

Usage
    python figures/make_calibration_figures.py

Output
    figures/calibration_reliability_4judge_llm.{pdf,png}        ELLIPSE
    figures/calibration_reliability_asap_4judge_llm.{pdf,png}   ASAP 2.0
    figures/calibration_brier_skill_4judge_llm.{pdf,png}        both corpora
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from interp_common import COLORS, CORPORA, CORPUS_LABEL, EXPERIMENT, ORDER, OUT
from paper_style import FULL_W, GREY, apply_style, save

RES = EXPERIMENT / "results" / "calibration"
SHORT = {
    "gpt-5.4-mini": "GPT",
    "claude-haiku-4.5": "Haiku",
    "gemini-3.6-flash": "Gemini",
    "deepseek-v4-flash": "DeepSeek",
}
TAG = {"ELLIPSE": "_4judge_llm", "ASAP-2": "_asap_4judge_llm"}


GATES = ("winner", "citation")


def _fmt(row, k: str) -> str:
    return f"{row[k]:.3f} [{row[k + '_lo']:.3f}, {row[k + '_hi']:.3f}]"


def reliability_figure(bins_by_judge: dict, table: pd.DataFrame, name: str, dataset: str) -> None:
    judges = list(bins_by_judge)
    t = table.set_index(["judge", "gate", "regime"])
    fig, axes = plt.subplots(
        2 * len(judges),
        len(GATES),
        figsize=(FULL_W, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * len(judges)},
        squeeze=False,
    )
    for r_, j in enumerate(judges):
        col = COLORS[j]
        for c, gate in enumerate(GATES):
            ax, axn = axes[2 * r_, c], axes[2 * r_ + 1, c]
            ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--", zorder=1)
            ins = bins_by_judge[j]["in_sample"][gate]
            ok = ins["n_per_bin"] > 0
            ax.plot(ins["p_mean"][ok], ins["y_mean"][ok], color=col, lw=0.8, alpha=0.45, zorder=2)
            b = bins_by_judge[j]["oof"][gate]
            ok = b["n_per_bin"] > 0
            yerr = np.vstack([b["y_mean"][ok] - b["y_lo"][ok], b["y_hi"][ok] - b["y_mean"][ok]])
            ax.errorbar(
                b["p_mean"][ok],
                b["y_mean"][ok],
                yerr=yerr,
                fmt="o",
                ms=3,
                color=col,
                mec="white",
                mew=0.5,
                elinewidth=0.7,
                capsize=0,
                zorder=3,
            )
            ax.text(
                0.03,
                0.97,
                f"OOF ECE {_fmt(t.loc[(j, gate, 'oof')], 'ece')}\n"
                f"in-sample ECE {t.loc[(j, gate, 'in_sample'), 'ece']:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            axn.bar(b["bin_centers"], b["n_per_bin"], width=0.09, color=col, alpha=0.6, linewidth=0)
            axn.set_yscale("log")
            axn.set_ylim(1, 3e4)
            axn.set_yticks([1e1, 1e3])
            if r_ == 0:
                ax.set_title(f"{gate} gate", fontsize=9)
            if c > 0:
                ax.set_yticklabels([])
                axn.set_yticklabels([])
        axes[2 * r_, 0].set_ylabel(f"{j}\nobserved rate", fontsize=7.5)
        axes[2 * r_ + 1, 0].set_ylabel("n")
    for c in range(len(GATES)):
        axes[-1, c].set_xlabel("predicted probability")
    handles = [
        Line2D(
            [], [], color="k", marker="o", ls="none", ms=3, label="5-fold OOF (95% bootstrap CI)"
        ),
        Line2D([], [], color="k", lw=0.8, alpha=0.45, label="in-sample"),
        Line2D([], [], color=GREY, lw=0.6, ls="--", label="perfect calibration"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    save(fig, OUT, name)
    plt.close(fig)


def brier_skill_figure(table: pd.DataFrame, name: str) -> None:
    judges, datasets = list(ORDER), [CORPUS_LABEL[c] for c in CORPORA]
    refs = (("bss_chance", "vs chance (p = 1/2)"), ("bss_emp", "vs empirical rate"))
    width = 0.19
    fig, axes = plt.subplots(len(refs), 2, figsize=(FULL_W, 3.9), sharex=True, squeeze=False)
    for r_, (ref, ref_label) in enumerate(refs):
        for c, gate in enumerate(GATES):
            ax = axes[r_, c]
            ax.axhline(0, color=GREY, lw=0.6)
            for d, ds in enumerate(datasets):
                sel = table[(table.dataset == ds) & (table.gate == gate)].set_index(
                    ["judge", "regime"]
                )
                for i, j in enumerate(judges):
                    x = d + (i - 1.5) * width
                    o = sel.loc[(j, "oof")]
                    v = float(o[ref])
                    ax.bar(x, v, width=width, color=COLORS[j], edgecolor="white", linewidth=0.8)
                    ax.errorbar(
                        x,
                        v,
                        yerr=[[v - o[f"{ref}_lo"]], [o[f"{ref}_hi"] - v]],
                        fmt="none",
                        ecolor="#333333",
                        elinewidth=0.7,
                        capsize=1.5,
                        capthick=0.7,
                    )
                    ax.plot(
                        x,
                        float(sel.loc[(j, "in_sample"), ref]),
                        marker="o",
                        ms=3.2,
                        mfc="white",
                        mec=COLORS[j],
                        mew=0.9,
                        ls="none",
                    )
            ax.set_xticks(
                [d + (i - 1.5) * width for d in range(len(datasets)) for i in range(len(judges))]
            )
            ax.set_xticklabels(
                [SHORT[j] for _ in datasets for j in judges], rotation=90, fontsize=6
            )
            if r_ == len(refs) - 1:
                for d, ds in enumerate(datasets):
                    ax.text(
                        d,
                        -0.42,
                        ds,
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=8,
                    )
            if r_ == 0:
                ax.set_title(f"{gate} gate", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{ref_label}\nBrier skill score")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[j], label=j) for j in judges]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            mfc="white",
            mec="k",
            ls="none",
            ms=3.2,
            label="in-sample; bars = 5-fold OOF ± 95% CI",
        )
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    save(fig, OUT, name)
    plt.close(fig)


def load_results():
    """Metrics table + reliability bins written by ``scripts/08_calibration.py``."""
    if not (RES / "metrics.csv").exists():
        raise FileNotFoundError(f"{RES} -- run scripts/08_calibration.py first")
    raw = json.loads((RES / "bins.json").read_text())
    bins = {
        ds: {
            j: {
                rg: {
                    g: {k: np.asarray(v, dtype=float) for k, v in b.items()}
                    for g, b in by_gate.items()
                }
                for rg, by_gate in by_regime.items()
            }
            for j, by_regime in by_judge.items()
        }
        for ds, by_judge in raw.items()
    }
    return pd.read_csv(RES / "metrics.csv"), bins


def main() -> int:
    apply_style()
    table, bins = load_results()
    for ds, tag in TAG.items():
        fname = f"calibration_reliability{tag}"
        reliability_figure(bins[ds], table[table.dataset == ds], fname, ds)
    brier_skill_figure(table, "calibration_brier_skill_4judge_llm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
