"""Produces Figures `fig:calibration_reliability_custom` and `fig:calibration_skill_custom`
(App. sec:synthetic-calibration; figures/calibration_{reliability,skill}.pdf).

    python figures/make_calibration_figures.py --config config.yaml

Reads only results/calibration/{bins.json,metrics.csv} -- step 12 does the fitting, this does
the drawing -- and writes calibration_reliability.{pdf,png} and calibration_skill.{pdf,png} next
to this file, printing for each the values its caption quotes.

Layout and resolution rc settings are applied on top of paper_style.apply_style(): both figures place their
own legend with tight_layout(rect=...), which constrained layout would override. Font and colour
settings still come from paper_style.

Both figures are 5.44 / 5.38 in wide after tight cropping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import BLUE, FULL_W, GREY, RED, apply_style  # noqa: E402

GATES = ("citation", "direction")
GATE_COLOR = {"citation": BLUE, "direction": RED}
REFS = [("bss_chance", "vs chance\n(p = 1/2)"), ("bss_emp", "vs empirical\nrate")]


def _fmt(r, k, d=3):
    return f"{r[k]:.{d}f} [{r[k + '_lo']:.{d}f}, {r[k + '_hi']:.{d}f}]"


def _save(fig, out: Path, name: str, caption: str):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    print(f"[saved] {name}.pdf/.png\n  caption values: {' '.join(caption.split())}")


def reliability(bins, t, K, out):
    fig, axes = plt.subplots(
        2, 2, figsize=(FULL_W, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for c, gate in enumerate(GATES):
        ax, axn, col = axes[0, c], axes[1, c], GATE_COLOR[gate]
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--", zorder=1)
        ins = bins[f"in_sample/{gate}"]
        ok = ins["n_per_bin"] > 0
        ax.plot(ins["p_mean"][ok], ins["y_mean"][ok], color=col, lw=0.8, zorder=2)
        b = bins[f"oof/{gate}"]
        ok = b["n_per_bin"] > 0
        yerr = np.vstack([b["y_mean"][ok] - b["y_lo"][ok], b["y_hi"][ok] - b["y_mean"][ok]])
        ax.errorbar(
            b["p_mean"][ok],
            b["y_mean"][ok],
            yerr=yerr,
            fmt="o-",
            ms=3.2,
            lw=0.9,
            color=col,
            mec="white",
            mew=0.5,
            elinewidth=0.8,
            capsize=1.5,
            zorder=3,
        )
        o = t.loc[(gate, "oof")]
        ax.text(
            0.03,
            0.97,
            f"OOF ECE {_fmt(o, 'ece')}\nin-sample ECE {t.loc[(gate, 'in_sample'), 'ece']:.3f}\n"
            f"n = {int(o['n']):,}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.8,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f"{gate} gate", fontsize=9)
        axn.bar(
            b["bin_centers"],
            np.maximum(b["n_per_bin"], 0.8),
            width=0.09,
            color=col,
            edgecolor="white",
            linewidth=0.8,
        )
        axn.set_yscale("log")
        axn.set_ylim(0.8, 1e5)
        axn.set_yticks([1e1, 1e3])
        axn.set_xlabel("predicted probability")
    axes[0, 0].set_ylabel("observed rate")
    axes[1, 0].set_ylabel("n cells")
    fig.legend(
        handles=[
            Line2D([], [], color="k", marker="o", ms=3, label="5-fold OOF (95% bootstrap CI)"),
            Line2D([], [], color="k", lw=0.8, label="in-sample"),
            Line2D([], [], color=GREY, lw=0.6, ls="--", label="perfect calibration"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    oc, od = t.loc[("citation", "oof")], t.loc[("direction", "oof")]
    _save(
        fig,
        out,
        "calibration_reliability",
        "Out-of-fold reliability of the comparative feature-discovery model's two gates "
        f"(5-fold over comparisons, K={K} criteria). Left: citation gate, scored on all "
        f"{int(oc['n']):,} (comparison, criterion) cells; right: direction gate, scored on "
        f"the {int(od['n']):,} cited cells. Points are the observed rate per equal-width bin "
        "of predicted probability with 95% comparison-bootstrap CIs; the thin line is the "
        "in-sample curve and the dashed diagonal is perfect calibration. OOF ECE: citation "
        f"{_fmt(oc, 'ece')}, direction {_fmt(od, 'ece')}. Bottom: number of cells per bin "
        "(log scale).",
    )
    return oc, od


def skill(t, K, out, oc, od):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5))
    for c, gate in enumerate(GATES):
        ax, col = axes[c], GATE_COLOR[gate]
        o = t.loc[(gate, "oof")]
        for x, (key, _) in enumerate(REFS):
            v = float(o[key])
            ax.bar(x, v, width=0.55, color=col, edgecolor="white", linewidth=0.8)
            ax.errorbar(
                x,
                v,
                yerr=[[v - o[f"{key}_lo"]], [o[f"{key}_hi"] - v]],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
            )
            ax.annotate(
                f"{v:.3f}\n[{o[key + '_lo']:.3f}, {o[key + '_hi']:.3f}]",
                (x, max(v, 0)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.3,
            )
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_xticks(range(len(REFS)))
        ax.set_xticklabels([lab for _, lab in REFS])
        lo = min(0.0, *(float(o[f"{k}_lo"]) for k, _ in REFS))
        hi = max(*(float(o[f"{k}_hi"]) for k, _ in REFS))
        ax.set_ylim(lo - 0.05 * abs(hi - lo), hi + 0.28 * abs(hi - lo))
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"{gate} gate", fontsize=9)
    axes[0].set_ylabel("Brier skill score")
    # No legend: one series per panel, its colour is the gate the title already names, and a
    # neutral swatch reads as a key to a bar that is not drawn (or to the zero line, same grey).
    fig.tight_layout()
    ic, idr = t.loc[("citation", "in_sample")], t.loc[("direction", "in_sample")]
    _save(
        fig,
        out,
        "calibration_skill",
        "Out-of-fold Brier skill score, BSS = 1 - Brier(model)/Brier(baseline), of the "
        f"comparative feature-discovery model's citation and direction gates (5-fold over "
        f"comparisons, K={K}). Baselines: chance (p = 1/2, Brier 0.25) and the empirical rate "
        "(per-criterion training-fold citation rate, i.e. the intercept-only model; "
        "per-criterion training-fold rate of favouring the first-shown letter among cited "
        "cells). BSS > 0 means the model beats the baseline. Bars: 5-fold out-of-fold "
        "estimates with 95% comparison-bootstrap CIs. Citation: vs chance "
        f"{_fmt(oc, 'bss_chance')}, vs empirical {_fmt(oc, 'bss_emp')}; direction: vs chance "
        f"{_fmt(od, 'bss_chance')}, vs empirical {_fmt(od, 'bss_emp')}. In-sample values are "
        f"not shown and are higher throughout: citation vs chance {ic['bss_chance']:.3f}, vs "
        f"empirical {ic['bss_emp']:.3f}; direction vs chance {idr['bss_chance']:.3f}, vs "
        f"empirical {idr['bss_emp']:.3f}. The in-sample-to-out-of-fold gap is the model's "
        "optimism, and it is far larger for the citation gate.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default=None, help="default: next to this script")
    args = ap.parse_args()
    root = Path(args.config).parent
    cal = root / "results" / "calibration"
    out = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    apply_style()
    # See the module docstring for why these are set here and not in paper_style.
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    raw = json.loads((cal / "bins.json").read_text())
    bins = {k: {f: np.asarray(v) for f, v in b.items()} for k, b in raw.items()}
    t = pd.read_csv(cal / "metrics.csv").set_index(["gate", "regime"])
    K = int(json.loads((cal / "summary.json").read_text())["K"])

    oc, od = reliability(bins, t, K, out)
    skill(t, K, out, oc, od)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
