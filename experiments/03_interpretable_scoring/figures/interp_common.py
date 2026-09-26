"""Shared loading and drawing for the App. ``app:verdict-extension`` figures.

Every figure reads the *reliable* refit of each run (``results/fit_reliable``, step 6), the
refit on the criteria that passed the (rho, kappa) screen; the radars additionally read the
cross-judge constructs of step 7 (``results/cross_judge/<corpus>_clusters.json``).

  * :func:`criteria_frame` — one row per reliable criterion of a run: definition, gamma,
    raw and denoised spread sigma_k with delta-method SEs, rho, citation share.
  * :func:`radar_axes` / :func:`draw_radar` — the cross-judge radar of denoised sigma_k,
    used by the two-panel paper radar (``make_cross_judge_radar.py``).
  * :func:`essays_in_fit_order` — the sampled essays aligned to the rows of ``s``.
"""

from __future__ import annotations

import json
import pickle
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
OUT = Path(__file__).resolve().parent  # figures land next to their scripts

# ``fit.pkl`` holds a ``core.model.holistic.FitResult``, so unpickling needs the repository
# root importable, and the run layout lives in ``scripts/_config.py``.
for _p in (str(ROOT), str(EXPERIMENT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _config import (  # noqa: E402
    CORPORA,
    CORPUS_LABEL,
    JUDGE_LABEL,
    JUDGES,
    load_run,
    run_name,
)
from core.model.holistic.diagnostics import score_spread  # noqa: E402
from core.model.holistic.inference import std_errors  # noqa: E402
from paper_style import BLUE, RED  # noqa: E402

GREEN, PURPLE = "#4E8A5B", "#7B5EA7"  # colourblind-safe hues alongside BLUE / RED
ORDER = [JUDGE_LABEL[j] for j in JUDGES]  # display names, fixed order
COLORS = dict(zip(ORDER, [BLUE, RED, GREEN, PURPLE]))
Z95 = 1.959963985
MAX_AXES = 16

__all__ = [
    "CORPORA",
    "CORPUS_LABEL",
    "ORDER",
    "COLORS",
    "OUT",
    "EXPERIMENT",
    "MAX_AXES",
    "Z95",
    "load_fit",
    "criteria_frame",
    "essays_in_fit_order",
    "load_clusters",
    "radar_axes",
    "draw_radar",
]


def _results(judge: str, corpus: str) -> Path:
    return load_run(run_name(judge, corpus)).results


def load_fit(judge: str, corpus: str, reliable: bool = True):
    fdir = "fit_reliable" if reliable else "fit"
    with open(_results(judge, corpus) / fdir / "fit.pkl", "rb") as f:
        return pickle.load(f)


def criteria_frame(judge: str, corpus: str) -> pd.DataFrame:
    """One row per reliable criterion of the run, in the column order of ``s``.

    The spread is recomputed from the reliable fit's own ``s`` and sandwich covariance,
    never read from ``fit/criterion_diagnostics.parquet`` (which describes the all-K fit).
    ``std`` is ddof=0, matching rho's convention.
    """
    res = _results(judge, corpus)
    active = pd.read_parquet(res / "pipeline_reliable/active_criteria.parquet")
    fit = load_fit(judge, corpus)
    cov = np.load(res / "fit_reliable/cov.npy")
    se = std_errors(fit, cov)
    defs = {
        c["name"]: c["definition"]
        for c in json.loads((res / "taxonomy/taxonomy.json").read_text())["criteria"]
    }
    names = list(active["canonical_name"])
    assert fit.s.shape[1] == len(names) == len(fit.gamma), f"order mismatch in {res}"
    sp = score_spread(fit, cov, se["s"])
    return pd.DataFrame(
        {
            "criterion": names,
            "definition": [defs[n] for n in names],
            "gamma": np.asarray(fit.gamma, float),
            "gamma_se": np.asarray(se["gamma"], float),
            "std_s": sp.std,
            "std_s_se": sp.se_std,
            "std_s_denoised": sp.std_denoised,
            "std_s_denoised_se": sp.se_std_denoised,
            "rho": sp.rho,
            "citation_share": active["share"].to_numpy(float),
        }
    )


def essays_in_fit_order(judge: str, corpus: str) -> pd.DataFrame:
    """The sampled essays, reindexed so row i is the essay behind ``s[i]``."""
    res = _results(judge, corpus)
    eidx = json.loads((res / "pipeline/essay_index.json").read_text())
    order = [e for e, _ in sorted(eidx.items(), key=lambda kv: kv[1])]
    return pd.read_parquet(res / "graph/essays.parquet").set_index("essay_id").loc[order]


def load_clusters(corpus: str) -> dict:
    path = EXPERIMENT / "results/cross_judge" / f"{corpus}_clusters.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run scripts/07_cross_judge_cluster.py first")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Paper radar (two panels, constructs shared by >= 3 judges)
# --------------------------------------------------------------------------- #
def radar_axes(corpus: str, min_judges: int = 1, max_axes: int = MAX_AXES):
    """Axis specs for one corpus: label, per-judge (value, SE), and judge coverage.

    Returns ``(axes, total)``: the constructs with at least ``min_judges`` judges, ordered
    by judge count then by the largest citation share among their members, capped at
    ``max_axes``; and the total number of constructs.
    """
    vals: dict[tuple[str, str], tuple[float, float]] = {}
    share: dict[tuple[str, str], float] = {}
    for judge, label in zip(JUDGES, ORDER):
        f = criteria_frame(judge, corpus)
        for r in f.itertuples(index=False):
            vals[(label, r.criterion)] = (float(r.std_s_denoised), float(r.std_s_denoised_se))
            share[(label, r.criterion)] = float(r.citation_share)

    axes = []
    for c in load_clusters(corpus)["clusters"]:
        members = {m["judge"]: m["name"] for m in c["members"]}
        axes.append(
            {
                "label": c["name"],
                "size": len(members),
                "max_share": max(share[(j, n)] for j, n in members.items()),
                # `members` holds only the judges present in this construct; the rest are None.
                "vals": {j: (vals.get((j, members[j])) if j in members else None) for j in ORDER},
            }
        )
    total = len(axes)
    axes = [a for a in axes if a["size"] >= min_judges]
    axes.sort(key=lambda a: (-a["size"], -a["max_share"]))
    return axes[:max_axes], total


def draw_radar(
    ax, axes, order, colors, *, panel_label=None, z=Z95, grey="#7F7F7F", wrap=12, label_size=7.0
):
    """Draw one corpus's radar onto a polar axis. Returns (lo, hi) of the radial range."""
    series = {j: [None if a["vals"][j] is None else a["vals"][j][0] for a in axes] for j in order}
    errs = {j: [None if a["vals"][j] is None else a["vals"][j][1] for a in axes] for j in order}
    present = [v for j in order for v in series[j] if v is not None]
    lo, hi = 0.0, max(present) * 1.12

    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed = ang + ang[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Small hole at the centre so judges missing a criterion do not all collapse onto the
    # origin; the floor VALUE stays at lo.
    ax.set_ylim(lo - 0.12 * (hi - lo), hi)

    for j in order:
        raw = series[j]
        v = [lo if x is None else x for x in raw]
        ax.plot(closed, v + v[:1], color=colors[j], lw=1.1, label=j, zorder=3)
        ax.fill(closed, v + v[:1], color=colors[j], alpha=0.08, zorder=2)
        absent = [(a_, lo) for a_, x in zip(ang, raw) if x is None]
        if absent:
            ax.scatter(
                *zip(*absent), s=14, facecolors="none", edgecolors=colors[j], lw=0.9, zorder=4
            )
        cap = 0.035 * 2 * np.pi / max(n, 1)
        for a_, x, e in zip(ang, raw, errs[j]):
            if x is None or e is None or not np.isfinite(e):
                continue
            b_lo, b_hi = max(lo, x - z * e), min(hi, x + z * e)
            ax.plot([a_, a_], [b_lo, b_hi], color=colors[j], lw=0.8, zorder=6)
            for end in (b_lo, b_hi):
                ax.plot([a_ - cap, a_ + cap], [end, end], color=colors[j], lw=0.7, zorder=6)

    ax.set_xticks(ang)
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(a["label"], wrap, break_long_words=False)) for a in axes],
        fontsize=label_size,
    )
    from matplotlib.ticker import MaxNLocator

    ax.set_yticks([t for t in MaxNLocator(nbins=3).tick_values(lo, hi) if lo <= t <= hi])
    ax.tick_params(axis="y", labelsize=7)
    peak = [
        max((series[j][i] for j in order if series[j][i] is not None), default=lo) for i in range(n)
    ]
    ax.set_rlabel_position(np.degrees(ang[int(np.argmin(peak))]) + 180 / n)
    ax.grid(color=grey, lw=0.4, alpha=0.5)
    ax.spines["polar"].set_alpha(0.3)
    if panel_label:
        # y in axes coords keeps both panels' labels on the same line regardless of
        # how far their longest tick label extends.
        ax.set_title(panel_label, fontsize=9, y=1.28, fontweight="medium")
    return lo, hi
