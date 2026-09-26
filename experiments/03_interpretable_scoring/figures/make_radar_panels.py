"""Per-corpus radars of the denoised spread sigma_k, every cross-judge construct shown.

Produces Figures ``fig:radar_ellipse`` and ``fig:radar_asap`` (App. ``sec:extension-appendix``). Where the two-panel
``make_cross_judge_radar.py`` keeps only constructs three or more judges share, these show
every construct with two or more judges plus the highest-citation-share singletons, up to a
16-axis cap -- on ASAP the cap leaves out three single-judge constructs (listed on stdout).

Axes are the step-7 constructs (grouping by MEANING). Within a judge count, constructs are
ordered by the mean cosine similarity of their members' ``"name. definition"`` embeddings
(text-embedding-3-large), and axis labels carry "(k/4 judges)" or "(<judge> only)". The
embeddings play no part in the grouping; they only fix the order, which is why the 81
vectors needed are shipped in ``raw/embed_cache/`` (no API call).

The plotted quantity is the denoised sigma_k on the reliable panels, with 95% delta-method
intervals from the sandwich covariance.

Usage
    python figures/make_radar_panels.py

Output
    figures/radar_std_s_denoised_reliable_4judge_llm.{pdf,png}        ELLIPSE
    figures/radar_std_s_denoised_reliable_asap_4judge_llm.{pdf,png}   ASAP 2.0
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from interp_common import (
    COLORS,
    CORPORA,
    EXPERIMENT,
    MAX_AXES,
    ORDER,
    OUT,
    Z95,
    criteria_frame,
    load_clusters,
)
from paper_style import FULL_W, GREY, apply_style, save

from core.embeddings import embed_texts

JUDGE_OF = dict(
    zip(ORDER, ["gpt_5_4_mini", "claude_haiku_4_5", "gemini_3_6_flash", "deepseek_v4_flash"])
)
FNAME = {
    "ellipse": "radar_std_s_denoised_reliable_4judge_llm",
    "asap2": "radar_std_s_denoised_reliable_asap_4judge_llm",
}
EMBED_CACHE = EXPERIMENT / "raw" / "embed_cache"


def construct_table(corpus: str):
    """Criteria of all four judges, their cosine matrix, and one row per construct."""
    df = pd.concat(
        [criteria_frame(JUDGE_OF[j], corpus).assign(judge=j) for j in ORDER], ignore_index=True
    )
    texts = list(df["criterion"] + ". " + df["definition"])
    E = embed_texts(texts, cache_dir=EMBED_CACHE)
    S = E @ E.T  # unit-norm rows -> cosine
    pos = {(r.judge, r.criterion): i for i, r in enumerate(df.itertuples())}

    clusters = []
    for c in load_clusters(corpus)["clusters"]:
        clusters.append((sorted(pos[(m["judge"], m["name"])] for m in c["members"]), c["name"]))
    placed = sorted(i for c, _ in clusters for i in c)
    assert placed == list(range(len(df))), "constructs do not partition the criteria"
    clusters.sort(key=lambda t: (-len(t[0]), t[0][0]))

    rows = []
    for cid, (c, name) in enumerate(clusters):
        judges = [df.loc[i, "judge"] for i in c]
        assert len(judges) == len(set(judges)), "a construct holds two criteria of one judge"
        sims = [S[x, y] for x in c for y in c if x < y]
        row = {
            "cluster": cid,
            "cluster_name": name,
            "size": len(c),
            "mean_cos": float(np.mean(sims)) if sims else np.nan,
            "max_share": float(df.loc[list(c), "citation_share"].max()),
        }
        for j in ORDER:
            m = [i for i in c if df.loc[i, "judge"] == j]
            row[j] = df.loc[m[0], "criterion"] if m else None
        rows.append(row)
    return df, pd.DataFrame(rows)


def radar_axes(df: pd.DataFrame, t: pd.DataFrame):
    multi = t[t["size"] >= 2].sort_values(["size", "mean_cos"], ascending=[False, False])
    solo = t[t["size"] == 1].sort_values("max_share", ascending=False)
    keep = pd.concat([multi, solo]).head(MAX_AXES)

    lookup = {(r.judge, r.criterion): r for r in df.itertuples()}
    axes = []
    for _, r in keep.iterrows():
        present = [j for j in ORDER if pd.notna(r[j])]
        label = str(r["cluster_name"])
        if len(present) == 1:
            label += f"\n({present[0].split('-')[0]} only)"
        elif len(present) < len(ORDER):
            label += f"\n({len(present)}/{len(ORDER)} judges)"
        axes.append(
            {
                "label": label,
                "vals": {j: (lookup[(j, r[j])] if pd.notna(r[j]) else None) for j in ORDER},
            }
        )
    omitted = t.loc[~t.index.isin(keep.index), "cluster_name"].tolist()
    return axes, omitted


def radar(
    axes,
    fname: str,
    field="std_s_denoised",
    err_field="std_s_denoised_se",
    legend_title=r"denoised $\sigma_k$",
    z=Z95,
):
    series = {
        j: [None if a["vals"][j] is None else float(getattr(a["vals"][j], field)) for a in axes]
        for j in ORDER
    }
    errs = {
        j: [None if a["vals"][j] is None else float(getattr(a["vals"][j], err_field)) for a in axes]
        for j in ORDER
    }
    present = [v for j in series for v in series[j] if v is not None]
    # A single thinly-cited criterion can have a large, noisily estimated spread that
    # compresses every other axis; cap the radial axis at a robust bound and report any
    # value beyond it rather than silently distorting.
    cap = float(np.quantile(present, 0.90)) * 1.35
    hi = min(max(present) * 1.12, max(cap, float(np.median(present)) * 2))
    lo = 0.0
    clipped = [
        (a["label"].split("\n")[0], j, v)
        for j in ORDER
        for a, v in zip(axes, series[j])
        if v is not None and v > hi
    ]

    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed = ang + ang[:1]

    fig, ax = plt.subplots(figsize=(FULL_W, 4.4), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # A small hole at the centre: absent criteria sit at `lo`, and without it every judge
    # missing a criterion collapses onto the origin. The floor VALUE is unchanged.
    buffer = 0.12 * (hi - lo)
    ax.set_ylim(lo - buffer, hi)
    ticks = [
        t
        for t in MaxNLocator(nbins=4, prune=None).tick_values(lo, hi)
        if lo - 1e-9 <= t <= hi + 1e-9
    ]
    if ticks:
        ax.set_yticks(ticks)

    for j in ORDER:
        raw = series[j]
        v = [lo if x is None else min(x, hi) for x in raw]
        ax.plot(closed, v + v[:1], color=COLORS[j], lw=1.2, label=j, zorder=3)
        ax.fill(closed, v + v[:1], color=COLORS[j], alpha=0.08, zorder=2)
        absent = [(a_, lo) for a_, x in zip(ang, raw) if x is None]
        if absent:
            ax.scatter(
                *zip(*absent), s=20, facecolors="none", edgecolors=COLORS[j], lw=1.1, zorder=4
            )
        # Radial +/- z*SE segments, per vertex: four shaded bands would be unreadable.
        capw = 0.035 * 2 * np.pi / max(n, 1)
        for a_, x, e in zip(ang, raw, errs[j]):
            if x is None or e is None or not np.isfinite(e):
                continue
            b_lo, b_hi = max(lo, x - z * e), min(hi, x + z * e)
            ax.plot(
                [a_, a_],
                [b_lo, b_hi],
                color=COLORS[j],
                lw=1.0,
                alpha=0.95,
                solid_capstyle="butt",
                zorder=6,
            )
            for endpoint in (b_lo, b_hi):
                ax.plot(
                    [a_ - capw, a_ + capw],
                    [endpoint, endpoint],
                    color=COLORS[j],
                    lw=0.8,
                    alpha=0.95,
                    solid_capstyle="butt",
                    zorder=6,
                )

    handles = ax.get_legend_handles_labels()[0]
    labels = list(ORDER)
    handles += [Line2D([], [], color=GREY, lw=0, marker="o", mfc="none", ms=4.5)]
    labels += ["absent for that judge"]

    ax.set_xticks(ang)
    ax.set_xticklabels(["\n".join(textwrap.wrap(a["label"], 13)) for a in axes], fontsize=6.8)
    ax.tick_params(axis="y", labelsize=7)
    # Radial tick labels go on the emptiest spoke, clear of the data polygons.
    peak = [
        max((series[j][i] for j in ORDER if series[j][i] is not None), default=lo) for i in range(n)
    ]
    ax.set_rlabel_position(np.degrees(ang[int(np.argmin(peak))]) + 180 / n)
    ax.grid(color=GREY, lw=0.5, alpha=0.5)
    ax.spines["polar"].set_alpha(0.3)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=7,
        title=legend_title,
        title_fontsize=8,
    )
    save(fig, OUT, fname)
    plt.close(fig)
    for lab, j, v in clipped:
        print(f"  NOTE: {j} on '{lab}' is {v:.2f}, clipped to the {hi:.2f} radial cap")


def main() -> int:
    apply_style()
    for corpus in CORPORA:
        df, table = construct_table(corpus)
        axes, omitted = radar_axes(df, table)
        print(
            f"{corpus}: {len(axes)} axes of {len(table)} constructs"
            + (f"; omitted (16-axis cap): {', '.join(omitted)}" if omitted else "")
        )
        radar(axes, FNAME[corpus])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
