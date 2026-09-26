"""Produces Figures `fig:cmv-win-rates` (figures/cmv_win_rates.pdf) and `fig:cmv-win-rates-full`
(figures/cmv_win_rates_full.pdf, with ``--full``).

    python scripts/11_win_rates.py                     # writes results/win_rates.csv
    python figures/make_cmv_win_rates.py               # compact, main text
    python figures/make_cmv_win_rates.py --full        # every row named, with intervals (appendix)

Requires a LaTeX installation (``latex`` and ``dvipng`` on PATH); ``--no-tex`` gives an
on-screen approximation only.

All 29 discovered challenger-side criteria, scored pointwise on the 800 matched held-out
exchanges, one dot per criterion, with the rows banded into five families. The compact version
spends its vertical space on the grouping; ``--full`` names every row and draws the intervals.

The numbers are not recomputed here: both versions read the table step 11 wrote and bin q
through ``paper_style.winrate_colour``, so a dot is the same number and the same colour as the
matching row of `tab:cmv-win-rates`.

Encodings, all of which the caption has to state:

  position   the held-out win rate, ties split
  colour     the BH q bin across the 29 tests; grey = not significant
  row band   one family; rows inside it are sorted by win rate, best at the top
  chance     the vertical stripe: 0.5 +- the BH critical margin across the 29 tests, the zone
             in which a criterion is not distinguishable from chance

The q bins are keyed by a legend row under the x-axis label; by default it shows only the bins
this panel actually contains (see :func:`add_legend`).

Whether a criterion passed the kappa/rho discovery screen is NOT drawn -- all 29 dots are filled
alike. The screen is a property of how a criterion was found, not of what the held-out pairs say
about it, and the two readings competed for the same marker. The counts are still printed.

Layout is solved, not guessed. Row pitch, family padding and the inter-family gap are set in
POINTS and the axes is then sized so one data unit is exactly one point; the build re-renders
until no two y tick labels overlap and no dot runs into a family name, and fails if it cannot.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, MultipleLocator  # noqa: E402

EXPERIMENT = Path(__file__).resolve().parent.parent
REPO = EXPERIMENT.parents[1]  # repository root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXPERIMENT / "figures"))
sys.path.insert(0, str(REPO))

from core.cmv import win_rates as WR  # noqa: E402
from paper_style import (
    CRITERION_LABELS,
    FULL_W,
    WINRATE_THEME,
    apply_style,  # noqa: E402
    save,
    tex_escape,
    winrate_colour,
)

#: The panel: the 16 gate-kept criteria + the 13 the gate dropped, both scored pointwise on
#: ``feature_bt_root_full``'s graph; the table is read from results/win_rates.csv (step 11).
CONFIGS = None

#: The cell whose criteria passed the kappa/rho discovery screen. Not drawn; the build reports
#: the split. Cell names come back in the table's ``cell`` column.
KEPT_CELL = "pointwise_discovered_full"

#: family -> its member criteria, by their FULL name in the score matrices. Every one of the 29
#: challenger-side criteria appears exactly once; the build asserts it. Edit only here.
FAMILIES = {
    "Evidence & cases": [
        "Concrete illustration",
        "Counterexample use",
        "Challenger evidentiary support",
        "Analogical reasoning",
    ],
    "Rebuttal strategy": [
        "Qualified rebuttal",
        "Structured rebuttal development",
        "Scope broadening",
        "Reframing and normalization",
        "Targeted rebuttal",
        "Alternative proposal and guidance",
        "Concessionary rebuttal",
    ],
    "Explanation & analysis": [
        "Structural explanation",
        "Conceptual clarification",
        "Mechanistic explanation",
        "Meta-level epistemic challenge",
        "Practical consequences and feasibility",
        "Consistency challenge",
    ],
    "Framing": [
        "Abstract or technical framing",
        "Exchange empirical framing",
        "Historical framing and legacy",
        "Legal and institutional framing",
        "Psychological framing and diagnosis",
    ],
    "Tone & appeals": [
        "Rhetorical vividness",
        "Audience tailoring",
        "Emotional appeal and empathy",
        "Normative and rights appeal",
        "Challenger experiential grounding",
        "Rhetorical questioning",
        "Confrontational tone",
    ],
}

XLABEL = r"$\mathbb{P}(\Delta\text{-winning reply scores higher})$"
XMIN = 0.45  # compact mode; with whiskers the left edge follows the data
XMAX0 = 0.67  # starting right edge; widened if a family name would reach a dot
XMAX_CAP = 0.78

ROW_PITCH_PT = 4.25  # centre-to-centre of consecutive rows, compact mode (most rows unnamed)
ROW_PITCH_FULL_PT = 11.0  # ... and with every row named, where the pitch is a line of 8pt text
RECT_PAD_PT = 3.0  # band padding above the first row and below the last
GAP_PT0 = 6.0  # white gap between consecutive bands
GAP_PT_CAP = 14.0
PITCH_CAP = 20.0
DOT_MS = 3.0  # marker diameter, points
WHISKER_LW = 0.9  # 95% Wald interval, drawn as a plain round-capped rule -- no
# T-caps, which at this scale only add ink
WHISKER_PAD = 0.005  # x headroom past the widest interval end
ROUND_PT = 2.5  # band corner radius
BAND_ALPHA = 0.075  # family band: black at 7.5%
CHANCE_ALPHA = 0.035  # chance stripe, drawn under the family bands so the two tints compound
LABEL_PT = 8  # y tick labels and family names; the minimum text size
NAME_PAD_FRAC = 0.012  # family name inset from the band's right edge, in axes fractions
NAME_CLEAR_PT = 3.0  # clearance a dot must keep from the family-name column
LEGEND_GAP_PT = 2.0  # gap between the axes and the legend stack below its right edge

#: The BH q bins, darkest first, as (upper bound, theme key, label). ``None`` is the open top bin.
Q_BINS = [
    (1e-3, "blue_hi", r"$q<0.001$"),
    (1e-2, "blue_mid", r"$q<0.01$"),
    (0.05, "blue_lo", r"$q<0.05$"),
    (None, "null", "n.s."),
]
MAX_H_IN = 2.9  # compact mode: it shares a page with body text
MAX_H_FULL_IN = 9.0  # full mode: an appendix figure may have the page to itself
SOFT_H_IN = 4.5  # a note is printed past this height


def check_families(features: list[str]) -> None:
    """Every criterion in the panel is in exactly one family, and no family invents one."""
    assigned = [c for members in FAMILIES.values() for c in members]
    dupes = sorted({c for c in assigned if assigned.count(c) > 1})
    missing = sorted(set(features) - set(assigned))
    unknown = sorted(set(assigned) - set(features))
    if dupes or missing or unknown:
        raise SystemExit(
            "[err] FAMILIES does not partition the panel"
            + (f"\n  in two families: {dupes}" if dupes else "")
            + (f"\n  never assigned:  {missing}" if missing else "")
            + (f"\n  not in the data: {unknown}" if unknown else "")
        )
    assert len(assigned) == len(features), (len(assigned), len(features))


def family_table(tab):
    """Families ordered by mean win rate descending, members within a family likewise."""
    wr = dict(zip(tab.feature, tab.win_rate))
    lo = dict(zip(tab.feature, tab.lo))
    hi = dict(zip(tab.feature, tab.hi))
    rows = []
    for fam, members in FAMILIES.items():
        ordered = sorted(members, key=lambda c: -wr[c])
        rows.append(
            dict(
                family=fam,
                k=len(members),
                members=ordered,
                values=np.array([wr[c] for c in ordered]),
                los=np.array([lo[c] for c in ordered]),
                his=np.array([hi[c] for c in ordered]),
                mean=float(np.mean([wr[c] for c in members])),
            )
        )
    return sorted(rows, key=lambda r: -r["mean"])


def solve_layout(fams, pitch: float, gap: float, pad: float):
    """Row and band geometry in POINTS from the top of the axes.

    Points, not row indices, because every constraint here is a point distance -- the pitch, the
    padding, the gap, the text height a label needs. Keeping the unit means the axes can be sized
    so one data unit is one point, and the geometry survives to the PDF unchanged.
    """
    rows, bands, cursor = [], [], 0.0
    for f in fams:
        top = cursor
        ys = [top + pad + j * pitch for j in range(f["k"])]
        bottom = ys[-1] + pad
        rows += [
            dict(criterion=c, value=v, lo=lo, hi=hi, y=y, family=f["family"])
            for c, v, lo, hi, y in zip(f["members"], f["values"], f["los"], f["his"], ys)
        ]
        bands.append(dict(family=f["family"], top=top, bottom=bottom, first=ys[0], last=ys[-1]))
        cursor = bottom + gap
    return rows, bands, cursor - gap


def build(
    tab,
    fams,
    pitch: float,
    gap: float,
    xlim: tuple[float, float],
    half_width: float,
    all_bins: bool = False,
    label_all: bool = False,
    whiskers: bool = False,
):
    """Render one candidate figure. Returns it with the measurements the caller checks."""
    rows, bands, h_pts = solve_layout(fams, pitch, gap, RECT_PAD_PT)
    axes_h_in = h_pts / 72
    fig = plt.figure(figsize=(FULL_W, axes_h_in + 0.6), layout=None)
    ax = fig.add_subplot(111)

    q = dict(zip(tab.feature, tab.q))

    # The chance stripe runs the full height, including the white gaps between family bands, so
    # it reads as one column crossing them rather than as a patch belonging to any family.
    ax.axvspan(
        0.5 - half_width, 0.5 + half_width, color="black", alpha=CHANCE_ALPHA, lw=0, zorder=-1
    )
    ax.axvline(0.5, color=WINRATE_THEME["chance"], lw=0.7, zorder=2)  # over bands, under dots
    for r in rows:
        c = winrate_colour(q[r["criterion"]])
        if whiskers:  # above the 0.5 rule -- the interval is data, the rule is furniture
            ax.plot(
                [r["lo"], r["hi"]],
                [r["y"]] * 2,
                color=c,
                lw=WHISKER_LW,
                solid_capstyle="round",
                zorder=2.5,
            )
        ax.plot(r["value"], r["y"], "o", ms=DOT_MS, zorder=3, color=c, mfc=c, mec=c, mew=0)

    ax.set_xlim(*xlim)
    ax.set_ylim(h_pts, 0)  # data unit = 1 pt, first row at the top
    # Compact mode names only each band's first and last row; full mode names all 29.
    ys = [r["y"] for r in rows] if label_all else [b[k] for b in bands for k in ("first", "last")]
    by_y = {r["y"]: r["criterion"] for r in rows}
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [tex_escape(CRITERION_LABELS.get(by_y[y], by_y[y])) for y in ys], fontsize=LABEL_PT
    )
    ax.set_xlabel(XLABEL)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 0.6]))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.6)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(False)

    leg, unused = add_legend(ax, tab, all_bins, h_pts)
    by_y_row = {r["y"]: r for r in rows}
    ax._row_family = {
        tex_escape(CRITERION_LABELS.get(by_y[y], by_y[y])): by_y_row[y]["family"] for y in ys
    }
    fit_axes(fig, ax, axes_h_in, leg)
    names = add_bands(fig, ax, bands, h_pts)
    return fig, ax, rows, names, h_pts, unused


def fit_axes(fig, ax, axes_h_in: float, leg=None) -> None:
    """Pin the axes to exactly ``axes_h_in`` and size the figure around the rendered text.

    The height is the layout's, so the figure height is whatever that plus the measured x-axis
    furniture comes to -- never the other way round, which would rescale the row pitch.
    """
    for _ in range(3):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        box = ax.get_window_extent()
        need_l = box.x0 - min(
            [t.get_window_extent(rend).x0 for t in ax.get_yticklabels()] + [box.x0]
        )
        below = (
            [ax.xaxis.label.get_window_extent(rend)]
            + [t.get_window_extent(rend) for t in ax.get_xticklabels()]
            + ([leg.get_window_extent(rend)] if leg is not None else [])
        )
        need_b = box.y0 - min(b.y0 for b in below)
        pad_px = 1.5
        fig_h = axes_h_in + (need_b + 2 * pad_px) / fig.dpi
        fig.set_size_inches(FULL_W, fig_h)
        w_px, h_px = FULL_W * fig.dpi, fig_h * fig.dpi
        left = (need_l + pad_px) / w_px
        ax.set_position(
            [left, (need_b + pad_px) / h_px, 1 - left - pad_px / w_px, axes_h_in / fig_h]
        )


def add_bands(fig, ax, bands, h_pts: float):
    """One pale rounded rectangle per family, plus its name inset at the right.

    Both live in axes fractions so they survive the 72-dpi re-render the PDF backend does. A
    FancyBboxPatch rounds its corners in its own coordinate space, which here is anisotropic, so
    ``mutation_aspect`` is set to the axes' display aspect -- without it the corners come out as
    ellipses stretched along x.

    Returns the name artist per family, which is what the clearance check measures against.
    """
    fig.canvas.draw()
    box = ax.get_window_extent()
    w_pts, h_pts_disp = box.width / fig.dpi * 72, box.height / fig.dpi * 72
    names = {}
    for b in bands:
        y0, y1 = 1 - b["bottom"] / h_pts, 1 - b["top"] / h_pts
        patch = FancyBboxPatch(
            (0, y0),
            1,
            y1 - y0,
            transform=ax.transAxes,
            boxstyle=f"round,pad=0,rounding_size={ROUND_PT / w_pts}",
            mutation_aspect=w_pts / h_pts_disp,
            facecolor="black",
            alpha=BAND_ALPHA,
            edgecolor="none",
            lw=0,
            zorder=0,
        )
        patch.set_clip_on(False)
        ax.add_patch(patch)
        names[b["family"]] = ax.text(
            1 - NAME_PAD_FRAC,
            0.5 * (y0 + y1),
            r"\textbf{" + tex_escape(b["family"]) + "}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=LABEL_PT,
            color=WINRATE_THEME["chance"],
            zorder=1,
        )
    return names


def q_bin(q: float) -> int:
    """Index into :data:`Q_BINS`. Mirrors ``paper_style.winrate_colour``, which picks the colour."""
    for i, (hi, _, _) in enumerate(Q_BINS):
        if hi is None or q < hi:
            return i
    return len(Q_BINS) - 1


def add_legend(ax, tab, all_bins: bool, h_pts: float):
    """Key the q colours in a stack at the bottom right, under the axes' right-hand corner.

    It sits beside the x-axis label rather than below it: the label is centred and the stack is
    right-aligned, so the two share the strip under the axes and the figure grows by only the
    difference between their heights.

    By default only the bins that actually occur are keyed. A swatch with no dot behind it reads
    as a missing category -- this panel has nothing in q in [0.01, 0.05), and showing that bin
    would have a reader hunting the figure for a colour that is not in it. ``all_bins`` keys all
    four bins.
    """
    used = {q_bin(q) for q in tab.q}
    shown = [i for i in range(len(Q_BINS)) if all_bins or i in used]
    handles = [
        Line2D([], [], marker="o", ls="", ms=DOT_MS, mew=0, color=WINRATE_THEME[Q_BINS[i][1]])
        for i in shown
    ]
    leg = ax.legend(
        handles,
        [Q_BINS[i][2] for i in shown],
        loc="upper right",
        bbox_to_anchor=(1.0, -LEGEND_GAP_PT / h_pts),
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=False,
        fontsize=LABEL_PT,
        handletextpad=0.35,
        labelspacing=0.32,
        borderaxespad=0,
        borderpad=0,
        handlelength=1.0,
    )
    return leg, [Q_BINS[i][2] for i in range(len(Q_BINS)) if i not in used]


def furniture_overlaps(fig, ax) -> list[str]:
    """Whether the legend stack runs into the x-axis label or a tick label.

    The stack shares the strip under the axes with the x label, which is what keeps the figure
    short; that only works while the two stay clear of each other, so it is checked, not assumed.
    """
    leg = ax.get_legend()
    if leg is None:
        return []
    rend = fig.canvas.get_renderer()
    box = leg.get_window_extent(rend)
    hits = ["x-axis label"] if box.overlaps(ax.xaxis.label.get_window_extent(rend)) else []
    hits += [
        f"tick {t.get_text()}"
        for t in ax.get_xticklabels()
        if t.get_text() and box.overlaps(t.get_window_extent(rend))
    ]
    return hits


def interval_vs_colour(tab) -> list[str]:
    """Criteria whose 95% interval clears 0.5 but whose colour says not significant, or vice versa.

    Not a bug: the intervals are uncorrected and the colour is BH-corrected across 29 tests, so
    a criterion can clear 0.5 on its own and still be grey. It is worth reporting because a
    reader takes "interval misses 0.5" as significance, and if it ever happens the caption has to
    say which of the two is the claim.
    """
    excludes = (tab.lo > 0.5) | (tab.hi < 0.5)
    return sorted(tab.feature[excludes != (tab.q < 0.05)])


def label_overlaps(fig, ax) -> list[tuple[str, str, bool]]:
    """Pairs of y tick labels whose rendered boxes touch, and whether they are in one family.

    Which it is decides the fix: two rows of the same family are a row-pitch apart, so only a
    bigger pitch separates them, while rows in different families are separated by the gap.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fam = getattr(ax, "_row_family", {})
    ts = sorted(ax.get_yticklabels(), key=lambda t: t.get_window_extent(rend).y0)
    out = []
    for a, b in zip(ts, ts[1:]):
        if a.get_window_extent(rend).overlaps(b.get_window_extent(rend)):
            ta, tb = a.get_text(), b.get_text()
            out.append((ta, tb, fam.get(ta) == fam.get(tb)))
    return out


def name_collisions(fig, ax, rows, names, whiskers: bool = False) -> list[str]:
    """Families whose name reaches into their band's dots; that column has to stay clear of data.

    The test is horizontal and runs over the whole band, not just the name's own line: a family
    name is vertically centred, so comparing boxes would only ever catch a dot on the middle row
    and would pass a layout whose top row runs straight into the text.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    clear_px = (DOT_MS / 2 + NAME_CLEAR_PT) / 72 * fig.dpi
    hits = []
    for family, t in names.items():
        left = t.get_window_extent(rend).x0 - clear_px
        key = "hi" if whiskers else "value"  # an interval reaches further than its dot
        reach = max(
            (ax.transData.transform((r[key], r["y"]))[0] for r in rows if r["family"] == family),
            default=-np.inf,
        )
        if reach >= left:
            hits.append(family)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--full",
        action="store_true",
        help="appendix version: name every criterion on the y axis, at a row pitch "
        "that fits a line of text; the figure gets much taller",
    )
    ap.add_argument(
        "--pitch",
        type=float,
        default=None,
        help="row pitch, points (default: 4.25 compact, 11 full)",
    )
    ap.add_argument("--gap", type=float, default=GAP_PT0, help="inter-family gap, points")
    ap.add_argument(
        "--out", default=None, help="file stem under figures/ (default: cmv_win_rates[_full])"
    )
    ap.add_argument(
        "--whiskers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="95%% Wald intervals on each dot (default: on with --full, off compact)",
    )
    ap.add_argument(
        "--all-bins",
        action="store_true",
        help="key all four q bins even where this panel has none in that bin",
    )
    ap.add_argument("--no-tex", action="store_true", help="mathtext fallback; NOT camera-ready")
    a = ap.parse_args()
    pitch = a.pitch if a.pitch is not None else (ROW_PITCH_FULL_PT if a.full else ROW_PITCH_PT)
    # The paper's filenames.
    out = a.out or ("cmv_win_rates_full" if a.full else "cmv_win_rates")
    max_h = MAX_H_FULL_IN if a.full else MAX_H_IN
    whiskers = a.full if a.whiskers is None else a.whiskers

    have_tex = shutil.which("latex") and shutil.which("dvipng")
    if not a.no_tex and not have_tex:
        raise SystemExit(
            "[err] no TeX toolchain (latex/dvipng): the paper figure must be usetex. "
            "Pass --no-tex for an on-screen approximation (STIX, not the paper font)."
        )
    os.environ["FIGURES_USETEX"] = "0" if a.no_tex else "1"

    # Read the table step 11 wrote, rather than re-deriving it: one source of truth means a
    # dot in this figure and a row of tab:cmv-win-rates cannot disagree.
    tab = pd.read_csv(EXPERIMENT / "results/win_rates.csv").sort_values("win_rate")
    meta = json.loads((EXPERIMENT / "results/win_rates_meta.json").read_text())
    length_ref, n = meta["length_reference_win_rate"], meta["n_pairs"]
    check_families(list(tab.feature))
    fams = family_table(tab)

    half_width = WR.bh_critical_margin(tab, n)
    if bad := WR.band_is_consistent(tab, half_width):
        raise SystemExit(
            f"[err] the chance stripe contradicts the BH verdict for {bad}; the "
            "figure would tell the reader the opposite of the test"
        )

    apply_style()
    # With intervals drawn, the leftmost one can reach past 0.45, so the left edge follows the
    # data; without them it stays at the compact figure's 0.45 so the two line up.
    xmin = min(XMIN, float(tab.lo.min()) - WHISKER_PAD) if whiskers else XMIN
    gap, xmax, notes = a.gap, XMAX0, []
    while True:
        fig, ax, rows, names, h_pts, unused = build(
            tab, fams, pitch, gap, (xmin, xmax), half_width, a.all_bins, a.full, whiskers
        )
        hit = name_collisions(fig, ax, rows, names, whiskers)
        bad = label_overlaps(fig, ax)
        if clash := furniture_overlaps(fig, ax):
            raise SystemExit(f"[err] the legend runs into {clash}; move it or shorten its labels")
        if hit and xmax < XMAX_CAP:
            plt.close(fig)
            xmax = round(xmax + 0.005, 3)
            notes.append(f"widened x to {xmax} ({hit[0]} reached the name column)")
            continue
        if bad:
            first, second, same_family = bad[0]
            if same_family and pitch < PITCH_CAP:
                plt.close(fig)
                pitch += 0.5
                notes.append(f"pitch -> {pitch:.2f}pt ({first} / {second}, same family)")
                continue
            if not same_family and gap < GAP_PT_CAP:
                plt.close(fig)
                gap += 1.0
                notes.append(f"gap -> {gap:.0f}pt ({first} / {second}, across families)")
                continue
        if hit:
            raise SystemExit(f"[err] family name still collides with a dot at xmax={xmax}: {hit}")
        if bad:
            raise SystemExit(f"[err] y labels still overlap at pitch={pitch}pt gap={gap}pt: {bad}")
        break

    h_in = float(fig.get_size_inches()[1])
    if h_in > max_h:
        raise SystemExit(f"[err] {h_in:.2f}in exceeds the {max_h}in budget; lower --pitch")
    if h_in > SOFT_H_IN:
        print(
            f"[note] {h_in:.2f}in is past the {SOFT_H_IN}in soft limit for a "
            "full-width figure -- intended here, this is the appendix version"
        )
    save(fig, EXPERIMENT / "figures", out)

    for note in notes:
        print(f"[layout] {note}")
    print(
        f"[layout] {'full (29 named)' if a.full else 'compact (10 named)'} | pitch "
        f"{pitch}pt | band pad {RECT_PAD_PT}pt | gap {gap:.0f}pt | "
        f"axes {h_pts:.0f}pt | x [{xmin:.3f}, {xmax}] | "
        f"{'95% Wald intervals' if whiskers else 'no intervals'} | "
        "no label overlaps, nothing in the name column"
    )
    if whiskers and (odd := interval_vs_colour(tab)):
        print(
            f"[note] uncorrected interval and BH colour disagree for {odd}: their interval "
            "misses 0.5 but BH across 29 tests does not call them significant"
        )
    if unused and not a.all_bins:
        print(
            f"[legend] bins not keyed because this panel has none: {', '.join(unused)} "
            "(pass --all-bins to key them anyway)"
        )
    print(
        f"[chance] not distinguishable from chance: 0.5 +/- {half_width:.4f} "
        f"= [{0.5 - half_width:.4f}, {0.5 + half_width:.4f}]"
    )
    print(
        f"[data] {n} held-out pairs | {len(tab)} criteria | {int((tab.q < 0.05).sum())} at "
        f"q<0.05 | {int((tab.cell == KEPT_CELL).sum())} of {len(tab)} passed the kappa/rho "
        f"screen (not encoded) | length-only ref {length_ref:.4f}"
    )
    print()
    if a.full:  # every row is named, so report every row
        print(f"{'family':24s}{'mean':>7}   rows top to bottom")
        for f in fams:
            print(f"{f['family']:24s}{f['mean']:>7.4f}")
            for c, v in zip(f["members"], f["values"]):
                print(f"{'':24s}{v:>7.4f}   {CRITERION_LABELS.get(c, c)}")
    else:
        print(f"{'family':24s}{'mean':>7}   {'labelled top row':<36}labelled bottom row")
        for f in fams:
            top, bot = f["members"][0], f["members"][-1]
            top_s = f"{CRITERION_LABELS.get(top, top)} {f['values'][0]:.4f}"
            bot_s = f"{CRITERION_LABELS.get(bot, bot)} {f['values'][-1]:.4f}"
            print(f"{f['family']:24s}{f['mean']:>7.4f}   {top_s:<36}{bot_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
