"""Shared figure style for this experiment's paper figures.

Every figure script imports ``apply_style()`` and the width constants from here;
per-figure rcParams overrides for fonts or sizes are not used.

The math font: the paper loads the ``times`` package, which restyles the TEXT font
only and leaves MATH in Computer Modern, so the target is **Times text with
Computer Modern math**. The default path is therefore matplotlib's own text engine
with **STIXGeneral** (the Times-metric-compatible text face) and
``mathtext.fontset="cm"``.

``FIGURES_USETEX=1`` switches to real TeX, loading the same ``times`` package
for the same reason. It is NOT the default even where TeX is installed: usetex
rejects a raw ``%`` in a label, which several of these figures use (the paired
panels annotate win rates as ``96%``), and it silently drops them.
"""

from __future__ import annotations

import os

import matplotlib as mpl

# The paper's text block. Figures are rendered at final print size.
FULL_W = 5.5  # full text width, inches
HALF_W = 2.65  # two-up half width, inches

# Single hue for single-series plots; the extra hues are reserved
# for genuinely categorical splits and are colourblind-safe.
BLUE = "#4878A8"
RED = "#B4444E"
GREY = "#7F7F7F"

# ---------------------------------------------------------------------------
# Colormaps. Both are built from the SAME two poles as the categorical colours
# above, so every figure in the paper carries one blue identity.
#
# They are deliberately NOT the same colormap: the encodings differ, and the
# rule is sequential = one hue light->dark (magnitude), diverging = two hues
# with a neutral midpoint (polarity). A diverging map on a 0..1 probability
# would invent a meaningful midpoint at 0.5; a sequential map on a signed
# correlation would hide the sign.
# ---------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

#: magnitude on [0, 1] -- one hue, light -> dark, anchored on the paper blue.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "paper_seq", ["#F7FAFC", "#C9DAEA", "#8FB4D1", BLUE, "#2F5476"]
)

#: signed quantities on [-1, 1] -- blue pole, neutral midpoint, red pole.
DIV_CMAP = LinearSegmentedColormap.from_list(
    "paper_div", ["#21425F", BLUE, "#C9DAEA", "#F2F2F2", "#EBC3C6", RED, "#7C2A32"]
)

# ---------------------------------------------------------------------------
# Experiment 2 (ChangeMyView): a third arm hue, the win-rate q bins, and the short
# criterion labels its win-rate figures need.
# ---------------------------------------------------------------------------
#: Third arm hue, for the data-informed (S) panel alongside BLUE (D) and RED (P).
#: Purple, not green: green vs RED is deuteranopia-confusable (OKLab dE 4.7, below the
#: dE>=8 floor). This pair reads dE 14.3 under protanopia and 18.0 with normal vision.
PURPLE = "#6E4B9E"

WINRATE_THEME = {
    "blue_hi": "#0B2E52",  # q < 0.001
    "blue_mid": "#2C6BA8",  # q < 0.01
    "blue_lo": "#7BAAD4",  # q < 0.05
    "null": "#9AA0A6",  # q >= 0.05
    "chance": "#4D4D4D",  # the x = 0.50 reference line
}


def winrate_colour(q: float, alpha: float = 0.05) -> str:
    """BH q-value -> dot colour. Darker blue is stronger evidence; grey is not significant.

    Both win-rate figures bin q here, so a criterion cannot be blue in one and grey in the other.
    """
    if q >= alpha:
        return WINRATE_THEME["null"]
    return WINRATE_THEME["blue_hi" if q < 1e-3 else "blue_mid" if q < 1e-2 else "blue_lo"]


#: Criterion -> the short name printed in figures. Anything absent prints in full.
#: The full names are the ones in the score matrices and must not be edited here.
CRITERION_LABELS = {
    "Challenger evidentiary support": "Evidentiary support",
    "Structured rebuttal development": "Structured rebuttal",
    "Abstract or technical framing": "Abstract/technical framing",
    "Practical consequences and feasibility": "Practical consequences",
    "Exchange empirical framing": "Empirical framing",
    "Alternative proposal and guidance": "Alternative proposal",
    "Challenger experiential grounding": "Experiential grounding",
    # the 13 criteria the kappa/rho gate dropped (pointwise-29 panel only)
    "Reframing and normalization": "Reframing/normalization",
    "Meta-level epistemic challenge": "Meta-epistemic challenge",
    "Historical framing and legacy": "Historical framing",
    "Legal and institutional framing": "Legal/institutional framing",
    "Emotional appeal and empathy": "Emotional appeal",
    "Psychological framing and diagnosis": "Psychological framing",
}


def tex_escape(s: str) -> str:
    """Escape the characters that are special to LaTeX in the short labels we print."""
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


_BASE = {
    "font.family": "serif",
    # Sizes are FINAL PRINT sizes.
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_USETEX = {
    "text.usetex": True,
    # `times` is exactly what the paper loads: Times text, Computer Modern math. amsmath gives
    # \text{} inside math -- without it a hyphen in an $...$ label renders as a minus -- and
    # amssymb gives \mathbb.
    "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}\usepackage{amssymb}",
}

_NOTEX = {
    "text.usetex": False,
    # Times-compatible text face + Computer Modern math, matching `\usepackage{times}`.
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
}


def apply_style() -> None:
    """Install the paper style. Honors FIGURES_USETEX=1 when TeX exists."""
    mpl.rcParams.update(_BASE)
    mpl.rcParams.update(_USETEX if os.environ.get("FIGURES_USETEX") == "1" else _NOTEX)


def save(fig, out_dir, name: str) -> None:
    """Save ``name`` as vector PDF (the paper figure) and PNG (for on-screen review)."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=200, pad_inches=0.02)
    w, h = fig.get_size_inches()
    assert w <= FULL_W + 1e-6, f"{name}: width {w:.2f}in exceeds the {FULL_W}in text block"
    print(f"[fig] {name}.pdf  ({w:.2f} x {h:.2f} in)")
