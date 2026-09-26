"""Shared figure style (fonts, sizes, colours, colormaps) for the paper's figures.

Every figure script imports ``apply_style()`` and the width constants from here;
figures do not override font or size rcParams individually.

THE MATH FONT: Computer Modern, not STIX
----------------------------------------
The paper loads ``\\usepackage{iclr2026_conference,times}``, not ``mathptmx``.
``times`` restyles the TEXT font only and leaves MATH in Computer Modern, so the
target is **Times text with Computer Modern math** -- and a figure whose math is
Times-derived does not match the body it sits in.

The default path is therefore matplotlib's own text engine with **STIXGeneral**
(the Times-metric-compatible text face) and ``mathtext.fontset="cm"``, because the
paper body's math is Computer Modern.

``FIGURES_USETEX=1`` switches to real TeX, loading the same ``times`` package
for the same reason. It is NOT the default even where TeX is installed: usetex
rejects a raw ``%`` in a label, which several of these figures use (the paired
panels annotate win rates as ``96%``), and it silently drops them.
"""

from __future__ import annotations

import os

import matplotlib as mpl

# The ICLR text block. Figures are rendered at final print size.
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

_BASE = {
    "font.family": "serif",
    # Sizes are FINAL PRINT sizes (figures are rendered at their width in the paper).
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
    """Save a vector PDF (the paper figure) and a PNG for on-screen review."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=200, pad_inches=0.02)
    w, h = fig.get_size_inches()
    assert w <= FULL_W + 1e-6, f"{name}: width {w:.2f}in exceeds the {FULL_W}in text block"
    print(f"[fig] {name}.pdf  ({w:.2f} x {h:.2f} in)")
