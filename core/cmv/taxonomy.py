"""Rendering of the cited-dimension listing that the discovery taxonomy prompt embeds."""

from __future__ import annotations

import pandas as pd


def render_dimension_listing(g: pd.DataFrame) -> str:
    """The ``[[INDEXED_DIMENSION_LIST]]`` block: one ``[i] name — description (cited n)`` line
    per row of the dimension table, in table order."""
    return "\n".join(
        f"[{i}] {r.name} — {r.description} (cited {int(r.n)})"
        for i, r in enumerate(g.itertuples(index=False))
    )
