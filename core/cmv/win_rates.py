"""Matched-pair win rates: *does the Δ-winning reply score higher on this criterion?*

The statistic behind ``fig:cmv-win-rates``. For one criterion and one heldout pair, compare the
frozen score of the reply that earned the Δ with its matched loser's; over the 800 pairs the win
rate is the fraction where the winner is higher, **with ties split** (a tie counts half).

Ties are not a technicality here. BT scores are continuous and never tie, so for a BT panel the
tie term is exactly zero. Pointwise ratings are integers on 0-10 and tie on 15-48% of pairs
depending on the criterion; counting a tie as a loss (the ``>`` shortcut) would drag every
pointwise criterion toward 0.5 by half its tie rate and is not comparable across arms.

Significance follows the same split: the point estimate splits ties, and the p-value is the
two-sided exact **sign test** over the non-tied pairs, which is the test the null "the criterion
is symmetric between winner and loser" actually implies. With no ties the two agree with the
plain binomial test on all n pairs, so BT numbers are unchanged.

Missing ratings are handled per criterion (pairwise-complete): a pair whose winner or loser has
no rating on a criterion drops from *that* criterion's rate only, and ``n`` records what was used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "panel_scores",
    "matched_pair_diffs",
    "win_rate_table",
    "length_reference",
    "table_from_configs",
    "bh_critical_margin",
]


def panel_scores(cells: list[Path]) -> pd.DataFrame:
    """Heldout scores for one panel, indexed by ``item_id``, one column per criterion.

    ``cells`` are experiment-cell directories whose ``results/scores_test.parquet`` is read in
    order. The first cell defines the panel; each later cell contributes the criteria named in
    its ``features.json`` ``"dropped"`` list (the gate-dropped criteria are rated in their own
    cell, which also re-rates a couple of kept ones as anchors -- those must not be taken), or,
    failing that list, whatever columns the earlier cells do not already carry.

    ``.attrs["source"]`` maps each criterion to the cell that rated it, so a pooled panel can
    still say where a column came from.

    The caller is responsible for checking that the cells share an item graph: ``item_id`` is
    positional per cell, so joining across graphs is silently wrong.
    """
    out: pd.DataFrame | None = None
    source: dict[str, str] = {}
    for cell in cells:
        cell = Path(cell)
        s = pd.read_parquet(cell / "results/scores_test.parquet").set_index("item_id")
        if out is None:
            out, take = s, list(s.columns)
        else:
            f = cell / "features.json"
            take = json.loads(f.read_text()).get("dropped") if f.exists() else None
            take = take or [c for c in s.columns if c not in out.columns]
            out = out.join(s[take])
        source.update({c: cell.name for c in take})
    assert out is not None, "panel_scores needs at least one cell"
    out.attrs["source"] = source
    return out


def matched_pair_diffs(scores: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Per-pair ``winner - loser`` score difference, one row per heldout pair, NaN where unrated."""
    te = items[items.split == "test"]
    w = te[te.delta == 1].set_index("pair_id")
    l = te[te.delta == 0].set_index("pair_id").reindex(w.index)
    assert not l.item_id.isna().any(), "a heldout pair is missing its delta==0 side"
    W = scores.loc[w.item_id].to_numpy(float)
    L = scores.loc[l.item_id].to_numpy(float)
    return pd.DataFrame(W - L, index=w.index, columns=scores.columns)


def win_rate_table(diffs: pd.DataFrame) -> pd.DataFrame:
    """Win rate (ties split), 95% Wald CI, exact sign-test p, BH q -- ascending by win rate.

    Columns: ``feature, win_rate, lo, hi, p, q, n, n_tie``. Sorted ascending so that plotting
    row *i* at *y = i* puts the most discriminative criterion at the top of the axes.
    """
    rows = []
    for f in diffs.columns:
        d = diffs[f].to_numpy(float)
        d = d[np.isfinite(d)]
        n = len(d)
        n_win, n_loss = int((d > 0).sum()), int((d < 0).sum())
        n_tie = n - n_win - n_loss
        wr = (n_win + 0.5 * n_tie) / n
        se = np.sqrt(wr * (1 - wr) / n)
        p = stats.binomtest(n_win, n_win + n_loss, 0.5).pvalue if n_win + n_loss else 1.0
        rows.append(
            dict(
                feature=f,
                win_rate=wr,
                lo=wr - 1.96 * se,
                hi=wr + 1.96 * se,
                p=float(p),
                n=n,
                n_tie=n_tie,
            )
        )
    tab = pd.DataFrame(rows)

    o = np.argsort(tab.p.values)  # Benjamini-Hochberg across the criteria
    q = tab.p.values[o] * len(tab) / (np.arange(len(tab)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tab.loc[tab.index[o], "q"] = np.clip(q, 0, 1)
    return tab.sort_values("win_rate").reset_index(drop=True)


def length_reference(items: pd.DataFrame) -> float:
    """ "The longer reply wins": the same statistic on raw word count, ties split."""
    te = items[items.split == "test"]
    w = te[te.delta == 1].set_index("pair_id")
    l = te[te.delta == 0].set_index("pair_id").reindex(w.index)
    d = w.path_wc.to_numpy(float) - l.path_wc.to_numpy(float)
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def table_from_configs(config_paths, repo: Path) -> tuple[pd.DataFrame, float, int]:
    """Win-rate table for the panel described by one or more cell configs.

    The single entry point both win-rate figures go through, so the forest plot and the compact
    family plot cannot drift apart. The first config supplies the item graph and the first block
    of criteria; each later config adds its gate-dropped criteria. They must all name the SAME
    graph -- ``item_id`` is positional per cell, so a cross-graph join lines up unrelated replies.

    Returns ``(table, length_only_reference, n_pairs)``; the table carries a ``cell`` column
    naming which cell rated each criterion.
    """
    import yaml

    cfgs = [yaml.safe_load((repo / c).read_text()) for c in config_paths]
    graphs = {c["graph"]["out_dir"] for c in cfgs}
    if len(graphs) > 1:
        raise ValueError(
            "cells do not share an item graph, so their item_ids are not "
            f"comparable: {sorted(graphs)}"
        )

    items = pd.read_parquet(repo / cfgs[0]["graph"]["out_dir"] / "items.parquet")
    scores = panel_scores([repo / Path(c).parent for c in config_paths])
    diffs = matched_pair_diffs(scores, items)
    tab = win_rate_table(diffs)
    tab["cell"] = tab.feature.map(scores.attrs["source"])
    return tab, length_reference(items), len(diffs)


def bh_critical_margin(tab: pd.DataFrame, n: int, alpha: float = 0.05) -> float:
    """Half-width of the "indistinguishable from chance" band, in win-rate units.

    Benjamini-Hochberg rejects the ``k*`` smallest p-values, the last of which had to clear
    ``alpha * k* / m``; that is the level every rejected test is effectively held to once the
    panel's ``m`` tests are accounted for. The margin is the displacement from 0.5 that meets it
    under the null at ``n`` pairs, ``z_{1 - alpha k*/(2m)} * sqrt(0.25 / n)``.

    The null SE is the no-tie one, ``sqrt(0.25/n)``, which is the widest a criterion's null can
    be: ties pull a split-tie win rate toward 0.5, so a criterion with many ties has a *narrower*
    null and can clear the band from closer in. The band is therefore conservative, and
    :func:`band_is_consistent` is what checks it against the per-criterion verdicts.
    """
    p = np.sort(tab.p.values)
    m = len(p)
    k = np.arange(1, m + 1)
    hit = p <= alpha * k / m
    if not hit.any():
        return float(stats.norm.isf(alpha / (2 * m)) * np.sqrt(0.25 / n))  # nothing rejected
    k_star = int(k[hit].max())
    return float(stats.norm.isf(alpha * k_star / (2 * m)) * np.sqrt(0.25 / n))


def band_is_consistent(tab: pd.DataFrame, half_width: float, alpha: float = 0.05) -> list[str]:
    """Criteria the band would classify against their own BH verdict (empty = the band is safe).

    A reader takes "outside the shaded band" to mean "significant", so a dot on the wrong side of
    it makes the figure lie. Callers should treat a non-empty return as a hard failure.
    """
    outside = (tab.win_rate - 0.5).abs() > half_width
    return sorted(tab.feature[outside != (tab.q < alpha)])
