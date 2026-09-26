"""Comparison graphs with forbidden edges: the CMV cells never pair a thread with itself.

Every CMV comparison graph forbids the edge between a thread's delta-winner and its own loser,
so each comparison is cross-topic (the discovery prompt tells the judge the two exchanges
concern different posters, and the BT anchor graph inherits the rule). The shared
:mod:`core.graph.matching` has no such option, because no other experiment forbids edges.

:func:`build_nested_pairs` below adds a ``forbid_edges`` argument. It composes
:func:`core.graph.matching.random_perfect_matching` exactly as the shared builder does, the only
difference being that the global ``used`` set starts out holding the forbidden edges, so the
matchings route around them. With ``forbid_edges=None`` it is draw-for-draw identical to
:func:`core.graph.matching.build_nested_pairs` (``tests/test_cmv_replay.py`` checks this).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.graph.matching import _edge, random_perfect_matching

__all__ = ["build_nested_pairs", "same_op_edges"]


def same_op_edges(pair_ids) -> set:
    """``(lo, hi)`` index edges joining the two sides of one matched pair.

    ``pair_ids`` is aligned with the ``essay_ids`` handed to :func:`build_nested_pairs`, so the
    edges are in the same 0..n-1 index space.
    """
    by_pair: dict = {}
    for i, pid in enumerate(pair_ids):
        by_pair.setdefault(pid, []).append(i)
    return {_edge(m[0], m[1]) for m in by_pair.values() if len(m) == 2}


def build_nested_pairs(
    essay_ids,
    bands,
    n_matchings: int,
    *,
    seed: int = 42,
    cross_band_bias: float = 0.0,
    prefer_edges: set | None = None,
    forbid_edges: set | None = None,
) -> pd.DataFrame:
    """Ordered union of ``n_matchings`` random matchings, never using a ``forbid_edges`` edge.

    Same contract and output columns as :func:`core.graph.matching.build_nested_pairs`
    (``pair_index, essay_a_id, essay_b_id, shown_first, band_a, band_b, score_gap``).
    Feasibility is the caller's: the graph must stay constructible at ``n_matchings``-regular
    once the forbidden edges are removed (trivially so for one forbidden edge per item).
    """
    essay_ids = list(map(str, essay_ids))
    n = len(essay_ids)
    bands = np.asarray(bands)
    if len(bands) != n:
        raise ValueError("bands must align with essay_ids")
    rng = np.random.RandomState(int(seed))
    nodes = list(range(n))

    used: set = set(forbid_edges) if forbid_edges else set()
    idx_pairs: list[tuple[int, int]] = []
    for _m in range(n_matchings):
        matching = random_perfect_matching(
            nodes,
            used,
            rng,
            bands=bands,
            cross_band_bias=cross_band_bias,
            prefer_edges=prefer_edges,
        )
        for e in matching:
            used.add(e)
        idx_pairs.extend(matching)

    # A/B assignment: independent, per-pair, frozen (build order is fixed), seeded.
    rng_ab = np.random.RandomState(int(seed) + 1)
    rows = []
    for k, (i, j) in enumerate(idx_pairs):
        a_id, b_id = essay_ids[i], essay_ids[j]
        ba, bb = float(bands[i]), float(bands[j])
        shown_first = a_id if rng_ab.randint(2) == 0 else b_id
        rows.append((k, a_id, b_id, shown_first, ba, bb, abs(ba - bb)))
    return pd.DataFrame(
        rows,
        columns=[
            "pair_index",
            "essay_a_id",
            "essay_b_id",
            "shown_first",
            "band_a",
            "band_b",
            "score_gap",
        ],
    )
