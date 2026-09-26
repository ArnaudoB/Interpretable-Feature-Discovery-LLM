"""Pairwise comparison graph via an ordered union of random matchings.

The comparison design is the union of ``d`` edge-disjoint perfect matchings of
the items: every item enters exactly ``d`` comparisons and ``N = nd/2`` pairs are
judged. The build unit is a **random perfect matching** of the ``n`` items:
``n/2`` disjoint pairs, every item appearing exactly once, so each matching adds
+1 to every item's degree. After ``r`` matchings the graph is exactly
``r``-regular, connected and duplicate-free.

The pair list is built in matching order, so a prefix ending on a matching
boundary is itself a regular graph on the same items. That is what
:func:`verify_budgets` checks.

Guarantees, all seeded/deterministic:
  * no self-pairs;
  * no duplicate unordered pair anywhere in the graph (global ``used`` set);
  * regular at every matching boundary;
  * A/B position frozen per pair (a pair in the 3k prefix has the same A/B at 10k).

Pure data wrangling — no LLM calls, no cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.graph._unionfind import UnionFind


def _edge(i: int, j: int) -> tuple[int, int]:
    """Canonical (lo, hi) key for an unordered pair."""
    return (i, j) if i < j else (j, i)


def _repair(a, remaining: set, pairs: list, local: set, used: set, rng) -> bool:
    """One-step augmenting swap when node ``a`` has no available partner left.

    ``a`` is already removed from ``remaining`` and every node still in
    ``remaining`` forms a used/duplicate edge with it. Find an existing pair
    ``(x, y)`` such that ``a``–``x`` is allowed and ``y`` can re-pair with some
    free node ``r``; rewire to ``(a, x)`` + ``(y, r)``. Returns True on success.
    """
    rem = sorted(remaining)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    for pi in order:
        c, d = pairs[pi]
        for x, y in ((c, d), (d, c)):
            ax = _edge(a, x)
            if a == x or ax in used or ax in local:
                continue
            for r in rem:
                if r == y:
                    continue
                yr = _edge(y, r)
                if yr in used or yr in local:
                    continue
                local.discard(pairs[pi])
                pairs[pi] = ax
                local.add(ax)
                pairs.append(yr)
                local.add(yr)
                remaining.discard(r)
                return True
    return False


def random_perfect_matching(
    nodes,
    used: set,
    rng,
    *,
    bands=None,
    cross_band_bias: float = 0.0,
    prefer_edges: set | None = None,
    max_whole_retries: int = 200,
) -> list[tuple[int, int]]:
    """One perfect matching over ``nodes`` (even count): ``n/2`` disjoint pairs.

    None is a self-pair and none is in ``used`` (a set of ``(lo, hi)`` keys, NOT
    mutated here). Greedy random with a one-step swap repair; whole-matching retry
    on a dead end. ``cross_band_bias`` ∈ [0,1] gently prefers cross-band partners
    (different ``bands[i]``) — 0 = pure random. ``prefer_edges`` (a set of ``(lo, hi)``
    keys), when given, restricts a node's candidate partners to those forming a preferred
    edge whenever at least one is still available — used to reuse an existing graph's edges
    inside a fresh nested matching decomposition (falls back to random when none remain).
    All choices iterate sorted structures so the result is reproducible from ``rng``.
    """
    nodes = list(nodes)
    if len(nodes) % 2:
        raise ValueError("perfect matching needs an even number of nodes")
    for _attempt in range(max_whole_retries):
        remaining = set(nodes)
        pairs: list[tuple[int, int]] = []
        local: set = set()
        ok = True
        while remaining:
            a = sorted(remaining)[rng.randint(len(remaining))]
            remaining.discard(a)
            cands = [
                b for b in sorted(remaining) if _edge(a, b) not in used and _edge(a, b) not in local
            ]
            if cands and prefer_edges is not None:
                pref = [b for b in cands if _edge(a, b) in prefer_edges]
                if pref:
                    cands = pref
            if cands and cross_band_bias > 0.0 and bands is not None:
                cross = [b for b in cands if bands[b] != bands[a]]
                if cross and rng.random() < cross_band_bias:
                    cands = cross
            if cands:
                b = cands[rng.randint(len(cands))]
                remaining.discard(b)
                e = _edge(a, b)
                pairs.append(e)
                local.add(e)
            elif not _repair(a, remaining, pairs, local, used, rng):
                ok = False
                break
        if ok and not remaining:
            return pairs
    raise RuntimeError(
        f"failed to build a perfect matching after {max_whole_retries} retries "
        f"(|used|={len(used)}); graph may be too dense for this n"
    )


def build_nested_pairs(
    essay_ids,
    bands,
    n_matchings: int,
    *,
    seed: int = 42,
    cross_band_bias: float = 0.0,
    prefer_edges: set | None = None,
) -> pd.DataFrame:
    """Ordered union of ``n_matchings`` random matchings → one nested pair list.

    Args:
        essay_ids: sequence of ``n`` unique essay-id strings (used as-is for index
            0..n-1; sort upstream if you want a canonical mapping).
        bands: per-essay integer/float score band, aligned to ``essay_ids``.
        n_matchings: number of matchings to concatenate (graph becomes this-regular).
        seed: RNG seed; matching construction and A/B assignment use seed and seed+1.
        cross_band_bias: mild cross-band pairing preference (default 0 = pure random).
        prefer_edges: optional set of ``(lo, hi)`` node-index edges to reuse preferentially
            inside the matchings (default ``None`` = pure random, unchanged behaviour). The
            keys are in the SAME 0..n-1 index space as the returned pairs, i.e. relative to
            the ``essay_ids`` order passed here.

    Returns:
        DataFrame in build order with columns ``pair_index, essay_a_id, essay_b_id,
        shown_first, band_a, band_b, score_gap``. ``essay_a_id``/``essay_b_id`` are the
        canonical (i<j) members; ``shown_first`` is the id placed in prompt position A.
    """
    essay_ids = list(map(str, essay_ids))
    n = len(essay_ids)
    bands = np.asarray(bands)
    if len(bands) != n:
        raise ValueError("bands must align with essay_ids")
    rng = np.random.RandomState(int(seed))
    nodes = list(range(n))

    used: set = set()
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


def verify_budgets(pairs: pd.DataFrame, essay_ids, budgets) -> tuple[pd.DataFrame, dict]:
    """Per-budget graph diagnostics on each prefix ``pairs.iloc[:budget]``.

    Returns ``(table, gaps_by_budget)`` where ``table`` has columns
    ``budget, n_matchings, components, fiedler, deg_min, deg_max, deg_mean,
    mean_score_gap`` and ``gaps_by_budget`` maps each budget to its array of
    per-pair ``|band_a − band_b|`` for the histogram.
    """
    essay_ids = list(map(str, essay_ids))
    idx = {eid: i for i, eid in enumerate(essay_ids)}
    n = len(essay_ids)
    half = n // 2
    rows, gaps_by_budget = [], {}
    for B in budgets:
        sub = pairs.iloc[:B]
        ai = sub["essay_a_id"].astype(str).map(idx).to_numpy()
        bi = sub["essay_b_id"].astype(str).map(idx).to_numpy()
        uf = UnionFind(n)
        deg = np.zeros(n, dtype=int)
        A = np.zeros((n, n), dtype=float)
        for a, b in zip(ai, bi):
            a, b = int(a), int(b)
            uf.union(a, b)
            deg[a] += 1
            deg[b] += 1
            A[a, b] = A[b, a] = 1.0
        lap = np.diag(deg.astype(float)) - A
        fiedler = float(np.linalg.eigvalsh(lap)[1])  # 2nd-smallest eigenvalue
        gaps = sub["score_gap"].to_numpy()
        rows.append(
            {
                "budget": int(B),
                "n_matchings": int(B // half),
                "components": int(uf.n_components()),
                "fiedler": fiedler,
                "deg_min": int(deg.min()),
                "deg_max": int(deg.max()),
                "deg_mean": float(deg.mean()),
                "mean_score_gap": float(gaps.mean()),
            }
        )
        gaps_by_budget[int(B)] = gaps
    return pd.DataFrame(rows), gaps_by_budget
