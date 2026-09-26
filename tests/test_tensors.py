"""Tests for core.tensors — share-floor filtering and observation-tensor assembly.

The phrase -> criterion map comes from the LLM taxonomy step, so only the filtering and
tensor-assembly functions are exercised here.
"""

from __future__ import annotations

import pandas as pd

from core.tensors import build_tensors, filter_active_criteria


def test_filter_share_floor():
    """Two clusters above floor, one below; verify only above-floor survive."""
    # Cluster 0 → cited in 70 comparisons, cluster 1 → 25, cluster 2 → 3.
    cluster_assignments = pd.DataFrame(
        {
            "phrase": ["pa0", "pa1", "pb0", "pc0"],
            "cluster_id": [0, 0, 1, 2],
            "canonical_name": ["pa0", "pa0", "pb0", "pc0"],
            "distance_to_centroid": [0.0, 0.05, 0.0, 0.0],
        }
    )
    qualities = [["pa0"]] * 70 + [["pb0"]] * 25 + [["pc0"]] * 3
    judgments = pd.DataFrame({"qualities": qualities})
    out = filter_active_criteria(cluster_assignments, judgments, share_floor=0.05)
    # 70 / (70+25+3) = 0.713 → keep
    # 25 / 98 = 0.255 → keep
    # 3  / 98 = 0.031 → drop
    assert len(out) == 2
    assert set(out["canonical_name"]) == {"pa0", "pb0"}
    # Bigger share gets cluster_id=0 (renumbered).
    assert out.iloc[0]["canonical_name"] == "pa0"
    assert "raw_cluster_id" in out.columns


# ---------------------------------------------------------------------------
# build_tensors — shapes + indexing
# ---------------------------------------------------------------------------


def test_build_tensors_shapes():
    judgments = pd.DataFrame(
        {
            "pair_id": [f"p{i:04d}" for i in range(8)],
            "essay_first_id": ["e1", "e2", "e3", "e1", "e4", "e5", "e2", "e3"],
            "essay_second_id": ["e2", "e3", "e1", "e4", "e5", "e1", "e5", "e2"],
            "winner": ["A", "B", "A", "B", "A", "A", "B", "A"],
            "qualities": [
                ["q0"],
                ["q1"],
                ["q0", "q2"],
                ["q3"],
                ["q0"],
                ["q1", "q2"],
                ["q3"],
                ["q0"],
            ],
            "parse_failed": [False] * 8,
        }
    )
    cluster_assignments = pd.DataFrame(
        {
            "phrase": ["q0", "q1", "q2", "q3"],
            "cluster_id": [10, 11, 12, 13],
            "canonical_name": ["q0", "q1", "q2", "q3"],
            "distance_to_centroid": [0.0] * 4,
        }
    )
    # Mock active criteria: keep clusters 10, 11, 13.
    active = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "canonical_name": ["q0", "q1", "q3"],
            "n_members": [1, 1, 1],
            "total_mentions": [5, 3, 3],
            "share": [0.5, 0.3, 0.3],
            "raw_cluster_id": [10, 11, 13],
        }
    )
    t = build_tensors(judgments, active, cluster_assignments)
    assert t["w"].shape == (8,)
    assert t["r"].shape == (8, 3)
    assert t["pairs"].shape == (8, 2)
    assert t["n"] == 5
    assert t["K"] == 3
    assert t["T"] == 8
    # Verify the essay_index covers e1..e5.
    assert set(t["essay_index"].keys()) == {"e1", "e2", "e3", "e4", "e5"}
    # Winner labels correct: 'A' → 1, 'B' → 0.
    assert (t["w"] == [1, 0, 1, 0, 1, 1, 0, 1]).all()
    # Mention check: row 0 mentioned only q0 → r[0] == [1, 0, 0].
    assert (t["r"][0] == [1, 0, 0]).all()
    # Row 2 mentioned q0 and q2; q2 is NOT active → only q0 column on.
    assert (t["r"][2] == [1, 0, 0]).all()
