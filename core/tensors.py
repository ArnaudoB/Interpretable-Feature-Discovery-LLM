"""Assemble the holistic model's observation tensors from canonical judgments.

The counterpart of :func:`core.fit.build_X`, which does the same job for the gated model.
Two steps, in order:

* :func:`filter_active_criteria` decides which canonicalized criteria are worth a column.
  A criterion's ``total_mentions`` counts the number of *comparisons* citing it at least
  once — comparison-level, not phrase-occurrence-level, so a judge naming the same
  construct twice in one rationale does not inflate it. Criteria whose share of all
  mentions falls below ``share_floor`` are dropped and the survivors are renumbered
  ``0..K-1`` in descending share.
* :func:`build_tensors` turns the judgments into ``w`` (which slot won), ``r`` (which
  criteria were cited), and ``pairs`` (integer item indices), which is exactly the triple
  :func:`core.model.holistic.fit` consumes.

The mention channel is dense: every ``(pair, criterion)`` cell is an observation, because a
non-citation is informative under the holistic likelihood. This is why kappa rarely binds
here — all T comparisons inform every column — where in the gated model a rarely cited
criterion is starved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.canonical import normalize_quality


def filter_active_criteria(
    cluster_assignments: pd.DataFrame,
    judgments: pd.DataFrame,
    share_floor: float = 0.05,
) -> pd.DataFrame:
    """Filter clusters to those exceeding the share floor.

    Args:
        cluster_assignments: output of ``cluster_phrases`` (one row per unique
            normalized phrase) with columns
            ``phrase, cluster_id, canonical_name, distance_to_centroid``.
        judgments: canonical judgment DataFrame (one row per comparison;
            ``qualities`` is a list[str] of raw rationale phrases).
        share_floor: minimum share a cluster must have to be active.

    Returns:
        DataFrame with columns ``cluster_id, canonical_name, n_members,
        total_mentions, share``, sorted by ``share`` descending. ``cluster_id``
        is *renumbered* 0..K-1 over the kept (active) clusters.
    """
    # Map normalized phrase → cluster_id (raw, before renumbering).
    phrase_to_cluster = dict(zip(cluster_assignments["phrase"], cluster_assignments["cluster_id"]))

    # For each comparison, derive the *set* of clusters cited.
    cluster_counts: dict[int, int] = {}
    for raw_list in judgments["qualities"]:
        if raw_list is None:
            continue
        cited_clusters: set[int] = set()
        for raw in raw_list:
            norm = normalize_quality(raw)
            if norm in phrase_to_cluster:
                cited_clusters.add(int(phrase_to_cluster[norm]))
        for cid in cited_clusters:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    if not cluster_counts:
        return pd.DataFrame(
            columns=["cluster_id", "canonical_name", "n_members", "total_mentions", "share"]
        )

    total = sum(cluster_counts.values())
    # n_members and canonical_name per raw cluster_id.
    member_counts = cluster_assignments.groupby("cluster_id").size().rename("n_members").to_dict()
    canonical_names = dict(
        cluster_assignments[["cluster_id", "canonical_name"]]
        .drop_duplicates(subset=["cluster_id"])
        .to_records(index=False)
    )

    rows = []
    for raw_cid, count in cluster_counts.items():
        share = count / total
        if share < share_floor:
            continue
        rows.append(
            {
                "raw_cluster_id": int(raw_cid),
                "canonical_name": str(canonical_names[int(raw_cid)]),
                "n_members": int(member_counts[int(raw_cid)]),
                "total_mentions": int(count),
                "share": float(share),
            }
        )

    out = pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)
    out.insert(0, "cluster_id", out.index.astype(int))
    # Keep raw mapping in case downstream wants to re-key.
    return out[
        ["cluster_id", "canonical_name", "n_members", "total_mentions", "share", "raw_cluster_id"]
    ]


def build_tensors(
    judgments: pd.DataFrame,
    active_criteria: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> dict:
    """Build the integer-indexed mention tensor + pair indices."""
    if "raw_cluster_id" not in active_criteria.columns:
        raise ValueError(
            "active_criteria must include 'raw_cluster_id' (the pre-renumber cluster_id "
            "from cluster_phrases) — produced by filter_active_criteria."
        )
    if "parse_failed" in judgments.columns and judgments["parse_failed"].any():
        raise ValueError("judgments must be pre-filtered for parse failures before build_tensors")

    # Raw cluster_id (from cluster_phrases) -> active column index.
    raw_to_col = {
        int(rc): int(k)
        for k, rc in zip(active_criteria["cluster_id"], active_criteria["raw_cluster_id"])
    }
    K = len(raw_to_col)

    # Normalized phrase -> raw cluster_id.
    phrase_to_raw = dict(zip(cluster_assignments["phrase"], cluster_assignments["cluster_id"]))

    # Essay index: sorted unique IDs across both slot columns.
    all_ids = pd.unique(
        pd.concat([judgments["essay_first_id"], judgments["essay_second_id"]], ignore_index=True)
    )
    all_ids = sorted(str(x) for x in all_ids)
    essay_index = {eid: i for i, eid in enumerate(all_ids)}
    n = len(all_ids)

    T = len(judgments)
    w = np.zeros(T, dtype=np.int64)
    r = np.zeros((T, K), dtype=np.int8)
    pairs = np.zeros((T, 2), dtype=np.int64)

    winner_arr = judgments["winner"].astype(str).to_numpy()
    a_id = judgments["essay_first_id"].astype(str).to_numpy()
    b_id = judgments["essay_second_id"].astype(str).to_numpy()
    qualities = judgments["qualities"].to_list()

    for t in range(T):
        pairs[t, 0] = essay_index[a_id[t]]
        pairs[t, 1] = essay_index[b_id[t]]
        w[t] = 1 if winner_arr[t] == "A" else 0
        if qualities[t] is None:
            continue
        cited_cols: set[int] = set()
        for raw in qualities[t]:
            norm = normalize_quality(raw)
            raw_cid = phrase_to_raw.get(norm)
            if raw_cid is None:
                continue
            col = raw_to_col.get(int(raw_cid))
            if col is not None:
                cited_cols.add(col)
        for col in cited_cols:
            r[t, col] = 1

    return {
        "w": w,
        "r": r,
        "pairs": pairs,
        "essay_index": essay_index,
        "K": K,
        "n": n,
        "T": T,
        "active_criteria": active_criteria,
    }
