"""Split-half refits of the per-feature BT panel, for the noise control in ``dimensionality``.

A BT score aggregates many comparisons, so its measurement error is small but not zero, and the
only way to measure it is to build two scores that share no data. :func:`split_half_scores` does
that: the anchor comparisons are split in two, each half fits its own anchor panel, and every test
item is then frozen-scored twice -- once per half, from that half's anchors and that half of its
own test comparisons. The two resulting score matrices are independent given the latent scores, so
their cross-correlation (``dimensionality.cross_half_corr``) estimates the latent correlation and
their diagonal the reliability.

The fitting arithmetic is that of step 6's BT fit (same ``fit_feature_bt`` on the anchor graph,
same ``score_query_frozen`` against frozen anchors); :func:`fit_panel` is that fit as a function,
so a half is fitted the same way the published full panel was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bt import fit_feature_bt, score_query_frozen


def fit_panel(
    judg: pd.DataFrame,
    items: pd.DataFrame,
    names: list[str],
    *,
    ridge: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int = 2000,
):
    """(anchor scores, test scores, beta) from a judgment table -- the published fit, as a function.

    ``judg`` carries ``group`` (anchor/test), ``feature``, ``shown_first``, ``other``, ``winner``.
    Test items with no comparison for a feature in this subset get NaN.
    """
    split_map = dict(zip(items["item_id"], items["split"]))
    train_ids = items[items["split"] == "train"]["item_id"].tolist()
    test_ids = items[items["split"] == "test"]["item_id"].tolist()
    aidx = {i: k for k, i in enumerate(train_ids)}

    anchor = judg[judg["group"] == "anchor"]
    S_anchor = np.full((len(train_ids), len(names)), np.nan)
    beta: dict[str, float] = {}
    for k, f in enumerate(names):
        jk = anchor[anchor["feature"] == f]
        pairs = np.array([[aidx[a], aidx[b]] for a, b in zip(jk["shown_first"], jk["other"])])
        w = (jk["winner"].to_numpy() == "A").astype(float)
        s, bk = fit_feature_bt(pairs, w, len(train_ids), ridge=ridge, tol=tol, max_iter=max_iter)
        S_anchor[:, k] = s
        beta[f] = bk
    sA = pd.DataFrame(S_anchor, index=train_ids, columns=names)

    tj = judg[judg["group"] == "test"].copy()
    tj["q_is_first"] = tj["shown_first"].map(split_map).eq("test")
    tj["q_id"] = np.where(tj["q_is_first"], tj["shown_first"], tj["other"])
    tj["anchor_id"] = np.where(tj["q_is_first"], tj["other"], tj["shown_first"])
    tj["sign"] = np.where(tj["q_is_first"], 1.0, -1.0)
    tj["w"] = (tj["winner"].to_numpy() == "A").astype(float)
    long = sA.reset_index(names="item_id").melt(
        "item_id", var_name="feature", value_name="s_anchor"
    )
    tj = tj.merge(
        long, left_on=["anchor_id", "feature"], right_on=["item_id", "feature"], how="left"
    )

    S_test = np.full((len(test_ids), len(names)), np.nan)
    tpos = {i: k for k, i in enumerate(test_ids)}
    kpos = {f: k for k, f in enumerate(names)}
    for (qid, f), grp in tj.groupby(["q_id", "feature"], sort=False):
        S_test[tpos[qid], kpos[f]] = score_query_frozen(
            grp["sign"].to_numpy(),
            grp["s_anchor"].to_numpy(),
            grp["w"].to_numpy(),
            beta[f],
            ridge=ridge,
        )
    return sA, pd.DataFrame(S_test, index=test_ids, columns=names), beta


def split_half_scores(
    judg: pd.DataFrame, items: pd.DataFrame, names: list[str], *, seed: int = 0, **fit_kw
):
    """Two independent test-score matrices from disjoint halves of the comparisons.

    The split is by ``pair_index`` -- one comparison carries all features, so splitting by feature
    would leave the halves sharing exchanges' verdicts. Anchor and test comparisons are split
    separately, and within test, per query item, so each half keeps a comparable number of
    comparisons for every item (8 of 16 at the flagship's ``test_anchors``).
    """
    rng = np.random.default_rng(seed)
    split_map = dict(zip(items["item_id"], items["split"]))

    anc = judg[judg["group"] == "anchor"]
    a_pairs = anc["pair_index"].unique()
    a_perm = rng.permutation(a_pairs)
    a_half = {0: set(a_perm[: len(a_perm) // 2]), 1: set(a_perm[len(a_perm) // 2 :])}

    tst = judg[judg["group"] == "test"].copy()
    q_is_first = tst["shown_first"].map(split_map).eq("test")
    tst["q_id"] = np.where(q_is_first, tst["shown_first"], tst["other"])
    pair_q = tst.drop_duplicates("pair_index")[["pair_index", "q_id"]]
    t_half: dict[int, set] = {0: set(), 1: set()}
    for _, grp in pair_q.groupby("q_id", sort=False):
        p = rng.permutation(grp["pair_index"].to_numpy())
        t_half[0].update(p[: len(p) // 2])
        t_half[1].update(p[len(p) // 2 :])

    out = []
    for h in (0, 1):
        keep = ((judg["group"] == "anchor") & judg["pair_index"].isin(a_half[h])) | (
            (judg["group"] == "test") & judg["pair_index"].isin(t_half[h])
        )
        _, sT, _ = fit_panel(judg[keep], items, names, **fit_kw)
        out.append(sT)
    return out[0], out[1]
