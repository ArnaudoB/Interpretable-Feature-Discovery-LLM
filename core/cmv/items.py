"""Build the (OP view, challenger reply) items for the comparative feature-discovery run.

Each *item* is one challenger's argument shown together with the OP view it answers:

    [OP VIEW]
    <op title>
    <op body>

    [CHALLENGE]
    <challenger's pair-path replies>   # OP interjections already excluded

Items are drawn from Tan's TRAINING matched pairs (`core.cmv.data.build_pair_units`).
Sampling `n_pairs` pairs and taking BOTH sides yields `2 * n_pairs` items with an exact 50/50
delta balance (one delta-winner + one loser per sampled pair), across `n_pairs` distinct OP
threads. Only pairs whose two challenges both fall within `[min_words, max_words]` are eligible,
so no text is truncated (the cap bounds batch context by *selection*, not by cutting arguments).

The delta label is carried on each item but is NOT shown to the judge; it is used only in the
downstream persuasion-outcome analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CONDITIONS


def _condition_columns(df: pd.DataFrame, condition: str):
    """Resolve a Tan text condition to ``(pos_col, neg_col, pos_wc, neg_wc)``.

    ``pos_wc``/``neg_wc`` are word-count Series aligned to ``df`` (read from the ``*_wc`` columns
    when present, e.g. ``pos_root_wc``; computed on the fly otherwise, e.g. for ``root_truncated``).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
    pos_col, neg_col = CONDITIONS[condition]

    def _wc(col):
        wcc = f"{col}_wc"
        return df[wcc] if wcc in df.columns else df[col].str.split().str.len()

    return pos_col, neg_col, _wc(pos_col), _wc(neg_col)


def format_item_text(op_title: str, op_body: str, challenge: str) -> str:
    """The single text blob the judge sees for one item."""
    op_title = (op_title or "").strip()
    op_body = (op_body or "").strip()
    challenge = (challenge or "").strip()
    return f"[OP VIEW]\n{op_title}\n{op_body}\n\n[CHALLENGE]\n{challenge}"


def build_items(
    pair_units: pd.DataFrame,
    n_pairs: int = 50,
    *,
    both_sides: bool = True,
    train_only: bool = True,
    min_words: int = 20,
    max_words: int = 1200,
    seed: int = 42,
    condition: str = "full_path",
) -> pd.DataFrame:
    """Return the item table (one row per item) for the comparison graph.

    ``condition`` selects the Tan text unit shown to the judge (``full_path`` = the concatenated
    challenger thread, the function default; ``root_reply`` = the single direct reply). Columns:
    ``item_id, text, delta, pair_id, side, op_author, op_title, path_wc``. ``item_id`` is a stable
    ``wa%04d`` id assigned in sorted order.
    """
    df = pair_units
    if train_only:
        df = df[df["train"].astype(bool)]

    # eligibility: BOTH sides' challenge word counts within [min_words, max_words] for the condition
    pos_col, neg_col, pos_wc, neg_wc = _condition_columns(df, condition)
    ok = pos_wc.between(min_words, max_words) & neg_wc.between(min_words, max_words)
    eligible = df[ok].reset_index(drop=True)
    if len(eligible) < n_pairs:
        raise ValueError(
            f"only {len(eligible)} eligible pairs for n_pairs={n_pairs} "
            f"(condition={condition}, min_words={min_words}, max_words={max_words})"
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), size=n_pairs, replace=False)
    sampled = eligible.iloc[np.sort(idx)].reset_index(drop=True)

    sides = [("pos", pos_col, 1), ("neg", neg_col, 0)] if both_sides else [("pos", pos_col, 1)]
    rows = []
    for _, r in sampled.iterrows():
        for side, col, delta in sides:
            challenge = r[col]
            rows.append(
                {
                    "text": format_item_text(r["op_title"], r["op_body"], challenge),
                    "delta": delta,
                    "pair_id": r["pair_id"],
                    "side": side,
                    "op_author": r["op_author"],
                    "op_title": r["op_title"],
                    "path_wc": int(len(str(challenge).split())),
                }
            )
    items = pd.DataFrame(rows)
    # stable ids in (pair_id, side) order so the set is deterministic
    items = items.sort_values(["pair_id", "side"], ascending=[True, False]).reset_index(drop=True)
    items.insert(0, "item_id", [f"wa{i:04d}" for i in range(len(items))])
    return items


def build_matched_items(
    pair_units: pd.DataFrame,
    n_pairs: int = 200,
    *,
    include_pair_ids=None,
    test_frac: float = 0.3,
    train_only: bool = True,
    min_words: int = 20,
    max_words: int = 1200,
    seed: int = 42,
    condition: str = "full_path",
    test_from_heldout: bool = False,
    n_test: int | None = None,
) -> pd.DataFrame:
    """Item table for the per-feature-BT experiment, with a train/test split at the PAIR level.

    Samples ``n_pairs`` eligible matched pairs (both sides within ``[min_words, max_words]`` on the
    chosen ``condition``), force-including ``include_pair_ids`` (e.g. the feature-discovery panel) so
    their exchanges recur. Each pair expands to 2 items (``pos`` = delta-winner, ``neg`` = loser).

    Two split modes:

    - ``test_from_heldout=False`` (default): draw everything from the TRAIN split; force-included
      pairs go to **train**; a ``test_frac`` fraction of the *additional* pairs go to **test**.
    - ``test_from_heldout=True`` (flagship): sample ``n_pairs`` anchor/train pairs from the TRAIN
      split and take the **test** set from the reserved HELDOUT split (all eligible heldout pairs, or
      a random ``n_test`` of them). This keeps the test set = Tan's heldout and train/test disjoint.

    Columns: ``item_id, text, delta, pair_id, side, op_author, split, path_wc``.
    """
    df = pair_units
    pos_col, neg_col, pos_wc, neg_wc = _condition_columns(df, condition)
    ok = pos_wc.between(min_words, max_words) & neg_wc.between(min_words, max_words)
    elig_all = df[ok].reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if test_from_heldout:
        is_tr = elig_all["train"].astype(bool)
        train_pool = elig_all[is_tr].reset_index(drop=True)
        test_pool = elig_all[~is_tr].reset_index(drop=True)
        forced = [p for p in (include_pair_ids or []) if p in set(train_pool["pair_id"])]
        pool = train_pool[~train_pool["pair_id"].isin(forced)].reset_index(drop=True)
        n_more = n_pairs - len(forced)
        if len(pool) < n_more:
            raise ValueError(f"only {len(pool)} extra eligible TRAIN pairs for n_more={n_more}")
        extra = pool.iloc[np.sort(rng.choice(len(pool), n_more, replace=False))]["pair_id"].tolist()
        train_ids = set(forced) | set(extra)
        test_ids = test_pool["pair_id"].tolist()
        if n_test is not None and n_test < len(test_ids):
            sel_idx = np.sort(rng.choice(len(test_pool), n_test, replace=False))
            test_ids = test_pool.iloc[sel_idx]["pair_id"].tolist()
        test_ids = set(test_ids)
    else:
        elig = elig_all[elig_all["train"].astype(bool)] if train_only else elig_all
        elig = elig.reset_index(drop=True)
        forced = [p for p in (include_pair_ids or []) if p in set(elig["pair_id"])]
        pool = elig[~elig["pair_id"].isin(forced)].reset_index(drop=True)
        n_more = n_pairs - len(forced)
        if len(pool) < n_more:
            raise ValueError(f"only {len(pool)} extra eligible pairs for n_more={n_more}")
        extra = pool.iloc[np.sort(rng.choice(len(pool), n_more, replace=False))]["pair_id"].tolist()
        n_te = round(test_frac * n_pairs) if n_test is None else n_test
        test_ids = set(rng.choice(extra, size=min(n_te, len(extra)), replace=False).tolist())
        train_ids = set(forced) | set(extra)

    chosen = train_ids | test_ids
    sel = elig_all[elig_all["pair_id"].isin(chosen)]
    rows = []
    for _, r in sel.iterrows():
        split = "test" if r["pair_id"] in test_ids else "train"
        for side, col, delta in [("pos", pos_col, 1), ("neg", neg_col, 0)]:
            rows.append(
                {
                    "text": format_item_text(r["op_title"], r["op_body"], r[col]),
                    "delta": delta,
                    "pair_id": r["pair_id"],
                    "side": side,
                    "op_author": r["op_author"],
                    "split": split,
                    "path_wc": int(len(str(r[col]).split())),
                }
            )
    items = (
        pd.DataFrame(rows)
        .sort_values(["pair_id", "side"], ascending=[True, False])
        .reset_index(drop=True)
    )
    items.insert(0, "item_id", [f"wa{i:04d}" for i in range(len(items))])
    return items
