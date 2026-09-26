"""Sampling and packing for the data-informed feature elicitation (the ``S`` arm).

The prior arm (``feature_prior_root``) asks the model to name dimensions having seen **nothing**.
This arm asks the same question having seen **exchanges**, and changes nothing else: the panel it
produces is scored by the byte-identical pointwise prompt the other PW arms use. So the S-vs-P
contrast isolates one factor — whether the elicitor looked at the data.

Everything the packing has to get right is a property of this corpus, which is why it lives here
rather than in the producer script:

- **The winner is always row 1.** ``items.parquet`` is sorted ``(pair_id, side desc)``, so the
  delta winner is the first row of all 1,200 pairs. Rendering pairs in table order would hand the
  elicitor a 100%-predictive positional cue. :func:`build_pair_blocks` therefore swaps A/B per
  pair under its own seed — deliberately NOT the seed-42 deck of
  ``pairtask.assign_signs``, which every LR design shares; correlating the presentation order with
  the modelling sign deck would be a silent dependence between the elicitation and the fit.
- **Only 914 distinct OP blobs back the 1,200 pairs.** 193 OPs serve up to 8 pairs each and 479
  pairs come from a repeated OP, so uniform pair sampling re-shows the same OP text and spends
  20-30% of a token budget on it. :func:`select_pairs` keeps at most one pair per distinct OP.
- **Pair length is heavy-tailed** (mean 1,213 tokens, p99 3,015, max 7,143). A greedy fill that
  *stops* at the first oversize pair would truncate the sample on one outlier, and a greedy fill
  that takes pairs in length order would bias the panel toward short replies — a real content
  risk, since ``#words`` alone predicts the outcome at 59.2%. :func:`pack_to_budget` skips
  oversize pairs and keeps going, and reports the realized length distribution so the bias is
  visible rather than assumed.
- **Chars per token is 4.37 here, not 5.1.** Back-solved from the shipped ``pointwise_prior``
  cache: a mean 8,458-char prompt measured 1,821.9 input tokens (whole-prompt ratio 4.642), and
  removing the 4,543-char boilerplate leaves the Reddit body near 4.37. Reddit markdown and URLs
  tokenize worse than the clean synthetic prose the 5.096 constant came from. The estimate is only
  used to *choose* a sample; the realized count comes back in ``usage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Chars per token for CMV body text — see the module docstring. Only used to size a sample.
CHARS_PER_TOKEN = 4.37

#: Seed for the A/B presentation order. Distinct from ``pairtask.SEED`` (42) on purpose.
PRESENTATION_SEED = 20260923

_OP_MARK, _CH_MARK = "[OP VIEW]", "[CHALLENGE]"


def split_item_text(text: str) -> tuple[str, str]:
    """``item.text`` -> ``(op_block, challenge_body)``.

    ``format_item_text`` writes ``"[OP VIEW]\\n...\\n\\n[CHALLENGE]\\n..."``, so the OP block is
    everything before the challenge marker and the reply is what follows it.
    """
    i = text.find(_CH_MARK)
    if i < 0:
        raise ValueError(f"item text has no {_CH_MARK} marker")
    return text[:i].rstrip("\n"), text[i + len(_CH_MARK) :].strip()


def build_pair_blocks(items: pd.DataFrame, *, seed: int = PRESENTATION_SEED) -> pd.DataFrame:
    """One row per matched pair: the shared OP plus its two replies in randomized order.

    Returns columns ``pair_id, split, op, reply_a, reply_b, a_is_winner, chars, est_tokens``.
    ``a_is_winner`` is recorded for the audit trail only — it must never reach a prompt.

    Raises if a pair's two items disagree about the OP text, which would mean the pairing is not
    what the matched-pair design claims.
    """
    if "delta" not in items.columns:
        raise ValueError("items must carry `delta` so the winner can be identified and then hidden")
    rng = np.random.default_rng(seed)
    rows = []
    for pair_id, d in items.groupby("pair_id", sort=True):
        if len(d) != 2:
            raise ValueError(f"pair {pair_id} has {len(d)} items, expected 2")
        parts = [split_item_text(t) for t in d["text"]]
        ops = {p[0] for p in parts}
        if len(ops) != 1:
            raise ValueError(f"pair {pair_id}: the two items carry different OP text")
        op = parts[0][0]
        win = d["delta"].to_numpy() == 1
        if win.sum() != 1:
            raise ValueError(f"pair {pair_id}: expected exactly one delta winner, got {win.sum()}")
        replies = [p[1] for p in parts]
        # swap so the winner is not systematically A (items.parquet sorts side desc -> pos first)
        flip = bool(rng.integers(2))
        order = [1, 0] if flip else [0, 1]
        a_is_winner = bool(win[order[0]])
        reply_a, reply_b = replies[order[0]], replies[order[1]]
        chars = len(op) + len(reply_a) + len(reply_b)
        rows.append(
            {
                "pair_id": pair_id,
                "split": d["split"].iloc[0],
                "op": op,
                "reply_a": reply_a,
                "reply_b": reply_b,
                "a_is_winner": a_is_winner,
                "chars": chars,
                "est_tokens": chars / CHARS_PER_TOKEN,
            }
        )
    out = pd.DataFrame(rows)
    # A balanced-enough shuffle; a large imbalance would still be a usable cue.
    frac = out.a_is_winner.mean()
    if not 0.4 <= frac <= 0.6:
        raise ValueError(f"A/B shuffle is unbalanced: winner is A in {frac:.1%} of pairs")
    return out


def select_pairs(
    blocks: pd.DataFrame, *, dedup_op: bool = True, seed: int = PRESENTATION_SEED
) -> pd.DataFrame:
    """Candidate pairs, at most one per distinct OP, ordered by length for stratified packing.

    Which pair survives an OP with several is chosen at random rather than by length, so the
    dedup does not itself bias the length distribution.
    """
    b = blocks
    if dedup_op:
        rng = np.random.default_rng(seed + 1)
        b = (
            b.assign(_r=rng.random(len(b)))
            .sort_values("_r")
            .drop_duplicates("op", keep="first")
            .drop(columns="_r")
        )
    return b.sort_values(["est_tokens", "pair_id"]).reset_index(drop=True)


def pack_to_budget(
    candidates: pd.DataFrame,
    budget_tokens: float,
    *,
    train_test_ratio: tuple[int, int] | None = (1, 2),
) -> pd.DataFrame:
    """Greatest stratified subset of ``candidates`` fitting ``budget_tokens``.

    Stratified sampling: sort by the spanning key (here pair length) and take ``np.linspace``
    indices, so the sample spans short to long rather than piling up at one end. ``train_test_ratio`` holds the pooled corpus's 400:800 proportion.

    Oversize pairs are skipped, never stopped on. Returns the selected rows with the packing
    order preserved.
    """
    if train_test_ratio is None:
        return _pack_one(candidates, budget_tokens)
    tr, te = train_test_ratio
    share = tr / (tr + te)
    a = _pack_one(candidates[candidates.split == "train"], budget_tokens * share)
    b = _pack_one(candidates[candidates.split == "test"], budget_tokens * (1 - share))
    return pd.concat([a, b]).sort_values(["est_tokens", "pair_id"]).reset_index(drop=True)


def _pack_one(pool: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Largest k whose linspace-spanning subset of ``pool`` fits ``budget`` tokens."""
    pool = pool.sort_values(["est_tokens", "pair_id"]).reset_index(drop=True)
    n = len(pool)
    if n == 0 or budget <= 0:
        return pool.iloc[:0]
    lo, hi, best = 1, n, 0
    while lo <= hi:  # monotone in k, so bisect
        k = (lo + hi) // 2
        if _span(pool, k).est_tokens.sum() <= budget:
            best, lo = k, k + 1
        else:
            hi = k - 1
    return _span(pool, best)


def _span(pool: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        return pool.iloc[:0]
    idx = np.unique(np.linspace(0, len(pool) - 1, k).round().astype(int))
    return pool.iloc[idx]


def render_exchanges(sample: pd.DataFrame) -> str:
    """The ``Exchange k:`` block handed to the elicitor. Carries no label, ever."""
    out = []
    for i, r in enumerate(sample.itertuples(index=False), 1):
        out.append(f"Exchange {i}:\n{r.op}\n\n[REPLY A]\n{r.reply_a}\n\n[REPLY B]\n{r.reply_b}")
    return "\n\n".join(out)


def sample_report(sample: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Coverage and the length-bias check the docstring promises."""
    q = (0.1, 0.5, 0.9)
    return {
        "n_pairs": int(len(sample)),
        "n_pool": int(len(pool)),
        "coverage": float(len(sample) / len(pool)) if len(pool) else 0.0,
        "est_tokens": float(sample.est_tokens.sum()),
        "by_split": {k: int(v) for k, v in sample.split.value_counts().items()},
        "len_quantiles_sample": [float(sample.est_tokens.quantile(x)) for x in q],
        "len_quantiles_pool": [float(pool.est_tokens.quantile(x)) for x in q],
        "winner_is_a": float(sample.a_is_winner.mean()),
    }
