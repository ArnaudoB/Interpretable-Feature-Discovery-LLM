"""ELLIPSE corpus loading, and the band-balanced selector every run uses.

The corpus (Crossley et al., 2023) ships ~6,500 English-language-learner essays scored for
overall holistic proficiency and for six analytic traits. It is carried in this repository
at ``raw/corpus/ELLIPSE_Final_github_train.csv`` so the sample regenerates offline; see that
directory's ``README.md`` for the corpus's own description and citation.

Two things about the sampling are deliberate and easy to get wrong:

* **Stratification is on the CSV's own holistic ``Overall`` column**, not on the mean of the
  six trait columns, which is a different number and would give a different sample.
* **The sample is not the corpus distribution.** ELLIPSE is unimodal at 3.0; the selection
  balances across the 1-5 bands so the pairwise comparisons span the full proficiency range,
  which lifts the rare extremes far above their natural frequency. The
  selected and pool histograms are both recorded in each run's ``graph/build_report.json``.

Selection is greedy and deterministic given the seed: cover every writing prompt first, then
fill by least-represented band and, within a band, least-represented prompt. Prompt coverage
is a hard constraint; band balance is best-effort, because bands 1 and 5 are scarce enough
to cap out (9 and 28 essays in the pool).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ELLIPSE_TRAIT_COLS = [
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
]


def load_candidates(csv_path: str) -> pd.DataFrame:
    """Read the ELLIPSE CSV → candidate pool with essay_id, full_text, word_count,
    6 traits, overall (CSV holistic ``Overall``), prompt, band (= round(Overall))."""
    raw = pd.read_csv(csv_path)
    id_col = "text_id" if "text_id" in raw.columns else "text_id_kaggle"
    required = [id_col, "full_text", "Overall", "prompt"] + ELLIPSE_TRAIT_COLS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}; got {list(raw.columns)}")
    raw = raw.dropna(subset=required).copy()

    out = pd.DataFrame(
        {
            "essay_id": raw[id_col].astype(str).to_numpy(),
            "full_text": raw["full_text"].astype(str).to_numpy(),
            "prompt": raw["prompt"].astype(str).to_numpy(),
            "overall": raw["Overall"].astype(np.float64).to_numpy(),
        }
    )
    out["word_count"] = out["full_text"].str.split().str.len().astype(np.int64)
    for col in ELLIPSE_TRAIT_COLS:
        out[col] = raw[col].astype(np.float64).to_numpy()
    # Integer score band for stratification & cross-band reporting (1..5).
    out["band"] = out["overall"].round().astype(int)
    return out.drop_duplicates(subset="essay_id").reset_index(drop=True)


def select_balanced_essays(pool: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """Select ``n_target`` essays: every prompt represented (≥1), score-band histogram
    soft-balanced. Greedy & deterministic:
      (A) one essay per prompt, choosing the currently-rarest band;
      (B) fill the rest by repeatedly drawing from the least-represented band, and within
          it from the least-represented prompt.
    Scarce bands (1, 5) cap out naturally — balance is best-effort, coverage is hard.
    """
    rng = random.Random(seed)
    pool = pool.reset_index(drop=True)
    band_of = pool["band"].tolist()
    prompt_of = pool["prompt"].tolist()
    idx_by_prompt: dict[str, list[int]] = defaultdict(list)
    idx_by_band: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pool)):
        idx_by_prompt[prompt_of[i]].append(i)
        idx_by_band[band_of[i]].append(i)
    bands = sorted(idx_by_band)

    selected: set[int] = set()
    band_count: Counter = Counter()
    prompt_count: Counter = Counter()

    # Phase A — cover every prompt.
    for pr in sorted(idx_by_prompt):
        cands = list(idx_by_prompt[pr])
        rng.shuffle(cands)
        pick = min(cands, key=lambda i: band_count[band_of[i]])
        selected.add(pick)
        band_count[band_of[pick]] += 1
        prompt_count[pr] += 1

    # Phase B — fill to n_target, balancing bands then spreading prompts.
    while len(selected) < n_target:
        avail_bands = [b for b in bands if any(i not in selected for i in idx_by_band[b])]
        if not avail_bands:
            break
        b = min(avail_bands, key=lambda bb: band_count[bb])
        pool_b = [i for i in idx_by_band[b] if i not in selected]
        rng.shuffle(pool_b)
        pick = min(pool_b, key=lambda i: prompt_count[prompt_of[i]])
        selected.add(pick)
        band_count[b] += 1
        prompt_count[prompt_of[pick]] += 1

    return pool.iloc[sorted(selected)].reset_index(drop=True)


def assert_graph_invariants(pairs: pd.DataFrame, table: pd.DataFrame, budgets, half: int) -> None:
    """Hard checks: nestedness, regularity, connectivity, no self-pairs, no duplicates."""
    for b1, b2 in zip(budgets, budgets[1:]):
        assert pairs.iloc[:b1].equals(pairs.iloc[:b2].iloc[:b1]), f"prefix {b1} not ⊂ {b2}"
    assert (pairs["essay_a_id"] != pairs["essay_b_id"]).all(), "self-pair present"
    keys = pairs.apply(lambda r: frozenset((r["essay_a_id"], r["essay_b_id"])), axis=1)
    assert keys.nunique() == len(pairs), "duplicate unordered pair present"
    assert pairs.apply(
        lambda r: r["shown_first"] in (r["essay_a_id"], r["essay_b_id"]), axis=1
    ).all(), "shown_first not a member of its pair"
    for _, row in table.iterrows():
        B, r = int(row["budget"]), int(row["budget"]) // half
        assert row["components"] == 1, f"budget {B}: {row['components']} components (need 1)"
        assert row["deg_min"] == row["deg_max"] == r, (
            f"budget {B}: degree not exactly {r} (min={row['deg_min']}, max={row['deg_max']})"
        )
