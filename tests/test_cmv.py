"""Regression tests for Experiment 2 (ChangeMyView) results.

Guards the split every arm is evaluated on, every arm's held-out count (exact: step 09 seeds
liblinear's global RNG once and fits the arms in a fixed order, so there is no tolerance), and
the win-rate summary behind fig:cmv-win-rates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "02_changemyview"
pytestmark = pytest.mark.skipif(
    not (EXP / "results" / "accuracy.csv").exists(), reason="experiment 2 results not present"
)

#: arm -> correct held-out pairs out of 800 (tab:cmv-predictive, tab:cmv-predictive-2);
#: zero-shot scores an order-inconsistent pair 0.5.
EXPECTED_COUNT = {
    "Emb. (full)": 519,
    "Tan": 510,
    "D + PW": 501,
    "Dr + PW": 497,
    "Emb.": 493,
    "Dr + BT": 492,
    "D + BT": 491,
    "BOW (full)": 490,
    "P + PW": 490,
    "P + BT": 489,
    "POS (full)": 484,
    "S + PW": 479,
    "#words": 474,
    "Zero--shot": 455.5,
}


def _acc():
    return pd.read_csv(EXP / "results" / "accuracy.csv").set_index("arm")


def _cmv_config():
    """Experiment 2's ``_config``, loaded under a unique name.

    Every experiment has a ``scripts/_config.py``, so a plain ``import _config`` resolves to
    whichever one another test imported first -- Experiment 3's, in suite order. These pass in
    isolation and fail in the suite unless the module is loaded explicitly.
    """
    import importlib.util

    for d in (str(EXP / "scripts"), str(EXP)):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("cmv_config", EXP / "scripts" / "_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arm,n_correct", sorted(EXPECTED_COUNT.items()))
def test_every_arm_reproduces_its_published_count(arm, n_correct):
    r = _acc().loc[arm]
    assert float(r.n_correct) == n_correct, f"{arm}: {r.n_correct} correct, expected {n_correct}"
    assert int(r.n_test) == 800


def test_the_split_is_the_flagship_one():
    C = _cmv_config()
    mp = C.matched_pairs()
    assert int(mp.train.sum()) == 400
    assert int((~mp.train).sum()) == 800
    assert mp.pair_id.is_unique


def test_item_ids_are_not_compared_across_graphs():
    """`items_for` must refuse cells that do not share a graph — the positional-id trap."""
    C = _cmv_config()
    C.items_for("pw_discovered", "pw_rep_a")  # same graph: fine
    with pytest.raises(ValueError, match="span graphs"):
        C.items_for("pw_discovered", "pw_discovered_full")


def test_win_rates_summary():
    wm = json.loads((EXP / "results" / "win_rates_meta.json").read_text())
    assert (wm["n_criteria"], wm["n_significant_bh05"]) == (29, 19)
    assert wm["all_significant_are_positive"]
