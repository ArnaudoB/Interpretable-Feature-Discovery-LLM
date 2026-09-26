"""Experiment 1 calibration numbers (sec:synthetic-calibration) must match the artifacts.

Step 12 recomputes ``results/calibration/metrics.csv`` from the fit; this checks the reported
ECE / Brier-skill values and cell counts, so a change to the model, the folds, the bins or the
bootstrap is caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "01_synthetic_cover_letters"
CAL = EXP / "results" / "calibration"
pytestmark = pytest.mark.skipif(
    not (CAL / "metrics.csv").exists(), reason="experiment 1 calibration artifacts not present"
)

#: (regime, gate, column) -> value reported in sec:synthetic-calibration, at 3 decimals.
PAPER = {
    ("oof", "citation", "n"): 84000,
    ("oof", "direction", "n"): 29030,
    ("oof", "direction", "ece"): 0.010,
    ("oof", "direction", "ece_lo"): 0.008,
    ("oof", "direction", "ece_hi"): 0.014,
    ("in_sample", "direction", "ece"): 0.019,
    ("oof", "citation", "ece"): 0.046,
    ("oof", "citation", "ece_lo"): 0.044,
    ("oof", "citation", "ece_hi"): 0.049,
    ("in_sample", "citation", "ece"): 0.030,
    ("oof", "citation", "bss_emp"): 0.069,
    ("oof", "citation", "bss_emp_lo"): 0.063,
    ("oof", "citation", "bss_emp_hi"): 0.076,
    ("in_sample", "citation", "bss_emp"): 0.168,
    ("oof", "citation", "bss_chance"): 0.266,
    ("oof", "citation", "bss_chance_lo"): 0.260,
    ("oof", "citation", "bss_chance_hi"): 0.272,
    ("oof", "direction", "bss_chance"): 0.548,
    ("oof", "direction", "bss_chance_lo"): 0.538,
    ("oof", "direction", "bss_chance_hi"): 0.557,
    ("oof", "direction", "bss_emp"): 0.545,
    ("oof", "direction", "bss_emp_lo"): 0.536,
    ("oof", "direction", "bss_emp_hi"): 0.554,
    ("in_sample", "direction", "bss_chance"): 0.623,
}


@pytest.mark.parametrize("key", sorted(PAPER), ids=lambda k: "/".join(map(str, k)))
def test_paper_calibration_number(key):
    tab = pd.read_csv(CAL / "metrics.csv").set_index(["regime", "gate"])
    regime, gate, col = key
    got, want = tab.loc[(regime, gate), col], PAPER[key]
    if col == "n":
        assert int(got) == want
    else:
        assert round(float(got), 3) == want, f"{key}: paper {want}, artifact {float(got):.6f}"


def test_the_calibration_is_on_the_papers_fit():
    """Step 12 refits on all 4,000 comparisons and must land on results/scores.parquet."""
    s = json.loads((CAL / "summary.json").read_text())
    assert s["full_refit_maxdiff_vs_scores_parquet"] < 1e-8
    assert s["K"] == 21 and s["T"] == 4000 and s["n"] == 200
    assert len(s["folds"]) == 5 and all(f["converged"] for f in s["folds"])
