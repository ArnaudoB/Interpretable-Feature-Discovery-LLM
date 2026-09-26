"""Shared pieces for the recovery heatmaps, so the panels are directly comparable.

Both `make_recovery_heatmap_comparative.py` and `make_recovery_heatmap_pointwise.py`
import from here. The point is that the two figures must use ONE dial order: if
each seriates its own columns, the same column position means a different dial in
each panel and the two cannot be read side by side.

Convention:
  * the canonical dial (column) order is derived ONCE from the comparative arm,
    which has a near one-to-one structure, and is then imposed on every panel;
  * within a panel, features (rows) are ordered so the panel is as diagonal as
    possible GIVEN that fixed column order -- each row sits at the column where
    it peaks, ties broken by peak strength.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP = Path(__file__).resolve().parent.parent  # the experiment directory

DIAL_LABEL = {
    "experience": "Experience",
    "education": "Education level",
    "school_prestige": "School prestige",
    "employer_prestige": "Employer prestige",
    "n_languages": "Language breadth",
    "n_launches": "Production launches",
    "skill_dsa_depth": "Algorithms depth",
    "skill_python": "Python",
    "skill_ai_agents_experience": "Applied AI",
    "skill_security_assessments": "Security assessments",
    "skill_vuln_triage_automation": "Vulnerability automation",
    "skill_agentic_tooling_design": "Agentic tooling",
    "side_project": "Side project",
    "open_source": "Open source",
    "security_cert": "Security certification",
    "ctf": "Capture-the-flag",
}


def _local_imports():
    sys.path[:0] = [str(EXP), str(EXP.parents[1])]  # experiment dir + repo root
    from dataset.groundtruth import build_ground_truth, relevant_dials  # noqa: PLC0415
    from core import metrics  # noqa: PLC0415

    return build_ground_truth, relevant_dials, metrics


def ground_truth():
    build_ground_truth, relevant_dials, _ = _local_imports()
    ak = pd.read_parquet(EXP / "results/answer_key.parquet").set_index("letter_id")
    return relevant_dials(ak), build_ground_truth(ak)


def corr(F: pd.DataFrame, G: pd.DataFrame, dials) -> pd.DataFrame:
    _, _, metrics = _local_imports()
    common = sorted(set(F.index) & set(G.index))
    return metrics.corr_matrix(F.loc[common], G.loc[common, dials])


def comparative_scores() -> pd.DataFrame:
    """The kept-criterion score matrix, with the operating point asserted."""
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    kmax = float(cfg["model"]["kappa_max"])
    rthr = float(cfg["model"]["rho_threshold"])
    fs = json.loads((EXP / "results/fit_summary.json").read_text())
    assert float(fs["kappa_max"]) == kmax and float(fs["rho_threshold"]) == rthr, (
        f"scores.parquet was fit at kappa_max={fs['kappa_max']}, rho={fs['rho_threshold']} "
        f"but config says {kmax}/{rthr} -- re-run scripts/04_fit_comparative.py"
    )
    return pd.read_parquet(EXP / "results/scores.parquet").set_index("letter_id")


def canonical_dial_order(dials, G) -> list[str]:
    """Column order shared by every recovery heatmap.

    Derived from the comparative arm: its rows are put in descending peak-|r|
    order, then each dial is placed at the row where it peaks. Dials no criterion
    peaks on keep their relative position at the end, so they never vanish.
    """
    r = corr(comparative_scores(), G, dials)
    r = r.iloc[np.argsort(-np.abs(r.values).max(axis=1))]
    A = np.abs(r.values)
    order = np.lexsort((-A.max(axis=0), A.argmax(axis=0)))
    return [dials[j] for j in order]


def order_rows_for_diagonal(r: pd.DataFrame) -> pd.DataFrame:
    """Given FIXED columns, order rows so the strong cells run down the diagonal.

    A plain argmax sort interleaves features that peak on nothing (their argmax is
    noise), which punches gaps in the diagonal. Instead assign each column ONE
    owning row by Hungarian on |r| -- so column j's owner sits at row j -- and send
    the leftover rows (more features than dials) to the bottom, strongest first.
    """
    from scipy.optimize import linear_sum_assignment

    A = np.abs(r.values)
    rows_i, cols_j = linear_sum_assignment(-A)  # maximize total |r|
    owner = {int(j): int(i) for i, j in zip(rows_i, cols_j)}
    diag = [owner[j] for j in range(A.shape[1]) if j in owner]
    rest = [i for i in range(A.shape[0]) if i not in set(diag)]
    rest.sort(key=lambda i: -A[i].max())
    return r.iloc[diag + rest]
