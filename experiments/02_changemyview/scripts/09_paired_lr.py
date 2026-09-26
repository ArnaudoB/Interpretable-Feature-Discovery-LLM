"""Step 9: the paired persuasion task — every arm of tab:cmv-predictive{,-2}. No LLM calls.

    python scripts/09_paired_lr.py

Given a feature set's per-exchange scores, the task is an L1 logistic regression on the
within-pair score difference ``sign * (s_winner - s_loser)``, no intercept, with the sign dealt
from a balanced seed-42 deck so chance is exactly 50%. C = 1/lambda is chosen by 5-fold
GroupKFold cross-validation grouped by original poster over 13 log-spaced values, fitted on the
400 anchor pairs and evaluated once on the 800 heldout pairs.

Score arms are refitted here, PW arms on the mean of their two scoring runs:

  D  + PW / BT   the 29 challenger-side discovered features
  Dr + PW / BT   the 16 of them the kappa/rho gate keeps (derived and checked below)
  P  + PW / BT   the 16 prior-elicited features
  S  + PW        the 42 sample-elicited features (see raw/cells/README.md)

The other arms are read from the per-pair predictions their own drivers saved: Tan, #words,
BOW (full), POS (full), Emb., Emb. (full), and the order-averaged zero-shot verdicts.

**Determinism.** liblinear shuffles coordinates, drawing from numpy's GLOBAL RNG when no
``random_state`` is passed. The published table was produced by seeding that RNG once
(``np.random.seed(42)``) and then fitting the score arms in one fixed order, which is what this
script does -- so it reproduces every published count exactly. Changing the order, adding an
arm before the others, or passing ``random_state`` moves individual arms by about one pair.

**Intervals.** One 4,000 x 800 resample matrix of the heldout pairs (seed 0) is shared by every
arm, as the caption says.

Writes results/accuracy.csv (one row per arm) and results/predictions/<arm>.parquet (per-pair
correctness).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C
from core.cmv import pairtask as PT
from core.cmv.designs import fit_select, signed_diff_design
from core.cmv.paired import average_runs, boot_indices, marginal_cis

warnings.filterwarnings("ignore")  # sklearn's penalty= deprecation flood

#: Arms read from saved per-pair predictions: arm -> (cell, file, arm key, dims).
SAVED_ARMS = {
    "Tan": ("bt_discovered", "pair_predictions.parquet", "all-Tan", 3205),
    "#words": ("bt_discovered", "pair_predictions.parquet", "#words", 1),
    "BOW (full)": ("bt_full", "pair_predictions.parquet", "BOW", None),
    "POS (full)": ("bt_full", "pair_predictions.parquet", "POS", None),
    "Emb.": ("embeddings", "pair_predictions_n400.parquet", "emb", 3072),
    "Emb. (full)": ("embeddings", "pair_predictions_n3411.parquet", "emb", 3072),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else C.RESULTS
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    task = C.cfg()["task"]
    mp = C.matched_pairs()
    train = mp.train.to_numpy()
    groups = mp[task["group_by"]].to_numpy()
    signs = PT.assign_signs(mp, seed=int(task["seed"]))
    y = (signs == 1).astype(int)
    test_ids = mp.loc[~mp.train, "pair_id"].to_numpy()
    cs = C.c_grid()
    kept = C.reliable_features()
    print(
        f"[lr] {int(train.sum())} anchor + {len(test_ids)} heldout pairs; grouped by "
        f"{task['group_by']}; D_r = {len(kept)} gate-kept features"
    )

    correct, meta = {}, {}
    np.random.seed(int(task["liblinear_global_seed"]))  # liblinear's RNG; see the docstring
    for arm, (run_a, run_b) in C.SCORE_ARMS.items():
        S = C.panel(run_a)
        if run_b is not None:
            S = average_runs(S, C.panel(run_b))
        cols = list(S.columns)
        if arm.startswith("Dr"):
            assert set(cols) == set(kept), arm
        r = fit_select(signed_diff_design(mp, S, cols, signs), y, train, groups, cs=cs)
        correct[arm] = pd.Series((r["pred"] == r["y_test"]).astype(float), index=test_ids)
        nan = float(S.loc[list(mp.winner_id) + list(mp.loser_id)].isna().to_numpy().mean())
        meta[arm] = {
            "dims": len(cols),
            "nnz": r["nnz"],
            "best_C": r["best_C"],
            "lambda": 1.0 / r["best_C"],
            "nan_cells": nan,
            "source": "refitted here: "
            + " + ".join(
                "+".join(C.cfg()["cells"][c] for c in run) for run in (run_a, run_b) if run
            )
            + (" (two-run mean)" if run_b else ""),
        }

    for arm, (cell, fn, key, dims) in SAVED_ARMS.items():
        d = pd.read_parquet(C.cell(cell) / "results" / fn).query("arm == @key").set_index("pair_id")
        correct[arm] = (d.pred == d.y).astype(float).reindex(test_ids)
        meta[arm] = {"dims": dims, "source": f"{C.cfg()['cells'][cell]}/results/{fn}[{key}]"}
    # BOW/POS (full): dims and lambda live in the arm-sparsity summary, not the predictions.
    sp = pd.read_parquet(C.cell("bt_full") / "results" / "arm_sparsity.parquet").set_index("arm")
    for arm, key in (("BOW (full)", "BOW"), ("POS (full)", "POS")):
        meta[arm] |= {
            "dims": int(sp.loc[key, "n_features"]),
            "best_C": float(sp.loc[key, "best_C"]),
            "lambda": 1.0 / float(sp.loc[key, "best_C"]),
        }
    tan = pd.read_parquet(C.cell("bt_discovered") / "results" / "arm_sparsity.parquet").set_index(
        "arm"
    )
    for arm, key in (("Tan", "all-Tan"), ("#words", "#words")):
        meta[arm] |= {
            "best_C": float(tan.loc[key, "best_C"]),
            "lambda": 1.0 / float(tan.loc[key, "best_C"]),
        }
    for arm, fn in (("Emb.", "summary_n400"), ("Emb. (full)", "summary_n3411")):
        e = (
            pd.read_parquet(C.cell("embeddings") / "results" / f"{fn}.parquet")
            .set_index("arm")
            .loc["emb"]
        )
        meta[arm] |= {"best_C": float(e.best_C), "lambda": 1.0 / float(e.best_C)}

    # Zero-shot: the judge queried in both presentation orders; a pair scores 1, 0.5 or 0.
    v = pd.read_parquet(C.cell("zeroshot") / "results" / "verdicts.parquet")
    zs = v.groupby("pair_id")["picked_winner"].mean().reindex(test_ids)
    ref = json.loads((C.cell("zeroshot") / "results" / "summary.json").read_text())[
        "order_averaged_acc"
    ]
    assert abs(zs.mean() - ref) < 1e-12, "order-averaged zero-shot drifted from its summary"
    correct["Zero--shot"] = zs.astype(float)
    meta["Zero--shot"] = {
        "dims": None,
        "source": "zeroshot_pairwise/results/verdicts.parquet (both orders averaged per pair)",
    }

    Cm = pd.DataFrame(correct)
    assert Cm.notna().all().all() and len(Cm) == 800, "every arm must cover the 800 heldout pairs"
    marg = marginal_cis(Cm, boot_indices(len(Cm), int(task["n_boot"]), seed=0))
    rows = []
    for r in marg.itertuples():
        rows.append(
            {
                "arm": r.arm,
                "acc": r.acc,
                "ci_lo": r.lo,
                "ci_hi": r.hi,
                "n_correct": float(r.n_correct),
                "n_test": len(Cm),
                **meta[r.arm],
            }
        )
        pd.DataFrame({"pair_id": test_ids, "correct": Cm[r.arm].to_numpy()}).to_parquet(
            out / "predictions" / f"{C.slug(r.arm)}.parquet", index=False
        )
    df = (
        pd.DataFrame(rows).sort_values("acc", ascending=False, kind="stable").reset_index(drop=True)
    )
    df.to_csv(out / "accuracy.csv", index=False)
    for _, r in df.iterrows():
        lam = f"lam={r['lambda']:>7.4g}" if pd.notna(r["lambda"]) else " " * 11
        print(
            f"  {r.arm:<12} {lam}  {r.n_correct:>5.1f}/800  {100 * r.acc:6.2f}%  "
            f"[{100 * r.ci_lo:.2f}, {100 * r.ci_hi:.2f}]"
        )
    print(f"\n[lr] wrote results/accuracy.csv and results/predictions/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
