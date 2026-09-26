"""Step 10: effective dimensionality of each feature set — fig:cmv-dimension-curve{,-full}. No LLM calls.

    python scripts/10_dimensionality.py            # ~15-20 min: 7 arms x 2 metrics x 200 replicates

The participation ratio PR = (sum lambda)^2 / sum lambda^2 of the eigenvalues of a feature set's
score correlation matrix counts the effective number of uncorrelated dimensions it resolves.
Three choices make the comparison across feature sets meaningful:

  a) **rank correlations** -- BT scores live on a logit scale and pointwise ratings on a coarse
     0-10 one, and a covariance PR weights a feature by its variance, so Spearman is used;
  b) **matched size** -- a bigger set trivially has more room, so PR is averaged over k-subsets
     (enumerated while C(K, k) <= 3,000, else 3,000 draws) and read at matched k;
  c) **noise correction** -- PR cannot tell signal from independent measurement error, so each
     set is measured twice independently (BT: refits on disjoint halves of the 32,000
     comparisons; PW: the second scoring run, features reshuffled) and the cross-measurement
     correlation is disattenuated before PR is taken.

Bands: 95% percentile bootstrap over the 1,600 test exchanges, the cross-measurement
correlation rebuilt inside each of 200 replicates. Everything is seeded (``--seed``, default 0,
the published run's), so the output is deterministic.

The rows are read and joined in a fixed way (test split only, left joins, file order), because
the bootstrap resamples rows by position.

Writes results/dimensionality/{summary,reliability,curves}.parquet and
results/dimensionality.json (the values the text quotes). Feeds Figures
`fig:cmv-dimension-curve` (left) and `fig:cmv-dimension-curve-full`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401
import _config as C
from core.cmv import bt_halves as BH
from core.cmv import dimensionality as DIM

#: arm -> (cells joined with "+", mode, repeat cells for a PW arm). The D_r cells hold exactly the
#: 16 gate-kept features (09_paired_lr asserts it); the D_r PW repeat is the 29-feature repeat
#: restricted to those 16 by name.
ARMS = {
    "D + BT": (("bt_discovered", "bt_dropped"), "bt", None),
    "D + PW": (("pw_discovered", "pw_dropped"), "pw", ("pw_rep_a", "pw_rep_b")),
    "P + BT": (("bt_prior",), "bt", None),
    "P + PW": (("pw_prior",), "pw", ("pw_prior_rep",)),
    "S + PW": (("pw_sample",), "pw", ("pw_sample_rep",)),
    "Dr + BT": (("bt_discovered",), "bt", None),
    "Dr + PW": (("pw_discovered",), "pw", ("pw_rep_a", "pw_rep_b")),
}


def _one(name: str) -> pd.DataFrame:
    return pd.read_parquet(C.cell(name) / "results" / "scores_test.parquet").set_index("item_id")


def test_scores(cells: tuple[str, ...]) -> pd.DataFrame:
    """Test-split scores; a second cell contributes its ``dropped`` list (or its new columns)."""
    if len(cells) == 1:
        return _one(cells[0])
    C.items_for(*cells)
    a, b = cells
    A, B = _one(a), _one(b)
    take = json.loads((C.cell(b) / "features.json").read_text()).get("dropped") or [
        c for c in B.columns if c not in A.columns
    ]
    return A.join(B[take])


def measure(arm: str, seed: int, n_boot: int):
    """One arm: summary row, per-feature reliabilities, and the noise-corrected curves."""
    cells, mode, rep = ARMS[arm]
    S = test_scores(cells)
    X = S.to_numpy(float)
    ok = np.isfinite(X).all(1)
    R = DIM.spearman_corr(X[ok])
    pr, _ = DIM.pr_from_corr(R)
    row = {
        "arm": arm,
        "K": X.shape[1],
        "n_items": int(ok.sum()),
        "mode": mode,
        "pr_rank": pr,
        "effrank_rank": DIM.effrank_from_corr(R),
    }
    names = list(S.columns)

    if mode == "bt":  # split-half refit on the cells' own judgments
        m = yaml.safe_load((C.cell(cells[0]) / "config.yaml").read_text())["model"]
        items = C.items_for(*cells)
        judg = pd.concat(
            [pd.read_parquet(C.cell(c) / "results" / "judgments.parquet") for c in cells]
        )
        judg = judg[judg["feature"].isin(names)].drop_duplicates(
            ["pair_index", "feature"], keep="first"
        )
        A, B = BH.split_half_scores(
            judg,
            items,
            names,
            seed=seed,
            ridge=float(m["ridge"]),
            tol=float(m["tol"]),
            max_iter=int(m["max_iter"]),
        )
        good = np.isfinite(A.to_numpy(float)).all(1) & np.isfinite(B.to_numpy(float)).all(1)
        MA, MB = A.to_numpy(float)[good], B.to_numpy(float)[good]
    else:  # cross-run: original ratings vs the reshuffled repeat
        S2 = test_scores(rep)[names]
        both = S.index.intersection(S2.index)
        A, B = S.loc[both].to_numpy(float), S2.loc[both].to_numpy(float)
        good = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        MA, MB = A[good], B[good]

    Rl, rel = DIM.cross_half_corr(MA, MB)
    pr_l, neg = DIM.pr_from_corr(Rl)
    row |= {
        "pr_latent": pr_l,
        "effrank_latent": DIM.effrank_from_corr(Rl),
        "median_reliability": float(np.median(rel)),
        "min_reliability": float(rel.min()),
        "neg_eig_mass": neg,
        "n_items_halves": int(good.sum()),
    }
    rels = [{"arm": arm, "feature": c, "reliability": float(r)} for c, r in zip(names, rel)]
    curves = []
    for metric in ("pr", "effrank"):
        mu = DIM.subset_curve(Rl, metric=metric, seed=seed)
        lo, hi = DIM.bootstrap_subset_curves(MA, MB, metric=metric, n_boot=n_boot, seed=seed)
        curves += [
            {"arm": arm, "metric": metric, "k": i + 1, "value": v, "ci_lo": lo[i], "ci_hi": hi[i]}
            for i, v in enumerate(mu)
        ]
    print(
        f"  {arm:<8} K={row['K']:>2}  PR raw {pr:5.2f} -> corrected {pr_l:5.2f}  "
        f"reliability median {row['median_reliability']:.3f}  neg-eig {neg:.3f}",
        flush=True,
    )
    return row, rels, curves


def quoted_values(summ: pd.DataFrame, rel: pd.DataFrame, curves: pd.DataFrame) -> dict:
    """The per-arm values the text quotes, from the summary and the curves."""
    res = {}
    at16 = curves[(curves.metric == "pr") & (curves.k == 16)].set_index("arm").value.round(2)
    res["pr_at_k16"] = at16.to_dict()

    s = summ.set_index("arm")
    shrink = (100 * (1 - s.pr_latent / s.pr_rank)).round(1)
    res["noise_correction_pct"] = shrink.to_dict()

    med = s.median_reliability.round(3)
    res["median_reliability"] = med.to_dict()

    er16 = curves[(curves.metric == "effrank") & (curves.k == 16)].set_index("arm").value
    res["effrank_at_k16"] = er16.round(2).to_dict()

    # Spearman-Brown step-up of the split-half median reliability, for the BT arms
    res["spearman_brown_bt"] = {
        a: round(2 * s.median_reliability[a] / (1 + s.median_reliability[a]), 2)
        for a in ("D + BT", "P + BT")
    }
    res["neg_eig_mass"] = s.neg_eig_mass.to_dict()
    res["order_at_k16"] = {
        "pr": at16.sort_values().index.tolist(),
        "effrank": er16.sort_values().index.tolist(),
    }
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=None, help="default: config dimensionality.n_boot")
    args = ap.parse_args()
    n_boot = args.boot or int(C.cfg()["dimensionality"]["n_boot"])
    out = C.RESULTS / "dimensionality"
    out.mkdir(parents=True, exist_ok=True)

    rows, rels, curves = [], [], []
    for arm in ARMS:
        r, rl, cv = measure(arm, args.seed, n_boot)
        rows.append(r)
        rels += rl
        curves += cv
    summ, rel, cur = pd.DataFrame(rows), pd.DataFrame(rels), pd.DataFrame(curves)
    summ.to_parquet(out / "summary.parquet", index=False)
    rel.to_parquet(out / "reliability.parquet", index=False)
    cur.to_parquet(out / "curves.parquet", index=False)

    res = quoted_values(summ, rel, cur)
    res |= {"seed": args.seed, "n_boot": n_boot}
    (C.RESULTS / "dimensionality.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n[dimensionality] wrote results/dimensionality/ and dimensionality.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
