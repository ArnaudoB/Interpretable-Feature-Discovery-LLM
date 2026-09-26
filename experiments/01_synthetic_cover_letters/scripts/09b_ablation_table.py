"""Step 9b: recovery metrics for the pointwise-canonical ablation (tab:main rows 6-9).

    python scripts/09b_ablation_table.py --config config.yaml

The companion of step 09. Step 09 scores the methods that discover their own features; this
scores the four arms that reuse OURS and change only the scoring mechanism, which is what
tab:main's "Ablations" block reports:

  PW-32 (1 rep)   Pairwise comparisons + LLM scoring                 rep 0 of the k32 frame
  PW-32 (n*)      Pairwise comparisons + LLM scoring--Multi-score    mean over n* k32 reps
  PW-21 (1 rep)   Reliable features + LLM scoring                    rep 0 of the k21 frame
  PW-21 (n*)      Reliable features + LLM scoring--Multi-score       mean over n* k21 reps

A separate table from `tab_results.csv` because these arms are not methods of their own --
they are ablations of one row of it.
Step 10 joins the two into the paper's nine-row tab:main.

COST. Each arm's total is the shared discovery spend (the comparative judging pass plus the one
taxonomy call -- these arms rate OUR criteria, so they cannot exist without it) plus their own
scoring spend, and the 1-rep arms carry 1/n* of the latter because they are one rep of the same
frame. Everything is on the Batch-API basis and the price table in ``core/judging/pricing.py``,
the same basis as every other cost in tab:main.

Semantic matching uses text-embedding-3-large (cached under .embed_cache/, so this runs offline
once seeded); theta and tau are read from config.metrics, as in step 09. Writes
results/tab_pointwise_ablation.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics
from core.embeddings import cosine_matrix
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials

CACHE = None  # set in main to results-relative .embed_cache

# (row label, arm, reps) -- "1" reads rep 0 off the long frame, "n*" the rep-mean wide frame.
ARMS = [
    ("PW-32 (1 rep)", "k32", 1),
    ("PW-32 (n*)", "k32", None),
    ("PW-21 (1 rep)", "k21", 1),
    ("PW-21 (n*)", "k21", None),
]

# The tab:main values of each row; the script exits non-zero if a recomputed cell drifts.
PAPER = {
    "PW-32 (1 rep)": dict(cost=3.67, K=32, SF=0.938, SM=0.763, SR=0.875, AF=0.406, LK=0.093),
    "PW-32 (n*)": dict(cost=6.55, K=32, SF=1.000, SM=0.860, SR=1.000, AF=0.438, LK=0.104),
    "PW-21 (1 rep)": dict(cost=3.60, K=21, SF=0.938, SM=0.792, SR=0.938, AF=0.667, LK=0.098),
    "PW-21 (n*)": dict(cost=6.54, K=21, SF=1.000, SM=0.868, SR=1.000, AF=0.667, LK=0.103),
}


def _cos(crit_text: dict, dials) -> pd.DataFrame:
    crits = list(crit_text)
    M = cosine_matrix([crit_text[c] for c in crits], [DIAL_TEXT[d] for d in dials], cache_dir=CACHE)
    return pd.DataFrame(M, index=crits, columns=dials)


def _scores(res: Path, arm: str, reps):
    """The arm's letter x criterion score frame: rep 0 if reps == 1, else the rep-mean."""
    d = res / f"pointwise_canonical_{arm}"
    if reps == 1:
        long = pd.read_parquet(d / "scores_long.parquet")
        return (
            long[long["rep"] == long["rep"].min()]
            .groupby(["letter_id", "dimension"])["score"]
            .mean()
            .unstack()
        )
    F = pd.read_parquet(d / "scores.parquet")
    F.index.name = "letter_id"
    return F


def _discovery_usd(res: Path) -> float:
    """The comparative pass + taxonomy call, i.e. what "Ours" costs and every PW arm inherits.

    Read from 06d's cost-match solve, which is the same figure tab:main prints for Ours and is
    what the pointwise reps were budget-matched against.
    """
    cm = json.loads((res / "rubric_stability/taxo_pool/cost_match.json").read_text())
    return float(cm["target_ours_usd"])


def _cost(res: Path, arm: str, reps, discovery: float) -> float:
    led = json.loads((res / f"pointwise_canonical_{arm}" / "cost_ledger.json").read_text())
    scoring = float(led["realized_usd_batch"])
    if reps == 1:
        scoring /= int(led["n_star"])  # one rep's share of the same frame
    return discovery + scoring


def main() -> int:
    global CACHE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="max allowed gap to the paper's printed value before this fails",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    CACHE = root / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)  # D = 16 (fluency excluded)
    G = build_ground_truth(ak)
    tax = json.loads((res / cfg["pointwise_canonical"]["criteria_source"]).read_text())
    ctext_all = {c["name"]: f"{c['name']}. {c['definition']}" for c in tax["criteria"]}
    disc = _discovery_usd(res)
    print(f"[ablation] D={len(dials)} relevant dials; theta={theta}, tau={tau}")
    print(f"[ablation] shared discovery cost ${disc:.4f} (comparative pass + taxonomy call)\n")

    rows = []
    for label, arm, reps in ARMS:
        F = _scores(res, arm, reps)
        ctext = {c: ctext_all[c] for c in F.columns if c in ctext_all}
        missing = [c for c in F.columns if c not in ctext_all]
        if missing:
            print(
                f"[warn] {label}: {len(missing)} scored columns absent from the taxonomy, "
                f"dropped: {missing[:3]}"
            )
        common = sorted(set(F.index) & set(G.index))
        sc = metrics.scorecard(
            F.loc[common, list(ctext)], G.loc[common, dials], _cos(ctext, dials), theta, tau
        )
        sc["method"] = label
        sc["cost"] = _cost(res, arm, reps, disc)
        rows.append(sc)

    df = pd.DataFrame(rows)[["method", "cost", "K", "SF", "SM", "SR", "AF", "LK"]].reset_index(
        drop=True
    )
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("=== tab:main, Ablations block ===")
    print(df.to_string(index=False))

    # Fail loudly on drift from the paper rather than writing a table nobody checks.
    bad = []
    for r in df.itertuples(index=False):
        for k, want in PAPER[r.method].items():
            got = getattr(r, k)
            if abs(got - want) > (0.5 if k == "K" else args.tolerance):
                bad.append(f"{r.method}.{k}: paper {want}, computed {got:.6f}")
    if bad:
        print("\n[ablation] DRIFT from the paper:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print(f"\n[ablation] all 28 cells agree with paper tab:main within {args.tolerance}")

    (res / "tab_pointwise_ablation.csv").write_text(df.to_csv(index=False))
    print("[ablation] wrote results/tab_pointwise_ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
