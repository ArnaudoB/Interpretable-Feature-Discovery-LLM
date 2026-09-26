"""Step 8c: run-variance bootstrap for the taxonomy-fusion arm (paper method 2a).

    python scripts/08c_bootstrap_taxo_arm.py --config config.yaml

The resampling unit is the ELICITATION: each replicate is one of the R taxonomy fusions from
``06b`` (a random E-subset of the 50 rubric elicitations), scored once by ``07b``. So this band
covers the variance that actually dominates this method -- which elicitations you happened to
run, and how the LLM fused them -- exactly as ``08b`` covers discovery variance for Ours.

Cost per replicate is booked on the batch basis: E elicitations + that replicate's own fusion
call + one scoring pass at that replicate's K.

Writes results/bootstrap_taxo_arm.json (the 2a band of tab:main-bands). No LLM calls
(everything is cached).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics
from core.embeddings import cosine_matrix
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials

BAND_KEYS = ["SF", "SM", "SR", "AF", "LK", "K", "USD"]


def _band(rows):
    out = {}
    for k in BAND_KEYS:
        v = np.array(
            [r[k] for r in rows if r.get(k) is not None and not np.isnan(float(r.get(k, np.nan)))],
            float,
        )
        if len(v):
            out[k] = (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--reps",
        type=int,
        default=1,
        help="scoring passes to bill per replicate (1 = method 2a, n* = 2b)",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    cache = root / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])
    E = int(cfg["judge"].get("m3_elicitations", 27))

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)

    pool = res / "rubric_stability" / "taxo_pool"
    boot = res / "pointwise_taxo_boot"
    cm = json.loads((pool / "cost_match.json").read_text())
    a, b_ = cm["per_spec_model"]["a"], cm["per_spec_model"]["b"]
    fusion_usd = {
        c["run"]: (c["input_tokens"] * 2.5 + c["output_tokens"] * 10.0) / 1e6 * 0.5
        for c in json.loads((pool / "cost_ledger.json").read_text())["calls"]
    }
    elic_rows = list(csv.DictReader(open(res / "rubric_stability" / "cost_ledger.csv")))
    elic_usd = np.array([float(r["usd"]) * 0.5 for r in elic_rows])  # sync ledger -> batch basis
    rng = np.random.default_rng(0)

    rows = []
    for sf in sorted(boot.glob("scores_b*.parquet")):
        b = int(sf.stem.split("b")[-1])
        run = json.loads((pool / f"run_{b:03d}.json").read_text())
        ctext = {f["name"]: f"{f['name']}. {f.get('description', '')}" for f in run["features"]}
        F = pd.read_parquet(sf)
        F.index.name = "letter_id"
        cols = [c for c in F.columns if c in ctext]
        common = sorted(set(F.index) & set(G.index))
        Fc = F.loc[common, cols]
        M = cosine_matrix([ctext[c] for c in cols], [DIAL_TEXT[d] for d in dials], cache_dir=cache)
        cos = pd.DataFrame(M, index=cols, columns=dials)
        row = metrics.scorecard(Fc, G.loc[common, dials], cos, theta, tau)
        K = int(run["n_features"])
        row["USD"] = (
            rng.choice(elic_usd, E, replace=True).sum()
            + fusion_usd.get(b, float(np.mean(list(fusion_usd.values()))))
            + args.reps * 200 * (a + b_ * K)
        )
        row["b"] = b
        rows.append(row)

    if not rows:
        raise SystemExit("no replicate score matrices — run 07b_score_taxo_boot.py collect first")

    band = _band(rows)
    print(
        f"[taxo-arm] B={len(rows)} replicates, E={E} elicitations each, "
        f"{args.reps} scoring pass(es), D={len(dials)}"
    )
    print("[taxo-arm] bands (mean [2.5, 97.5]):")
    for k in BAND_KEYS:
        if k in band:
            mu, lo, hi = band[k]
            print(f"  {k:4s} {mu:7.3f} [{lo:7.3f}, {hi:7.3f}]")

    out = res / "bootstrap_taxo_arm.json"
    out.write_text(
        json.dumps(
            {
                "B": len(rows),
                "E": E,
                "reps_billed": args.reps,
                "theta": theta,
                "tau": tau,
                "resampling_unit": "elicitation subset (which E of the 50 elicitations, and the LLM fusion)",
                "bands": {k: list(v) for k, v in band.items()},
                "per_replicate": rows,
            },
            indent=2,
        )
    )
    print(f"[taxo-arm] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
