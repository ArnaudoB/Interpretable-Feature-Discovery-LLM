"""Step 6f: the permutation floor and the stability sweep of the tracking threshold tau.

    python scripts/06f_tau_robustness.py --config config.yaml

Feeds App. app:metrics: the tracking threshold tau against the permutation floor under
letter-shuffled nulls, and the stability of the metrics for tau in [0.3, 0.5].

Two analyses, on the comparative arm ("Ours") AND the pointwise arms (1b whole-dataset rubric,
2b canonical taxonomy fusion), no API needed after first run (correlations are pure; the SR cos
matrices are cached):

1. PERMUTATION FLOOR — letter-shuffled null. Shuffle G's letter labels B times; each shuffle
   destroys any feature<->dial relation, so the tracking statistic max_k |r_{k,d}| measures what
   |r| reaches by CHANCE. Report the null distribution of the per-dial tracking statistic — its
   mean/95th pct are the "floor" tau must clear. (Same statistic as the permutation null of App. app:metrics:
   statistic = per-dial max over criteria of |corr|, GT rows permuted.)

2. TAU SWEEP — recompute SF/SM/SR/AF/LK over tau in {0.30, 0.35, 0.40, 0.45, 0.50}, per arm.
   "Stable" = the metrics (and the recovered-dial set) barely move across the band.

Writes results/tau_robustness.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics
from core.embeddings import cosine_matrix
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials

TAUS = [0.30, 0.35, 0.40, 0.45, 0.50]


def _load_arms(res):
    """{arm: (feature_score_df, {feature_name: text})} for Ours / 1b / 2b."""
    arms = {}
    tax = json.loads((res / "gated/llm_taxonomy/taxonomy.json").read_text())
    ct = {c["name"]: f"{c['name']}. {c['definition']}" for c in tax["criteria"]}
    arms["Ours"] = (pd.read_parquet(res / "scores.parquet").set_index("letter_id"), ct)
    full = json.loads((res / "rubric_stability/full_dataset_rubric.json").read_text())
    ft = {f["name"]: f"{f['name']}. {f['description']}" for f in full["features"]}
    m2 = pd.read_parquet(res / "pointwise/scores.parquet")
    m2.index.name = "letter_id"
    arms["1b"] = (m2, ft)
    run0 = json.loads((res / "rubric_stability/taxo_pool/run_000.json").read_text())
    pt = {f["name"]: f"{f['name']}. {f.get('description', '')}" for f in run0["features"]}
    s2 = pd.read_parquet(res / "pointwise_taxo/scores.parquet")
    s2.index.name = "letter_id"
    arms["2b"] = (s2, pt)
    return arms


def _floor(F, G, B, seed):
    rng = np.random.default_rng(seed)
    per_dial_max = []
    for _ in range(B):
        Gs = G.iloc[rng.permutation(len(G))]
        Gs.index = G.index
        per_dial_max.append(metrics.corr_matrix(F, Gs).abs().max(axis=0).values)
    pooled = np.array(per_dial_max).ravel()
    return {
        "per_dial_max_mean": float(pooled.mean()),
        "per_dial_max_p95": float(np.percentile(pooled, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    res = Path(args.config).parent / "results"
    cache = Path(args.config).parent / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    dtxt_of = lambda dials: [DIAL_TEXT[d] for d in dials]

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    Gall = build_ground_truth(ak)
    arms = _load_arms(res)

    out = {
        "B": args.B,
        "seed": args.seed,
        "tau_used": float(cfg["metrics"]["tau"]),
        "taus": TAUS,
        "arms": {},
    }
    for name, (F, textmap) in arms.items():
        cols = [c for c in F.columns if c in textmap]
        common = sorted(set(F.index) & set(Gall.index))
        Fa = F.loc[common, cols].astype(float)
        G = Gall.loc[common, dials]
        M = cosine_matrix([textmap[c] for c in cols], dtxt_of(dials), cache_dir=cache)
        cos = pd.DataFrame(M, index=cols, columns=dials)

        floor = _floor(Fa, G, args.B, args.seed)
        sweep = []
        for t in TAUS:
            sc = metrics.scorecard(Fa, G, cos, theta, t)
            n_rec = int((metrics.corr_matrix(Fa, G).abs().max(axis=0) > t).sum())
            sweep.append(
                {
                    "tau": t,
                    "dials_recovered": n_rec,
                    **{k: sc[k] for k in ["SF", "SM", "SR", "AF", "LK"]},
                }
            )
        spread = {
            k: float(max(r[k] for r in sweep) - min(r[k] for r in sweep))
            for k in ["SF", "SM", "SR", "AF", "LK"]
        }
        out["arms"][name] = {
            "K": len(cols),
            "permutation_floor": floor,
            "tau_sweep": sweep,
            "sweep_spread": spread,
        }

        print(f"\n=== {name} (K={len(cols)}) ===")
        print(
            f"  permutation floor: per-dial max|r| mean {floor['per_dial_max_mean']:.4f}, "
            f"p95 {floor['per_dial_max_p95']:.4f}   (tau=0.4 is "
            f"{0.4 / floor['per_dial_max_mean']:.1f}x the mean)"
        )
        print(f"  {'tau':>5} {'#rec':>5} {'SF':>6} {'SM':>6} {'SR':>6} {'AF':>6} {'LK':>6}")
        for r in sweep:
            print(
                f"  {r['tau']:5.2f} {r['dials_recovered']:5d} {r['SF']:6.3f} {r['SM']:6.3f} "
                f"{r['SR']:6.3f} {r['AF']:6.3f} {r['LK']:6.3f}"
            )
        print("  max-min over [0.3,0.5]: " + "  ".join(f"{k} {v:.3f}" for k, v in spread.items()))

    (res / "tau_robustness.json").write_text(json.dumps(out, indent=2))
    print("\n[tau] wrote results/tau_robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
