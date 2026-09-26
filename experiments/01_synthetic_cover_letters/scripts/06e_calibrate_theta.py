"""Step 6e: calibrate the semantic-acceptance threshold theta (paper: eta).

    python scripts/06e_calibrate_theta.py --config config.yaml

theta is "the 95th percentile of the alignment of the (dial, feature) pairs the Hungarian
assignment REJECTS" -- the empirical background of unrelated pairs (core.metrics.calibrate_theta,
which pools core.metrics.rejected_cosines over one or more cos matrices).

Calibrated over the DISCOVERED feature set of each of the three method families, pooled:
  * Ours    -- all comparative criteria from the taxonomy (results/gated/llm_taxonomy/taxonomy.json).
               The FULL discovered set (K=33), NOT the kappa/rho-kept 21: theta is a property of
               name-alignment at discovery, independent of the downstream reliability filter.
  * 1a/1b   -- the whole-dataset pointwise rubric (rubric_stability/full_dataset_rubric.json).
  * 2a/2b   -- the canonical taxonomy-fusion rubric (rubric_stability/taxo_pool/run_000.json).

Reproduces the value set in core.metrics.THETA and config.metrics.theta (0.406); eta in
App. app:metrics. Needs OPENAI_API_KEY only for the first run (feature/dial embeddings are then
cached under .embed_cache/).

Writes results/theta_calibration.json.
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
from dataset.groundtruth import DIAL_TEXT, relevant_dials


def _cos(textmap, dtxt, dials, cache):
    cc = list(textmap)
    M = cosine_matrix([textmap[c] for c in cc], dtxt, cache_dir=cache)
    return pd.DataFrame(M, index=cc, columns=dials)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--q", type=float, default=95.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    cache = root / ".embed_cache"

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    dtxt = [DIAL_TEXT[d] for d in dials]

    tax = json.loads((res / "gated/llm_taxonomy/taxonomy.json").read_text())
    ours = {c["name"]: f"{c['name']}. {c['definition']}" for c in tax["criteria"]}  # all 33
    full = json.loads((res / "rubric_stability/full_dataset_rubric.json").read_text())
    r1b = {f["name"]: f"{f['name']}. {f['description']}" for f in full["features"]}
    run0 = json.loads((res / "rubric_stability/taxo_pool/run_000.json").read_text())
    r2b = {f["name"]: f"{f['name']}. {f.get('description', '')}" for f in run0["features"]}

    mats = {
        "Ours(K=%d)" % len(ours): _cos(ours, dtxt, dials, cache),
        "1a/1b(K=%d)" % len(r1b): _cos(r1b, dtxt, dials, cache),
        "2a/2b(K=%d)" % len(r2b): _cos(r2b, dtxt, dials, cache),
    }

    theta = metrics.calibrate_theta(list(mats.values()), args.q)
    cfg_theta = float(cfg["metrics"]["theta"])
    print(f"[theta] pooled over: {', '.join(mats)}")
    for k, m in mats.items():
        print(f"   {k:14s} alone -> {metrics.calibrate_theta([m], args.q):.4f}")
    print(f"[theta] q={args.q}  ->  theta = {theta:.6f}  (rounds to {round(theta, 3)})")
    print(
        f"[theta] config/hardcoded value = {cfg_theta}   match@3dp = {round(theta, 3) == cfg_theta}"
    )

    out = res / "theta_calibration.json"
    out.write_text(
        json.dumps(
            {
                "theta": theta,
                "rounded_3dp": round(theta, 3),
                "q": args.q,
                "config_theta": cfg_theta,
                "pooled_over": list(mats),
                "note": "95th pct of Hungarian-rejected (dial,feature) cosines, pooled over the three "
                "method families' DISCOVERED feature sets (Ours uses the full K=33, pre kappa/rho).",
            },
            indent=2,
        )
    )
    print(f"[theta] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
