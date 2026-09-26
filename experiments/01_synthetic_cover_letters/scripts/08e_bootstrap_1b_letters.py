"""Step 8e: letter-resampled bootstrap for the frozen-rubric pointwise arms (1b AND 2b).

    python scripts/08e_bootstrap_1b_letters.py --config config.yaml            # 1b (default)
    python scripts/08e_bootstrap_1b_letters.py --config config.yaml --arm 2b   # 2b

``--arm`` selects the arm and the output path is ``results/bootstrap_{arm}_letters.json``, so
``--arm 2b`` writes ``bootstrap_2b_letters.json`` (read by 10_paper_tables.py for the 2b band of
tab:main-bands) and the default writes the 1b band.
  * 1b: whole-dataset rubric (rubric_stability/full_dataset_rubric.json) x 20 passes.
  * 2b: canonical taxonomy fusion (rubric_stability/taxo_pool/run_000.json) x 6 passes.

Ours vs 1a is not a fair-budget comparison ($3.450 vs $0.259). 1b is: the same whole-dataset
rubric, with the budget difference spent on repeated scoring passes (20 reps, $3.419). This
script gives 1b a bootstrap so that comparison can carry a CI.

Design is deliberately SYMMETRIC with ``08d`` (frozen-taxonomy variant), and uses the SAME 50
letter resamples, so the two are paired:

  * both freeze their discovery step  — 1b's rubric is fixed, Ours' taxonomy is fixed;
  * both resample letters, drawn from gated/m1_boot_resamples.json;
  * both evaluate on the drawn multiset, duplicates included.

So the omitted variance component is the same on both sides (discovery), which is what makes the
paired difference interpretable. Note this is NOT comparable to 1a's bootstrap, which DOES
re-elicit the rubric per replicate.

Free: 1b's 20 scoring passes over all 200 letters are already in results/pointwise/.
Writes results/bootstrap_{arm}_letters.json  (bootstrap_1b_letters.json or bootstrap_2b_letters.json).
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

BAND_KEYS = ["SF", "SM", "SR", "AF", "LK", "K"]


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
        "--arm",
        default="1b",
        choices=["1b", "2b"],
        help="1b = whole-dataset rubric x 20 passes; "
        "2b = canonical taxonomy fusion (run_000) x 6 passes",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    cache = root / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)

    if args.arm == "1b":
        rub_f = res / "rubric_stability/full_dataset_rubric.json"
        sc_f = res / "pointwise/scores.parquet"
        led_f = res / "pointwise/cost_ledger.json"
    else:
        rub_f = res / "rubric_stability/taxo_pool/run_000.json"
        sc_f = res / "pointwise_taxo/scores.parquet"
        led_f = res / "pointwise_taxo/cost_ledger.json"
    n_reps = json.loads(led_f.read_text())["R"]
    rub = json.loads(rub_f.read_text())
    ctext = {f["name"]: f"{f['name']}. {f.get('description', '')}" for f in rub["features"]}
    S = pd.read_parquet(sc_f)  # mean over the arm's passes
    S.index.name = "letter_id"
    cols = [c for c in S.columns if c in ctext]
    S = S[cols]
    M = cosine_matrix([ctext[c] for c in cols], [DIAL_TEXT[d] for d in dials], cache_dir=cache)
    cos = pd.DataFrame(M, index=cols, columns=dials)

    resamp = json.loads((res / "gated/m1_boot_resamples.json").read_text())
    letter_ids, resamples = resamp["letter_ids"], resamp["resamples"]
    tag = f"{args.arm}-letters"
    print(
        f"[{tag}] {len(resamples)} letter resamples (shared with Ours/1a), "
        f"frozen rubric K={len(cols)}, {n_reps} scoring passes averaged"
    )

    rows = []
    for b, draw in enumerate(resamples):
        drawn = [letter_ids[i] for i in draw]  # multiset, duplicates kept
        drawn = [l for l in drawn if l in S.index]
        F = S.loc[drawn].reset_index(drop=True)
        Gc = G.loc[drawn, dials].reset_index(drop=True)
        row = metrics.scorecard(F, Gc, cos, theta, tau)
        row["b"] = b
        rows.append(row)

    band = _band(rows)
    print(f"\n[{tag}] B={len(rows)}; bands (mean [2.5, 97.5]):")
    for k in BAND_KEYS:
        if k in band:
            mu, lo, hi = band[k]
            print(f"  {k:4s} {mu:7.3f} [{lo:7.3f}, {hi:7.3f}]")

    out = res / f"bootstrap_{args.arm}_letters.json"
    out.write_text(
        json.dumps(
            {
                "B": len(rows),
                "arm": args.arm,
                "frozen_rubric": True,
                "reps_averaged": n_reps,
                "resampling_unit": "letter (shared with Ours 08d and 1a -> paired)",
                "pair_with": "results/bootstrap_ours_letters.json (frozen-taxonomy Ours: symmetric)",
                "caveat": "omits rubric-elicitation variance, exactly as the paired Ours run omits "
                "taxonomy variance. NOT comparable to 1a's bootstrap, which re-elicits.",
                "bands": {k: list(v) for k, v in band.items()},
                "per_replicate": rows,
            },
            indent=2,
        )
    )
    print(f"[{tag}] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
