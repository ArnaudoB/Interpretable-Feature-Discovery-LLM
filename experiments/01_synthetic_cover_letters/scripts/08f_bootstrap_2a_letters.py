"""Step 8f: letter-resampled bootstrap for method 2a, pairable with the comparative arm.

    python scripts/08f_bootstrap_2a_letters.py --config config.yaml

``08c`` bootstraps 2a over ELICITATION SUBSETS on a fixed corpus. That cannot be paired with the
comparative arm, whose replicates are indexed by letter draw -- there is no correspondence between
replicate b in one and replicate b in the other, and the existing elicitations cannot be recycled
onto a resampled corpus (an elicitation reads a 40-letter sample; the chance it falls entirely
inside a ~126-letter resample is (126/200)^40 ~ 1e-8). Re-eliciting per resample would cost ~$122.

So this script pairs them the only way that is free: **freeze discovery on both sides and resample
letters**. 2a's fusion is held at the canonical run (run_000, the first E=27 elicitations -- the
same slice 06_pool_rubrics.py takes), exactly as the paired comparative run holds its taxonomy
fixed. The omission is symmetric, which is what makes the difference interpretable.

Because run_000 is only one draw from the fusion distribution, the script also reports a
sensitivity sweep: the same paired difference recomputed with each of the 50 fusions frozen in
turn. Every replicate of every fusion was already scored on all 200 letters by 07b, so this is
free.

Writes results/bootstrap_2a_letters.json. Feeds tab:paired (b), fig:paired (b) and
App. app:paired-design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics
from core.embeddings import cosine_matrix
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials

METRICS = ["SF", "SM", "SR", "AF", "LK"]
BAND_KEYS = METRICS + ["K"]
CANONICAL = 0  # run_000 = the first E=27 elicitations


def _band(rows):
    out = {}
    for k in BAND_KEYS:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        if len(v):
            out[k] = (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def _rows_for_fusion(res, cache, bidx, draws, lids, G, dials, theta, tau):
    """Metrics per letter resample, with fusion `bidx` frozen."""
    S = pd.read_parquet(res / f"pointwise_taxo_boot/scores_b{bidx:03d}.parquet")
    S.index.name = "letter_id"
    run = json.loads((res / f"rubric_stability/taxo_pool/run_{bidx:03d}.json").read_text())
    ctext = {f["name"]: f"{f['name']}. {f.get('description', '')}" for f in run["features"]}
    cols = [c for c in S.columns if c in ctext]
    S = S[cols]
    M = cosine_matrix([ctext[c] for c in cols], [DIAL_TEXT[d] for d in dials], cache_dir=cache)
    cos = pd.DataFrame(M, index=cols, columns=dials)
    rows = []
    for b, dr in enumerate(draws):
        drawn = [lids[i] for i in dr if lids[i] in S.index]
        row = metrics.scorecard(
            S.loc[drawn].reset_index(drop=True),
            G.loc[drawn, dials].reset_index(drop=True),
            cos,
            theta,
            tau,
        )
        row["b"] = b
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    cache = root / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)
    resamp = json.loads((res / "gated/m1_boot_resamples.json").read_text())
    lids, draws = resamp["letter_ids"], resamp["resamples"]

    rows = _rows_for_fusion(res, cache, CANONICAL, draws, lids, G, dials, theta, tau)
    band = _band(rows)
    print(f"[2a-letters] B={len(rows)} letter resamples, fusion frozen at run_{CANONICAL:03d}")
    for k in BAND_KEYS:
        mu, lo, hi = band[k]
        print(f"  {k:4s} {mu:7.3f} [{lo:7.3f}, {hi:7.3f}]")

    # sensitivity: freeze each fusion in turn, recompute the paired difference vs Ours
    sens = {}
    if not args.no_sensitivity:
        oursf = {
            r["b"]: r
            for r in json.loads((res / "bootstrap_ours_letters_frozen.json").read_text())[
                "per_replicate"
            ]
        }
        n_f = len(sorted((res / "pointwise_taxo_boot").glob("scores_b*.parquet")))
        acc = {k: [] for k in METRICS}  # per-fusion MEAN difference
        pool = {k: [] for k in METRICS}  # every (fusion, letter-draw) difference
        for bi in range(n_f):
            rr = {
                r["b"]: r
                for r in _rows_for_fusion(res, cache, bi, draws, lids, G, dials, theta, tau)
            }
            common = sorted(set(oursf) & set(rr))
            for k in METRICS:
                d = [oursf[b][k] - rr[b][k] for b in common]
                acc[k].append(float(np.mean(d)))
                pool[k].extend(d)
        for k in METRICS:
            v = np.array(acc[k])
            pv = np.array(pool[k], float)
            fav = float(np.mean(v < 0) if k == "LK" else np.mean(v > 0))
            favp = float(np.mean(pv < 0) if k == "LK" else np.mean(pv > 0))
            lo, hi = np.percentile(pv, [2.5, 97.5])
            sens[k] = {
                "mean": float(v.mean()),
                "min": float(v.min()),
                "max": float(v.max()),
                "frac_favouring_ours": fav,
                # pooled over BOTH sources: a fusion draw x a corpus draw
                "pooled_mean": float(pv.mean()),
                "pooled_lo": float(lo),
                "pooled_hi": float(hi),
                "pooled_frac_favouring_ours": favp,
                "pooled_n": int(pv.size),
            }
        print(f"\n[2a-letters] sensitivity over {n_f} fusions (mean paired diff Ours - 2a):")
        for k, s in sens.items():
            print(
                f"  {k:4s} per-fusion mean {s['mean']:+.3f} range "
                f"[{s['min']:+.3f}, {s['max']:+.3f}] ({s['frac_favouring_ours']:.0%}) | "
                f"POOLED {s['pooled_mean']:+.3f} [{s['pooled_lo']:+.3f}, {s['pooled_hi']:+.3f}] "
                f"({s['pooled_frac_favouring_ours']:.0%} of {s['pooled_n']})"
            )

    out = res / "bootstrap_2a_letters.json"
    out.write_text(
        json.dumps(
            {
                "B": len(rows),
                "frozen_fusion": CANONICAL,
                "resampling_unit": "letter (shared -> paired)",
                "pair_with": "results/bootstrap_ours_letters_frozen.json (symmetric: both freeze discovery)",
                "caveat": "2a cannot be paired with Ours in the varying-discovery regime: its replicates "
                "are indexed by elicitation subset, not letter draw, and elicitations cannot be "
                "recycled onto a resampled corpus. Re-eliciting per resample would cost ~$122.",
                "bands": {k: list(v) for k, v in band.items()},
                "sensitivity_over_fusions": sens,
                "per_replicate": rows,
            },
            indent=2,
        )
    )
    print(f"\n[2a-letters] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
