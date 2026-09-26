"""Step 5: per-criterion identifiability, and the reliable panel the refit uses.

    python scripts/05_diagnostics.py --run all

No API spend. Reads the full-K fit, computes rho and kappa from the sandwich covariance
(core/model/holistic/diagnostics.py), selects the criteria passing both gates, and writes a reduced
tensor set so step 6 (``04_fit.py --reliable``) can refit on the reliable subset. Feeds the
reliability screen of App. ``sec:extension-appendix``. Needs ``fit/cov.npy`` from step 4.

Dropping criteria genuinely changes the model: the winner predictor is ``sum_k Delta_k``,
so a refit on K' < K criteria is a different (smaller) model, not a projection of the
first. That is why we refit rather than slice the fitted s.

Writes:
  fit/criterion_diagnostics.parquet   rho, kappa, gamma+/-SE, citation counts, keep flag
  pipeline_reliable/tensors.npz       r restricted to the kept columns
  pipeline_reliable/active_criteria.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import resolve  # noqa: E402

from core.model.holistic.diagnostics import per_criterion_diagnostics  # noqa: E402
from core.model.holistic.inference import std_errors  # noqa: E402


def screen(run, args) -> int:
    print(f"\n=== {run.name}")
    cfg = run.cfg
    res = run.results
    pipe = Path(args.pipeline_dir) if args.pipeline_dir else res / "pipeline"
    fit_dir = Path(args.fit_dir) if args.fit_dir else res / "fit"
    out = res / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["diagnostics"]
    rho_min = float(args.rho_min if args.rho_min is not None else dcfg["rho_min"])
    kappa_max = float(args.kappa_max if args.kappa_max is not None else dcfg["kappa_max"])

    npz = np.load(pipe / "tensors.npz")
    w, r, pairs = npz["w"], npz["r"], npz["pairs"]
    n, K, T = int(npz["n"]), int(npz["K"]), int(npz["T"])
    with open(fit_dir / "fit.pkl", "rb") as f:
        fit = pickle.load(f)
    cov = np.load(fit_dir / "cov.npy")
    active = pd.read_parquet(pipe / "active_criteria.parquet")

    ses = std_errors(fit, cov)
    d = per_criterion_diagnostics(fit, w, r, pairs, ses["s"])
    keep = (d.rho > rho_min) & (d.kappa < kappa_max)

    tab = pd.DataFrame(
        {
            "cluster_id": active["cluster_id"].to_numpy(),
            "canonical_name": active["canonical_name"].to_numpy(),
            "n_cited_pairs": d.n_cited_pairs,
            "citation_rate": d.n_cited_pairs / T,
            "std_s": np.sqrt(d.var),
            "noise_sd": np.sqrt(d.noise),
            "rho": d.rho,
            "kappa": d.kappa,
            "lambda_min": d.lambda_min,
            "lambda_max": d.lambda_max,
            "gamma": fit.gamma,
            "gamma_se": ses["gamma"],
            "sign_corr": fit.sign_corrs,
            "keep": keep,
        }
    ).sort_values("rho", ascending=False)
    tab.to_parquet(fit_dir / "criterion_diagnostics.parquet", index=False)

    print(f"gates: rho > {rho_min}  and  kappa < {kappa_max}")
    print(
        f"kappa observed range: {d.kappa.min():.2f} – {d.kappa.max():.2f}"
        f"   (winner-channel reference kappa = {d.kappa_bt:.1f})"
    )
    print(f"\nkeeping {int(keep.sum())}/{K} criteria:")
    for row in tab.itertuples(index=False):
        print(
            f"  {'KEEP' if row.keep else 'drop'}  rho={row.rho:5.3f}  kappa={row.kappa:6.2f}  "
            f"gamma={row.gamma:+6.2f}  {row.canonical_name}"
        )

    if not keep.any():
        print("no criteria survive the gates", file=sys.stderr)
        return 1

    idx = np.flatnonzero(keep)
    np.savez_compressed(out / "tensors.npz", w=w, r=r[:, idx], pairs=pairs, n=n, K=len(idx), T=T)
    kept = active.iloc[idx].copy().reset_index(drop=True)
    kept["raw_cluster_id"] = kept["cluster_id"].to_numpy()
    kept["cluster_id"] = np.arange(len(kept))
    kept.to_parquet(out / "active_criteria.parquet", index=False)
    (out / "selection.json").write_text(
        json.dumps(
            {
                "rho_min": rho_min,
                "kappa_max": kappa_max,
                "K_before": K,
                "K_after": int(keep.sum()),
                "kept": kept["canonical_name"].tolist(),
                "dropped": active.iloc[np.flatnonzero(~keep)]["canonical_name"].tolist(),
            },
            indent=2,
        )
    )
    print(f"\nwrote → {out}  (K {K} → {len(idx)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run name, a comma list, or 'all'")
    ap.add_argument("--pipeline-dir", default=None)
    ap.add_argument("--fit-dir", default=None)
    ap.add_argument("--out-subdir", default="pipeline_reliable")
    ap.add_argument("--rho-min", type=float, default=None)
    ap.add_argument("--kappa-max", type=float, default=None)
    args = ap.parse_args()
    return max(screen(run, args) for run in resolve(args.run))


if __name__ == "__main__":
    raise SystemExit(main())
