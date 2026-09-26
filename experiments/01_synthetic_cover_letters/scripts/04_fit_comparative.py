"""Step 4: fit the gated symmetric Bradley--Terry model to the comparative judgments, filter, refit.

    python scripts/04_fit_comparative.py --config config.yaml

Pipeline (taxonomy-driven, no clustering):
  * map each cited dimension -> canonical criterion (gated/llm_taxonomy/mapping.json),
  * tally A/B per (pair, criterion) -> design tensor X in {-1, 0, +1}^(T x K) (core.fit.build_X),
  * fit the gated model (core.model.gated.fit_model),
  * per-criterion identification kappa + reliability rho; DROP kappa > model.kappa_max or
    rho <= model.rho_threshold,
  * refit on the reliable criteria; save per-letter scores.

Writes results/{scores.parquet, criterion_diagnostics.parquet, fit_summary.json}. This is the
"comparative" arm ("Ours"): tab:main row Ours, the kept column of tab:feats-ours, and
fig:recovery-heatmap-comparative. Pure numeric — no API key, runs from the shipped inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.fit import build_X, eff_rank, kappa_rho
from core.model.gated import fit_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--taxonomy",
        default="gated/llm_taxonomy",
        help="results-relative dir holding mapping.json (default: gated/llm_taxonomy)",
    )
    ap.add_argument(
        "--out",
        default="scores.parquet",
        help="results-relative output parquet (default: scores.parquet)",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    res = Path(args.config).parent / "results"

    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    pair_order = json.loads((res / "pair_order.json").read_text())
    p2i = {str(p): i for i, p in enumerate(pair_order)}
    eidx = json.loads((res / "essay_index.json").read_text())
    n = len(eidx)
    mapping = json.loads((res / args.taxonomy / "mapping.json").read_text())
    crit_names = sorted(set(mapping.values()))
    K = len(crit_names)
    cof = {norm: crit_names.index(c) for norm, c in mapping.items()}

    X = build_X(phrases, cof, K, p2i)
    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])
    print(
        f"[fit] K={K} criteria, T={X.shape[0]} pairs, n={n} letters; "
        f"citation rate={(X != 0).mean():.3f}"
    )

    fit = fit_model(X, pairs, K=K, n=n, ridge=ridge, tol=tol, max_iter=mi)
    kappa, rho = kappa_rho(fit, X, pairs, n, ridge)
    keep = (kappa <= kmax) & (rho > rthr)
    diag = pd.DataFrame(
        {
            "criterion": crit_names,
            "kappa": kappa,
            "rho": rho,
            "keep": keep,
            "citation_rate": (X != 0).mean(0),
        }
    )
    diag.to_parquet(res / "criterion_diagnostics.parquet", index=False)
    print(diag.sort_values(["keep", "citation_rate"], ascending=False).to_string(index=False))

    kept = [i for i in range(K) if keep[i]]
    names_k = [crit_names[i] for i in kept]
    print(f"\n[fit] kept {len(kept)}/{K} criteria (kappa <= {kmax:g}, rho > {rthr})")
    fitR = fit_model(X[:, kept], pairs, K=len(kept), n=n, ridge=ridge, tol=tol, max_iter=mi)
    id2l = {i: lid for lid, i in eidx.items()}
    scores = pd.DataFrame(fitR.s, columns=names_k)
    scores.insert(0, "letter_id", [id2l[i] for i in range(n)])
    scores.to_parquet(res / args.out, index=False)

    er = eff_rank(fitR.s) if len(kept) > 1 else 1.0
    (res / "fit_summary.json").write_text(
        json.dumps(
            {
                "K_total": K,
                "K_kept": len(kept),
                "kept": names_k,
                "dropped": [crit_names[i] for i in range(K) if not keep[i]],
                "eff_rank_kept": er,
                "beta": float(fit.beta),
                "kappa_max": kmax,
                "rho_threshold": rthr,
                "n_pairs": int(X.shape[0]),
                "n_letters": n,
            },
            indent=2,
        )
    )
    print(f"[fit] eff_rank(kept scores) = {er:.2f}; beta_hat = {fit.beta:+.4f}")
    print(
        f"[fit] wrote results/{args.out} ({n} x {len(kept)}), criterion_diagnostics.parquet, fit_summary.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
