"""Step 10a: sandwich standard errors and Wald CIs for beta and gamma_k.

    python scripts/10a_sandwich_ci.py --config config.yaml

Evaluated ONCE at the point estimate (the same K-criterion fit that produces
kappa/rho and beta in 04_fit_comparative.py) -- there is no resampling here.

  V = H^{-1} B H^{-1}
    H = penalized EXPECTED (Fisher) Hessian at the fit (core.model.gated
        .expected_hessian_full), the same information matrix the rho noise term
        uses, so beta/gamma SEs and rho rest on one definition;
    B = sum_t U_t U_t^T, the query-clustered outer product of per-query scores
        (s-entries use dL/dDelta, gamma entries use r_cite, and the beta entry is
        sum_k d_dir -- beta is absent from the citation gate).

Uses core.model.sandwich.gamma_beta_V, which is sandwich_V restricted to the
(gamma, beta) columns without materializing four dense P x P matrices.

Writes results/sandwich_ci.json and results/criterion_gamma_sandwich.csv (tab:feats-ours:
gamma_k with its 95% CI per criterion, and beta with its CI in the caption).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.fit import build_X, kappa_rho
from core.model.gated import fit_model
from core.model.sandwich import gamma_beta_V

Z95 = 1.959963984540054


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--taxonomy", default="gated/llm_taxonomy")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    res = Path(args.config).parent / "results"

    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])

    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    pair_order = json.loads((res / "pair_order.json").read_text())
    p2i = {str(p): i for i, p in enumerate(pair_order)}
    n = len(json.loads((res / "essay_index.json").read_text()))
    mapping = json.loads((res / args.taxonomy / "mapping.json").read_text())
    crit_names = sorted(set(mapping.values()))
    K = len(crit_names)
    cof = {norm: crit_names.index(c) for norm, c in mapping.items()}

    X = build_X(phrases, cof, K, p2i)
    fit = fit_model(X, pairs, K=K, n=n, ridge=ridge, tol=tol, max_iter=mi)
    gamma = np.asarray(fit.gamma, dtype=float).ravel()
    beta = float(fit.beta)
    kappa, rho = kappa_rho(fit, X, pairs, n, ridge)
    keep = (kappa <= kmax) & (rho > rthr)
    print(
        f"[ci] K={K} criteria, beta_hat={beta:+.6f}, {int(keep.sum())} kept "
        f"(kappa<={kmax:g}, rho>{rthr})"
    )

    V = gamma_beta_V(fit.s, gamma, beta, X, pairs, ridge)  # (K+1, K+1)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    se_gamma, se_beta = se[:K], float(se[K])

    lo, hi = beta - Z95 * se_beta, beta + Z95 * se_beta
    print(f"[ci] beta = {beta:+.5f}  SE {se_beta:.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    out = pd.DataFrame(
        {
            "criterion": crit_names,
            "gamma": gamma,
            "se_gamma": se_gamma,
            "gamma_lo": gamma - Z95 * se_gamma,
            "gamma_hi": gamma + Z95 * se_gamma,
            "kappa": kappa,
            "rho": rho,
            "keep": keep,
            "citation_rate": (X != 0).mean(0),
        }
    )
    out.to_csv(res / "criterion_gamma_sandwich.csv", index=False)
    (res / "sandwich_ci.json").write_text(
        json.dumps(
            {
                "estimator": "sandwich H^-1 B H^-1 (expected Fisher bread, query-clustered OPG meat)",
                "ridge_in_bread": ridge,
                "K": K,
                "n": n,
                "n_pairs": int(X.shape[0]),
                "beta": beta,
                "se_beta": se_beta,
                "beta_lo": lo,
                "beta_hi": hi,
            },
            indent=2,
        )
    )

    covered = bool(((out.gamma >= out.gamma_lo) & (out.gamma <= out.gamma_hi)).all())
    print(f"[ci] every gamma_hat inside its CI: {covered}")
    print("[ci] wrote results/sandwich_ci.json, results/criterion_gamma_sandwich.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
