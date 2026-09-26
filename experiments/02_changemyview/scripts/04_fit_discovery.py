"""Step 4: the gated discovery fit -- 45 criteria, the kappa/rho gate, sandwich CIs. No LLM calls.

    python scripts/04_fit_discovery.py --out-dir DIR

Maps every cited dimension of step 2 onto its step-3 criterion and tallies each comparison into
``X in {-1, 0, +1}^(6000 x 45)`` (``core.fit.build_X``), fits the gated symmetric model
(``core.model.gated.fit_model``), and computes each criterion's identification ``kappa`` and
reliability ``rho`` (``core.fit.kappa_rho``). The gate keeps ``kappa <= model.kappa_max`` and
``rho > model.rho_threshold`` (the discovery config's 50 / 0.75); the kept criteria are refit
alone for the per-item scores. At the same all-criteria point estimate, sandwich standard
errors for every gamma_k and beta (``core.model.sandwich.gamma_beta_V``).

The all-criteria model behind the ungated scores and the sandwich CIs is fit once. Which of
the kept criteria are challenger-side (the 16 the BT and pointwise cells score)
was decided by hand when those cells' ``features.json`` were written, and is not re-derived.

**Provenance.** The shipped ``criterion_diagnostics.parquet`` and ``fit_summary.json`` come from
a later refit with other settings (raw/cells/README.md); their kappa/rho diagnostics are not the
ones the paper uses. ``criterion_gamma_sandwich.csv`` and ``scores.parquet`` are the run of
record. kappa and rho move in their 3rd-4th significant digit on a refit (the
L-BFGS-B stopping point); the kept set and the scores do not.

Writes <discovery cell>/results/{scores.parquet, scores_ungated.parquet,
criterion_diagnostics.parquet, fit_summary.json, criterion_gamma_sandwich.csv, sandwich_ci.json}.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C
from core.fit import build_X, eff_rank, kappa_rho
from core.model.gated import fit_model
from core.model.sandwich import gamma_beta_V

CELL = "discovery"
TAXONOMY = "llm_taxonomy"
Z95 = 1.959963984540054


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None, help="mirror the cell tree under DIR")
    ap.add_argument("--in-place", action="store_true", help="overwrite the shipped cell")
    a = ap.parse_args()
    C.check_writable(a.out_dir, a.in_place)

    m = C.cell_config(CELL)["model"]
    src = C.cell(CELL) / "results"
    inp = lambda rel: C.input_path(src / rel, a.out_dir)  # noqa: E731
    res = C.work_path(src, a.out_dir)
    res.mkdir(parents=True, exist_ok=True)

    phrases = pd.read_parquet(inp("phrases.parquet"))
    pairs = np.load(inp("pairs.npy"))
    p2i = {str(p): i for i, p in enumerate(json.loads(inp("pair_order.json").read_text()))}
    eidx = json.loads(inp("essay_index.json").read_text())
    n = len(eidx)
    mapping = json.loads(inp(f"gated/{TAXONOMY}/mapping.json").read_text())
    crit = sorted(set(mapping.values()))
    K = len(crit)
    X = build_X(phrases, {nm: crit.index(c) for nm, c in mapping.items()}, K, p2i)
    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])
    print(
        f"[fit] K={K} criteria, T={X.shape[0]} comparisons, n={n} items; citation rate "
        f"{(X != 0).mean():.3f}"
    )

    # ---- all-criteria fit, the gate, and the sandwich at that point estimate ----
    fit = fit_model(X, pairs, K=K, n=n, ridge=ridge, tol=tol, max_iter=mi)
    kappa, rho = kappa_rho(fit, X, pairs, n, ridge)
    keep = (kappa <= kmax) & (rho > rthr)
    cite = (X != 0).mean(0)
    pd.DataFrame(
        {"criterion": crit, "kappa": kappa, "rho": rho, "keep": keep, "citation_rate": cite}
    ).to_parquet(res / "criterion_diagnostics.parquet", index=False)
    id2l = {i: l for l, i in eidx.items()}
    ungated = pd.DataFrame(fit.s, columns=crit)
    ungated.insert(0, "letter_id", [id2l[i] for i in range(n)])
    ungated.to_parquet(res / "scores_ungated.parquet", index=False)

    gamma = np.asarray(fit.gamma, dtype=float).ravel()
    beta = float(fit.beta)
    V = gamma_beta_V(fit.s, gamma, beta, X, pairs, ridge)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    se_g, se_b = se[:K], float(se[K])
    pd.DataFrame(
        {
            "criterion": crit,
            "gamma": gamma,
            "se_gamma": se_g,
            "gamma_lo": gamma - Z95 * se_g,
            "gamma_hi": gamma + Z95 * se_g,
            "kappa": kappa,
            "rho": rho,
            "keep": keep,
            "citation_rate": cite,
        }
    ).to_csv(res / "criterion_gamma_sandwich.csv", index=False)
    (res / "sandwich_ci.json").write_text(
        json.dumps(
            {
                "beta": beta,
                "se_beta": se_b,
                "beta_lo": beta - Z95 * se_b,
                "beta_hi": beta + Z95 * se_b,
                "K": K,
                "n_kept": int(keep.sum()),
            },
            indent=2,
        )
    )
    print(
        f"[fit] beta = {beta:+.5f} (SE {se_b:.5f}); kept {int(keep.sum())}/{K} "
        f"(kappa <= {kmax:g}, rho > {rthr})"
    )

    # ---- refit on the kept criteria: the per-item discovery scores ----
    kept = [i for i in range(K) if keep[i]]
    names = [crit[i] for i in kept]
    fitR = fit_model(X[:, kept], pairs, K=len(kept), n=n, ridge=ridge, tol=tol, max_iter=mi)
    scores = pd.DataFrame(fitR.s, columns=names)
    scores.insert(0, "letter_id", [id2l[i] for i in range(n)])
    scores.to_parquet(res / "scores.parquet", index=False)
    er = eff_rank(fitR.s) if len(kept) > 1 else 1.0
    (res / "fit_summary.json").write_text(
        json.dumps(
            {
                "K_total": K,
                "K_kept": len(kept),
                "kept": names,
                "dropped": [crit[i] for i in range(K) if not keep[i]],
                "eff_rank_kept": er,
                "beta": beta,
                "kappa_max": kmax,
                "rho_threshold": rthr,
                "n_pairs": int(X.shape[0]),
                "n_letters": n,
                "taxonomy_dir": TAXONOMY,
            },
            indent=2,
        )
    )
    print(f"[fit] eff_rank(kept scores) = {er:.2f} -> {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
