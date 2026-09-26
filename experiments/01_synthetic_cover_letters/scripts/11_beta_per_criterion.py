"""Step 11: the per-criterion position-bias refit (App. app:beta-orthogonality, "Empirical check").

    python scripts/11_beta_per_criterion.py --config config.yaml

The paper's model shares ONE position-bias scalar beta across criteria, in the direction term
z_dir = Delta + beta (which slot wins, given the criterion is cited), and argues that the
balanced slot design makes the scores essentially insensitive to that choice. This script is
the empirical check: refit with beta free per criterion (z_dir = Delta + beta_k) and ask
whether the scores move.

Read the result precisely. It supports the claim the paper makes -- the scores barely move
(minimum per-criterion column correlation 0.988 between the two fits, so no criterion is
re-ordered) -- and it does NOT support the stronger claim that the beta_k are homogeneous: the
likelihood-ratio test rejects a common beta, and 14 of the 21 beta_k are individually
significant. The position bias differs by criterion; the scores do not care.

Self-contained: it re-derives the design exactly as 04_fit_comparative (same taxonomy, same
kept set) and fits two models on it -- the global-beta model (as a reference, must reproduce
beta_hat = +0.0051) and a beta_k variant. It imports the model helpers from core.model.gated
and only generalizes the beta gradient to a K-vector. The new gradient is checked against
finite differences before use.

Reports: the beta_k spread vs their own standard errors, a likelihood-ratio test of
beta_k-vs-global (chi^2, K-1 df), and whether the scores move. Writes
results/beta_check/beta_per_criterion.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import chi2, spearmanr

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.fit import build_X, kappa_rho
from core.model.gated import fit_model, log2cosh

# --- flat layout for the beta_k model: [vec(S), gamma(K), beta(K)] ----------- #


def _unpack_bk(p, n, K):
    nK = n * K
    return p[:nK].reshape(n, K), p[nK : nK + K], p[nK + K : nK + 2 * K]


def _nll_grad_bk(p, X, pairs, K, n, ridge):
    """NLL + analytic grad for z_dir = Delta + beta_k (beta a K-vector).

    Identical to core.model.gated.nll_and_grad except beta broadcasts per column and its
    gradient is a column sum rather than a global sum.
    """
    s, gamma, beta = _unpack_bk(p, n, K)
    a, b = pairs[:, 0], pairs[:, 1]
    delta = s[a] - s[b]  # (T, K)
    z_cite = log2cosh(delta) + gamma[None, :]
    z_dir = delta + beta[None, :]  # <-- per-criterion bias

    cited = (X != 0).astype(np.float64)
    pos = (X == 1).astype(np.float64)
    neg = (X == -1).astype(np.float64)

    ll_cite = cited * log_expit(z_cite) + (1.0 - cited) * log_expit(-z_cite)
    ll_dir = pos * log_expit(z_dir) + neg * log_expit(-z_dir)
    pen = 0.5 * ridge * float((s * s).sum())
    val = float(-(ll_cite + ll_dir).sum() + pen)

    p_cite = expit(z_cite)
    p_dir = expit(z_dir)
    r_cite = cited - p_cite
    d_dir = pos - cited * p_dir
    dL_dDelta = r_cite * np.tanh(delta) + d_dir

    g_s = np.zeros_like(s)
    g_term = -dL_dDelta
    np.add.at(g_s, a, g_term)
    np.add.at(g_s, b, -g_term)
    g_s += ridge * s
    g_gamma = -(r_cite.sum(0))
    g_beta = -(d_dir.sum(0))  # <-- (K,), per-criterion
    return val, np.concatenate([g_s.ravel(order="C"), g_gamma, g_beta])


def _fit_bk(X, pairs, K, n, ridge, tol, mi, s0, gamma0, beta0):
    p0 = np.concatenate([s0.ravel(order="C"), gamma0, beta0])
    res = minimize(
        lambda p: _nll_grad_bk(p, X, pairs, K, n, ridge),
        p0,
        method="L-BFGS-B",
        jac=True,
        options={"ftol": tol * 1e-2, "gtol": tol, "maxiter": mi},
    )
    return res


def _beta_k_se(s, beta, pairs, X):
    """Direction-term Fisher SE for each beta_k: 1/sqrt(sum_{cited} p_dir(1-p_dir))."""
    a, b = pairs[:, 0], pairs[:, 1]
    delta = s[a] - s[b]
    p_dir = expit(delta + beta[None, :])
    cited = (X != 0).astype(np.float64)
    info = (cited * p_dir * (1.0 - p_dir)).sum(0)  # (K,)
    return 1.0 / np.sqrt(np.maximum(info, 1e-12)), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    root = Path(args.config).parent
    res = root / "results"
    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])

    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    p2i = {str(p): i for i, p in enumerate(json.loads((res / "pair_order.json").read_text()))}
    n = len(json.loads((res / "essay_index.json").read_text()))
    mapping = json.loads((res / "gated/llm_taxonomy/mapping.json").read_text())
    crit_names = sorted(set(mapping.values()))
    K0 = len(crit_names)
    cof = {norm: crit_names.index(c) for norm, c in mapping.items()}
    X0 = build_X(phrases, cof, K0, p2i)

    # reproduce the paper's kept set exactly
    fit0 = fit_model(X0, pairs, K=K0, n=n, ridge=ridge, tol=tol, max_iter=mi)
    kap, rho = kappa_rho(fit0, X0, pairs, n, ridge)
    kept = [i for i in range(K0) if (kap[i] <= kmax and rho[i] > rthr)]
    names = [crit_names[i] for i in kept]
    X = X0[:, kept]
    K = len(kept)

    # reference: global-beta model on the kept design (must reproduce beta_hat)
    ref = fit_model(X, pairs, K=K, n=n, ridge=ridge, tol=tol, max_iter=mi)
    s_g, gamma_g, beta_g = ref.s, ref.gamma, ref.beta
    print(f"[beta] kept K={K}; global-beta refit: beta_hat={beta_g:+.5f}, nll={ref.nll:.3f}")

    # --- gradient self-check on the beta_k objective (finite differences) ----- #
    rng = np.random.default_rng(0)
    p_chk = np.concatenate([s_g.ravel(order="C"), gamma_g, rng.normal(0, 0.1, K)])
    v0, ganl = _nll_grad_bk(p_chk, X, pairs, K, n, ridge)
    gnum = np.zeros_like(p_chk)
    eps = 1e-6
    idx = np.r_[0:3, n * K : n * K + 2, n * K + K : n * K + K + 3]  # a few s, gamma, beta coords
    for i in idx:
        d = np.zeros_like(p_chk)
        d[i] = eps
        gnum[i] = (
            _nll_grad_bk(p_chk + d, X, pairs, K, n, ridge)[0]
            - _nll_grad_bk(p_chk - d, X, pairs, K, n, ridge)[0]
        ) / (2 * eps)
    gerr = float(np.max(np.abs(ganl[idx] - gnum[idx])))
    print(f"[beta] beta_k gradcheck max|analytic-fd| = {gerr:.2e}")
    assert gerr < 1e-4, "beta_k gradient disagrees with finite differences"

    # --- fit the beta_k model, warm-started at the global solution ------------ #
    fitk = _fit_bk(X, pairs, K, n, ridge, tol, mi, s_g, gamma_g, np.full(K, beta_g))
    s_k, gamma_k, beta_k = _unpack_bk(fitk.x, n, K)
    nll_k = float(fitk.fun)
    se_k, info_k = _beta_k_se(s_k, beta_k, pairs, X)

    # likelihood-ratio test: beta_k (K params) vs global beta (1 param), K-1 extra df
    lr = 2.0 * (ref.nll - nll_k)
    dof = K - 1
    p_lr = float(chi2.sf(lr, dof))

    # how much do the scores move? (per-criterion correlation + max abs shift)
    col_corr = [float(np.corrcoef(s_g[:, j], s_k[:, j])[0, 1]) for j in range(K)]
    max_shift = float(np.abs(s_g - s_k).max())

    # The shift RELATIVE to each criterion's own score spread, and its correlation with
    # |beta_k| (App. app:beta-orthogonality, "Empirical check"). The relative shift is
    # summarised both over cells and over criteria.
    D = np.abs(s_g - s_k)  # (n, K) absolute score shift
    spread = s_g.std(axis=0)  # (K,) per-criterion score spread
    rel_cell = D / spread  # cell-wise, relative to the column's spread
    rel_crit = D.mean(axis=0) / spread  # per-criterion mean, same normalisation
    rel = {
        "cellwise_median": float(np.median(rel_cell)),
        "cellwise_p90": float(np.percentile(rel_cell, 90)),
        "per_criterion_median": float(np.median(rel_crit)),
        "per_criterion_p90": float(np.percentile(rel_crit, 90)),
    }
    shift_vs_beta = {
        "pearson_mean_abs_shift_vs_abs_beta_k": float(
            np.corrcoef(D.mean(axis=0), np.abs(beta_k))[0, 1]
        ),
        "pearson_median_abs_shift_vs_abs_beta_k": float(
            np.corrcoef(np.median(D, axis=0), np.abs(beta_k))[0, 1]
        ),
        "spearman_mean_abs_shift_vs_abs_beta_k": float(
            spearmanr(D.mean(axis=0), np.abs(beta_k)).statistic
        ),
    }

    order = np.argsort(-np.abs(beta_k))
    print(f"\n[beta] beta_k (sorted by |beta_k|), global beta = {beta_g:+.4f}:")
    print(f"  {'criterion':38s} {'beta_k':>8s} {'SE':>7s} {'z':>6s} {'cited':>6s}")
    for j in order:
        z = beta_k[j] / se_k[j]
        print(
            f"  {names[j][:38]:38s} {beta_k[j]:+8.3f} {se_k[j]:7.3f} {z:+6.2f} {int(info_k[j]):>6d}"
        )

    wstd = float(
        np.sqrt(np.average((beta_k - np.average(beta_k, weights=info_k)) ** 2, weights=info_k))
    )
    print(
        f"\n[beta] spread of beta_k: raw sd={beta_k.std():.4f}, "
        f"precision-weighted sd={wstd:.4f}; typical SE={np.median(se_k):.3f}"
    )
    print(f"[beta] LR test beta_k vs global: chi2={lr:.2f}, df={dof}, p={p_lr:.3g}")
    print(
        f"[beta] scores barely move: min col corr={min(col_corr):.4f}, "
        f"max |Delta s|={max_shift:.4f}; beta_k refit nll={nll_k:.3f} vs global {ref.nll:.3f}"
    )
    n_sig = int((np.abs(beta_k / se_k) > 1.96).sum())
    print(
        f"[beta] {n_sig}/{K} criteria have |beta_k/SE| > 1.96 "
        f"(expected ~{0.05 * K:.1f} by chance if all truly zero)"
    )

    print(f"\n[beta] shift relative to each criterion's score spread:")
    print(
        f"  cell-wise:      median {rel['cellwise_median']:.4f} "
        f"({rel['cellwise_median'] * 100:.0f}%)   p90 {rel['cellwise_p90']:.4f} "
        f"({rel['cellwise_p90'] * 100:.0f}%)"
    )
    print(
        f"  per-criterion:  median {rel['per_criterion_median']:.4f} "
        f"({rel['per_criterion_median'] * 100:.0f}%)   p90 {rel['per_criterion_p90']:.4f} "
        f"({rel['per_criterion_p90'] * 100:.0f}%)"
    )
    print(
        f"[beta] corr(mean |Delta s_k|, |beta_k|) = "
        f"{shift_vs_beta['pearson_mean_abs_shift_vs_abs_beta_k']:.4f} Pearson, "
        f"{shift_vs_beta['spearman_mean_abs_shift_vs_abs_beta_k']:.4f} Spearman"
    )

    # App. app:beta-orthogonality: "each letter's first-slot share has standard deviation sqrt(0.25/40) = 0.079,
    # matching the realized imbalance". pairs.npy is in presentation order (first shown first).
    first = np.bincount(pairs[:, 0], minlength=n)
    degree = first + np.bincount(pairs[:, 1], minlength=n)
    share = first / degree
    slot = {
        "degree": sorted(set(degree.tolist())),
        "theoretical_sd": float(np.sqrt(0.25 / 40)),
        "realized_sd": float(share.std()),
        "mean_first_share": float(share.mean()),
    }
    print(
        f"[beta] first-slot share: realized sd {slot['realized_sd']:.4f} vs "
        f"sqrt(0.25/40) = {slot['theoretical_sd']:.4f}"
    )

    out = res / "beta_check" / "beta_per_criterion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "note": "Per-criterion position bias beta_k: the refit behind the paper's claim that the balanced slot design leaves the scores insensitive to beta.",
                "K": K,
                "beta_global": float(beta_g),
                "nll_global": float(ref.nll),
                "nll_beta_k": nll_k,
                "LR_chi2": float(lr),
                "LR_df": dof,
                "LR_p": p_lr,
                "beta_k_raw_sd": float(beta_k.std()),
                "beta_k_precision_weighted_sd": wstd,
                "n_beta_k_significant_1p96": n_sig,
                "max_abs_score_shift": max_shift,
                "min_score_col_corr": float(min(col_corr)),
                "relative_shift": rel,
                "shift_vs_beta_k": shift_vs_beta,
                "slot_balance": slot,
                "per_criterion": [
                    {
                        "criterion": names[j],
                        "beta_k": float(beta_k[j]),
                        "se": float(se_k[j]),
                        "z": float(beta_k[j] / se_k[j]),
                        "n_cited": int(info_k[j]),
                    }
                    for j in range(K)
                ],
            },
            indent=2,
        )
    )
    print(f"\n[beta] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
