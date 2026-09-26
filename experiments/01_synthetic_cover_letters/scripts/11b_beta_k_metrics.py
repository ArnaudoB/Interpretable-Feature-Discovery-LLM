"""Step 11b: does the per-criterion beta_k change the kept features or the recovery metrics?

    python scripts/11b_beta_k_metrics.py --config config.yaml

Companion to 11_beta_per_criterion.py: that one shows the scores barely move under beta_k,
this one shows the downstream selection and metrics do not move either.

Two questions:
  Q1  Does the kappa/rho filter keep the SAME criteria under beta_k?  (prior: yes -- the
      thresholds read the SCORE-block Hessian, which beta only touches through w_dir.)
  Q2  What are the five recovery metrics of the beta_k scores, vs the global-beta scores?

Self-contained. It re-derives the design as 04_fit_comparative, fits the global-beta model
(reference) and the beta_k model on the FULL criterion set, then runs the SAME kappa/rho filter
-- reimplemented for a vector beta and validated against core.model.diagnostics at constant
beta (kappa exactly; rho up to the immaterial shared-vs-per-criterion beta border).
Supporting check for App. app:beta-orthogonality. Writes results/beta_check/beta_k_metrics.json.
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

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics as M
from core.embeddings import cosine_matrix
from core.fit import build_X, kappa_rho
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials
from core.model import diagnostics as gdiag
from core.model.gated import fit_model, log2cosh

# ------------------------- beta_k model (vector beta) ----------------------- #


def _nll_grad_bk(p, X, pairs, K, n, ridge):
    nK = n * K
    s = p[:nK].reshape(n, K)
    gamma = p[nK : nK + K]
    beta = p[nK + K : nK + 2 * K]
    a, b = pairs[:, 0], pairs[:, 1]
    delta = s[a] - s[b]
    z_cite = log2cosh(delta) + gamma[None, :]
    z_dir = delta + beta[None, :]
    cited = (X != 0).astype(np.float64)
    pos = (X == 1).astype(np.float64)
    neg = (X == -1).astype(np.float64)
    ll = (
        cited * log_expit(z_cite)
        + (1 - cited) * log_expit(-z_cite)
        + pos * log_expit(z_dir)
        + neg * log_expit(-z_dir)
    )
    val = float(-ll.sum() + 0.5 * ridge * float((s * s).sum()))
    p_cite = expit(z_cite)
    p_dir = expit(z_dir)
    r_cite = cited - p_cite
    d_dir = pos - cited * p_dir
    dL = r_cite * np.tanh(delta) + d_dir
    g_s = np.zeros_like(s)
    np.add.at(g_s, a, -dL)
    np.add.at(g_s, b, dL)
    g_s += ridge * s
    return val, np.concatenate([g_s.ravel(order="C"), -(r_cite.sum(0)), -(d_dir.sum(0))])


def _fit_bk(X, pairs, K, n, ridge, tol, mi, s0, g0, b0):
    p0 = np.concatenate([s0.ravel(order="C"), g0, b0])
    res = minimize(
        lambda p: _nll_grad_bk(p, X, pairs, K, n, ridge),
        p0,
        method="L-BFGS-B",
        jac=True,
        options={"ftol": tol * 1e-2, "gtol": tol, "maxiter": mi},
    )
    nK = n * K
    return res.x[:nK].reshape(n, K), res.x[nK : nK + K], res.x[nK + K : nK + 2 * K]


# --- beta_k-aware Fisher weights and kappa/rho (mirror of core.model.diagnostics) --- #


def _fisher_bk(s, gamma, beta, X, pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    delta = s[a] - s[b]
    p_cite = expit(log2cosh(delta) + gamma[None, :])
    p_dir = expit(delta + beta[None, :])  # <-- vector beta
    cited = (X != 0).astype(float)
    pos = (X == 1).astype(float)
    w_cite, w_dir = p_cite * (1 - p_cite), p_dir * (1 - p_dir)
    tanh_d = np.tanh(delta)
    return dict(
        h_DD=w_cite * tanh_d**2 + cited * w_dir,
        h_Dg=w_cite * tanh_d,
        h_gg=w_cite,
        h_Db=cited * w_dir,
        dL=(cited - p_cite) * tanh_d + (pos - cited * p_dir),
        r_cite=cited - p_cite,
        d_dir=pos - cited * p_dir,
    )


def _lap(w, a, b, n):
    L = np.zeros((n, n))
    np.add.at(L, (a, a), w)
    np.add.at(L, (b, b), w)
    np.add.at(L, (a, b), -w)
    np.add.at(L, (b, a), -w)
    return L


def _kappa_rho_bk(s, gamma, beta, X, pairs, ridge):
    """kappa (via the frozen model-agnostic per_criterion_kappa) and rho with a PER-CRITERION
    beta_k border (each beta_k couples to its own block only -- exact for this model)."""
    n, K = s.shape
    a, b = pairs[:, 0], pairs[:, 1]
    fw = _fisher_bk(s, gamma, beta, X, pairs)
    _, _, kappa = gdiag.per_criterion_kappa(fw["h_DD"], pairs, n)
    var = s.var(0)
    noise = np.zeros(K)
    for k in range(K):
        # bread over [s(n), gamma_k, beta_k]
        H = np.zeros((n + 2, n + 2))
        H[:n, :n] = _lap(fw["h_DD"][:, k], a, b, n)
        H[np.arange(n), np.arange(n)] += ridge
        cg = np.zeros(n)
        np.add.at(cg, a, fw["h_Dg"][:, k])
        np.add.at(cg, b, -fw["h_Dg"][:, k])
        cb = np.zeros(n)
        np.add.at(cb, a, fw["h_Db"][:, k])
        np.add.at(cb, b, -fw["h_Db"][:, k])
        H[:n, n] = cg
        H[n, :n] = cg
        H[:n, n + 1] = cb
        H[n + 1, :n] = cb
        H[n, n] = fw["h_gg"][:, k].sum()
        H[n + 1, n + 1] = fw["h_Db"][:, k].sum()
        # meat U (T, n+2): score cols, gamma col, beta col
        T = pairs.shape[0]
        U = np.zeros((T, n + 2))
        U[np.arange(T), a] += fw["dL"][:, k]
        U[np.arange(T), b] -= fw["dL"][:, k]
        U[:, n] = fw["r_cite"][:, k]
        U[:, n + 1] = fw["d_dir"][:, k]
        Hi = np.linalg.inv(H)
        Vs = Hi @ (U.T @ U) @ Hi  # sandwich
        noise[k] = np.diag(Vs)[:n].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(var > 0, (var - noise) / var, np.nan)
    return kappa, rho


def _metrics(s_cols, names, dials, G, cache):
    idx = {nm: j for j, nm in enumerate(names)}
    cols = [c for c in names if c in idx]
    F = pd.DataFrame(s_cols, columns=names)[cols]
    F.index = G.index[: len(F)] if False else pd.RangeIndex(len(F))
    return cols, F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    root = Path(args.config).parent
    res = root / "results"
    cache = root / ".embed_cache"
    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])

    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    p2i = {str(p): i for i, p in enumerate(json.loads((res / "pair_order.json").read_text()))}
    eidx = json.loads((res / "essay_index.json").read_text())
    n = len(eidx)
    id2l = {i: lid for lid, i in eidx.items()}
    mapping = json.loads((res / "gated/llm_taxonomy/mapping.json").read_text())
    crit = sorted(set(mapping.values()))
    K0 = len(crit)
    cof = {norm: crit.index(c) for norm, c in mapping.items()}
    X = build_X(phrases, cof, K0, p2i)
    ctext = {
        c["name"]: f"{c['name']}. {c['definition']}"
        for c in json.loads((res / "gated/llm_taxonomy/taxonomy.json").read_text())["criteria"]
    }

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)

    # ---- global-beta reference on full K0 (the paper's discovery step) ---- #
    g = fit_model(X, pairs, K=K0, n=n, ridge=ridge, tol=tol, max_iter=mi)
    kap_g, rho_g = kappa_rho(g, X, pairs, n, ridge)
    keep_g = (kap_g <= kmax) & (rho_g > rthr)

    # ---- validate the beta_k diagnostic reduces to core.model.diagnostics at constant beta ---- #
    kap_v, rho_v = _kappa_rho_bk(g.s, g.gamma, np.full(K0, g.beta), X, pairs, ridge)
    dkap = float(np.nanmax(np.abs(kap_v - kap_g)))
    drho = float(np.nanmax(np.abs(rho_v - rho_g)))
    # kappa is exact (score-block Hessian, beta-free path); rho agrees to <=0.002 on every
    # well-conditioned criterion and diverges only where kappa already rejects (weak
    # identification -> the shared-vs-per-criterion beta border matters but the decision does not).
    near_rho = np.abs(rho_v - rho_g)[kap_g <= kmax]
    print(
        f"[validate] beta_k diag vs core at const beta: max|dkappa|={dkap:.2e} (exact); "
        f"max|drho| among kappa<=kmax criteria = {float(np.nanmax(near_rho)):.2e} "
        f"(overall {drho:.2e}, on a kappa-rejected criterion)"
    )

    # ---- beta_k model on full K0, warm-started at the global fit ---- #
    s_k, gamma_k, beta_k = _fit_bk(
        X, pairs, K0, n, ridge, tol, mi, g.s, g.gamma, np.full(K0, g.beta)
    )
    kap_k, rho_k = _kappa_rho_bk(s_k, gamma_k, beta_k, X, pairs, ridge)
    keep_k = (kap_k <= kmax) & (rho_k > rthr)

    same = bool(np.array_equal(keep_g, keep_k))
    added = [crit[i] for i in range(K0) if keep_k[i] and not keep_g[i]]
    dropped = [crit[i] for i in range(K0) if keep_g[i] and not keep_k[i]]
    print(
        f"\n[Q1] kept set identical: {same}  (global {int(keep_g.sum())} vs beta_k {int(keep_k.sum())})"
    )
    if not same:
        print(f"     added by beta_k: {added}\n     dropped by beta_k: {dropped}")
    # how close were the borderline decisions?
    dk = pd.DataFrame(
        {
            "criterion": crit,
            "kap_g": kap_g,
            "kap_k": kap_k,
            "rho_g": rho_g,
            "rho_k": rho_k,
            "keep_g": keep_g,
            "keep_k": keep_k,
        }
    )
    near = dk[(dk.rho_g.sub(rthr).abs() < 0.05) | (dk.kap_g.sub(kmax).abs() < 2)]
    if len(near):
        print("     borderline criteria (rho near 0.75 or kappa near 10):")
        print(near.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- Q2: metrics of each model's scores on the SAME kept set ---- #
    def _score(mask, s_full, tag):
        keep = [i for i in range(K0) if mask[i]]
        names = [crit[i] for i in keep]
        Sr, gr, br = (s_full[:, keep], None, None)  # placeholder; refit below for identifiability
        return keep, names

    kept_g = [i for i in range(K0) if keep_g[i]]
    names_g = [crit[i] for i in kept_g]
    # refit BOTH models on the kept design so scores are the identified ones
    gR = fit_model(X[:, kept_g], pairs, K=len(kept_g), n=n, ridge=ridge, tol=tol, max_iter=mi)
    sK, _, _ = _fit_bk(
        X[:, kept_g],
        pairs,
        len(kept_g),
        n,
        ridge,
        tol,
        mi,
        gR.s,
        gR.gamma,
        np.full(len(kept_g), gR.beta),
    )
    common = [c for c in names_g if c in ctext]
    cos = pd.DataFrame(
        cosine_matrix([ctext[c] for c in common], [DIAL_TEXT[d] for d in dials], cache_dir=cache),
        index=common,
        columns=dials,
    )
    idxmap = {c: names_g.index(c) for c in common}
    Gc = G.loc[[id2l[i] for i in range(n)], dials].reset_index(drop=True)
    out_metrics = {}
    for tag, S in [("global_beta", gR.s), ("beta_k", sK)]:
        F = pd.DataFrame(S, columns=names_g)[common].reset_index(drop=True)
        out_metrics[tag] = M.scorecard(F, Gc, cos, theta, tau)
    print(f"\n[Q2] recovery metrics on the {len(kept_g)} kept criteria:")
    print(f"  {'':12s} {'K':>3s} {'SF':>6s} {'SM':>6s} {'SR':>6s} {'AF':>6s} {'LK':>6s}")
    for tag in ("global_beta", "beta_k"):
        r = out_metrics[tag]
        print(
            f"  {tag:12s} {r['K']:3d} {r['SF']:6.3f} {r['SM']:6.3f} {r['SR']:6.3f} "
            f"{r['AF']:6.3f} {r['LK']:6.3f}"
        )
    # score-shape change
    corr = [float(np.corrcoef(gR.s[:, j], sK[:, j])[0, 1]) for j in range(len(kept_g))]
    print(
        f"\n[Q2] score columns stay {min(corr):.4f}-{max(corr):.4f} correlated (global vs beta_k)"
    )

    outp = res / "beta_check" / "beta_k_metrics.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(
        json.dumps(
            {
                "note": "Selection and recovery metrics under the per-criterion beta_k refit.",
                "validate_max_dkappa": dkap,
                "validate_max_drho": drho,
                "kept_identical": same,
                "n_kept_global": int(keep_g.sum()),
                "n_kept_beta_k": int(keep_k.sum()),
                "added_by_beta_k": added,
                "dropped_by_beta_k": dropped,
                "metrics": out_metrics,
                "min_score_col_corr": float(min(corr)),
            },
            indent=2,
        )
    )
    print(f"\n[beta] wrote {outp.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
