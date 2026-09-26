"""Step 12: out-of-fold calibration of the feature-discovery model's two gates.

    python scripts/12_calibration.py --config config.yaml

Backs the paper's "Model calibration" subsection (Appendix, sec:synthetic-calibration). The
fitted model turns each (comparison, criterion) cell into two probability forecasts of
observable binary events, and this scores both as forecasts:

  citation gate   P(criterion k is cited on pair p)  = sigma(log 2cosh(Delta_pk) + gamma_k)
                  defined on all T x K = 84,000 cells
  direction gate  P(first-shown letter favored | cited) = sigma(Delta_pk + beta)
                  defined on the 29,030 cited cells only

Both are scored out of fold -- 5-fold KFold over comparisons, so every cell is predicted by
parameters estimated without it -- with 95% percentile bootstrap intervals over comparisons.
The in-sample counterparts quantify the optimism of the fit.

Two references for the Brier skill score. Chance is p = 1/2 (Brier 0.25). The empirical rate
is estimated on the training folds and is the demanding one: for the citation gate it is the
maximum-likelihood fit of the gap-independent submodel f = 0, whose gate sigma(gamma_k) is
constant within a criterion, so skill over it is a direct out-of-fold test of whether the
merit gap carries information about citation at all.

Before scoring anything, the script refits on all 4,000 comparisons and asserts the scores
equal results/scores.parquet -- the guard that the calibration is on the run the paper
reports, not on a lookalike.

Reuses core.model.gated.{fit_model,gate_probabilities} and
core.evaluation.calibration.{pair_bootstrap_calibration,compute_ece,STAT_KEYS}; the latter is
the same module Experiment 3's 08_calibration.py uses on the sibling model, so the two share
their output schema. Runs offline: no API calls, no embeddings.

Writes results/calibration/{metrics.csv, per_criterion.csv, bins.json, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.fit import build_X
from core.model.gated import fit_model, gate_probabilities
from core.evaluation.calibration import STAT_KEYS, pair_bootstrap_calibration, compute_ece

GATES = ("citation", "direction")
N_FOLDS, SPLIT_SEED = 5, 42
N_BINS, N_BOOT, BOOT_SEED = 10, 2000, 0
EPS = 1e-12


def _fmt(r, k, d=3):
    return f"{r[k]:.{d}f} [{r[k + '_lo']:.{d}f}, {r[k + '_hi']:.{d}f}]"


def _load(root, cfg):
    """Design tensor over the kept criteria, plus the pair list and the reference scores."""
    res = root / "results"
    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    pair_order = json.loads((res / "pair_order.json").read_text())
    p2i = {str(p): i for i, p in enumerate(pair_order)}
    eidx = json.loads((res / "essay_index.json").read_text())
    n = len(eidx)
    mapping = json.loads((res / "gated/llm_taxonomy/mapping.json").read_text())
    names_all = sorted(set(mapping.values()))
    cof = {nm: names_all.index(c) for nm, c in mapping.items()}

    fs = json.loads((res / "fit_summary.json").read_text())
    m = cfg["model"]
    assert (fs["K_total"], fs["kappa_max"], fs["rho_threshold"]) == (
        len(names_all),
        float(m["kappa_max"]),
        float(m["rho_threshold"]),
    ), fs
    kept = fs["kept"]
    X = build_X(phrases, cof, len(names_all), p2i)[:, [names_all.index(c) for c in kept]]
    return X, pairs, n, kept, eidx, len(names_all)


def _metrics(p, y, ref, mask=None):
    """Point metrics + pair-bootstrap CIs, and the reliability bins."""
    cal = pair_bootstrap_calibration(
        p, y, ref, mask=mask, n_bins=N_BINS, n_boot=N_BOOT, seed=BOOT_SEED
    )
    sel = np.ones(np.shape(p), bool) if mask is None else np.asarray(mask, bool)
    pf, yf = np.asarray(p)[sel], np.asarray(y)[sel]
    # independent restatement of ECE, as a guard on the bootstrap's own point estimate
    assert abs(cal["ece"] - compute_ece(pf, yf, n_bins=N_BINS)) < 1e-10
    pc = np.clip(pf, EPS, 1 - EPS)
    out = {
        "n": int(sel.sum()),
        "rate": float(yf.mean()),
        "mean_p": float(pf.mean()),
        "brier_chance": 0.25,
        **{f"{k}{s}": cal[f"{k}{s}"] for k in STAT_KEYS for s in ("", "_lo", "_hi")},
        "logloss": float(-(yf * np.log(pc) + (1 - yf) * np.log(1 - pc)).mean()),
        "acc": float(((pf > 0.5) == (yf > 0.5)).mean()),
    }
    return out, cal["bins"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write under DIR instead of results/calibration/, so a re-run can be "
        "diffed against the shipped artifacts",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    out = Path(args.out_dir) if args.out_dir else (res / "calibration")
    out.mkdir(parents=True, exist_ok=True)

    X, pairs, n, kept, eidx, K_all = _load(root, cfg)
    T, K = X.shape
    m = cfg["model"]
    fit_kw = dict(ridge=float(m["ridge"]), tol=float(m["tol"]), max_iter=int(m["max_iter"]))
    warnings.filterwarnings("ignore", message=r"\[gated fit\]")  # per-obs grad note

    # ---- the guard: the full refit must be the paper's run ------------------ #
    full = fit_model(X, pairs, K=K, n=n, **fit_kw)
    id2l = {i: l for l, i in eidx.items()}
    ref_scores = pd.read_parquet(res / "scores.parquet").set_index("letter_id")
    S = pd.DataFrame(full.s, columns=kept, index=[id2l[i] for i in range(n)])
    maxdiff = float(np.abs(S.loc[ref_scores.index, kept].values - ref_scores[kept].values).max())
    assert maxdiff < 1e-8, f"full refit does not reproduce results/scores.parquet: {maxdiff}"

    CITED, POS = (X != 0), (X == 1)
    print(
        f"[calibration] T={T} pairs, n={n} letters, K={K} kept of {K_all} "
        f"(kappa<={m['kappa_max']}, rho>{m['rho_threshold']})"
    )
    print(f"[calibration] full refit reproduces results/scores.parquet (max|diff|={maxdiff:.1e})")
    print(
        f"[calibration] citation rate={CITED.mean():.3f}  cited cells={int(CITED.sum()):,}  "
        f"A|cited={POS.sum() / CITED.sum():.3f}  beta={full.beta:+.4f}"
    )

    # ---- out-of-fold predictions and training-fold references --------------- #
    P = {g: np.full((T, K), np.nan) for g in GATES}
    REF = {g: np.full((T, K), np.nan) for g in GATES}
    health = []
    folds = KFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED).split(np.arange(T))
    for f, (tr, te) in enumerate(folds):
        assert np.unique(pairs[tr]).size == n, f"fold {f}: a letter is absent from training"
        fit = fit_model(X[tr], pairs[tr], K=K, n=n, **fit_kw)
        P["citation"][te], P["direction"][te] = gate_probabilities(
            fit.s, fit.gamma, fit.beta, pairs[te]
        )
        ct, pt = CITED[tr], POS[tr]
        REF["citation"][te] = ct.mean(axis=0)  # the f=0 submodel
        REF["direction"][te] = pt.sum(axis=0) / np.maximum(
            ct.sum(axis=0), 1
        )  # per-criterion A|cited
        health.append(
            {
                "fold": f,
                "n_train": len(tr),
                "n_test": len(te),
                "beta": float(fit.beta),
                "n_iter": int(fit.n_iter),
                "converged": bool(fit.converged),
                "grad_inf_per_obs": fit.grad_norm / (len(tr) * K),
            }
        )
    assert not any(np.isnan(v).any() for v in (*P.values(), *REF.values()))

    P_IN = dict(zip(GATES, gate_probabilities(full.s, full.gamma, full.beta, pairs)))
    REF_IN = {
        "citation": np.broadcast_to(CITED.mean(axis=0), (T, K)),
        "direction": np.broadcast_to(POS.sum(axis=0) / CITED.sum(axis=0), (T, K)),
    }
    Y = {"citation": CITED.astype(float), "direction": POS.astype(float)}
    MASK = {"citation": None, "direction": CITED}
    print("\n" + pd.DataFrame(health).to_string(index=False))

    # ---- pooled metrics, both regimes -------------------------------------- #
    rows, bins = [], {}
    for regime, (PP, RR) in {"oof": (P, REF), "in_sample": (P_IN, REF_IN)}.items():
        for gate in GATES:
            r, b = _metrics(PP[gate], Y[gate], RR[gate], MASK[gate])
            rows.append({"regime": regime, "gate": gate, **r})
            bins[(regime, gate)] = b
    TAB = pd.DataFrame(rows)
    assert TAB.set_index(["regime", "gate"]).loc[("oof", "direction"), "n"] == CITED.sum()

    # ---- per-criterion, out of fold ---------------------------------------- #
    crit = []
    for gate in GATES:
        for k, name in enumerate(kept):
            mk = None if MASK[gate] is None else MASK[gate][:, k]
            r, _ = _metrics(P[gate][:, k], Y[gate][:, k], REF[gate][:, k], mk)
            crit.append({"regime": "oof", "gate": gate, "k": k, "criterion": name, **r})
    CRIT = pd.DataFrame(crit)

    TAB.to_csv(out / "metrics.csv", index=False)
    CRIT.to_csv(out / "per_criterion.csv", index=False)
    (out / "bins.json").write_text(
        json.dumps(
            {
                f"{rg}/{g}": {k: np.asarray(v).tolist() for k, v in b.items()}
                for (rg, g), b in bins.items()
            },
            indent=2,
            default=float,
        )
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "model": f"{m['type']}, K={K} of {K_all} (kappa<={m['kappa_max']}, "
                f"rho>{m['rho_threshold']})",
                "K": K,
                "kept": kept,
                "T": T,
                "n": n,
                "oof": f"{N_FOLDS}-fold KFold over pairs (shuffle, seed {SPLIT_SEED}); kept set fixed; "
                f"references from the training folds",
                "ece": f"{N_BINS} equal-width bins; citation pooled over all (pair, criterion); "
                f"direction over cited cells",
                "bss": "1 - Brier(model)/Brier(reference); chance reference p=1/2 (Brier 0.25); "
                "empirical reference is the per-criterion citation rate (= the f=0 submodel) "
                "and the per-criterion A|cited rate",
                "ci": f"95% percentile bootstrap over comparisons, {N_BOOT} replicates, seed {BOOT_SEED}",
                "full_refit_maxdiff_vs_scores_parquet": maxdiff,
                "beta_full": float(full.beta),
                "folds": health,
                "metrics": rows,
            },
            indent=2,
            default=float,
        )
    )

    pd.set_option("display.width", 250)
    for regime in ("oof", "in_sample"):
        sub = TAB[TAB.regime == regime].set_index("gate")
        show = pd.DataFrame(
            {
                k: sub.apply(lambda r, k=k: _fmt(r, k), axis=1)
                for k in ("ece", "brier", "brier_emp", "bss_chance", "bss_emp")
            }
        )
        show.insert(0, "n", sub["n"])
        show.insert(1, "rate", sub["rate"].round(3))
        print(f"\n== {regime}  (Brier of chance = 0.250)\n" + show.to_string())
    print(
        f"\n[calibration] wrote {out}/{{metrics.csv, per_criterion.csv, bins.json, summary.json}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
