"""Step 8: calibration (ECE, Brier skill) of the holistic model's two gates, all eight runs.

    python scripts/08_calibration.py        # ~5 min: 40 fold refits + the bootstraps

Gates, on each run's reliable refit (``results/fit_reliable``):
  winner   : P(w_t = 1)          = sig( sum_k Delta_{t,k} + beta )
  citation : P(r_{t,k} = 1 | w_t) = sig( eta_t Delta_{t,k} + gamma_k )

Regimes
  oof       : 5-fold out-of-fold predictions (RandomState(seed_split).permutation(T)),
              each fold predicted by a refit on the other four with the run's model config.
  in_sample : results/fit_reliable/fit.pkl evaluated on all T pairs.

Metrics per (run, gate, regime): 10-bin equal-width ECE, Brier, Brier skill vs chance
(p = 1/2, Brier 0.25) and vs the empirical rate (winner: mean(w); citation: per-criterion
mean(r[:, k]), i.e. the gamma-only "habit" submodel). In ``oof`` the empirical rates come
from the training folds only.

95% CIs: percentile bootstrap over comparisons (2000 replicates). The K citations of one
comparison are resampled together; predictions are held fixed (no refit per replicate).

Writes ``results/calibration/{metrics.csv, per_criterion.csv, summary.json, bins.json}``;
``figures/make_calibration_figures.py`` draws the three paper figures from them
(``fig:calibration_reliability_ellipse``, ``fig:calibration_reliability_asap``,
``fig:calibration_skill``).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import CORPORA, CORPUS_LABEL, EXPERIMENT, JUDGE_LABEL, JUDGES, load_run, run_name  # noqa: E402
from core.evaluation.calibration import STAT_KEYS, compute_ece, pair_bootstrap_calibration  # noqa: E402
from core.model.holistic import fit, gate_probabilities  # noqa: E402

OUT = EXPERIMENT / "results" / "calibration"
N_BINS = 10
N_BOOT = 2000
BOOT_SEED = 0
EPS = 1e-12


def _metrics(p, y, p_ref) -> tuple[dict, dict]:
    """Point metrics with pair-bootstrap CIs, and the reliability bins (bootstrap CIs too)."""
    cal = pair_bootstrap_calibration(p, y, p_ref, n_bins=N_BINS, n_boot=N_BOOT, seed=BOOT_SEED)
    p, y = (np.asarray(x, float).ravel() for x in (p, y))
    assert abs(cal["ece"] - compute_ece(p, y, n_bins=N_BINS)) < 1e-10
    pc = np.clip(p, EPS, 1 - EPS)
    m = {
        "n": int(len(y)),
        "rate": float(y.mean()),
        "mean_p": float(p.mean()),
        "brier_chance": 0.25,
        **{f"{k}{s}": cal[f"{k}{s}"] for k in STAT_KEYS for s in ("", "_lo", "_hi")},
        "logloss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "acc": float(((p > 0.5) == (y > 0.5)).mean()),
    }
    return m, cal["bins"]


def _oof(w, r, pairs, n, K, mcfg, vcfg):
    """Out-of-fold gate probabilities + train-fold empirical rates for every pair."""
    T = len(w)
    seed = int(vcfg.get("seed_split", 42))
    folds = np.array_split(np.random.RandomState(seed).permutation(T), int(vcfg.get("cv_folds", 5)))
    p_win, p_cite = np.empty(T), np.empty((T, K))
    ref_win, ref_cite = np.empty(T), np.empty((T, K))
    health = []
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(T), test)
        assert np.unique(pairs[train]).size == n, f"fold {f}: an essay is absent from training"
        res = fit(
            w[train],
            r[train],
            pairs[train],
            K=K,
            n=n,
            ridge=float(mcfg.get("ridge", 1e-3)),
            include_gamma=bool(mcfg.get("include_gamma", True)),
            tol=float(mcfg.get("tol", 1e-8)),
            max_iter=int(mcfg.get("max_iter", 1000)),
            seed=seed + 1 + f,
        )
        assert res.converged and res.inf_diff < 1e-4, (f, res.converged, res.inf_diff)
        p_win[test], p_cite[test] = gate_probabilities(
            res.s, res.gamma, res.beta, w[test], pairs[test]
        )
        ref_win[test] = w[train].mean()
        ref_cite[test] = r[train].mean(axis=0)
        health.append({"fold": f, "n_iter": res.n_iter, "inf_diff": res.inf_diff})
    return p_win, p_cite, ref_win, ref_cite, health


def evaluate_cell(run):
    cfg = run.cfg
    res = run.results
    z = np.load(res / "pipeline_reliable/tensors.npz")
    w, r, pairs = z["w"].astype(float), z["r"].astype(float), z["pairs"].astype(np.int64)
    n, K, T = int(z["n"]), int(z["K"]), int(z["T"])
    names = list(
        pd.read_parquet(res / "pipeline_reliable/active_criteria.parquet")["canonical_name"]
    )
    with open(res / "fit_reliable/fit.pkl", "rb") as fh:
        full = pickle.load(fh)
    assert (full.n, full.K, full.T) == (n, K, T)

    # In-sample: the saved fit on all T pairs. Its log-loss must reproduce the fitted NLL.
    pw_in, pc_in = gate_probabilities(full.s, full.gamma, full.beta, w, pairs)
    nll = -(
        np.sum(w * np.log(pw_in) + (1 - w) * np.log(1 - pw_in))
        + np.sum(r * np.log(pc_in) + (1 - r) * np.log(1 - pc_in))
    )
    assert abs(nll - full.nll_unregularized) <= 1e-6 * abs(full.nll_unregularized), (
        run.name,
        nll,
        full.nll_unregularized,
    )

    pw_oof, pc_oof, rw_oof, rc_oof, health = _oof(
        w, r, pairs, n, K, cfg.get("model", {}), cfg.get("validation", {})
    )

    preds = {
        "in_sample": (pw_in, pc_in, np.full(T, w.mean()), np.broadcast_to(r.mean(axis=0), (T, K))),
        "oof": (pw_oof, pc_oof, rw_oof, rc_oof),
    }
    rows, crit_rows, bins = [], [], {}
    for regime, (pw, pc, rw, rc) in preds.items():
        m_w, b_w = _metrics(pw, w, rw)
        m_c, b_c = _metrics(pc, r, rc)
        rows.append({"regime": regime, "gate": "winner", **m_w})
        rows.append({"regime": regime, "gate": "citation", "K": K, **m_c})
        bins[regime] = {"winner": b_w, "citation": b_c}
        for k, name in enumerate(names):
            m, _ = _metrics(pc[:, k], r[:, k], rc[:, k])
            crit_rows.append({"regime": regime, "k": k, "criterion": name, **m})
    return rows, crit_rows, bins, {"nll_check": float(nll), "folds": health}


def compute_results():
    """Refit every OOF fold and bootstrap every metric (~5 min); writes all result files."""
    rows, crit_rows, diag = [], [], {}
    bins = {CORPUS_LABEL[c]: {} for c in CORPORA}
    for corpus in CORPORA:
        ds = CORPUS_LABEL[corpus]
        for judge in JUDGES:
            run = load_run(run_name(judge, corpus))
            j = JUDGE_LABEL[judge]
            print(f"[run] {run.name}")
            r_, c_, b_, d_ = evaluate_cell(run)
            rows += [{"dataset": ds, "judge": j, "cell": run.name, **x} for x in r_]
            crit_rows += [{"dataset": ds, "judge": j, **x} for x in c_]
            bins[ds][j] = b_
            diag[run.name] = d_

    table = pd.DataFrame(rows)
    crit = pd.DataFrame(crit_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "metrics.csv", index=False)
    crit.to_csv(OUT / "per_criterion.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "n_bins": N_BINS,
                "ece": "equal-width, pooled over (pair, criterion) for the citation gate",
                "citation_empirical_reference": "per-criterion rate (gamma-only model)",
                "oof": "5-fold, RandomState(seed_split).permutation(T); references from training folds",
                "ci": f"95% percentile bootstrap over comparisons, {N_BOOT} replicates, seed {BOOT_SEED}",
                "metrics": rows,
                "diagnostics": diag,
            },
            indent=2,
            default=float,
        )
    )
    (OUT / "bins.json").write_text(
        json.dumps(
            {
                ds: {
                    j: {
                        rg: {
                            g: {k: np.asarray(v).tolist() for k, v in b.items()}
                            for g, b in by_gate.items()
                        }
                        for rg, by_gate in by_regime.items()
                    }
                    for j, by_regime in by_judge.items()
                }
                for ds, by_judge in bins.items()
            }
        )
    )
    return table, bins


def main() -> int:
    table, _ = compute_results()
    pd.set_option("display.width", 250)
    for regime in ("oof", "in_sample"):
        sub = table[table.regime == regime].set_index(["dataset", "gate", "judge"])
        fmt = {
            k: sub.apply(
                lambda row, k=k: f"{row[k]:.3f} [{row[k + '_lo']:.3f}, {row[k + '_hi']:.3f}]",
                axis=1,
            )
            for k in ("ece", "brier", "bss_chance", "bss_emp")
        }
        print(f"\n== {regime}\n" + pd.DataFrame(fmt).to_string())
    print(f"\nwrote → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
