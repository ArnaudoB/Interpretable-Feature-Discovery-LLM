"""Step 4: fit the holistic model on a run's tensors and write a human-readable summary.

    python scripts/04_fit.py --run all               # all K criteria  -> results/fit/
    python scripts/04_fit.py --run all --reliable    # step 6: the survivors of step 5
                                                     #   -> results/fit_reliable/

Calls ``core.model.holistic.fit`` + ``sandwich_covariance`` and writes ``fit.pkl``,
``cov.npy``, ``std_errors.json``, ``heldout_metrics.json`` and ``summary.md``. The reliable
refit feeds every figure and number of App. ``app:verdict-extension``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import resolve  # noqa: E402

from core.model.holistic import fit, per_criterion_se, sandwich_covariance, std_errors  # noqa: E402

log = logging.getLogger("scripts.04_fit")


def _heldout_accuracy(w, r, pairs, K, n, model_cfg, holdout_frac: float, seed: int):
    """Refit on a (1-holdout_frac) split and report accuracy on the rest."""
    T = len(w)
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(T)
    n_test = int(round(holdout_frac * T))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    res_tr = fit(
        w[train_idx],
        r[train_idx],
        pairs[train_idx],
        K=K,
        n=n,
        ridge=float(model_cfg.get("ridge", 1e-3)),
        include_gamma=bool(model_cfg.get("include_gamma", True)),
        tol=float(model_cfg.get("tol", 1e-8)),
        max_iter=int(model_cfg.get("max_iter", 1000)),
        seed=seed + 1,
    )
    # Predict winner on test set via the BT predictor.
    a = pairs[test_idx, 0]
    b = pairs[test_idx, 1]
    z_bt = (res_tr.s[a] - res_tr.s[b]).sum(axis=1) + res_tr.beta
    pred = (z_bt > 0).astype(np.int64)
    acc = float((pred == w[test_idx]).mean())
    return acc, int(len(train_idx)), int(len(test_idx))


def _write_summary(
    out_dir: Path, cfg: dict, result, ses: dict, cov, heldout: dict, active: pd.DataFrame
) -> None:
    lines: list[str] = []
    label = cfg["cell"]["label"]
    lines.append(f"# Fit summary — `{label}`")
    lines.append("")
    lines.append(f"- judge: `{cfg['cell']['judge']}` ({cfg['cell']['judge_full_name']})")
    lines.append(f"- dataset: `{cfg['cell']['dataset']}`")
    lines.append(f"- prompt variant: `{cfg['cell']['prompt_variant']}`")
    lines.append(f"- n essays: {result.n}")
    lines.append(f"- K active criteria: {result.K}")
    lines.append(f"- T comparisons: {result.T}")
    lines.append("")

    lines.append("## Global parameters")
    lines.append("")
    lines.append(
        f"- β̂ = {result.beta:+.4f} (SE {ses['beta']:.4f}) ; σ(β̂) = {result.sigmoid_beta:.4f}"
    )
    lines.append(
        f"- NLL (regularized) = {result.nll:.4f} ; NLL (unregularized) = {result.nll_unregularized:.4f}"
    )
    lines.append(f"- L-BFGS-B iterations: {result.n_iter} ; converged: {result.converged}")
    lines.append(f"- Convex sanity: |L_primary − L_sanity| = {result.inf_diff:.3e}")
    lines.append("")

    lines.append("## Per-criterion table")
    lines.append("")
    lines.append(
        "| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for k in range(result.K):
        pc = (
            per_criterion_se(result, cov, k)
            if cov is not None
            else {
                "std_s": float(np.std(result.s[:, k])),
                "std_s_se": float("nan"),
                "gamma": (float(result.gamma[k]) if result.include_gamma else float("nan")),
                "gamma_se": float("nan"),
            }
        )
        name = str(active.iloc[k]["canonical_name"]) if k < len(active) else ""
        share = float(active.iloc[k]["share"]) if k < len(active) else float("nan")
        sc = float(result.sign_corrs[k])
        lines.append(
            f"| {k} | {name} | {share:.3f} | {pc['std_s']:.3f} | {pc['std_s_se']:.3f} | "
            f"{pc['gamma']:+.3f} | {pc['gamma_se']:.3f} | {sc:+.3f} |"
        )
    if (np.asarray(result.sign_corrs) < 0).any():
        bad = [int(k) for k in range(result.K) if result.sign_corrs[k] < 0]
        lines.append("")
        lines.append(f"⚠ negative sign correlations at criteria {bad} — see DERIVATION.md §3.3.")
    lines.append("")

    if heldout is not None:
        lines.append("## Held-out validation")
        lines.append("")
        lines.append(f"- n_train = {heldout['n_train']}  n_test = {heldout['n_test']}")
        lines.append(f"- BT-predictor winner accuracy = {heldout['acc']:.4f}")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))


def fit_run(run, reliable: bool) -> None:
    cfg = run.cfg
    pipeline_dir = run.results / ("pipeline_reliable" if reliable else "pipeline")
    out_dir = run.results / ("fit_reliable" if reliable else "fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== %s (%s)", run.name, out_dir.name)

    # Load pipeline outputs.
    npz = np.load(pipeline_dir / "tensors.npz")
    w = npz["w"]
    r = npz["r"]
    pairs = npz["pairs"]
    n = int(npz["n"])
    K = int(npz["K"])
    T = int(npz["T"])
    active = pd.read_parquet(pipeline_dir / "active_criteria.parquet")
    log.info("loaded tensors: n=%d K=%d T=%d", n, K, T)

    # Fit.
    model_cfg = cfg.get("model", {})
    result = fit(
        w,
        r,
        pairs,
        K=K,
        n=n,
        ridge=float(model_cfg.get("ridge", 1e-3)),
        include_gamma=bool(model_cfg.get("include_gamma", True)),
        tol=float(model_cfg.get("tol", 1e-8)),
        max_iter=int(model_cfg.get("max_iter", 1000)),
        seed=int(cfg["cell"].get("seed", 42)),
    )
    log.info(
        "fit: converged=%s n_iter=%d β=%.4f inf_diff=%.2e",
        result.converged,
        result.n_iter,
        result.beta,
        result.inf_diff,
    )

    # Sandwich.
    cov = sandwich_covariance(result, w, r, pairs)
    ses = std_errors(result, cov)

    # Held-out accuracy.
    val_cfg = cfg.get("validation", {})
    holdout_frac = float(val_cfg.get("holdout_frac", 0.2))
    seed_split = int(val_cfg.get("seed_split", 42))
    acc, n_train, n_test = _heldout_accuracy(
        w,
        r,
        pairs,
        K,
        n,
        model_cfg,
        holdout_frac,
        seed_split,
    )
    heldout = {
        "acc": acc,
        "n_train": n_train,
        "n_test": n_test,
        "holdout_frac": holdout_frac,
        "seed_split": seed_split,
    }
    log.info("held-out accuracy: %.4f (n_test=%d)", acc, n_test)

    # Save.
    with open(out_dir / "fit.pkl", "wb") as f:
        pickle.dump(result, f)
    np.save(out_dir / "cov.npy", cov)
    (out_dir / "std_errors.json").write_text(
        json.dumps(
            {
                "s": ses["s"].tolist(),
                "gamma": ses["gamma"].tolist() if ses["gamma"] is not None else None,
                "beta": ses["beta"],
            },
            indent=2,
        )
    )
    (out_dir / "heldout_metrics.json").write_text(json.dumps(heldout, indent=2))

    _write_summary(out_dir, cfg, result, ses, cov, heldout, active)
    print(f"fit complete → {out_dir}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="a run name, a comma list, or 'all'")
    p.add_argument(
        "--reliable",
        action="store_true",
        help="fit results/pipeline_reliable (step 6) instead of results/pipeline",
    )
    args = p.parse_args()
    for run in resolve(args.run):
        fit_run(run, args.reliable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
