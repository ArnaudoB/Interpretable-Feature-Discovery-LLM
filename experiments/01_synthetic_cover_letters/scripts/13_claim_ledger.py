"""Step 13: the numbers quoted in Sec. sec:feature_extraction_empirical_results and the
appendices, recomputed from the artifacts of steps 00-12.

    python scripts/13_claim_ledger.py --config config.yaml

One record per quoted number or group of numbers (``core.ledger``). No refitting: it reads
what steps 00-12 wrote, so run those first. Writes results/paper_numbers.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.ledger import Ledger

#: (regime, gate, column) cells of results/calibration/metrics.csv reported in
#: sec:synthetic-calibration.
CALIBRATION_CELLS = [
    ("oof", "citation", "n"),
    ("oof", "direction", "n"),
    ("oof", "direction", "ece"),
    ("oof", "direction", "ece_lo"),
    ("oof", "direction", "ece_hi"),
    ("in_sample", "direction", "ece"),
    ("oof", "citation", "ece"),
    ("oof", "citation", "ece_lo"),
    ("oof", "citation", "ece_hi"),
    ("in_sample", "citation", "ece"),
    ("oof", "citation", "bss_emp"),  # vs the f = 0 (gap-independent) submodel
    ("oof", "citation", "bss_emp_lo"),
    ("oof", "citation", "bss_emp_hi"),
    ("in_sample", "citation", "bss_emp"),
    ("oof", "citation", "bss_chance"),
    ("oof", "citation", "bss_chance_lo"),
    ("oof", "citation", "bss_chance_hi"),
    ("oof", "direction", "bss_chance"),
    ("oof", "direction", "bss_chance_lo"),
    ("oof", "direction", "bss_chance_hi"),
    ("oof", "direction", "bss_emp"),
    ("oof", "direction", "bss_emp_lo"),
    ("oof", "direction", "bss_emp_hi"),
    ("in_sample", "direction", "bss_chance"),
]

SEC = "Sec. sec:feature_extraction_empirical_results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    root = Path(args.config).parent
    res = root / "results"
    cfg = yaml.safe_load(Path(args.config).read_text())
    L = Ledger("01_synthetic_cover_letters")
    J = lambda f: json.loads((res / f).read_text())  # noqa: E731

    tab = pd.read_csv(res / "tab_results.csv").set_index("method")
    abl = pd.read_csv(res / "tab_pointwise_ablation.csv").set_index("method")
    cost = J("tables/costs.json")
    ours, m1 = tab.loc["Ours"], tab.loc["M1"]

    # ---- Section 5 prose ------------------------------------------------------ #
    trk = J("tracked_dials.json")
    L.record(
        "results.tracked",
        SEC,
        "dials tracked by Ours and by Direct LLM, the dials Direct LLM "
        "misses, and whether Direct LLM--Multi-score tracks the same set",
        {
            "Ours": trk["Ours"]["n"],
            "Direct LLM": trk["M1"]["n"],
            "Direct LLM missed": trk["M1"]["missed"],
            "same_set_Multi_score": trk["_same_tracked_M1_M2"],
        },
    )
    L.record(
        "results.attributable",
        SEC,
        "attributable features (AF x K) for Ours and Direct LLM, and K",
        {"Ours": round(ours.AF * ours.K), "Direct LLM": round(m1.AF * m1.K), "K": [ours.K, m1.K]},
    )
    L.record(
        "results.leakage",
        SEC,
        "leakage LK of Ours and of Direct LLM",
        [round(ours.LK, 4), round(m1.LK, 4)],
    )
    L.record(
        "results.cost",
        SEC,
        "cost in USD of Direct LLM and of Ours",
        [round(cost["1a"], 4), round(cost["Ours"], 4)],
    )

    rows = {
        k: abl.loc[v]
        for k, v in (
            ("32", "PW-32 (1 rep)"),
            ("32m", "PW-32 (n*)"),
            ("21", "PW-21 (1 rep)"),
            ("21m", "PW-21 (n*)"),
        )
    }
    n_attr = {k: round(r.AF * r.K) for k, r in rows.items()}
    lk = {k: round(r.LK, 4) for k, r in rows.items()}
    sm = {k: round(r.SM, 4) for k, r in rows.items()}
    L.record(
        "ablations.vs_ours",
        SEC,
        "ablation arms: attributable features, leakage and Stat.R. "
        "per arm, and the Multi-score arm's cost relative to Ours",
        {
            "attributable": n_attr,
            "LK": lk,
            "Stat.R.": sm,
            "cost_ratio_multi_score": round(cost["PWC-32m"] / cost["Ours"], 2),
        },
    )
    L.record(
        "ablations.filtering",
        SEC,
        "effect of reliability filtering and of statistical "
        "scoring: attributable features / K of the two Multi-score ablations and of Ours, "
        "and Tracked / Sem.R. of the filtered Multi-score arm",
        {
            "32m": [n_attr["32m"], int(rows["32m"].K)],
            "21m": [n_attr["21m"], int(rows["21m"].K)],
            "Ours": round(ours.AF * ours.K),
            "SF_SR_21m": [rows["21m"].SF, rows["21m"].SR],
        },
    )

    # ---- Appendix: dataset and metrics ---------------------------------------- #
    dc = J("dataset_checks.json")
    b, f = dc["balanced"], dc["full_corpus"]
    L.record(
        "dataset.orthogonality",
        "App. app:dataset",
        "max |r| over non-structural dial pairs "
        "and max |r| with letter length, on the balanced pool and on the full corpus",
        {
            "balanced": [round(b["max_nonstructural_abs_corr"], 4), round(b["max_length_corr"], 4)],
            "full": [round(f["max_nonstructural_abs_corr"], 4), round(f["max_length_corr"], 4)],
        },
    )
    th = J("theta_calibration.json")
    L.record(
        "metrics.eta",
        "App. app:metrics",
        "semantic-acceptance threshold eta (95th percentile of the Hungarian-rejected cosines)",
        round(th["theta"], 5),
    )
    tr = J("tau_robustness.json")
    floor = tr["arms"]["Ours"]["permutation_floor"]["per_dial_max_mean"]
    stable = all(
        len({round(r[m], 9) for r in v["tau_sweep"]}) == 1
        for v in tr["arms"].values()
        for m in ("SF", "SR", "AF")
    )
    L.record(
        "metrics.tau",
        "App. app:metrics",
        "tracking threshold tau, its permutation floor, the "
        "tau sweep, and whether Tracked / Sem.R. / Attrib. are constant over it",
        {
            "tau": cfg["metrics"]["tau"],
            "floor": round(floor, 3),
            "taus": tr["taus"],
            "Tracked_SemR_Attrib_constant_over_taus": stable,
        },
    )

    fs, sw = J("fit_summary.json"), J("sandwich_ci.json")
    L.record(
        "feats.gate",
        "tab:feats-ours",
        "surfaced and kept features and the kappa / rho gate "
        "thresholds (kept: kappa <= kappa_max and rho > rho)",
        {
            "K_total": fs["K_total"],
            "K_kept": fs["K_kept"],
            "kappa_max": fs["kappa_max"],
            "rho": fs["rho_threshold"],
        },
    )
    L.record(
        "feats.beta",
        "tab:feats-ours",
        "fitted position bias beta and its 95% sandwich CI",
        [round(sw["beta"], 5), round(sw["beta_lo"], 4), round(sw["beta_hi"], 4)],
    )

    # ---- Appendix: run variance and the paired design ------------------------- #
    bd = J("bootstrap_discovery_counts.json")
    L.record(
        "variance.discovered",
        "App. app:variance-bands",
        "mean number of discovered features "
        "over the Ours bootstrap replicates, and the number of replicates",
        {"mean": round(bd["discovered_mean"], 3), "B": bd["B"]},
    )
    pd_ = J("tables/tab_paired_design.json")
    L.record(
        "paired.budget",
        "App. app:paired-design",
        "pairs retained and average degree under "
        "letter resampling, K of Ours at that budget, and the costs of Ours at degree 25 and "
        "40 and of Local LLMs",
        {
            k: round(pd_[k], 3)
            for k in (
                "pairs_retained",
                "average_degree",
                "ours_K_at_reduced_budget",
                "usd_comparative_at_degree_25",
                "usd_ours_degree_40",
                "usd_local_llms",
            )
        },
    )
    cf = pd_["canonical_fusion_mean_diff"]
    L.record(
        "paired.canonical_fusion",
        "App. app:paired-design",
        "mean paired differences (Ours - "
        "Local LLMs) per metric, conditioning on the canonical fusion",
        {k: round(v, 4) for k, v in cf.items()},
    )
    rng = pd_["per_fusion_range"]
    L.record(
        "paired.per_fusion_range",
        "App. app:paired-design",
        "range over fusions of the mean paired differences, per metric",
        {k: [round(x, 4) for x in v] for k, v in rng.items()},
    )

    # ---- Appendix: the per-feature position-bias refit ------------------------ #
    bk = J("beta_check/beta_per_criterion.json")
    rs = bk["relative_shift"]
    L.record(
        "beta.significant",
        "App. app:beta-orthogonality",
        "likelihood-ratio test of equal "
        "per-feature position biases beta_k, and the number of individually significant beta_k",
        {
            "LR_chi2": round(bk["LR_chi2"], 1),
            "df": bk["LR_df"],
            "p": bk["LR_p"],
            "n_significant": f"{bk['n_beta_k_significant_1p96']} of {bk['K']}",
        },
    )
    L.record(
        "beta.median_shift",
        "App. app:beta-orthogonality",
        "median absolute score shift under "
        "the beta_k refit, relative to the per-feature score spread (cell-wise)",
        round(rs["cellwise_median"], 4),
    )
    L.record(
        "beta.p90_shift",
        "App. app:beta-orthogonality",
        "90th percentile of the relative score "
        "shift, cell-wise and per feature, and the per-feature median",
        {
            "cellwise_p90": round(rs["cellwise_p90"], 4),
            "per_feature_p90": round(rs["per_criterion_p90"], 4),
            "per_feature_median": round(rs["per_criterion_median"], 4),
        },
    )
    L.record(
        "beta.shift_vs_beta",
        "App. app:beta-orthogonality",
        "Pearson r between a feature's mean absolute score shift and |beta_k|",
        round(bk["shift_vs_beta_k"]["pearson_mean_abs_shift_vs_abs_beta_k"], 4),
    )
    sb = bk["slot_balance"]
    L.record(
        "beta.slot_imbalance",
        "App. app:beta-orthogonality",
        "theoretical and realized standard deviation of each letter's first-slot share",
        {k: round(v, 4) if isinstance(v, float) else v for k, v in sb.items()},
    )

    # ---- Appendix: model calibration ------------------------------------------ #
    m = pd.read_csv(res / "calibration" / "metrics.csv").set_index(["regime", "gate"])
    L.record(
        "calibration.all",
        "App. sec:synthetic-calibration",
        "cell counts, ECE and Brier skill "
        "scores (with 95% bootstrap CIs) of the citation and direction gates, out of fold and "
        "in sample; keys are regime/gate/column of results/calibration/metrics.csv",
        {
            "/".join(k): (
                int(m.loc[k[:2], k[2]]) if k[2] == "n" else round(float(m.loc[k[:2], k[2]]), 3)
            )
            for k in CALIBRATION_CELLS
        },
    )

    L.write(res / "paper_numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
