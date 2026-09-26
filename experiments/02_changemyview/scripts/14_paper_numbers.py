"""Step 14: every number the CMV sections quote, recomputed from the shipped artifacts.

    python scripts/14_paper_numbers.py

Feeds the numbers in the text of Sec. `sec:cmv` and Apps. `app:experiment-details-cmv` /
`app:additional-results`. No API spend, no refitting beyond what steps 9-12 already wrote.
Writes results/paper_numbers.json (schema: ``core.ledger``).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401
import _config as C
from core.ledger import Ledger

SEC = "Sec. sec:cmv"
DATA = "App. app:cmv-dataset"
METHODS = "App. app:methods-cmv"
CONTAM = "App. app:cmv-contamination"
MEASURES = "App. app:additional-measures"


def main() -> int:
    R = C.RESULTS
    L = Ledger("02_changemyview")
    disc = C.cell("discovery") / "results"
    bld = json.loads((disc / "graph" / "build_report.json").read_text())
    fit = json.loads((disc / "fit_summary.json").read_text())
    ph = pd.read_parquet(disc / "phrases.parquet")
    bt = json.loads(
        (C.cell("bt_discovered") / "results" / "graph" / "build_report.json").read_text()
    )

    # ---- dataset and design -------------------------------------------------- #
    from core.cmv.data import build_pair_units, CONDITIONS

    pu = build_pair_units()
    split = pu.train.value_counts().to_dict()
    L.record(
        "dataset.tan_split",
        DATA,
        "Tan et al.'s training / held-out pair counts",
        {"train": int(split[1]), "heldout": int(split[0])},
    )

    pos, neg = CONDITIONS["root_reply"]
    wc = lambda c: pu[c].fillna("").str.split().str.len()
    elig = wc(pos).between(20, 1200) & wc(neg).between(20, 1200)
    n_el_test = int((elig & (pu.train == 0)).sum())
    n_el_train = int((elig & (pu.train == 1)).sum())
    L.record(
        "dataset.eligible",
        DATA,
        "pairs whose two root replies both have 20-1200 words, held-out and training",
        {"eligible_heldout": n_el_test, "eligible_train": n_el_train},
    )

    L.record(
        "design.discovery_graph",
        METHODS,
        "discovery graph: items, matchings, comparisons, forbidden same-thread edges",
        {
            k: bld[k]
            for k in (
                "n_items",
                "n_matchings",
                "n_pairs",
                "n_forbidden_same_op_edges",
                "n_same_op_pairs",
            )
        },
    )

    per_cmp = len(ph) / bld["n_pairs"]
    L.record(
        "design.citations",
        METHODS,
        "dimensions cited by the discovery judge: total, distinct phrases, per comparison",
        {
            "citations": int(len(ph)),
            "distinct_phrases": int(ph.phrase_norm.nunique()),
            "per_comparison": round(per_cmp, 3),
        },
    )

    L.record(
        "design.criteria",
        f"{SEC}; {METHODS}",
        "criteria in the discovery taxonomy, and the challenger-side criteria the D panels "
        "score (step 9 asserts the column count)",
        {"K_total": fit["K_total"], "challenger_side_used": 29},
    )

    L.record(
        "design.bt_graph",
        METHODS,
        "BT scoring graph: train / test exchanges, anchor and test-anchor comparisons, degrees",
        {
            k: bt[k]
            for k in (
                "train_ex",
                "test_ex",
                "anchor_pairs",
                "test_anchor_pairs",
                "anchor_degree",
                "test_anchors",
            )
        },
    )

    L.record(
        "design.shared_protocol",
        METHODS,
        "exchanges scored by every arm",
        {"n_items": bt["n_items"]},
    )

    # the prior panel's open-count draws: run 2's panels are shipped; run 1's per-draw counts
    # are recorded in the table of the cell's REPORT.md
    op = C.cell("prior_open") / "results"
    run2 = [len(p["criteria"]) for p in json.loads((op / "panels_blind.json").read_text())]
    assert run2 == [
        len(p["criteria"]) for p in json.loads((op / "panels_blind_run2.json").read_text())
    ], "panels_blind.json is run 2 (the canonical copy); both files should hold it"
    rep = (C.cell("prior_open") / "REPORT.md").read_text()
    run1 = [int(x) for x in re.search(r"\| run 1 \| ([\d, ]+) \|", rep).group(1).split(",")]
    pooled = run1 + run2
    stats = {
        name: {
            "counts": v,
            "mean": round(float(np.mean(v)), 2),
            "sd": round(float(np.std(v, ddof=1)), 2),
        }
        for name, v in (("run1", run1), ("run2", run2), ("pooled", pooled))
    }
    L.record(
        "prior.open_count",
        METHODS,
        "features named per draw when the prior elicitation chooses the count: two runs of "
        "10 draws, mean and SD per run and pooled",
        stats,
    )

    # the sample-elicited (S) feature set
    sp = json.loads((C.cell("sample_panel") / "results" / "panels_sample.json").read_text())[0]
    sv = {
        "pairs": sp["sample"]["n_pairs"],
        "replies": 2 * sp["sample"]["n_pairs"],
        "input_tokens": sp["usage"]["input_tokens"],
        "features": len(sp["criteria"]),
        "by_split": sp["sample"]["by_split"],
    }
    L.record(
        "sample.elicitation",
        METHODS,
        "S elicitation: pairs and replies shown, input tokens, features named, pairs per split",
        sv,
    )

    # D_r: the gate's survivors among the 29
    kept = C.reliable_features()
    g = yaml.safe_load((C.cell("discovery") / "discovery.yaml").read_text())["model"]
    L.record(
        "design.reliable_subset",
        f"{METHODS}; tab:cmv-win-rates",
        "reliability gate (kappa_max, rho threshold) and the D_r features it keeps",
        {
            "kappa_max": g["kappa_max"],
            "rho_threshold": g["rho_threshold"],
            "n": len(kept),
            "features": kept,
        },
    )

    acc = pd.read_csv(R / "accuracy.csv").set_index("arm")
    lam = {
        a: round(float(acc.loc[a, "lambda"]), 1)
        for a in ("D + BT", "D + PW", "P + BT", "P + PW", "Tan", "Emb.", "#words")
    }
    L.record("lambda.selected", METHODS, "cross-validated lambda = 1/C selected per arm", lam)
    L.record(
        "lambda.selected_dr_s",
        METHODS,
        "cross-validated lambda = 1/C selected for the D_r and S arms",
        {a: float(acc.loc[a, "lambda"]) for a in ("Dr + BT", "Dr + PW", "S + PW")},
    )

    # ---- contamination, win rates, dimensionality ----------------------------- #
    ct = json.loads((R / "contamination.json").read_text())
    lf = ct["lift"]
    L.record(
        "contamination.lift",
        f"{CONTAM}; tab:cmv-contamination",
        "lift P2 - P1 in accuracy (pp) and its 95% CI",
        {"lift_pp": round(lf["pp"], 3), "ci_pp": [round(x, 2) for x in lf["ci_pp"]]},
    )
    L.record(
        "contamination.mcnemar",
        "tab:cmv-contamination",
        "McNemar discordant counts (right under P2 only / P1 only) and p-value",
        {
            "p2_only_right": lf["mcnemar_b"],
            "p1_only_right": lf["mcnemar_c"],
            "p": round(lf["mcnemar_p"], 3),
        },
    )
    p1 = ct["by_outcome"]["p1"]
    L.record(
        "contamination.asymmetry",
        CONTAM,
        "P1 accuracy (%) on no-delta and on delta exchanges",
        {"no_delta": round(100 * p1["no_delta"], 1), "delta": round(100 * p1["delta"], 1)},
    )
    p2 = ct["by_outcome"]["p2"]
    d_rng = [100 * p2["delta_heldout"], 100 * p2["delta_train"]]
    n_rng = [100 * p2["no_delta_train"], 100 * p2["no_delta_heldout"]]
    L.record(
        "contamination.p2_shift",
        CONTAM,
        "P2 accuracy (%) range over the train/held-out cells, delta and no-delta",
        {
            "delta_pct": [round(min(d_rng), 1), round(max(d_rng), 1)],
            "no_delta_pct": [round(min(n_rng), 1), round(max(n_rng), 1)],
        },
    )

    wm = json.loads((R / "win_rates_meta.json").read_text())
    L.record(
        "win_rates.significant",
        "fig:cmv-win-rates; tab:cmv-win-rates",
        "criteria tested, criteria with BH q < 0.05, and whether all of those are positive",
        {k: wm[k] for k in ("n_criteria", "n_significant_bh05", "all_significant_are_positive")},
    )
    wr = pd.read_csv(R / "win_rates.csv").set_index("feature")
    ct_row = wr.loc["Confrontational tone"]
    L.record(
        "win_rates.confrontational",
        SEC,
        "win rate and BH q of 'Confrontational tone'",
        {"win_rate": round(float(ct_row.win_rate), 3), "q": round(float(ct_row.q), 3)},
    )

    dm = json.loads((R / "dimensionality.json").read_text())
    at16, nc, med = dm["pr_at_k16"], dm["noise_correction_pct"], dm["median_reliability"]
    dims = [
        (
            "at 16 features, D and D_r span about eight e",
            f"{SEC}; fig:cmv-dimension-curve",
            "noise-corrected participation ratio at k = 16: D + BT, D_r + BT",
            [at16["D + BT"], at16["Dr + BT"]],
        ),
        (
            "... six to seven under pointwise scoring",
            f"{SEC}; fig:cmv-dimension-curve",
            "noise-corrected participation ratio at k = 16: D + PW, D_r + PW",
            [at16["D + PW"], at16["Dr + PW"]],
        ),
        (
            "... against about four for P",
            f"{SEC}; fig:cmv-dimension-curve",
            "noise-corrected participation ratio at k = 16: P + BT, P + PW",
            [at16["P + BT"], at16["P + PW"]],
        ),
        (
            "the correction removes 15-20% of the raw PR ",
            MEASURES,
            "share of the raw participation ratio removed by the noise correction (%), BT arms",
            {a: v for a, v in nc.items() if a.endswith("BT")},
        ),
        (
            "... and 25-28% for the pointwise ones",
            MEASURES,
            "share of the raw participation ratio removed by the noise correction (%), PW arms",
            {a: v for a, v in nc.items() if a.endswith("PW")},
        ),
        (
            "median reliability ('D + PW', 'P + PW')",
            MEASURES,
            "median per-feature cross-run reliability: D + PW, P + PW",
            [med["D + PW"], med["P + PW"]],
        ),
        (
            "median reliability ('D + BT', 'P + BT')",
            MEASURES,
            "median per-feature split-half reliability: D + BT, P + BT",
            [med["D + BT"], med["P + BT"]],
        ),
        (
            "Spearman-Brown full-set reliability of the B",
            MEASURES,
            "Spearman-Brown full-data reliability of the BT fits: D + BT, P + BT",
            list(dm["spearman_brown_bt"].values()),
        ),
        (
            "no disattenuated matrix carried any negative",
            MEASURES,
            "negative eigenvalue mass of each disattenuated correlation matrix",
            dm["neg_eig_mass"],
        ),
        (
            "'the order holds under both measures' (PR an",
            "fig:cmv-dimension-curve-full",
            "arms ordered by participation ratio and by effective rank at k = 16",
            dm["order_at_k16"],
        ),
    ]
    for key, where, what, value in dims:
        L.record(f"dimensionality.{key}", where, what, value)

    # ---- matched-k accuracy (step 10b) --------------------------------------- #
    mk = R / "matched_k"
    cur = pd.read_parquet(mk / "accuracy_curves.parquet")
    at = lambda arm, k: 100 * float(cur.loc[(cur.arm == arm) & (cur.k == k), "acc"].iloc[0])
    summ = lambda d: json.loads((mk / d / "summary.json").read_text())["primary_mean_over_k"]
    k16 = lambda d: json.loads((mk / d / "summary.json").read_text())["k16"]
    curve = lambda d: pd.read_parquet(mk / d / "matched_k_curve.parquet")
    for d, arms in (
        ("DRPW_vs_DPW", "D_r + PW minus D + PW"),
        ("DRBT_vs_DBT", "D_r + BT minus D + BT"),
    ):
        s_, c_ = summ(d), curve(d)
        L.record(
            f"matched_k.{d}",
            f"{SEC}; fig:cmv-diff-dr",
            f"matched-k accuracy difference {arms}: mean over k <= 16 (pp), paired 95% CI, "
            f"and whether it is positive at every k",
            {
                "mean_pp": round(100 * s_["estimate"], 2),
                "ci_pp": [round(100 * s_["lo"], 3), round(100 * s_["hi"], 3)],
                "positive_at_every_k": bool((c_.delta > 0).all()),
            },
        )
    v = {
        "Dr+PW@16": round(at("Dr + PW", 16), 2),
        "D+PW@29": round(at("D + PW", 29), 2),
        "Dr+BT@16": round(at("Dr + BT", 16), 2),
        "D+BT@29": round(at("D + BT", 29), 2),
    }
    L.record(
        "matched_k.dr_full",
        SEC,
        "held-out accuracy (%) of D_r with all 16 features and of D with all 29, per scoring",
        v,
    )
    gap16 = at("Dr + PW", 16) - at("S + PW", 16)
    s_ = summ("DPW_vs_SPW")
    ks = sorted(cur.k.unique())
    lowest = [k for k in ks if (cur[cur.k == k].sort_values("acc").arm.iloc[0] == "S + PW")]
    words = 100 * float(acc.loc["#words", "acc"])
    s_curve = cur[cur.arm == "S + PW"].set_index("k").acc * 100
    first_above = [int(k) for k, a in s_curve.items() if a >= words]
    L.record(
        "matched_k.sample",
        SEC,
        "S + PW at matched k: gap to D_r + PW at k = 16 (pp); D + PW minus S + PW averaged "
        "over k <= 29 with paired 95% CI (pp); number of k at which S + PW is the lowest set; "
        "k at which S + PW reaches the #words accuracy",
        {
            "gap_k16_pp": round(gap16, 2),
            "mean_gap_pp": round(100 * s_["estimate"], 2),
            "ci_pp": [round(100 * s_["lo"], 2), round(100 * s_["hi"], 2)],
            "k_where_S_lowest": f"{len(lowest)} of {len(ks)}",
            "S_reaches_words_at_k": first_above,
        },
    )

    s_, t_ = summ("DRPW_vs_SPW"), k16("DRPW_vs_SPW")
    L.record(
        "matched_k.DRPW_vs_SPW",
        f"{SEC}; fig:cmv-diff-dr",
        "matched-k accuracy difference D_r + PW minus S + PW (fraction): mean over k <= 16 "
        "and value at k = 16, each with its paired 95% CI",
        {
            "mean_over_k": {x: s_[x] for x in ("estimate", "lo", "hi")},
            "k16": {x: t_[x] for x in ("estimate", "lo", "hi")},
        },
    )
    t_ = k16("DRBT_vs_SPW")
    L.record(
        "matched_k.DRBT_vs_SPW",
        "fig:cmv-diff-dr",
        "matched-k accuracy difference D_r + BT minus S + PW (fraction) at k = 16, with its "
        "paired 95% CI",
        {"k16": {x: t_[x] for x in ("estimate", "lo", "hi")}},
    )

    L.write(R / "paper_numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
