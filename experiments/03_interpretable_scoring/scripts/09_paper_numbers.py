"""Step 9: the numbers quoted in Appendix ``app:verdict-extension``, recomputed from artifacts.

    python scripts/09_paper_numbers.py

No API spend, no refitting. Reads each run's graph, judgments, diagnostics and reliable
refit, the step-7 constructs, the step-8 calibration table, and the Spearman values written
by ``figures/make_total_score_distributions.py``. Writes ``results/paper_numbers.json``
(one record per quoted number or group of numbers; see ``core/ledger.py``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import (  # noqa: E402
    CORPORA,
    CORPUS_LABEL,
    EXPERIMENT,
    JUDGE_LABEL,
    JUDGES,
    load_run,
    run_name,
)
from core.ledger import Ledger  # noqa: E402
from core.model.holistic.diagnostics import score_spread  # noqa: E402
from core.model.holistic.inference import std_errors  # noqa: E402

RESULTS = EXPERIMENT / "results"
APP = "App. app:verdict-extension"
DETAILS = "App. sec:extension-appendix"


def rng(xs, nd=3):
    return [round(float(min(xs)), nd), round(float(max(xs)), nd)]


def _load(run):
    import pickle

    with open(run.results / "fit_reliable/fit.pkl", "rb") as f:
        fit = pickle.load(f)
    cov = np.load(run.results / "fit_reliable/cov.npy")
    names = list(
        pd.read_parquet(run.results / "pipeline_reliable/active_criteria.parquet")["canonical_name"]
    )
    return fit, cov, names


def main() -> int:
    L = Ledger("03_interpretable_scoring")
    runs = {(j, c): load_run(run_name(j, c)) for c in CORPORA for j in JUDGES}

    # ------------------------------------------------------------------ design
    for c in CORPORA:
        rep = json.loads((runs[(JUDGES[0], c)].graph / "build_report.json").read_text())
        L.record(
            f"design.{c}",
            DETAILS,
            f"{CORPUS_LABEL[c]} comparison design: essays, pairs, perfect matchings, "
            "per-essay degree, seed, selected band histogram",
            {
                "n_essays": rep["n_essays"],
                "n_pairs": rep["n_pairs"],
                "n_matchings": rep["n_matchings"],
                "degree": rep["degree_per_essay"],
                "seed": rep["seed"],
                "bands": rep["selected_band_histogram"],
            },
        )

    parsed = {
        r.name: int((~pd.read_parquet(r.results / "judgments.parquet")["parse_failed"]).sum())
        for r in runs.values()
    }
    L.record(
        "elicitation.parsed",
        DETAILS,
        "judge responses that parsed, in total and per run",
        {"parsed": sum(parsed.values()), "per_run": parsed},
    )

    # ------------------------------------------------------------------ gates
    kept, diag_rows = {}, []
    for (j, c), r in runs.items():
        sel = json.loads((r.results / "pipeline_reliable/selection.json").read_text())
        kept[r.name] = sel["K_after"]
        d = pd.read_parquet(r.results / "fit/criterion_diagnostics.parquet")
        d = d.assign(
            run=r.name,
            corpus=c,
            fail_kappa=d.kappa >= sel["kappa_max"],
            fail_rho=d.rho <= sel["rho_min"],
        )
        diag_rows.append(d)
    L.record(
        "gates.retained",
        DETAILS,
        "features retained by the (rho, kappa) screen: range over runs and per run",
        {"range": [min(kept.values()), max(kept.values())], "per_run": kept},
    )

    D = pd.concat(diag_rows, ignore_index=True)
    drops = D[~D.keep]
    by_run = {
        n: {
            "kappa_only": int((g.fail_kappa & ~g.fail_rho).sum()),
            "rho_only": int((g.fail_rho & ~g.fail_kappa).sum()),
            "both": int((g.fail_kappa & g.fail_rho).sum()),
        }
        for n, g in drops.groupby("run", sort=False)
    }
    n_kappa = int(drops.fail_kappa.sum())
    L.record(
        "gates.kappa_causes_most",
        DETAILS,
        "screened-out features: total, those failing kappa, those failing rho only; "
        "per-run breakdown",
        {
            "drops": int(len(drops)),
            "involving_kappa": n_kappa,
            "rho_only": int((drops.fail_rho & ~drops.fail_kappa).sum()),
            "per_run": by_run,
        },
    )

    fk, pk = D[D.fail_kappa], D[~D.fail_kappa]
    per_run_lower = {
        n: bool(g[g.fail_kappa].citation_rate.mean() < g[~g.fail_kappa].citation_rate.mean())
        for n, g in D.groupby("run", sort=False)
        if g.fail_kappa.any()
    }
    L.record(
        "gates.kappa_removes_frequent",
        DETAILS,
        "citation rate of features failing vs passing kappa; kappa failure rate by "
        "citation-rate bin; kappa failures cited in more than half the pairs",
        {
            "median_cite_rate_kappa_fail": round(float(fk.citation_rate.median()), 4),
            "median_cite_rate_kappa_pass": round(float(pk.citation_rate.median()), 4),
            "runs_where_kappa_fails_are_LESS_cited_on_average": f"{sum(per_run_lower.values())} of {len(per_run_lower)}",
            "kappa_fail_rate_by_citation_rate": {
                str(b): f"{int(g.sum())}/{len(g)}"
                for b, g in D.groupby(
                    pd.cut(D.citation_rate, [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]), observed=True
                ).fail_kappa
            },
            "frequent_kappa_failures": fk.loc[
                fk.citation_rate > 0.5, ["run", "canonical_name", "citation_rate"]
            ]
            .round(3)
            .to_dict("records"),
        },
    )

    # ------------------------------------------------------ gate sensitivity
    # The strict gate (rho > 0.70, kappa < 50) is applied to the same all-K diagnostics. A
    # strict survivor is always a reliable feature, so its cross-judge construct is read off
    # the shipped clustering (no new LLM call).
    cl = json.loads((RESULTS / "cross_judge/ellipse_clusters.json").read_text())
    strict = {}
    for (j, c), r in runs.items():
        if c != "ellipse":
            continue
        d = pd.read_parquet(r.results / "fit/criterion_diagnostics.parquet")
        s_ = set(d.loc[(d.rho > 0.70) & (d.kappa < 50), "canonical_name"])
        assert s_ <= set(d.loc[d.keep, "canonical_name"])
        strict[JUDGE_LABEL[j]] = s_
    assert set(strict) == set(cl["judges"]), "cluster judge ids must equal the judge labels"
    n_per = {k: len(v) for k, v in strict.items()}
    three = [
        g["name"]
        for g in cl["clusters"]
        if sum(m["name"] in strict[m["judge"]] for m in g["members"]) >= 3
    ]
    L.record(
        "gates.sensitivity",
        DETAILS,
        "ELLIPSE: features per judge and constructs shared by >= 3 judges, at the "
        "reported gate (rho > 0.50, kappa < 100) and at the strict gate "
        "(rho > 0.70, kappa < 50)",
        {
            "features_per_judge": n_per,
            "three_judge_constructs": three,
            "three_judge_constructs_at_reported_gate": sum(
                len(g["members"]) >= 3 for g in cl["clusters"]
            ),
            "features_per_judge_at_reported_gate": {
                JUDGE_LABEL[j]: kept[r.name] for (j, c), r in runs.items() if c == "ellipse"
            },
            "n_three_judge_constructs": len(three),
        },
    )
    dv = pd.read_parquet(
        runs[("deepseek_v4_flash", "ellipse")].results / "fit/criterion_diagnostics.parquet"
    )
    voc = dv[dv.canonical_name.str.contains("ocabular")].iloc[0]
    vg = next(g for g in cl["clusters"] if g["name"].lower().startswith("vocabular"))
    L.record(
        "gates.vocabulary",
        DETAILS,
        "ELLIPSE Vocabulary construct: its judges, and DeepSeek's vocabulary feature "
        "with its rho, kappa and screen outcome",
        {
            "construct_judges": [m["judge"] for m in vg["members"]],
            "deepseek_feature": voc.canonical_name,
            "rho": round(float(voc.rho), 3),
            "kappa": round(float(voc.kappa), 1),
            "kept": bool(voc.keep),
        },
    )

    # ------------------------------------------------------------------ spreads
    spread = {}  # (judge label, criterion) -> row
    for (j, c), r in runs.items():
        fit, cov, names = _load(r)
        sp = score_spread(fit, cov, std_errors(fit, cov)["s"])
        for k, nm in enumerate(names):
            spread[(c, JUDGE_LABEL[j], nm)] = {
                "std": float(sp.std[k]),
                "denoised": float(sp.std_denoised[k]),
                "rho": float(sp.rho[k]),
            }

    # Denoising, ELLIPSE reliable sets.
    ell = pd.DataFrame(
        [{"judge": jl, "criterion": nm, **v} for (c, jl, nm), v in spread.items() if c == "ellipse"]
    )
    ell["shrink_pct"] = 100 * (1 - ell.denoised / ell["std"])
    ell["rank_std"] = ell.groupby("judge")["std"].rank(ascending=False).astype(int)
    ell["rank_dn"] = ell.groupby("judge")["denoised"].rank(ascending=False).astype(int)
    moved = int((ell.rank_std != ell.rank_dn).sum())
    shrink = float(ell.shrink_pct.mean())
    L.record(
        "denoising.ellipse",
        DETAILS,
        "ELLIPSE: mean % shrinkage of sigma_k from denoising, and features whose "
        "within-judge rank changes",
        {"mean_shrink_pct": round(shrink, 2), "reordered": moved, "n_criteria": len(ell)},
    )

    consensus = {}
    for c in CORPORA:
        cl = json.loads((RESULTS / "cross_judge" / f"{c}_clusters.json").read_text())
        axes = []
        for g in cl["clusters"]:
            vals = {
                m["judge"]: spread[(c, m["judge"], m["name"])]["denoised"] for m in g["members"]
            }
            axes.append({"name": g["name"], "size": len(vals), "vals": vals})
        consensus[c] = axes

    for c in CORPORA:
        ax3 = [a for a in consensus[c] if a["size"] >= 3]
        gpt_min = sum(min(a["vals"], key=a["vals"].get) == "gpt-5.4-mini" for a in ax3)
        L.record(
            f"sigma.gpt_most_compressed.{c}",
            f"{APP}, fig:cross_judge_radar",
            f"{CORPUS_LABEL[c]}: consensus axes (>= 3 judges) on which GPT-5.4 mini has "
            "the smallest denoised sigma_k",
            {"gpt_smallest": gpt_min, "consensus_axes": len(ax3)},
        )

    def construct(c, name):
        return next(a for a in consensus[c] if a["name"] == name)

    th = construct("asap2", "Thesis control")
    L.record(
        "sigma.thesis_control",
        f"{APP}, fig:cross_judge_radar",
        "ASAP 2.0 Thesis control: range of denoised sigma_k over judges",
        {"range": rng(th["vals"].values(), 4), "n_judges": th["size"]},
    )
    sl = construct("asap2", "Sentence-level control")
    L.record(
        "sigma.sentence_level",
        f"{APP}, fig:cross_judge_radar",
        "ASAP 2.0 Sentence-level control: range of denoised sigma_k over judges",
        {"range": rng(sl["vals"].values(), 4), "n_judges": sl["size"]},
    )

    largest = {}
    for c in CORPORA:
        a, j = max(
            ((a, j) for a in consensus[c] for j in a["vals"]), key=lambda t: t[0]["vals"][t[1]]
        )
        largest[c] = {
            "construct": a["name"],
            "judge": j,
            "denoised_sigma": round(a["vals"][j], 3),
            "n_judges_on_axis": a["size"],
        }
    L.record(
        "sigma.largest_single_judge",
        APP,
        "per corpus: the largest denoised sigma_k, its construct, judge, and the number "
        "of judges on that construct",
        largest,
    )

    # ------------------------------------------------------------------ agreement
    agree = pd.read_csv(RESULTS / "total_score_agreement.csv")
    for c in CORPORA:
        v = agree.loc[agree.corpus == c, "spearman_rs"]
        L.record(
            f"agreement.spearman.{c}",
            f"{APP}, fig:total_score_distribution",
            f"{CORPUS_LABEL[c]}: Spearman r_s between the fitted total score and the human "
            "holistic score, range and per judge",
            {
                "range": rng(v, 4),
                "per_judge": dict(zip(agree.loc[agree.corpus == c, "judge"], v.round(4))),
            },
        )

    # ------------------------------------------------------------------ position bias
    pb = {}
    for (j, c), r in runs.items():
        fit, _, _ = _load(r)
        w = np.load(r.results / "pipeline_reliable/tensors.npz")["w"]
        pb.setdefault(c, []).append(
            {
                "judge": JUDGE_LABEL[j],
                "beta": round(float(fit.beta), 4),
                "first_position_win_rate": round(float(w.mean()), 4),
            }
        )
    betas = [abs(x["beta"]) for v in pb.values() for x in v]
    monotone = {
        c: bool(pd.DataFrame(v).sort_values("beta").first_position_win_rate.is_monotonic_increasing)
        for c, v in pb.items()
    }
    L.record(
        "position_bias",
        DETAILS,
        "fitted position coefficient beta per run, its largest magnitude, and whether "
        "ordering judges by beta orders their first-position win rates",
        {"max_abs_beta": max(betas), "rank_order_matches_win_rate": monotone, "per_run": pb},
    )

    # ------------------------------------------------------------------ calibration
    cal = pd.read_csv(RESULTS / "calibration" / "metrics.csv")
    oof, ins = cal[cal.regime == "oof"], cal[cal.regime == "in_sample"]

    def sub(t, ds, gate):
        return t[(t.dataset == ds) & (t.gate == gate)]

    for ds in ("ELLIPSE", "ASAP-2"):
        e = sub(oof, ds, "winner").ece
        L.record(
            f"calibration.winner_ece.{ds}",
            f"{DETAILS}, fig:calibration_reliability_{'ellipse' if ds == 'ELLIPSE' else 'asap'}",
            f"{ds}: range over judges of the winner gate's out-of-fold ECE",
            {"range": rng(e, 4)},
        )
    w_o = oof[oof.gate == "winner"].set_index(["dataset", "judge"]).ece
    w_i = ins[ins.gate == "winner"].set_index(["dataset", "judge"]).ece
    L.record(
        "calibration.winner_oof_not_worse",
        DETAILS,
        "runs whose winner-gate out-of-fold ECE is at most the in-sample ECE",
        {"oof_le_in_sample": f"{int((w_o <= w_i).sum())} of {len(w_o)}"},
    )
    for regime, t in (("in-sample", ins), ("out-of-fold", oof)):
        e = t[t.gate == "citation"].ece
        L.record(
            f"calibration.citation_ece.{regime}",
            DETAILS,
            f"range over runs of the citation gate's {regime} ECE",
            {"range": rng(e, 4)},
        )

    # One range per corpus, taken over both references (chance and empirical rate).
    for ds in ("ELLIPSE", "ASAP-2"):
        s = sub(oof, ds, "winner")
        both = list(s.bss_chance) + list(s.bss_emp)
        L.record(
            f"calibration.winner_bss.{ds}",
            f"{DETAILS}, fig:calibration_skill",
            f"{ds}: range over judges of the winner gate's out-of-fold Brier skill, vs "
            "chance, vs the empirical rate, and over both",
            {
                "vs_chance": rng(s.bss_chance),
                "vs_empirical": rng(s.bss_emp),
                "union_over_both_references": rng(both),
            },
        )
    s = oof[oof.gate == "citation"]
    L.record(
        "calibration.citation_bss_habit",
        f"{DETAILS}, fig:calibration_skill",
        "citation gate's out-of-fold Brier skill against the per-feature empirical "
        "rate (habit-only submodel): range overall and per corpus, smallest lower "
        "bootstrap bound",
        {
            "range": rng(s.bss_emp),
            **{
                f"range_{ds}": rng(sub(oof, ds, "citation").bss_emp) for ds in ("ELLIPSE", "ASAP-2")
            },
            "smallest_lower_bound": round(float(s.bss_emp_lo.min()), 3),
        },
    )
    g = sub(oof, "ASAP-2", "winner").set_index("judge").loc["gpt-5.4-mini"]
    L.record(
        "calibration.winner_skills_coincide",
        f"{DETAILS}, fig:calibration_skill",
        "GPT-5.4 mini on ASAP 2.0: winner-gate Brier skill vs chance and vs the "
        "empirical rate, and its first-shown win rate",
        {
            "gpt-5.4-mini_ASAP-2": {
                "vs_chance": round(float(g.bss_chance), 3),
                "vs_empirical": round(float(g.bss_emp), 3),
                "first_shown_rate": round(float(g.rate), 3),
            }
        },
    )

    L.write(RESULTS / "paper_numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
