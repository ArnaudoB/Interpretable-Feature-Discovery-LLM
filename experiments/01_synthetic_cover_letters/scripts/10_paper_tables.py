"""Step 10: emit the three paper tables as LaTeX.

    python scripts/10_paper_tables.py --config config.yaml

tab:main        -- the nine-row point-estimate table: Ours, the four benchmarks, and the four
                   pointwise-canonical ablation rows from 09b_ablation_table.py. No bands.
tab:main-bands  -- the five discovery methods again, each followed by its OWN run-variance
                   band. Those bands are NOT comparable across methods: the arms resample
                   different units (matchings / letters / elicitation subsets). They describe a
                   method's own stability, nothing more.
tab:paired      -- the two paired comparisons that ARE valid, because each shares its
                   resampling draws with the arm it is compared against.

Both tab:main and tab:main-bands take every point estimate (including 2b's) from
``tab_results.csv``; the bands table adds each method's band row beneath it.

Metric display names (Tracked / Stat.R. / Sem.R. / Attrib. / LK) come from
``core.metrics.DISPLAY_NAME``; the internal keys stay SF/SM/SR/AF/LK, which is what every
shipped artifact is keyed on.

Everything is read from results/, nothing is transcribed. Costs are on a Batch-API basis
throughout (see 06d_cost_match.py) and are reported without a CI: cost is fixed by a method's
design, and bootstrapping it only measures token jitter (and is distorted for 1a, whose resample
scores only ~126 unique letters).

Writes results/tables/tab_main.tex, tab_main_bands.tex, tab_paired.tex, tab_paired.json (the
values of fig:paired), costs.json and the values of Appendix app:paired-design
(tab_paired_design.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.metrics import DISPLAY_NAME, HIGHER_BETTER, PERCENT

METRICS = ["SF", "SM", "SR", "AF", "LK"]

# tab:main's row labels and grouping, as the paper prints them. The 1a/1b/2a/2b codes used
# throughout the code and in tab:main-bands map onto the paper's method names here.
MAIN_ROWS = [
    (None, "Ours", r"\textbf{Ours}"),
    ("Benchmarks", "1a", "Direct LLM"),
    ("Benchmarks", "1b", "Direct LLM--Multi-score"),
    ("Benchmarks", "2a", "Local LLMs"),
    ("Benchmarks", "2b", "Local LLMs--Multi-score"),
    ("Ablations", "PWC-32", "Pairwise comparisons + LLM scoring"),
    ("Ablations", "PWC-32m", "Pairwise comparisons + LLM scoring--Multi-score"),
    ("Ablations", "PWC-21", "Reliable features + LLM scoring"),
    ("Ablations", "PWC-21m", "Reliable features + LLM scoring--Multi-score"),
]


def _pt(res):
    """Point estimates keyed by paper name.

    The five discovery methods come from ``tab_results.csv`` (step 09) and the four ablation
    arms from ``tab_pointwise_ablation.csv`` (step 09b) -- two files because the ablations are
    not discovery methods of their own.
    """
    df = pd.read_csv(res / "tab_results.csv").set_index("method")
    ren = {"Ours": "Ours", "M1": "1a", "M2": "1b", "M3-taxo(1)": "2a", "M3-taxo(n*)": "2b"}
    out = {ren[k]: df.loc[k].to_dict() for k in df.index if k in ren}

    abl = res / "tab_pointwise_ablation.csv"
    if abl.exists():
        adf = pd.read_csv(abl).set_index("method")
        aren = {
            "PW-32 (1 rep)": "PWC-32",
            "PW-32 (n*)": "PWC-32m",
            "PW-21 (1 rep)": "PWC-21",
            "PW-21 (n*)": "PWC-21m",
        }
        out.update({aren[k]: adf.loc[k].to_dict() for k in adf.index if k in aren})
    return out


def _paired(a_rows, b_rows, keys=METRICS):
    A = {r["b"]: r for r in a_rows}
    B = {r.get("b", i): r for i, r in enumerate(b_rows)}
    common = sorted(set(A) & set(B))
    out = {}
    for k in keys:
        d = np.array([A[b][k] - B[b][k] for b in common], float)
        lo, hi = np.percentile(d, [2.5, 97.5])
        wins = float(np.mean(d < 0) if k == "LK" else np.mean(d > 0))
        out[k] = (float(d.mean()), float(lo), float(hi), wins, bool(lo > 0 or hi < 0))
    out["_n"] = len(common)
    return out


def _band(d, k):
    b = d.get("bands", d).get(k)
    return None if b is None else tuple(b)


def _f(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def _ci(band, n=3):
    if band is None:
        return "--"
    return f"{band[0]:.{n}f} \\,[{band[1]:.{n}f}, {band[2]:.{n}f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    root = Path(args.config).parent
    res = root / "results"
    out = res / "tables"
    out.mkdir(parents=True, exist_ok=True)

    pt = _pt(res)
    ours_b = json.loads((res / "bootstrap_discovery_counts.json").read_text())  # canonical Ours
    taxo_a = json.loads((res / "bootstrap_taxo_arm.json").read_text())  # 2a
    runvar = json.loads((res / "tab_runvar.json").read_text())  # 1a
    b1b = json.loads((res / "bootstrap_1b_letters.json").read_text())  # 1b
    b2a = json.loads((res / "bootstrap_2a_letters.json").read_text())  # 2a, letter-resampled
    b2b = json.loads((res / "bootstrap_2b_letters.json").read_text())  # 2b, letter-resampled
    ours_L = json.loads((res / "bootstrap_ours_letters.json").read_text())  # Ours, per-rep taxo
    ours_Lf = json.loads((res / "bootstrap_ours_letters_frozen.json").read_text())  # Ours, frozen
    cm = json.loads((res / "rubric_stability/taxo_pool/cost_match.json").read_text())

    # ---- costs (batch basis, deterministic) ------------------------------- #
    tb = cm["target_breakdown"]
    fb = cm["fixed_breakdown"]
    usd_ours = cm["target_ours_usd"]
    usd_pw = json.loads((res / "pointwise/cost_ledger.json").read_text())["realized_usd_batch"]
    usd_rub = 0.185 / 2
    usd_taxo_score = json.loads((res / "pointwise_taxo/cost_ledger.json").read_text())[
        "realized_usd_batch"
    ]
    cost = {
        "Ours": usd_ours,
        "1a": usd_rub + usd_pw / 20,
        "1b": usd_rub + usd_pw,
        "2a": fb["usd_elicitations"] + fb["usd_taxonomy_fusion"] + usd_taxo_score / 6,
        "2b": fb["usd_elicitations"] + fb["usd_taxonomy_fusion"] + usd_taxo_score,
    }

    ablation_cost = {
        m: pt[m]["cost"] for m in ("PWC-32", "PWC-32m", "PWC-21", "PWC-21m") if m in pt
    }
    (out / "costs.json").write_text(
        json.dumps({"basis": "Batch-API USD", **cost, **ablation_cost}, indent=2)
    )

    # ---- K: point + band --------------------------------------------------- #
    kband = {
        "Ours": tuple(ours_b["bands"]["kept"]),
        "1a": tuple(runvar["M1"]["K"]),
        "1b": _band(b1b, "K"),
        "2a": _band(taxo_a, "K"),
        "2b": _band(b2b, "K"),
    }
    mband = {
        "Ours": {k: tuple(ours_b["bands"][k]) for k in METRICS},
        "1a": {k: tuple(runvar["M1"][k]) for k in METRICS},
        "1b": {k: _band(b1b, k) for k in METRICS},
        "2a": {k: _band(taxo_a, k) for k in METRICS},
        "2b": {k: _band(b2b, k) for k in METRICS},
    }
    unit = {
        "Ours": "matching",
        "1a": "letter",
        "1b": "letter",
        "2a": "elicitation subset",
        "2b": "letter",
    }
    unit_plural = {
        "Ours": "matchings",
        "1a": "letters",
        "1b": "letters",
        "2a": "elicitation subsets",
        "2b": "letters",
    }
    label = {"Ours": "\\textbf{Ours}", "1a": "1a", "1b": "1b", "2a": "2a", "2b": "2b"}

    # =================== tab:main =========================================== #
    # Bold the best cell per column, all ties bolded. Cost and LK are
    # minimised, the other four maximised.
    def _best(key):
        vals = [pt[m][key] for _, m, _ in MAIN_ROWS if m in pt and pd.notna(pt[m].get(key))]
        if not vals:
            return None
        return min(vals) if key in ("cost", "LK") else max(vals)

    costs = {m: (cost[m] if m in cost else pt[m]["cost"]) for _, m, _ in MAIN_ROWS if m in pt}
    best = {k: _best(k) for k in METRICS}
    best_cost = min(costs.values()) if costs else None

    def _cell(v, key):
        """Format one metric cell: Tracked/Sem.R./Attrib. as a percentage, the rest as 3dp."""
        if v is None or pd.isna(v):
            return "--"
        txt = f"{v * 100:.1f}\\%" if key in PERCENT else f"{v:.3f}"
        tol = 5e-4 if key in PERCENT else 5e-4
        return f"\\textbf{{{txt}}}" if (best[key] is not None and abs(v - best[key]) < tol) else txt

    hdr = " & ".join(
        DISPLAY_NAME[k] + (r"$\downarrow$" if not HIGHER_BETTER[k] else "") for k in METRICS
    )
    L = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}c l r r ccccc@{}}",
        r"\toprule",
        r" & Method & Cost & $K$ & " + hdr + r" \\",
        r"\midrule",
    ]
    prev_group = None
    for group, m, label in MAIN_ROWS:
        if m not in pt:
            continue
        p_ = pt[m]
        if group != prev_group:
            L.append(r"\midrule")
            gcell = (
                ""
                if group is None
                else r"\multirow{4}{*}{\rotatebox{90}{\textcolor{black}{\scriptsize "
                + group
                + r"}}}"
            )
            prev_group = group
        else:
            gcell = ""
        c = costs[m]
        ctxt = f"{c:.2f}"
        if best_cost is not None and abs(c - best_cost) < 5e-3:
            ctxt = f"\\textbf{{{ctxt}}}"
        L.append(
            f"{gcell} & {label} & {ctxt} & {int(round(p_['K']))} & "
            + " & ".join(_cell(p_.get(k), k) for k in METRICS)
            + r" \\"
        )
    L += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Per-method point estimates on the single full-data run. Costs are "
        r"Batch-API prices in USD; $K$ is the number of features the method reports. "
        r"Tracked, Sem.R., Attrib. in \%; Stat.R. and LK are correlations.}",
        r"\label{tab:main}",
        r"\end{table}",
    ]
    (out / "tab_main.tex").write_text("\n".join(L) + "\n")

    # =================== tab:main-bands ===================================== #
    # The five discovery methods only: the ablation arms have no bootstrap, so no band rows.
    band_label = {m: lab for _, m, lab in MAIN_ROWS}

    def _v(x, k):
        return f"{100 * x:.1f}\\%" if k in PERCENT else f"{x:.3f}"

    def _bnd(b, k):
        return (
            f"[{100 * b[1]:.1f}, {100 * b[2]:.1f}]" if k in PERCENT else f"[{b[1]:.3f}, {b[2]:.3f}]"
        )

    L = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l r r ccccc}",
        r"\toprule",
        r"Method & Cost & $K$ & "
        + " & ".join(
            DISPLAY_NAME[k] + (r"$\downarrow$" if not HIGHER_BETTER[k] else "") for k in METRICS
        )
        + r" \\",
        r"\midrule",
    ]
    for i, m in enumerate(["Ours", "1a", "1b", "2a", "2b"]):
        p_ = pt[m]
        bold = m == "Ours"

        def _b(txt):
            return f"\\textbf{{{txt}}}" if bold else txt

        ctxt = f"{cost[m]:.2f}"
        if abs(cost[m] - min(cost.values())) < 5e-3:
            ctxt = f"\\textbf{{{ctxt}}}"
        L.append(
            f"{band_label[m]} & {ctxt} & {int(round(p_['K']))} & "
            + " & ".join(_b(_v(p_[k], k)) for k in METRICS)
            + r" \\"
        )
        kb = kband[m]
        kb_txt = f"[{kb[1]:.1f}, {kb[2]:.1f}]" if kb else "--"
        L.append(
            f"\\band{{\\quad resampling {unit_plural[m]}}} & & \\band{{{kb_txt}}} & "
            + " & ".join(f"\\band{{{_bnd(mband[m][k], k)}}}" for k in METRICS)
            + r" \\"
        )
        if i < 4:
            L.append(r"\addlinespace")
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-method point estimates with run-variance bands. Tracked, Sem.R., "
        r"Attrib. in \%; Stat.R. and LK are correlations; bands are in the same units as the "
        r"column. Below each method's "
        r"point estimates, the gray row gives that method's \textbf{own} run-variance band "
        r"($[2.5,\,97.5]$ percentiles across replicates), obtained by resampling the unit "
        r"named at the left of the row. These bands are \textbf{not} confidence intervals for "
        r"the point estimates above them -- the point estimates come from the single full-data "
        r"run, the bands from re-runs on resampled data, so a point estimate may fall outside "
        r"its band -- and they \textbf{must not be compared across methods}, since the arms "
        r"resample different units. Paired, comparable contrasts: Table~\ref{tab:paired}.}",
        r"\label{tab:main-bands}",
        r"\end{table}",
    ]
    (out / "tab_main_bands.tex").write_text("\n".join(L) + "\n")

    # =================== TABLE 2 ============================================ #
    p1a = _paired(ours_L["per_replicate"], runvar["M1_per_replicate"])
    p2a = _paired(ours_Lf["per_replicate"], b2a["per_replicate"])
    kd1a = _paired(ours_L["per_replicate"], runvar["M1_per_replicate"], ["K"])["K"]
    kd2a = _paired(ours_Lf["per_replicate"], b2a["per_replicate"], ["K"])["K"]
    sens = b2a["sensitivity_over_fusions"]
    deg = ours_L["bands"]["degree"][0]
    npair = ours_L["bands"]["n_pairs"][0]
    # cost of the comparative arm at this reduced degree, scaled to the full corpus
    frac = npair / 4000.0
    usd_deg25 = tb["comparative_judging"] * (2500.0 / 4000.0) + tb["taxonomy"]

    def num(x, n=3):
        return f"${x:+.{n}f}$".replace("+-", "-")

    def ci(lo, hi, n=3):
        return f"$[{lo:+.{n}f},\\ {hi:+.{n}f}]$"

    def row(k, mu, lo, hi, w):
        # the caption's rule: a star marks a 95% CI that excludes zero
        nm = DISPLAY_NAME[k]
        nm = f"\\textbf{{{nm}}}$^{{*}}$" if (lo > 0 or hi < 0) else nm
        return f"$\\Delta$ {nm} & {num(mu)} & {ci(lo, hi)} & ${w * 100:.0f}\\%$ \\\\"

    L = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l r c r}",
        r"\toprule",
        r"& Difference & 95\% CI & Ours better \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{(a) Ours $-$ Direct LLM} \; "
        r"(discovery re-run per replicate on \emph{both} sides)} \\",
        r"\midrule",
        f"$\\Delta K$ & {num(kd1a[0], 2)} & {ci(kd1a[1], kd1a[2], 2)} & --- \\\\",
    ]
    L += [row(k, *p1a[k][:4]) for k in METRICS]
    L += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{(b) Ours $-$ Local LLMs} \; "
        r"(discovery frozen on \emph{both} sides; pooled over fusion \emph{and} corpus draws)} \\",
        r"\midrule",
        f"$\\Delta K$ & {num(kd2a[0], 2)} & {ci(kd2a[1], kd2a[2], 2)} & --- \\\\",
    ]
    L += [
        row(
            k,
            sens[k]["pooled_mean"],
            sens[k]["pooled_lo"],
            sens[k]["pooled_hi"],
            sens[k]["pooled_frac_favouring_ours"],
        )
        for k in METRICS
    ]
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Paired comparisons over $B=50$ bootstrap draws (with replacement) of the "
        r"$200$-letter corpus; every arm sees the same draws, and differences are taken within "
        r"replicate. Each row reports the mean difference, its $95\%$ percentile CI ($^{*}$: "
        r"excludes zero), and \emph{Ours better}: the share of replicates in which our method "
        r"wins in the metric's preferred direction ($\Delta K$ is descriptive and carries no "
        r"share). \textbf{The panels bracket discovery variance from opposite sides and are not "
        r"comparable to one another}. \textbf{Both panels score our method at a reduced budget "
        rf"(average degree ${deg:.1f}$ against the designed $40$)}}; it leads on several "
        r"metrics regardless. Design rationale, the conditional-fusion alternative, and budget "
        r"details: Appendix~\ref{app:paired-design}.}",
        r"\label{tab:paired}",
        r"\end{table}",
    ]
    (out / "tab_paired.tex").write_text("\n".join(L) + "\n")
    # the same values, as printed, for figures/make_paired_panels.py (fig:paired)
    r3 = lambda x: round(float(x), 3)
    fig = {
        "a": {
            "dK": [round(float(v), 2) for v in kd1a[:3]],
            "rows": [
                [
                    k,
                    r3(p1a[k][0]),
                    r3(p1a[k][1]),
                    r3(p1a[k][2]),
                    round(100 * p1a[k][3]),
                    bool(p1a[k][1] > 0 or p1a[k][2] < 0),
                    HIGHER_BETTER[k],
                ]
                for k in METRICS
            ],
        },
        "b": {
            "dK": [round(float(v), 2) for v in kd2a[:3]],
            "rows": [
                [
                    k,
                    r3(sens[k]["pooled_mean"]),
                    r3(sens[k]["pooled_lo"]),
                    r3(sens[k]["pooled_hi"]),
                    round(100 * sens[k]["pooled_frac_favouring_ours"]),
                    bool(sens[k]["pooled_lo"] > 0 or sens[k]["pooled_hi"] < 0),
                    HIGHER_BETTER[k],
                ]
                for k in METRICS
            ],
        },
    }
    (out / "tab_paired.json").write_text(json.dumps(fig, indent=2))

    # The numbers Appendix app:paired-design quotes.
    design = {
        "B": 50,
        "n_fusions": 50,
        "pooled_n": sens["SF"]["pooled_n"],
        "canonical_fusion_mean_diff": {k: p2a[k][0] for k in METRICS},
        "per_fusion_range": {k: [sens[k]["min"], sens[k]["max"]] for k in METRICS},
        "pairs_retained": npair,
        "of_pairs": 4000,
        "average_degree": deg,
        "share_of_comparisons": frac,
        "ours_K_at_reduced_budget": ours_L["bands"]["K"][0],
        "usd_comparative_at_degree_25": usd_deg25,
        "usd_ours_degree_40": cost["Ours"],
        "usd_local_llms": cost["2a"],
    }
    (out / "tab_paired_design.json").write_text(json.dumps(design, indent=2))

    print(f"[tables] degree {deg:.1f}, {npair:.0f} pairs, deg-25 cost ${usd_deg25:.2f}")
    print(
        f"[tables] wrote tab_main.tex, tab_main_bands.tex, tab_paired.tex, "
        f"tab_paired.json, costs.json and tab_paired_design.json under {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
