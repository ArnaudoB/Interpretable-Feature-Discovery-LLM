"""Step 13: emit the CMV tables as LaTeX from results/ and the shipped cells. No LLM calls.

    python scripts/13_paper_tables.py

Writes results/tables/:

  tab_cmv_predictive.tex     Table `tab:cmv-predictive`: every arm with all of its features
  tab_cmv_predictive_2.tex   Table `tab:cmv-predictive-2`: the same plus the (full) arms, best first
  tab_cmv_contamination.tex  Table `tab:cmv-contamination`: the memorization probe
  tab_cmv_win_rates.tex      Table `tab:cmv-win-rates`: the 29 discovered features, D_r starred
  tab_cmv_features_p.tex     Table `tab:cmv-features-p`: the 16 prior-elicited features, in order
  tab_cmv_features_s.tex     Table `tab:cmv-features-s`: the 42 sample-elicited features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C


def half_up(x: float, nd: int) -> str:
    """``x`` at ``nd`` decimals, halves rounded up. Accuracies are k/800, exact eighths of a
    percent, so half the rows sit on a tie that ``format`` would round to even; the paper rounds
    them up (62.625 -> 62.63). The epsilon guards against k/800 not being binary-exact."""
    s = 10**nd
    return f"{np.floor(s * x + 0.5 + 1e-6) / s:.{nd}f}"


def dims(d) -> str:
    return "--" if d is None or pd.isna(d) else f"{int(d):,}".replace(",", "{,}")


def emit(
    out: Path, name: str, label: str, head: list[str], rows: list[str], tail: list[str]
) -> None:
    (out / name).write_text("\n".join(head + rows + tail) + "\n")
    ours = [r for r in rows if "&" in r]
    print(f"  {label:<24} {len(ours):>2} rows -> {name}")


def acc_rows(acc: pd.DataFrame, arms: list[str], appendix: bool) -> list[str]:
    rows = []
    for arm in arms:
        r = acc.loc[arm]
        name = C.ARM_TEX.get(arm, arm)
        a = f"{half_up(100 * r.acc, 2)}\\%"
        ci = f"$[{half_up(100 * r.ci_lo, 2)}, {half_up(100 * r.ci_hi, 2)}]$"
        rows.append(
            f"{name:<22} & {dims(r.dims):<8} & {a} & {ci} \\\\"
            if appendix
            else f"    {name} & {dims(r.dims)} & {a} \\scriptsize{{{ci}}} \\\\"
        )
    return rows


def q_fmt(q: float) -> str:
    """BH q as the paper prints it: 3 decimals, 4 where 3 would hide which side of 0.05 it is."""
    if q < 0.001:
        return "$<$0.001"
    s = f"{q:.3f}"
    return f"{q:.4f}" if s == "0.050" else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    out = C.RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    acc = pd.read_csv(C.RESULTS / "accuracy.csv").set_index("arm")

    # ---- accuracy ------------------------------------------------------------ #
    emit(
        out,
        "tab_cmv_predictive.tex",
        "tab:cmv-predictive",
        [r"\begin{tabular}{@{}l r l@{}}", r"\toprule", r"Arm & Dims & Accuracy \\", r"\midrule"],
        acc_rows(acc, C.MAIN_ARMS, False),
        [r"\bottomrule", r"\end{tabular}"],
    )
    emit(
        out,
        "tab_cmv_predictive_2.tex",
        "tab:cmv-predictive-2",
        [
            r"\begin{tabular}{@{}l r r c@{}}",
            r"\toprule",
            r"Arm & Dims & Accuracy & $95\%$ CI \\",
            r"\midrule",
        ],
        acc_rows(acc, C.ARM_ORDER, True),
        [r"\bottomrule", r"\end{tabular}"],
    )

    # ---- contamination ------------------------------------------------------- #
    ct = json.loads((C.RESULTS / "contamination.json").read_text())
    lf = ct["lift"]
    rows = [
        f"{lab:<22} & ${100 * ct[k]['acc']:.1f}\\%$ \\scriptsize{{$[{100 * ct[k]['ci'][0]:.1f},"
        f"\\ {100 * ct[k]['ci'][1]:.1f}]$}} \\\\"
        for k, lab in (("p1", "P1 (unnamed, scrubbed)"), ("p2", "P2 (named)"))
    ]
    rows.append(
        f"Lift $\\mathrm{{P2}}-\\mathrm{{P1}}$ & ${lf['pp']:+.1f}$\\,pp \\scriptsize{{$[{lf['ci_pp'][0]:+.1f},"
        f"\\ {lf['ci_pp'][1]:+.1f}]$}} \\\\"
    )
    emit(
        out,
        "tab_cmv_contamination.tex",
        "tab:cmv-contamination",
        [r"\begin{tabular}{@{}l l@{}}", r"\toprule", r"Framing & Accuracy \\", r"\midrule"],
        rows[:2] + [r"\midrule"] + rows[2:],
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"% McNemar: b/c = {lf['mcnemar_b']}/{lf['mcnemar_c']}, p = {lf['mcnemar_p']:.3f}",
        ],
    )

    # ---- the 29 discovered features ------------------------------------------ #
    wr = pd.read_csv(C.RESULTS / "win_rates.csv").sort_values(
        "win_rate", ascending=False, kind="stable"
    )
    tax = json.loads(
        (C.cell("discovery") / "results" / "gated" / "llm_taxonomy" / "taxonomy.json").read_text()
    )
    desc = {c["name"]: c["definition"] for c in tax["criteria"]}
    kept = set(C.reliable_features())
    rows = []
    for _, r in wr.iterrows():
        nm = r.feature + ("$^{*}$" if r.feature in kept else "")
        nm = f"\\bfseries {nm}" if r.q < 0.05 else nm
        rows.append(
            f"{nm} & {desc[r.feature]} & {r.win_rate:.3f} "
            f"{{\\tiny$[{r.lo:.3f}, {r.hi:.3f}]$}} & {q_fmt(r.q)} \\\\"
        )
    emit(
        out,
        "tab_cmv_win_rates.tex",
        "tab:cmv-win-rates",
        [
            r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth} X r r@{}}",
            r"\toprule",
            r"Feature & Description & Win rate (95\% CI) & $q$ \\",
            r"\midrule",
        ],
        rows,
        [
            r"\bottomrule",
            r"\end{tabularx}",
            f"% {int((wr.q < 0.05).sum())} of {len(wr)} features have q < 0.05; "
            f"* = D_r ({len(kept)} features)",
        ],
    )

    # ---- the prior (P) and sample-elicited (S) feature sets -------------------- #
    for key, fn, label, name in (
        ("prior_panel", "features_blind.json", "tab:cmv-features-p", "p"),
        ("sample_panel", "features_sample.json", "tab:cmv-features-s", "s"),
    ):
        feats = json.loads((C.cell(key) / fn).read_text())["criteria"]
        emit(
            out,
            f"tab_cmv_features_{name}.tex",
            label,
            [
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth} X@{}}",
                r"\toprule",
                r"Feature & Description \\",
                r"\midrule",
            ],
            [f"{f['name']} & {f['definition']} \\\\" for f in feats],
            [r"\bottomrule", r"\end{tabularx}"],
        )

    print("[tables] wrote results/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
