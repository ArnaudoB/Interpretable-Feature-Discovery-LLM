"""Step 12: the memorization probe — tab:cmv-contamination. No LLM calls.

    python scripts/12_contamination.py

The CMV corpus predates the judge's training cutoff, so a skeptical reading of
tab:cmv-predictive is that the judge recalls which replies earned a delta rather than
inferring it. The probe asks the same question of identical de-identified text under two
framings: P1 names no source, P2 says the text is from r/ChangeMyView and explains the delta
mechanic. A memorized outcome is easier to retrieve once the source is named, so the lift
P2 - P1 on constant text is the recall signal.

1,600 exchanges in four cells of 400 ({delta, no-delta} x {train, heldout}), so the base rate
is 50% by construction. The lift is assessed by a paired McNemar test on the same exchanges.

Everything is recomputed here from the per-exchange verdicts; the cell's own
``probe_summary.json`` is used only as a cross-check. Writes results/contamination.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

import _paths  # noqa: F401
import _config as C
from core.cmv.designs import boot_ci


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else C.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    n_boot = int(C.cfg()["task"]["n_boot"])

    d = pd.read_parquet(C.cell("contamination") / "results" / "probe_results.parquet")
    wide = d.pivot(index="exch_id", columns="framing", values="ok")
    meta = d[d.framing == "p1"].set_index("exch_id")[["split", "label"]]
    assert len(wide) == 1600 and set(wide.columns) == {"p1", "p2"}
    assert meta.label.mean() == 0.5, "the four cells should make the base rate exactly 50%"

    res = {"n_exchanges": int(len(wide))}
    for f in ("p1", "p2"):
        acc = float(wide[f].mean())
        # Resample each framing's rows in the order the probe's analysis stage stored them
        # (probe_results.parquet), as the published intervals were; resampling the exch_id-sorted
        # pivot draws the same indices over differently ordered rows and moves a bound by 0.1pp.
        rows = d.loc[(d.framing == f) & d.ok.notna(), "ok"].to_numpy(float)
        assert len(rows) == 1600
        lo, hi = boot_ci(rows, n_boot=n_boot)
        res[f] = {"acc": acc, "ci": [lo, hi]}
        print(f"  {f.upper()}  acc={100 * acc:.2f}%  [{100 * lo:.1f}, {100 * hi:.1f}]")

    lift = (wide.p2 - wide.p1).to_numpy(float)
    lo, hi = boot_ci(lift, n_boot=n_boot)
    # McNemar on the discordant pairs: b = P2 right where P1 wrong, c = the reverse.
    b = int(((wide.p2 == 1) & (wide.p1 == 0)).sum())
    c = int(((wide.p2 == 0) & (wide.p1 == 1)).sum())
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    res["lift"] = {
        "pp": float(lift.mean() * 100),
        "ci_pp": [lo * 100, hi * 100],
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": p,
    }
    print(
        f"  lift P2-P1 = {100 * lift.mean():+.1f} pp  [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
        f"McNemar b/c = {b}/{c}, p = {p:.3f}"
    )

    # Is the residual above-chance accuracy recall-shaped, or just a prior toward "no change"?
    res["by_outcome"] = {}
    for f in ("p1", "p2"):
        cell_acc = {}
        for lab, nm in ((0, "no_delta"), (1, "delta")):
            m = meta.label == lab
            cell_acc[nm] = float(wide.loc[m.index[m], f].mean())
            for sp in ("train", "heldout"):
                mm = m & (meta.split == sp)
                cell_acc[f"{nm}_{sp}"] = float(wide.loc[mm.index[mm], f].mean())
        res["by_outcome"][f] = cell_acc
        print(
            f"  {f.upper()}  no-delta {100 * cell_acc['no_delta']:.1f}%   "
            f"delta {100 * cell_acc['delta']:.1f}%  "
            f"(delta by split: {100 * cell_acc['delta_train']:.1f} / "
            f"{100 * cell_acc['delta_heldout']:.1f};  no-delta: "
            f"{100 * cell_acc['no_delta_train']:.1f} / {100 * cell_acc['no_delta_heldout']:.1f})"
        )

    ref = json.loads((C.cell("contamination") / "results" / "probe_summary.json").read_text())
    res["crosscheck_vs_shipped_summary"] = {
        "p1_acc": abs(res["p1"]["acc"] - ref["p1_acc"]),
        "p2_acc": abs(res["p2"]["acc"] - ref["p2_acc"]),
        "lift": abs(res["lift"]["pp"] / 100 - ref["lift"]),
    }
    assert max(res["crosscheck_vs_shipped_summary"].values()) < 1e-12
    print("  [ok] point estimates match the cell's shipped summary exactly")

    (out / "contamination.json").write_text(json.dumps(res, indent=2))
    print(f"\n[contamination] wrote {out.name}/contamination.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
