"""Step 11: per-criterion effectiveness — tab:cmv-win-rates, fig:cmv-win-rates{,-full}. No LLM calls.

    python scripts/11_win_rates.py

For each of the 29 discovered criteria, the *effectiveness* is the fraction of held-out
exchanges in which the reply that earned the delta scores higher on that criterion, with ties
split. 0.5 is chance. Brackets are 95% Wald intervals; the exact sign test gives p, and q is
the Benjamini-Hochberg false-discovery rate across the 29 tests.

The panel is the pointwise one at full anchor budget: ``pointwise_discovered_full`` supplies
the 16 criteria the reliability screen kept, ``pointwise_dropped_full`` the 13 it dropped
(that cell also re-rates two kept criteria as anchors, which its ``features.json`` "dropped"
list excludes). Both cells share one item graph, so joining on ``item_id`` is safe here.

Writes results/win_rates.csv and results/win_rates_meta.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _paths  # noqa: F401
import _config as C
from core.cmv.win_rates import length_reference, matched_pair_diffs, panel_scores, win_rate_table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else C.RESULTS
    out.mkdir(parents=True, exist_ok=True)

    cells = [C.cell("pw_discovered_full"), C.cell("pw_dropped_full")]
    S = panel_scores(cells)
    items = C.items_for("pw_discovered_full", "pw_dropped_full")
    diffs = matched_pair_diffs(S, items)
    tab = win_rate_table(diffs)
    tab["cell"] = tab.feature.map(S.attrs["source"])

    n_sig = int((tab.q < 0.05).sum())
    length_wr = length_reference(items)
    print(f"[win-rates] {len(tab)} criteria over {len(diffs)} held-out pairs")
    print(
        f"[win-rates] {n_sig} clear BH q < 0.05, all of them above chance: "
        f"{bool((tab[tab.q < 0.05].win_rate > 0.5).all())}"
    )
    print(f"[win-rates] reply length alone wins {length_wr:.3f} of exchanges")
    print("\n  top 5 and bottom 3 by effectiveness:")
    for _, r in pd.concat([tab.tail(5)[::-1], tab.head(3)]).iterrows():
        print(f"    {r.feature[:46]:<46} {r.win_rate:.3f} [{r.lo:.3f}, {r.hi:.3f}]  q={r.q:.3g}")

    tab.to_csv(out / "win_rates.csv", index=False)
    (out / "win_rates_meta.json").write_text(
        json.dumps(
            {
                "n_criteria": int(len(tab)),
                "n_pairs": int(len(diffs)),
                "n_significant_bh05": n_sig,
                "all_significant_are_positive": bool((tab[tab.q < 0.05].win_rate > 0.5).all()),
                "length_reference_win_rate": float(length_wr),
                "cells": [c.name for c in cells],
                "note": "ties split; 95% Wald intervals; BH across all criteria",
            },
            indent=2,
        )
    )
    print(f"\n[win-rates] wrote {out.name}/win_rates.csv and win_rates_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
