"""Step 0: sample the essays and build the comparison graph.

    python scripts/00_build_pairs.py --run all        # or --run gpt_5_4_mini__asap2, ...

No API spend. Feeds the design described in App. ``sec:extension-appendix`` ("Data and
comparison design"). Writes, under ``runs/<run>/results/graph/``:

  essays.parquet          essay_id, full_text, word_count, overall, prompt, band
                          (+ the 6 ELLIPSE trait columns on ELLIPSE runs)
  pairs.parquet           pair_index, essay_a_id, essay_b_id, shown_first, band_a, band_b,
                          score_gap, pair_id, essay_first_id, essay_second_id
  comparison_plan.parquet pair_id, pair_index, essay_first_id, essay_second_id, score_gap
  build_report.json       selection + nested-budget diagnostics, all invariant checks

Design:
  * 100 essays, band-balanced on the human holistic score -- ELLIPSE's ``Overall``, with
    every writing prompt represented; ASAP 2.0's ``score`` (1-6), single prompt -- by
    ``dataset.ellipse.select_balanced_essays``. Deliberately not a uniform subsample, and on
    ELLIPSE deliberately not stratified on the 6-trait mean; see ``dataset/ellipse.py``.
  * The four runs of a corpus draw the SAME sample, pairs and slot order (seed 42), so
    within a corpus only the judge varies. ``--run all`` asserts this.
  * Union of 40 random perfect matchings -> 2000 pairs, every essay at degree exactly 40.
    Nested degree prefixes [5,10,...,40] are recorded so a pairs-scaling curve is free.

SLOT CONVENTION (load-bearing, do not change):
    shown_first == essay_first_id == prompt slot A,
    and the judge's ``overall_winner == "A"`` becomes ``w_t = 1`` downstream.
    ``pair_id`` is the request ``custom_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import resolve  # noqa: E402
from core.graph.matching import build_nested_pairs, verify_budgets  # noqa: E402
from dataset import (  # noqa: E402
    assert_graph_invariants,
    load_asap2_candidates,
    load_candidates,
    select_balanced_essays,
)


def build(run) -> None:
    cfg = run.cfg
    src, gcfg = cfg["source"], cfg["graph"]
    print(f"\n=== {run.name}")

    n_essays = int(src["n_essays"])
    n_matchings = int(gcfg["n_matchings"])
    seed = int(gcfg.get("seed", 42))
    if n_essays % 2:
        raise ValueError(f"n_essays must be even for perfect matchings, got {n_essays}")
    half = n_essays // 2
    n_pairs = n_matchings * half

    if not run.source_path.exists():
        raise FileNotFoundError(f"corpus not found: {run.source_path}")
    out_dir = run.graph
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. candidate pool + band-balanced selection -----------------------------
    if src["loader"] == "ellipse":
        pool = load_candidates(str(run.source_path))
    elif src["loader"] == "asap2":
        pool = load_asap2_candidates(
            str(run.source_path),
            prompt_name=src.get("prompt_filter"),
            score_col=src["band_col"],
            prompt_col=src.get("prompt_col", "prompt_name"),
        )
    else:
        raise ValueError(f"unknown source.loader {src['loader']!r} (expected ellipse|asap2)")
    print(f"pool: {len(pool)} essays, {pool['prompt'].nunique()} writing prompts")
    print("pool band histogram:", pool["band"].value_counts().sort_index().to_dict())

    sel = select_balanced_essays(pool, n_essays, seed)
    # Canonical node order 0..n-1 is sorted essay_id.
    sel = sel.sort_values("essay_id").reset_index(drop=True)
    essay_ids = sel["essay_id"].astype(str).tolist()
    bands = sel["band"].to_numpy()
    assert len(set(essay_ids)) == n_essays, "duplicate essay_id in selection"

    print(f"selected: {len(sel)} essays, {sel['prompt'].nunique()} prompts covered")
    print("selected band histogram:", sel["band"].value_counts().sort_index().to_dict())

    # --- 2. nested union of perfect matchings ------------------------------------
    pairs = build_nested_pairs(
        essay_ids,
        bands,
        n_matchings=n_matchings,
        seed=seed,
        cross_band_bias=float(gcfg.get("cross_band_bias", 0.0)),
    )
    assert len(pairs) == n_pairs, f"expected {n_pairs} pairs, got {len(pairs)}"

    pairs["pair_id"] = [f"p{i:05d}" for i in pairs["pair_index"]]
    pairs["essay_first_id"] = pairs["shown_first"].astype(str)
    pairs["essay_second_id"] = np.where(
        pairs["essay_first_id"] == pairs["essay_a_id"].astype(str),
        pairs["essay_b_id"].astype(str),
        pairs["essay_a_id"].astype(str),
    )

    # --- 3. nested-prefix diagnostics + hard invariants ---------------------------
    degrees = [int(d) for d in gcfg.get("degrees", [n_matchings])]
    budgets = [d * half for d in degrees]
    diag, gap_by_budget = verify_budgets(pairs, essay_ids, budgets)
    print("\nnested budget diagnostics:")
    print(diag.to_string(index=False))
    assert_graph_invariants(pairs, diag, budgets, half)

    # Degree at the full budget must be exactly n_matchings for every essay.
    deg = pd.concat([pairs["essay_a_id"], pairs["essay_b_id"]]).value_counts()
    assert deg.min() == deg.max() == n_matchings, (
        f"degree not exactly {n_matchings}: min={deg.min()} max={deg.max()}"
    )
    assert len(deg) == n_essays, f"only {len(deg)} essays appear in the graph"

    slot_a_share = float((pairs["essay_first_id"] == pairs["essay_a_id"].astype(str)).mean())
    print(f"\ndegree: exactly {n_matchings} for all {n_essays} essays  ✓")
    print(f"slot-A share (shown_first == essay_a): {slot_a_share:.3f}")

    # --- 4. write ------------------------------------------------------------------
    sel.to_parquet(out_dir / "essays.parquet", index=False)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    pairs[["pair_id", "pair_index", "essay_first_id", "essay_second_id", "score_gap"]].to_parquet(
        out_dir / "comparison_plan.parquet", index=False
    )

    gap_counts = pairs["score_gap"].value_counts().sort_index()
    report = {
        "n_essays": n_essays,
        "n_matchings": n_matchings,
        "n_pairs": int(len(pairs)),
        "degree_per_essay": int(deg.min()),
        "seed": seed,
        "loader": src["loader"],
        "band_col": src["band_col"],
        "csv_path": str(src["path"]),
        "prompt_filter": src.get("prompt_filter"),
        "prompts_covered": int(sel["prompt"].nunique()),
        "prompts_in_pool": int(pool["prompt"].nunique()),
        "selected_band_histogram": {
            int(k): int(v) for k, v in sel["band"].value_counts().sort_index().items()
        },
        "pool_band_histogram": {
            int(k): int(v) for k, v in pool["band"].value_counts().sort_index().items()
        },
        "word_count": {
            "mean": float(sel["word_count"].mean()),
            "median": float(sel["word_count"].median()),
            "min": int(sel["word_count"].min()),
            "max": int(sel["word_count"].max()),
        },
        "score_gap_distribution": {int(k): int(v) for k, v in gap_counts.items()},
        "mean_score_gap": float(pairs["score_gap"].mean()),
        "slot_a_share": slot_a_share,
        "nested_budget_diagnostics": diag.to_dict("records"),
        "invariants": {
            "degree_exactly_n_matchings": True,
            "all_prefixes_connected": True,
            "no_self_pairs": True,
            "no_duplicate_pairs": True,
            "prefixes_nested": True,
        },
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nwrote → {out_dir}")
    print(f"  essays.parquet          {len(sel)} rows")
    print(f"  pairs.parquet           {len(pairs)} rows")
    print(f"  comparison_plan.parquet {len(pairs)} rows")
    print(f"  build_report.json")


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run name, a comma list, or 'all'")
    args = ap.parse_args()
    runs = resolve(args.run)
    for run in runs:
        build(run)

    # Within a corpus only the judge may vary: same essays, pairs and slot order.
    for corpus in sorted({r.corpus for r in runs}):
        same = [r for r in runs if r.corpus == corpus]
        for f in ("essays.parquet", "pairs.parquet", "comparison_plan.parquet"):
            digests = {_md5(r.graph / f) for r in same}
            assert len(digests) == 1, f"{corpus}: {f} differs across runs"
        print(
            f"\n{corpus}: graph byte-identical across {len(same)} run(s) "
            f"(pairs md5 {_md5(same[0].graph / 'pairs.parquet')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
