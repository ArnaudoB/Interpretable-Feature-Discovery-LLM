"""Step 1: build the nested comparison graph for the cover-letter corpus.

    python scripts/01_build_pairs.py --config config.yaml

The union of ``n_matchings`` = 40 random perfect matchings over the 200 letters
(:func:`core.graph.build_nested_pairs`), so every letter enters exactly 40 comparisons
and N = 4000 pairs are judged. The result is verified to be exactly 40-regular,
connected, and free of duplicate and self pairs. Writes to ``results/``
(``graph.out_dir``):

  * essays.parquet          -- letter_id, text, is_anchor, template_id (judge inputs)
  * pairs.parquet           -- full 4000-pair list (pair_index, essay_a/b_id, shown_first, ...)
  * build_report.json       -- summary + invariant checks

Pure data wrangling -- no LLM calls, no cost. Feeds the comparison design of
Sec. sec:feature_extraction_empirical_results (App. app:methods-details).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.graph import build_nested_pairs, verify_budgets

# ad-relevant "substance" composite -> per-letter band (diagnostic score-gap only;
# cross_band_bias is 0 so it does not steer pairing).
_BAND_COLS = ["exp_lvl", "edu_lvl", "skillsk_lvl", "nlang_lvl", "nlaunch_lvl"]
_SKILL_PREFIX = "skill_"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    gcfg = cfg["graph"]
    root = Path(args.config).parent
    results_dir = root / "results"
    ak = pd.read_parquet(results_dir / "answer_key.parquet")
    letters = pd.read_parquet(results_dir / "letters.parquet")

    ak = ak.sort_values("letter_id").reset_index(drop=True)
    essay_ids = ak["letter_id"].tolist()
    n = len(essay_ids)
    half = n // 2
    n_matchings = int(gcfg["n_matchings"])

    skill_cols = [c for c in ak.columns if c.startswith(_SKILL_PREFIX)]
    band = (ak[_BAND_COLS].sum(axis=1) + ak[skill_cols].sum(axis=1)).to_numpy(float)

    print(
        f"[pairs] n={n} letters, half={half}, n_matchings={n_matchings} "
        f"-> {n_matchings * half} pairs"
    )
    pairs = build_nested_pairs(
        essay_ids,
        band,
        n_matchings,
        seed=int(gcfg["seed"]),
        cross_band_bias=float(gcfg.get("cross_band_bias", 0.0)),
    )
    assert len(pairs) == n_matchings * half, (len(pairs), n_matchings * half)

    table, _ = verify_budgets(pairs, essay_ids, [n_matchings * half])
    row = table.iloc[0]
    checks = {
        "regular": int(row["deg_min"]) == n_matchings == int(row["deg_max"]),
        "connected": int(row["components"]) == 1,
        "deg_min": int(row["deg_min"]),
        "deg_max": int(row["deg_max"]),
        "components": int(row["components"]),
        "fiedler": float(row["fiedler"]),
    }
    dup = pairs.duplicated(subset=["essay_a_id", "essay_b_id"]).sum()
    self_pairs = int((pairs["essay_a_id"] == pairs["essay_b_id"]).sum())
    all_ok = checks["regular"] and checks["connected"] and dup == 0 and self_pairs == 0

    out_dir = root / gcfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    letters[["letter_id", "text"]].merge(
        ak[["letter_id", "is_anchor", "template_id"]], on="letter_id"
    ).sort_values("letter_id").to_parquet(out_dir / "essays.parquet", index=False)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    (out_dir / "build_report.json").write_text(
        json.dumps(
            {
                "n_letters": n,
                "n_matchings": n_matchings,
                "n_pairs": int(len(pairs)),
                "seed": int(gcfg["seed"]),
                "n_duplicate_pairs": int(dup),
                "n_self_pairs": self_pairs,
                "graph": checks,
                "all_ok": bool(all_ok),
            },
            indent=2,
        )
    )

    print(table.to_string(index=False))
    print(f"[pairs] duplicates={dup}  self_pairs={self_pairs}")
    print(f"[pairs] wrote {out_dir}/ (essays, pairs, build_report)")
    print("[pairs] ALL INVARIANTS HOLD" if all_ok else "[pairs] INVARIANT FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
