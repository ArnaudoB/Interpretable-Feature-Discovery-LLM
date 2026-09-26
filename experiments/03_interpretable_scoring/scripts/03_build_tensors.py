"""Step 3: judgments + taxonomy -> the (w, r, pairs) tensors the model consumes.

    python scripts/03_build_tensors.py --run all

No API spend. The shared tensor builders ``core.tensors.filter_active_criteria`` and
``core.tensors.build_tensors`` consume a ``cluster_assignments`` frame (phrase, cluster_id,
canonical_name, distance_to_centroid). Here that frame is synthesized directly from the LLM
taxonomy's ``mapping.json`` ({phrase_norm: criterion_name}), so both run unchanged.

There is no embedder and no hierarchical clustering anywhere in this experiment: the
phrase -> criterion map comes from the LLM taxonomy of step 02.

Writes under ``runs/<run>/results/pipeline/``:
  cluster_assignments.parquet  phrase -> criterion (from the taxonomy)
  active_criteria.parquet      cluster_id, canonical_name, n_members, total_mentions,
                               share, raw_cluster_id
  tensors.npz                  w (T,), r (T,K), pairs (T,2), n, K, T
  essay_index.json             {essay_id: row index in s}
  pipeline_metadata.json       canonicalization settings and counts
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import resolve  # noqa: E402

from core.tensors import filter_active_criteria  # noqa: E402
from core.tensors import build_tensors  # noqa: E402

OTHER = "Other"  # taxonomy catch-all; dropped, never a criterion column


def build(run, args) -> None:
    print(f"\n=== {run.name}")
    cfg = run.cfg
    res = run.results
    tax_dir = res / "taxonomy"
    out = res / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)

    judgments = pd.read_parquet(res / "judgments.parquet")
    mapping = json.loads((tax_dir / "mapping.json").read_text())
    taxonomy = json.loads((tax_dir / "taxonomy.json").read_text())

    # --- drop parse failures (build_tensors refuses them) -------------------------
    n_fail = int(judgments["parse_failed"].sum())
    judgments = judgments[~judgments["parse_failed"]].reset_index(drop=True)
    judgments = judgments.drop(columns=["parse_failed"])
    print(f"judgments: {len(judgments)} usable ({n_fail} parse failures dropped)")

    # --- taxonomy -> cluster_assignments (replaces embed + HAC) -------------------
    kept = {p: c for p, c in mapping.items() if c != OTHER}
    n_other = len(mapping) - len(kept)
    names = sorted(set(kept.values()))
    cid = {c: i for i, c in enumerate(names)}
    cluster_assignments = pd.DataFrame(
        {
            "phrase": list(kept),  # already phrase_norm
            "cluster_id": [cid[kept[p]] for p in kept],
            "canonical_name": [kept[p] for p in kept],
            "distance_to_centroid": 0.0,  # unused downstream
        }
    )
    print(
        f"taxonomy: {len(names)} criteria over {len(kept)} phrases "
        f"({n_other} phrases in '{OTHER}', dropped)"
    )

    # --- filter + tensors (shared core code) --------------------------------------
    share_floor = float(
        args.share_floor if args.share_floor is not None else cfg["clustering"]["share_floor"]
    )
    active = filter_active_criteria(cluster_assignments, judgments, share_floor=share_floor)
    dropped = [n for n in names if n not in set(active["canonical_name"])]
    print(
        f"active criteria: {len(active)} at share_floor={share_floor}"
        + (f"  (dropped: {', '.join(dropped)})" if dropped else "")
    )

    tensors = build_tensors(judgments, active, cluster_assignments)
    w, r, pairs = tensors["w"], tensors["r"], tensors["pairs"]
    n, K, T = tensors["n"], tensors["K"], tensors["T"]

    np.savez_compressed(out / "tensors.npz", w=w, r=r, pairs=pairs, n=n, K=K, T=T)
    active.to_parquet(out / "active_criteria.parquet", index=False)
    cluster_assignments.to_parquet(out / "cluster_assignments.parquet", index=False)
    (out / "essay_index.json").write_text(json.dumps(tensors["essay_index"], indent=2))

    usage = json.loads((tax_dir / "usage.json").read_text())
    (out / "pipeline_metadata.json").write_text(
        json.dumps(
            {
                "cell_label": cfg["cell"]["label"],
                "judge": cfg["cell"]["judge_full_name"],
                "dataset": cfg["cell"]["dataset"],
                "prompt_variant": cfg["cell"]["prompt_variant"],
                "T_before": int(len(judgments) + n_fail),
                "T_after": int(T),
                "n": int(n),
                "K": int(K),
                "canonicalization": {
                    "method": "llm_taxonomy",  # NOT embedding clustering
                    "model": usage["model"],
                    "reasoning_effort": usage["reasoning_effort"],
                    "n_phrases_in": usage["n_phrases_in"],
                    "n_criteria_raw": len(taxonomy["criteria"]),
                    "n_criteria_after_other_and_floor": int(K),
                    "min_cite": taxonomy["min_cite"],
                    "citation_share_retained": usage["citation_share_retained"],
                    "n_phrases_in_other": n_other,
                    "share_floor": share_floor,
                },
            },
            indent=2,
        )
    )

    print(f"\ntensors: n={n} K={K} T={T}")
    print(f"  slot-A win rate : {w.mean():.3f}   (expect beta<0 if <0.5)")
    print(f"  citation rate   : {r.mean():.3f}   (mean over T*K cells)")
    print(f"  criteria per pair: {r.sum(1).mean():.2f}")
    print("\nper-criterion citation share:")
    for row in active.itertuples(index=False):
        print(
            f"  [{row.cluster_id:>2}] {row.canonical_name:<34} "
            f"{row.share:.3f}  ({row.total_mentions} pairs, {row.n_members} phrases)"
        )
    print(f"\nwrote → {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run name, a comma list, or 'all'")
    ap.add_argument(
        "--share-floor",
        type=float,
        default=None,
        help="override clustering.share_floor. NOTE this is a share of total "
        "CITATIONS, not a fraction of pairs; 0.0 keeps every criterion.",
    )
    ap.add_argument(
        "--out-subdir",
        default="pipeline",
        help="write under runs/<run>/results/<subdir> (for sensitivity runs)",
    )
    args = ap.parse_args()
    for run in resolve(args.run):
        build(run, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
