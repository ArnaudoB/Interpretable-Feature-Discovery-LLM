"""Step 1: the comparison graphs every judging step reads. No LLM calls.

    python scripts/01_build_pairs.py --out-dir DIR                 # every graph in pipeline.yaml
    python scripts/01_build_pairs.py --out-dir DIR --cell bt_discovered
    python scripts/01_build_pairs.py --out-dir DIR --skip bt_full   # the slow 3,411-pair graph

Two designs:

* **nested** (the discovery cell) -- the ordered union of
  ``graph.n_matchings`` = 40 random perfect matchings over the 300 step-0 items, so every
  degree-5..40 prefix is a regular, connected, duplicate-free sub-graph: 6,000 comparisons.
  Writes graph/{pairs.parquet, budget_cutoffs.json, verification_table.csv, build_report.json}.
* **matched** (every BT / pointwise cell with its own graph) --
  400 Tan-TRAIN anchor pairs (the 150 discovery pairs force-included) plus all 800 eligible
  heldout pairs as the test set, a 16-regular anchor graph over the 800 anchor exchanges, and
  16 anchors per test exchange for frozen scoring, shown-first balanced 8/8 per test item.
  Writes graph/{items.parquet, anchor_pairs.parquet, test_anchor_pairs.parquet, build_report.json}.

Both forbid the edge between a thread's winner and its own loser, so every comparison is
cross-topic (``core.cmv.graphs``). A few cells' graph/ holds only a copy of another cell's
items.parquet (pipeline.yaml ``build_pairs.items_copied``); those are copied, not rebuilt.

The matched design's discovery pairs are read from step 0's essays.parquet under --out-dir if
it was regenerated there, else from the shipped discovery cell.
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C
from core.cmv.data import build_pair_units
from core.cmv.graphs import build_nested_pairs, same_op_edges
from core.cmv.items import build_matched_items
from core.graph.matching import verify_budgets


def graph_dir(name: str):
    return C.source_path(C.cell_config(name)["graph"]["out_dir"])


def build_nested(name: str, out_dir) -> bool:
    """The discovery cell's nested comparison graph."""
    g = C.cell_config(name)["graph"]
    src, dst = graph_dir(name), C.work_path(graph_dir(name), out_dir)
    essays = pd.read_parquet(C.input_path(src / "essays.parquet", out_dir))
    essays = essays.sort_values("item_id").reset_index(drop=True)
    ids = essays["item_id"].tolist()
    n, half = len(ids), len(ids) // 2
    n_m, degrees = int(g["n_matchings"]), list(g["degrees"])
    budgets = [half * d for d in degrees]
    forbid = same_op_edges(essays["pair_id"]) if bool(g.get("exclude_same_op", True)) else set()

    pairs = build_nested_pairs(
        ids,
        essays["delta"].to_numpy(float),
        n_m,
        seed=int(g["seed"]),
        cross_band_bias=float(g.get("cross_band_bias", 0.0)),
        forbid_edges=forbid,
    )
    assert len(pairs) == n_m * half, (len(pairs), n_m * half)
    table, _ = verify_budgets(pairs, ids, budgets)

    checks = {}
    for d, (_, row) in zip(degrees, table.iterrows()):
        checks[d] = {
            "regular": int(row["deg_min"]) == d == int(row["deg_max"]),
            "connected": int(row["components"]) == 1,
            "deg_min": int(row["deg_min"]),
            "deg_max": int(row["deg_max"]),
            "components": int(row["components"]),
            "fiedler": float(row["fiedler"]),
        }
    dup = int(pairs.duplicated(subset=["essay_a_id", "essay_b_id"]).sum())
    self_pairs = int((pairs["essay_a_id"] == pairs["essay_b_id"]).sum())
    pair_of = dict(zip(essays["item_id"], essays["pair_id"]))
    same_op = int(
        sum(pair_of[a] == pair_of[b] for a, b in zip(pairs["essay_a_id"], pairs["essay_b_id"]))
    )
    ok = (
        all(c["regular"] and c["connected"] for c in checks.values())
        and dup == 0
        and self_pairs == 0
        and same_op == 0
    )

    dst.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(dst / "pairs.parquet", index=False)
    (dst / "budget_cutoffs.json").write_text(
        json.dumps({str(d): int(b) for d, b in zip(degrees, budgets)}, indent=2)
    )
    table.to_csv(dst / "verification_table.csv", index=False)
    (dst / "build_report.json").write_text(
        json.dumps(
            {
                "n_items": n,
                "n_matchings": n_m,
                "n_pairs": int(len(pairs)),
                "degrees": degrees,
                "budgets": budgets,
                "seed": int(g["seed"]),
                "n_duplicate_pairs": dup,
                "n_self_pairs": self_pairs,
                "n_same_op_pairs": same_op,
                "n_forbidden_same_op_edges": len(forbid),
                "cross_delta_fraction": float((pairs["band_a"] != pairs["band_b"]).mean()),
                "per_degree": checks,
                "all_ok": bool(ok),
            },
            indent=2,
        )
    )
    print(
        f"[nested {name}] {n} items x {n_m} matchings -> {len(pairs)} pairs; forbidden "
        f"same-OP edges {len(forbid)}; dup={dup} self={self_pairs} same_op={same_op} "
        f"-> {'ALL INVARIANTS HOLD' if ok else 'INVARIANT FAILURE'}"
    )
    return ok


def build_matched(name: str, out_dir) -> bool:
    """Items + anchor graph + test->anchor comparisons."""
    cfg = C.cell_config(name)
    s, g = cfg["source"], cfg["graph"]
    dst = C.work_path(graph_dir(name), out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(g["seed"]))

    # ---- items: the discovery pairs are force-included in the anchor/train split ----
    disc = pd.read_parquet(C.input_path(C.source_path(s["discovery_items"]), out_dir))
    include = disc["pair_id"].unique().tolist()
    items = build_matched_items(
        build_pair_units(),
        n_pairs=int(s["n_pairs"]),
        include_pair_ids=include,
        test_frac=float(s["test_frac"]),
        min_words=int(s["min_words"]),
        max_words=int(s["max_words"]),
        seed=int(s["seed"]),
        condition=s.get("challenge_field", "full_path"),
        test_from_heldout=bool(s.get("test_from_heldout", False)),
        n_test=(int(s["n_test"]) if s.get("n_test") is not None else None),
    )
    items.to_parquet(dst / "items.parquet", index=False)
    tr = items[items["split"] == "train"].reset_index(drop=True)
    te = items[items["split"] == "test"].reset_index(drop=True)
    print(
        f"[matched {name}] {len(items)} exchanges | train pairs {tr.pair_id.nunique()} "
        f"(incl. {items[items.pair_id.isin(include)].pair_id.nunique()}/{len(include)} "
        f"discovery) | test pairs {te.pair_id.nunique()}"
    )

    # ---- anchor graph over the train exchanges, same-OP edges forbidden ----
    train_ids = tr["item_id"].tolist()
    n, d_A = len(train_ids), int(g["anchor_degree"])
    forbid = same_op_edges(tr["pair_id"])
    anchor = build_nested_pairs(
        train_ids, tr["delta"].to_numpy(float), d_A, seed=int(g["seed"]), forbid_edges=forbid
    )
    anchor["group"] = "anchor"
    table, _ = verify_budgets(anchor, train_ids, [n // 2 * d_A])
    pair_of = dict(zip(tr["item_id"], tr["pair_id"]))
    same_op = int(
        sum(pair_of[a] == pair_of[b] for a, b in zip(anchor["essay_a_id"], anchor["essay_b_id"]))
    )
    dup = int(anchor.duplicated(subset=["essay_a_id", "essay_b_id"]).sum())
    row = table.iloc[0]
    reg_ok = int(row["deg_min"]) == d_A == int(row["deg_max"]) and int(row["components"]) == 1
    anchor.to_parquet(dst / "anchor_pairs.parquet", index=False)
    print(
        f"[matched {name}] anchor graph {len(anchor)} pairs | {d_A}-regular={reg_ok} "
        f"dup={dup} same_op={same_op} forbidden={len(forbid)}"
    )

    # ---- test -> anchor comparisons: m anchors from other OPs, shown-first a balanced deck ----
    m = int(g["test_anchors"])
    tr_op = dict(zip(tr["item_id"], tr["op_author"]))
    rows, pi = [], 0
    for _, t in te.iterrows():
        cand = [a for a in train_ids if tr_op[a] != t["op_author"]]
        anchors = rng.choice(cand, size=m, replace=False)
        test_first = np.array([True] * (m // 2) + [False] * (m - m // 2))
        rng.shuffle(test_first)
        for a, tf in zip(anchors, test_first):
            first, other = (t["item_id"], a) if tf else (a, t["item_id"])
            rows.append(
                {
                    "pair_index": pi,
                    "essay_a_id": first,
                    "essay_b_id": other,
                    "shown_first": first,
                    "test_id": t["item_id"],
                    "anchor_id": a,
                    "group": "test",
                }
            )
            pi += 1
    ta = pd.DataFrame(rows)
    ta.to_parquet(dst / "test_anchor_pairs.parquet", index=False)
    per_test = ta.groupby("test_id").size()

    ok = bool(reg_ok and dup == 0 and same_op == 0 and (per_test == m).all())
    (dst / "build_report.json").write_text(
        json.dumps(
            {
                "n_items": len(items),
                "n_pairs": int(items["pair_id"].nunique()),
                "train_ex": len(tr),
                "test_ex": len(te),
                "anchor_pairs": len(anchor),
                "test_anchor_pairs": len(ta),
                "anchor_degree": d_A,
                "test_anchors": m,
                "regular_ok": bool(reg_ok),
                "dup": dup,
                "same_op": same_op,
                "all_ok": ok,
            },
            indent=2,
        )
    )
    print(
        f"[matched {name}] {len(ta)} test->anchor pairs ({m} per test item) -> "
        f"{'ALL INVARIANTS HOLD' if ok else 'INVARIANT FAILURE'}"
    )
    return ok


def copy_items(name: str, owner: str, out_dir) -> bool:
    src = C.input_path(graph_dir(owner) / "items.parquet", out_dir)
    dst = C.work_path(graph_dir(name), out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst / "items.parquet")
    print(f"[copy {name}] items.parquet <- {owner}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", action="append", default=None, help="only these cells")
    ap.add_argument("--skip", action="append", default=[], help="skip these cells")
    ap.add_argument("--out-dir", default=None, help="mirror the cell tree under DIR")
    ap.add_argument("--in-place", action="store_true", help="overwrite the shipped cells")
    a = ap.parse_args()
    C.check_writable(a.out_dir, a.in_place)

    bp = C.pipeline()["build_pairs"]
    jobs = (
        [(c, lambda c=c: build_nested(c, a.out_dir)) for c in bp["nested"]]
        + [(c, lambda c=c: build_matched(c, a.out_dir)) for c in bp["matched"]]
        + [
            (c, lambda c=c, o=o: copy_items(c, o, a.out_dir))
            for c, o in bp.get("items_copied", {}).items()
        ]
    )
    ok = True
    for c, job in jobs:
        if (a.cell and c not in a.cell) or c in a.skip:
            continue
        ok &= job()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
