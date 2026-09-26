"""Step 0: the (OP view, challenger reply) items the discovery judge compares. No LLM calls.

    python scripts/00_build_items.py --out-dir DIR        # regenerate into DIR/<cell>/...
    python scripts/00_build_items.py --in-place           # rewrite the shipped cell (a real re-run)

Samples ``source.n_pairs`` (150) Tan TRAINING pairs whose two root replies both fall in the
20-1200 word band and takes both sides of each, so the 300 items are exactly half delta
winners. Each item's text is ``[OP VIEW] ... [CHALLENGE] ...``. Reads the derived pair table in
raw/data/, never the ConvoKit corpus.

The ``source.items_from`` option (copy another cell's items) is used by no cell the paper
cites and is not implemented.

Writes <cell>/results/graph/essays.parquet for each cell in pipeline.yaml ``build_items.cells``
(the discovery cell).
"""

from __future__ import annotations

import argparse

import _paths  # noqa: F401
import _config as C
from core.cmv.data import build_pair_units
from core.cmv.items import build_items


def build(name: str) -> "pd.DataFrame":  # noqa: F821
    cfg = C.cell_config(name)
    s = cfg["source"]
    if s.get("items_from"):
        raise NotImplementedError(
            f"{name}: source.items_from (copying another cell's items) is not implemented"
        )
    return build_items(
        build_pair_units(),
        n_pairs=int(s["n_pairs"]),
        both_sides=bool(s.get("both_sides", True)),
        train_only=bool(s.get("train_only", True)),
        min_words=int(s.get("min_words", 20)),
        max_words=int(s.get("max_words", 1200)),
        seed=int(s.get("seed", cfg["cell"]["seed"])),
        condition=s.get("challenge_field", "full_path"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cell",
        action="append",
        default=None,
        help="cell(s) to build (default: pipeline.yaml build_items.cells)",
    )
    ap.add_argument("--out-dir", default=None, help="mirror the cell tree under DIR")
    ap.add_argument("--in-place", action="store_true", help="overwrite the shipped cell")
    a = ap.parse_args()
    C.check_writable(a.out_dir, a.in_place)

    for name in a.cell or C.pipeline()["build_items"]["cells"]:
        items = build(name)
        g = C.work_path(C.source_path(C.cell_config(name)["graph"]["out_dir"]), a.out_dir)
        g.mkdir(parents=True, exist_ok=True)
        items.to_parquet(g / "essays.parquet", index=False)
        wc = items["path_wc"]
        print(
            f"[items {name}] {len(items)} items ({int(items.delta.sum())} delta=1) from "
            f"{items.op_author.nunique()} OP threads; words median {wc.median():.0f} "
            f"[{wc.min()}, {wc.max()}] -> {g / 'essays.parquet'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
