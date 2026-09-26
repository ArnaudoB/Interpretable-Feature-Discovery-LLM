"""The one place this experiment's paths and shared objects are built.

No stage constructs a path of its own, so the layout is defined here and nowhere else. The
same contract Experiment 3's ``_config.py`` follows.

The unit of organisation is a **cell**: one directory under ``raw/cells/`` holding a scoring
or elicitation run's ``results/`` (score frames, fit reports, graph) and its ``cache/`` of
content-addressed LLM responses. ``config.yaml``'s ``cells:`` block maps the paper's arm names
onto those directories.

Everything here is read-only and offline.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (puts the repo root and this experiment on sys.path)

#: This experiment's directory, derived from __file__ rather than taken from ``_paths``.
#: Every experiment ships a module of that name, so in a full test run ``_paths`` may already
#: be another experiment's -- and then every path here would silently point at it.
EXPERIMENT = Path(__file__).resolve().parents[1]

CONFIG = EXPERIMENT / "config.yaml"
RESULTS = EXPERIMENT / "results"
CELLS = EXPERIMENT / "raw" / "cells"
FIGURES = EXPERIMENT / "figures"

#: Score arms, IN FITTING ORDER (the order is part of the result: see 09_paired_lr):
#: arm -> (run A cells, run B cells or None). Cells joined with "+" share one graph.
SCORE_ARMS = {
    "D + PW": (("pw_discovered", "pw_dropped"), ("pw_rep_a", "pw_rep_b")),
    "D + BT": (("bt_discovered", "bt_dropped"), None),
    "Dr + PW": (("pw_discovered",), ("pw_rep_a", "pw_rep_b")),  # run B restricted to the 16
    "Dr + BT": (("bt_discovered",), None),
    "P + PW": (("pw_prior",), ("pw_prior_rep",)),
    "P + BT": (("bt_prior",), None),
    "S + PW": (("pw_sample",), ("pw_sample_rep",)),
}

#: Row order of tab:cmv-predictive (main text), as the paper prints it.
MAIN_ARMS = [
    "Tan",
    "D + PW",
    "Emb.",
    "D + BT",
    "Dr + PW",
    "Dr + BT",
    "P + PW",
    "P + BT",
    "S + PW",
    "#words",
    "Zero--shot",
]

#: Row order of tab:cmv-predictive-2 (appendix): every arm, best first.
ARM_ORDER = [
    "Emb. (full)",
    "Tan",
    "D + PW",
    "Dr + PW",
    "Emb.",
    "Dr + BT",
    "D + BT",
    "BOW (full)",
    "P + PW",
    "P + BT",
    "POS (full)",
    "S + PW",
    "#words",
    "Zero--shot",
]

#: How the paper typesets an arm name.
ARM_TEX = {
    "Dr + PW": r"D$_{\mathrm{r}}$ + PW",
    "Dr + BT": r"D$_{\mathrm{r}}$ + BT",
    "#words": r"\#words",
}


def slug(arm: str) -> str:
    """Filesystem-safe arm name: ``"Dr + PW"`` -> ``"Dr_p_PW"``."""
    return (
        arm.replace(" + ", "_p_")
        .replace("#", "n")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("--", "_")
        .replace(" ", "_")
    )


@lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def cell(name: str) -> Path:
    """Directory of the cell registered under ``cells.<name>`` in config.yaml."""
    c = cfg()["cells"]
    if name not in c:
        raise KeyError(f"no cell '{name}' in config.yaml; known: {sorted(c)}")
    d = CELLS / c[name]
    if not d.exists():
        raise FileNotFoundError(f"cell '{name}' -> {d} is not present")
    return d


def scores(name: str) -> pd.DataFrame:
    """A cell's item x criterion score frame, anchors and test items together.

    WARNING: ``item_id`` (``wa%04d``) is assigned POSITIONALLY per cell. Two cells' ids mean
    the same item only when they were built from the same graph -- which the config's comments
    record. Across unrelated cells, join on ``(pair_id, side)`` from ``items()`` instead.
    """
    d = cell(name) / "results"
    return pd.concat(
        [pd.read_parquet(d / f"scores_{s}.parquet") for s in ("anchor", "test")]
    ).set_index("item_id")


def graph_owner(name: str) -> str:
    """The cell whose graph defines ``name``'s ``item_id`` space (itself, unless mapped)."""
    return cfg().get("graph_of", {}).get(name, name)


def items(name: str = "pw_discovered") -> pd.DataFrame:
    """The item table (item_id, pair_id, side, delta, op_author, split) for a cell's graph.

    Resolved through ``graph_of`` in config.yaml, because ``item_id`` is positional per graph:
    the (full) arms, for instance, are built on 3,411 training pairs and their ids mean
    something different from the 400-pair cells'.
    """
    owner = graph_owner(name)
    p = cell(owner) / "results" / "graph" / "items.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"cell '{name}' resolves to graph owner '{owner}', which has no "
            f"results/graph/items.parquet. Add the right owner to config.yaml's graph_of."
        )
    return pd.read_parquet(p)


def items_for(*names: str) -> pd.DataFrame:
    """Item table shared by several cells; raises if they do not share one graph."""
    owners = {graph_owner(n) for n in names}
    if len(owners) != 1:
        raise ValueError(
            f"{names} span graphs {sorted(owners)}; item_id is not comparable "
            f"across them -- join on (pair_id, side) instead"
        )
    return items(next(iter(owners)))


def c_grid() -> np.ndarray:
    lo, hi, n = cfg()["task"]["c_grid_log10"]
    return np.logspace(lo, hi, n)


def matched_pairs():
    """The flagship split: 400 anchor/train pairs + all 800 eligible heldout pairs.

    Built by ``core.cmv.designs.matched_pairs_from_split`` from the shipped derived tables, so
    it needs neither the corpus nor a network. The config it is handed is the discovery-BT
    cell's, because that cell defines the split every arm is evaluated on.
    """
    from core.cmv.designs import matched_pairs_from_split

    cell_cfg = yaml.safe_load((cell("bt_discovered") / "config.yaml").read_text())

    # The cell config's `discovery_items` is a path in the layout the run was executed in. It
    # force-includes the 150 discovery pairs in the 400 anchor pairs, so if it silently fails
    # to resolve the split changes and every accuracy moves -- hence the assert rather than the
    # library's tolerant `if dpath.exists()`.
    disc = (cell("discovery") / "results" / "graph" / "essays.parquet").relative_to(EXPERIMENT)
    assert (EXPERIMENT / disc).exists(), f"discovery items missing at {disc}"
    cell_cfg["source"]["discovery_items"] = str(disc)

    mp, _ = matched_pairs_from_split(cell_cfg, EXPERIMENT)
    assert int(mp.train.sum()) == 400, f"expected 400 anchor pairs, got {int(mp.train.sum())}"
    assert int((~mp.train).sum()) == 800, (
        f"expected 800 heldout pairs, got {int((~mp.train).sum())}"
    )
    return mp


def panel(cells: tuple[str, ...]) -> pd.DataFrame:
    """Join several cells' score frames on item_id, each feature from the cell that scored it.

    A later cell contributes its ``features.json`` ``dropped`` list if it has one (the dropped
    cells re-score two kept features as anchors, which must not be taken twice), otherwise its
    columns not already present. Joining on item_id is safe only because ``items_for`` has
    checked the cells share one graph.
    """
    items_for(*cells)
    S = scores(cells[0])
    for c in cells[1:]:
        B = scores(c)
        take = json.loads((cell(c) / "features.json").read_text()).get("dropped") or [
            x for x in B.columns if x not in S.columns
        ]
        S = S.join(B[take], how="outer")
    return S


def reliable_features() -> list[str]:
    """D_r: the discovered challenger-side features that pass the kappa/rho gate.

    The gate (kappa <= kappa_max, rho > rho_threshold, from the discovery cell's config) is
    applied to the canonical sandwich diagnostics; OP-side features also pass it but are not
    challenger-side, so D_r is the gate's survivors among D's 29. Checked against the columns of
    both D_r scoring cells, which were built from the gate's output at the time.
    """
    disc = cell("discovery")
    g = yaml.safe_load((disc / "discovery.yaml").read_text())["model"]
    d = pd.read_csv(disc / "results" / "criterion_gamma_sandwich.csv")
    passed = set(
        d.loc[(d.kappa <= float(g["kappa_max"])) & (d.rho > float(g["rho_threshold"])), "criterion"]
    )
    assert passed == set(d.loc[d.keep, "criterion"]), "gate disagrees with the shipped keep flags"
    d29 = list(panel(("pw_discovered", "pw_dropped")).columns)
    kept = [f for f in d29 if f in passed]
    for c in ("pw_discovered", "bt_discovered"):
        assert set(kept) == set(scores(c).columns), f"D_r != columns of {c}"
    return kept


# --------------------------------------------------------------------------- #
# The judging half (steps 00-08): cell configs, run-path mapping, workspaces
# --------------------------------------------------------------------------- #
#: Everything the steps 00-08 need that the cells' own configs do not record (elicitation
#: models, which cells each step regenerates, baseline arms). Kept out of config.yaml, which
#: describes the analysis half.
PIPELINE = EXPERIMENT / "pipeline.yaml"

#: Path prefix of the layout the cells were run in. Every path in a cell config reads
#: ``<prefix>/<cell>/...``; :func:`source_path` maps it onto this experiment's tree.
_SOURCE_PREFIX = ("experiments", "winning_args")


@lru_cache(maxsize=1)
def pipeline() -> dict:
    return yaml.safe_load(PIPELINE.read_text())


def cell_config_path(name: str) -> Path:
    """The cell's own config file (``discovery.yaml`` for the discovery cell)."""
    d = cell(name)
    for fn in ("config.yaml", "discovery.yaml"):
        if (d / fn).exists():
            return d / fn
    raise FileNotFoundError(f"cell '{name}' ({d}) ships no config.yaml / discovery.yaml")


def cell_config(name: str) -> dict:
    """The cell's config, exactly as the run read it (paths still in the run's layout)."""
    return yaml.safe_load(cell_config_path(name).read_text())


def source_path(p: str | Path) -> Path:
    """Map a run-layout path from a cell config onto this experiment's tree.

    ``<prefix>/data/...``   -> ``raw/data/...``
    ``<prefix>/<cell>/...`` -> ``raw/cells/<cell>/...``

    Raises for anything else rather than guessing: a silently wrong path here changes which
    items a step reads (see ``matched_pairs``'s assert for the case that matters most).
    """
    parts = Path(p).parts
    if parts[:2] != _SOURCE_PREFIX or len(parts) < 3:
        raise ValueError(f"{p!s} is not a run-layout cell path (experiments/winning_args/...)")
    rest = parts[2:]
    if rest[0] == "data":
        return EXPERIMENT / "raw" / "data" / Path(*rest[1:])
    return CELLS / Path(*rest)


def work_path(p: Path, out_dir: str | Path | None) -> Path:
    """Where a step WRITES ``p`` (a path under raw/cells/).

    With ``out_dir`` the cell tree is mirrored under it (``raw/cells/x/y`` -> ``out_dir/x/y``),
    so a replay can be diffed file by file against the shipped cell. Without it the step writes
    in place, which the step scripts allow only behind ``--in-place``.
    """
    p = Path(p)
    if out_dir is None:
        return p
    return Path(out_dir) / p.resolve().relative_to(CELLS.resolve())


def input_path(p: Path, out_dir: str | Path | None) -> Path:
    """Where a step READS ``p``: a regenerated copy under ``out_dir`` if an upstream step wrote
    one, else the shipped file. That is what lets 00 -> 08 run as a chain into a scratch tree
    while each step can also run alone against the shipped inputs."""
    w = work_path(p, out_dir)
    return w if w.exists() else Path(p)


def cell_cache(name: str) -> Path:
    """The cell's response-cache directory, as the run handed it to ``core.judging.cache``.

    ``judge.cache_dir`` from the cell config where there is one; the config-less elicitation and
    baseline cells record theirs in pipeline.yaml ``caches:`` (their drivers hardcoded it, and
    not uniformly: ``cache/`` for the zero-shot cell, ``cache/responses`` for the elicitations).
    The store is always the shipped one: records are keyed by request hash, so adding one never
    overwrites another, and a replay must read exactly what the run read.
    """
    rel = pipeline().get("caches", {}).get(name)
    if rel is not None:
        return cell(name) / rel
    j = cell_config(name).get("judge", {})
    if not j.get("cache_dir"):
        raise KeyError(f"cell '{name}' has no judge.cache_dir and no pipeline.yaml caches: entry")
    return source_path(j["cache_dir"])


def check_writable(out_dir, in_place: bool) -> None:
    """Refuse to write into raw/cells/ unless asked to explicitly.

    The shipped cells are the run of record; a replay goes to ``--out-dir``. ``--in-place`` is
    for a genuine re-run from the corpus, where regenerating the cell is the point.
    """
    if out_dir is None and not in_place:
        raise SystemExit(
            "refusing to write into raw/cells/ (the shipped run of record): pass "
            "--out-dir DIR to regenerate into a scratch tree, or --in-place to "
            "overwrite the cell deliberately"
        )


@lru_cache(maxsize=None)
def prompt(name: str):
    """This experiment's ``prompts/<name>.py``, loaded by file path under a unique module name.

    Not ``import prompts.<name>``: ``prompts`` is a namespace directory here, and Experiment 1
    ships a plain ``prompts.py`` module, so in a full test run a bare import resolves to
    whichever experiment reached ``sys.path`` first (the same trap as ``_config`` itself).
    """
    import importlib.util

    p = EXPERIMENT / "prompts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cmv_prompts_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
