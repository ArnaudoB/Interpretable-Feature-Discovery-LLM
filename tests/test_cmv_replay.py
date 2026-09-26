"""Experiment 2's LLM steps must rebuild the exact requests the shipped caches answer.

Each cached response is filed under ``sha256(canonical request)``. If a prompt in
``experiments/02_changemyview/prompts/``, a builder in ``core/judging/requests.py``, an input
table or a step's spec construction drifts by one byte, the rebuilt request misses and the
offline replay silently loses that comparison. This rebuilds every LLM step's requests exactly
as steps 02-08 do (``build_specs`` et al.) and checks each one hits the shipped cache:

  discovery comparisons 6,000 | three BT cells 3 x 32,000 | ten pointwise cells 2,400 / 8,422
  | the sample elicitation 1 | the zero-shot judge 1,600

Two LLM calls have no response cache, so they are checked by re-derivation instead
(see steps 03 and 05): the taxonomy call (its request is rebuilt -- 664 listed dimensions -- and
its saved criteria re-derive ``mapping.json`` byte for byte) and the five prior-panel draws.

Fast: specs are built and their cache files stat'ed; nothing is refitted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "02_changemyview"
SCRIPTS = EXP / "scripts"
pytestmark = pytest.mark.skipif(
    not (EXP / "raw" / "cells" / "feature_bt_root" / "cache").exists(),
    reason="experiment 2 response caches not present",
)

_SHARED = ("_paths", "_config", "_judge")


@lru_cache(maxsize=None)
def _shared(name: str):
    """This experiment's ``scripts/<name>.py`` under a unique module name.

    Every experiment ships a ``_paths``/``_config``, so a bare import resolves to whichever one
    another test loaded first.
    """
    saved = {k: sys.modules.get(k) for k in _SHARED}
    try:
        for dep in _SHARED[: _SHARED.index(name)]:
            sys.modules[dep] = _shared(dep)
        spec = importlib.util.spec_from_file_location(f"cmv{name}", SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@lru_cache(maxsize=None)
def _step(fname: str):
    """A numbered step script, with ``_paths``/``_config``/``_judge`` bound to this experiment's."""
    saved = {k: sys.modules.get(k) for k in _SHARED}
    try:
        for k in _SHARED:
            sys.modules[k] = _shared(k)
        spec = importlib.util.spec_from_file_location(
            f"cmv_step_{Path(fname).stem}", SCRIPTS / fname
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _assert_all_cached(specs, cache_dir, n_expected):
    from core.judging import cache

    assert len(specs) == n_expected
    missing = [
        s.custom_id for s in specs if not cache.cache_path(cache_dir, s.cache_key()).exists()
    ]
    assert not missing, f"{len(missing)} of {len(specs)} requests miss {cache_dir}: {missing[:5]}"
    rec = cache.get(cache_dir, specs[0].cache_key())  # and the store is what it says
    assert rec["request_hash"] == specs[0].cache_key() and not rec.get("error")


def test_discovery_comparisons_replay():
    S = _step("02_judge_discovery.py")
    C = _shared("_config")
    cfg, text, pairs = S.load()
    _assert_all_cached(S.build_specs(cfg, text, pairs), C.cell_cache("discovery"), 6000)


def test_taxonomy_request_rebuilds_and_its_mapping_rederives(tmp_path):
    S = _step("03_taxonomy.py")
    C = _shared("_config")
    spec, g = S.build()
    assert len(g) == 664  # the number of listed dimensions
    shipped = C.cell("discovery") / "results" / "gated" / "llm_taxonomy"
    tax = json.loads((shipped / "taxonomy.json").read_text())
    S.assemble(
        {"text": json.dumps({"criteria": tax["criteria"]}), "usage": tax["usage"]}, g, tmp_path
    )
    assert (tmp_path / "mapping.json").read_bytes() == (shipped / "mapping.json").read_bytes()


def test_prior_panel_rederives(tmp_path):
    S = _step("05_elicit_prior.py")
    C = _shared("_config")
    assert len(S.build_specs()) == C.pipeline()["prior"]["draws"]
    panels = json.loads((C.cell("prior_panel") / "results" / "panels_blind.json").read_text())
    shipped = json.loads((C.cell("prior_panel") / "features_blind.json").read_text())
    assert shipped["criteria"] == panels[0]["criteria"]  # draw 0, promoted by position


def test_sample_elicitation_replays():
    S = _step("05b_elicit_sample.py")
    C = _shared("_config")
    _assert_all_cached(S.build_specs(), C.cell_cache("sample_panel"), 1)


@pytest.mark.parametrize("cell", ["bt_discovered", "bt_dropped", "bt_prior"])
def test_bt_scoring_replays(cell):
    S = _step("06_score_bt.py")
    C = _shared("_config")
    cfg, items, pairs, criteria = S.load(cell)
    _assert_all_cached(S.build_specs(cfg, items, pairs, criteria), C.cell_cache(cell), 32000)


@pytest.mark.parametrize(
    "cell",
    [
        "pw_discovered",
        "pw_dropped",
        "pw_rep_a",
        "pw_rep_b",
        "pw_prior",
        "pw_prior_rep",
        "pw_sample",
        "pw_sample_rep",
        "pw_discovered_full",
        "pw_dropped_full",
    ],
)
def test_pointwise_scoring_replays(cell):
    S = _step("07_score_pointwise.py")
    C = _shared("_config")
    cfg, items, criteria = S.load(cell)
    _assert_all_cached(
        S.build_specs(cfg, items, criteria),
        C.cell_cache(cell),
        8422 if cell.endswith("_full") else 2400,
    )


def test_zeroshot_replays():
    S = _step("08_baselines.py")
    C = _shared("_config")
    _assert_all_cached(S.zeroshot_specs(S.zeroshot_pairs()), C.cell_cache("zeroshot"), 1600)


def test_forbid_free_graph_is_the_shared_builder():
    """core.cmv.graphs restores forbid_edges; with none it must be draw-for-draw core.graph's."""
    from core.cmv.graphs import build_nested_pairs as with_forbid
    from core.graph.matching import build_nested_pairs as shared

    ids = [f"x{i:03d}" for i in range(40)]
    bands = np.arange(40) % 2
    pd.testing.assert_frame_equal(with_forbid(ids, bands, 5, seed=3), shared(ids, bands, 5, seed=3))
    forbid = {(2 * i, 2 * i + 1) for i in range(20)}
    g = with_forbid(ids, bands, 5, seed=3, forbid_edges=forbid)
    idx = {x: i for i, x in enumerate(ids)}
    used = {tuple(sorted((idx[a], idx[b]))) for a, b in zip(g.essay_a_id, g.essay_b_id)}
    assert not used & forbid


def test_cell_config_paths_resolve_in_this_repository():
    C = _shared("_config")
    assert (
        C.source_path("experiments/winning_args/data/pair_units.parquet")
        == EXP / "raw" / "data" / "pair_units.parquet"
    )
    assert (
        C.source_path("experiments/winning_args/feature_bt_root/cache/responses")
        == EXP / "raw" / "cells" / "feature_bt_root" / "cache" / "responses"
    )
    with pytest.raises(ValueError):
        C.source_path("experiments/other/x")
