"""Experiment 1's pointwise-canonical ablation must replay entirely from the shipped cache.

These four arms are tab:main's "Ablations" block. Two things have to hold for them to be
reproducible offline, and each is easy to break silently:

1. **Cache keys.** Every response is filed under ``sha256(canonical request)``. If the prompt
   in ``prompts_pointwise_canonical.py``, the criterion panel, the permutation seeding or
   ``max_out`` drifts by one byte, the rebuilt request misses and step 07d quietly drops that
   letter-rep from the score frame. This rebuilds all 6,800 specs and checks each resolves.

2. **The panels themselves.** k32 must be the 33 canonical criteria minus the ``Other``
   catch-all, and k21 the kappa/rho-kept set.

Skipped when the caches are not present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1] / "experiments" / "01_synthetic_cover_letters"
pytestmark = pytest.mark.skipif(
    not (EXP / "results" / "pointwise_canonical_k21" / "scores.parquet").exists(),
    reason="experiment 1 pointwise-canonical artifacts not present",
)

EXPECTED = {"k32": (32, 14), "k21": (21, 20)}  # (panel size, n* from the config)


def _step():
    """Import step 07d the way the experiment's own scripts are imported."""
    import importlib.util

    for p in (str(EXP / "scripts"), str(EXP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "pwc_step", EXP / "scripts" / "07d_score_pointwise_canonical.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg():
    return yaml.safe_load((EXP / "config.yaml").read_text())


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_panel_is_the_documented_criterion_set(arm):
    step, cfg = _step(), _cfg()
    crit = step._criteria(cfg, EXP, arm)
    K, _ = EXPECTED[arm]
    assert len(crit) == K
    names = [c["name"] for c in crit]
    assert names == sorted(names), "the panel must be name-sorted; the prompt order depends on it"

    tax = json.loads((EXP / "results" / cfg["pointwise_canonical"]["criteria_source"]).read_text())
    all_names = {c["name"] for c in tax["criteria"]}
    if arm == "k32":
        # 33 surfaced criteria; the pointwise panel drops the Other catch-all
        assert len(all_names) == 33
        assert all_names - set(names) == {"Other"}
    else:
        kept = json.loads(
            (EXP / "results" / cfg["pointwise_canonical"]["gated_fit_summary"]).read_text()
        )["kept"]
        assert set(names) == set(kept)


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_n_star_comes_from_the_config(arm):
    """n* (repetitions per letter) is fixed in config.yaml, not re-solved from the price table."""
    step, cfg = _step(), _cfg()
    _, n_star = EXPECTED[arm]
    assert step._n_star(cfg, arm) == n_star


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_every_request_hits_the_shipped_cache(arm):
    from core.judging import cache

    step, cfg = _step(), _cfg()
    K, n_star = EXPECTED[arm]
    text = step._text(cfg, EXP)
    crit = step._criteria(cfg, EXP, arm)
    specs = step._score_specs(cfg, text, sorted(text), crit, n_star, arm)
    assert len(specs) == len(text) * n_star

    cache_dir = EXP / cfg["judge"]["cache_dir"]
    misses = [s.custom_id for s in specs if cache.get(cache_dir, s.cache_key()) is None]
    assert not misses, (
        f"{len(misses)}/{len(specs)} {arm} requests miss the cache, e.g. {misses[:3]}"
    )
