"""Experiment 3's request builders must reproduce the shipped cache keys, for every provider.

Each cached response is filed under ``sha256(canonical request)``. If a builder in
``core/judging/requests.py`` or a prompt in ``prompts/<corpus>.py`` drifts by one byte, the
rebuilt request misses the cache and the offline replay silently loses that pair (step 01
counts it as missing). This test rebuilds every run's requests exactly as step 01 does and
checks each key resolves to a record for the same ``custom_id`` -- 16,000 keys, four
providers (OpenAI Responses, Anthropic Messages, Gemini generateContent, DeepSeek chat).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "03_interpretable_scoring"
pytestmark = pytest.mark.skipif(
    not (EXP / "raw" / "cache").exists(), reason="experiment 3 response caches not present"
)


def _judge_module():
    sys.path.insert(0, str(EXP / "scripts"))
    spec = importlib.util.spec_from_file_location("judge_step", EXP / "scripts" / "01_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runs():
    sys.path.insert(0, str(EXP / "scripts"))
    import _config

    return _config.run_names()


@pytest.mark.parametrize("name", _runs())
def test_every_request_hits_the_shipped_cache(name):
    judge = _judge_module()
    import _config
    from core.judging import cache

    run = _config.load_run(name)
    if not (run.graph / "pairs.parquet").exists():
        pytest.skip("graph not built; run scripts/00_build_pairs.py --run all")
    judge.ELIC = _config.load_prompts(run)
    specs = judge.build_specs(run.cfg, run.graph)
    assert len(specs) == 2000
    for spec in specs:
        rec = cache.get(run.cache, spec.cache_key())
        assert rec is not None, f"{name}: {spec.custom_id} misses the cache"
        assert rec["custom_id"] == spec.custom_id and not rec.get("error")
