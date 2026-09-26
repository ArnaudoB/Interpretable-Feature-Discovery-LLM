"""Every experiment's ``results/paper_numbers.json`` must be well formed and match its snapshot.

Each experiment's last step (01: 13_claim_ledger, 02: 14_paper_numbers, 03: 09_paper_numbers)
recomputes the numbers quoted in the paper from the other steps' outputs and writes them in the
``core.ledger`` schema. ``tests/data/paper_numbers/<experiment>.json`` holds the values that
correspond to the paper; after re-running an experiment, any change shows up here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("01_synthetic_cover_letters", "02_changemyview", "03_interpretable_scoring")
SNAPSHOT = Path(__file__).resolve().parent / "data" / "paper_numbers"


def _load(exp):
    p = ROOT / "experiments" / exp / "results" / "paper_numbers.json"
    if not p.exists():
        pytest.skip(f"{exp}: paper_numbers.json not built")
    return json.loads(p.read_text())


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_schema(exp):
    d = _load(exp)
    assert d["experiment"] == exp
    ids = [r["id"] for r in d["numbers"]]
    assert len(ids) == len(set(ids)) == d["n"], "duplicate or miscounted record ids"
    for r in d["numbers"]:
        assert set(r) == {"id", "where", "what", "value"}, r.get("id")
        assert r["where"].strip() and r["what"].strip(), r["id"]


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_values_match_snapshot(exp):
    got = {r["id"]: r["value"] for r in _load(exp)["numbers"]}
    want = json.loads((SNAPSHOT / f"{exp}.json").read_text())
    assert got.keys() == want.keys(), set(got) ^ set(want)
    changed = {k: (want[k], got[k]) for k in want if got[k] != want[k]}
    assert not changed, changed
