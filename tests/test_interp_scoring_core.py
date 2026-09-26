"""Tests for the core pieces added for the interpretable-scoring experiment.

* ``core.model.holistic.diagnostics.score_spread`` — the denoised spread sigma_k;
* ``core.model.holistic.gate_probabilities`` — per-observation gate probabilities;
* ``core.evaluation.calibration`` — ECE / Brier skill with a pair bootstrap;
* ``core.analysis.cross_judge.validate_clusters`` — the consolidation checks;
* ``core.judging.requests`` — provider specs keep pre-existing cache keys stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.cross_judge import cluster_prompt, extract_json, validate_clusters
from core.evaluation.calibration import compute_ece, pair_bootstrap_calibration
from core.judging.requests import (
    DEEPSEEK_ENDPOINT,
    GENERATE_ENDPOINT,
    MESSAGES_ENDPOINT,
    build_anthropic_scoring_spec,
    build_deepseek_scoring_spec,
    build_gemini_scoring_spec,
    build_scoring_spec,
)
from core.model.holistic import fit, gate_probabilities, generate, sandwich_covariance, std_errors
from core.model.holistic.diagnostics import per_criterion_diagnostics, score_spread


@pytest.fixture(scope="module")
def fitted():
    data = generate(n=30, K=3, T=900, seed=0)
    res = fit(data.w, data.r, data.pairs, K=3, n=30, ridge=1e-3, tol=1e-10, max_iter=5000)
    cov = sandwich_covariance(res, data.w, data.r, data.pairs)
    return data, res, cov


# ---------------------------------------------------------------------------
# score_spread / gate_probabilities
# ---------------------------------------------------------------------------


def test_score_spread_matches_rho_and_population_std(fitted):
    data, res, cov = fitted
    se_s = std_errors(res, cov)["s"]
    sp = score_spread(res, cov, se_s)
    np.testing.assert_allclose(sp.std, res.s.std(axis=0), rtol=1e-12)
    diag = per_criterion_diagnostics(res, data.w, data.r, data.pairs, se_s)
    np.testing.assert_allclose(sp.rho, diag.rho, rtol=1e-10)
    ok = sp.rho > 0
    np.testing.assert_allclose(sp.std_denoised[ok], sp.std[ok] * np.sqrt(sp.rho[ok]), rtol=1e-10)
    # Mechanical: same sqrt(Var V_k) over a smaller spread.
    assert np.all(sp.se_std_denoised[ok] >= sp.se_std[ok])


def test_gate_probabilities_reproduce_fitted_nll(fitted):
    data, res, _ = fitted
    pw, pc = gate_probabilities(res.s, res.gamma, res.beta, data.w, data.pairs)
    w, r = data.w, data.r
    nll = -(
        np.sum(w * np.log(pw) + (1 - w) * np.log(1 - pw))
        + np.sum(r * np.log(pc) + (1 - r) * np.log(1 - pc))
    )
    assert nll == pytest.approx(res.nll_unregularized, rel=1e-9)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_bootstrap_ece_equals_loop_ece():
    rng = np.random.default_rng(0)
    p = rng.random((400, 5))
    y = (rng.random((400, 5)) < p).astype(float)
    cal = pair_bootstrap_calibration(p, y, np.full_like(p, 0.5), n_boot=50)
    assert cal["ece"] == pytest.approx(compute_ece(p, y), abs=1e-12)
    # The reference forecast is scored on the same cells: here it is the constant 1/2.
    assert cal["brier_emp"] == pytest.approx(np.mean((0.5 - y) ** 2))
    assert cal["ece_lo"] <= cal["ece"] <= cal["ece_hi"]


def test_all_ones_mask_is_bit_identical_to_no_mask():
    rng = np.random.default_rng(1)
    p = rng.random((200, 3))
    y = (rng.random((200, 3)) < p).astype(float)
    ref = np.full_like(p, 0.4)
    a = pair_bootstrap_calibration(p, y, ref, n_boot=40, seed=3)
    b = pair_bootstrap_calibration(p, y, ref, mask=np.ones_like(p), n_boot=40, seed=3)
    for k in ("ece", "brier", "bss_chance", "bss_emp", "ece_lo", "bss_emp_hi"):
        assert a[k] == b[k]


# ---------------------------------------------------------------------------
# cross-judge consolidation
# ---------------------------------------------------------------------------

ENTRIES = [
    (0, "j1", "Grammar", "d0"),
    (1, "j2", "Grammar accuracy", "d1"),
    (2, "j1", "Vocabulary", "d2"),
]


def test_validate_clusters_accepts_a_valid_grouping():
    clusters = [
        {
            "name": "Grammar",
            "members": [
                {"index": 0, "judge": "j1", "name": "Grammar"},
                {"index": 1, "judge": "j2", "name": "Grammar accuracy"},
            ],
        },
        {"name": "Vocabulary", "members": [{"index": 2, "judge": "j1", "name": "Vocabulary"}]},
    ]
    stats = validate_clusters(clusters, ENTRIES)
    assert stats == {"n_clusters": 2, "n_placed": 3, "clusters_by_size": {2: 1, 1: 1}}


@pytest.mark.parametrize(
    "clusters,msg",
    [
        (
            [
                {
                    "name": "x",
                    "members": [
                        {"index": 0, "judge": "j1", "name": "Grammar"},
                        {"index": 2, "judge": "j1", "name": "Vocabulary"},
                        {"index": 1, "judge": "j2", "name": "Grammar accuracy"},
                    ],
                }
            ],
            "two criteria from",
        ),
        (
            [
                {
                    "name": "x",
                    "members": [
                        {"index": 0, "judge": "j1", "name": "Grammar"},
                        {"index": 1, "judge": "j2", "name": "Grammar accuracy"},
                    ],
                }
            ],
            "never placed",
        ),
        (
            [
                {
                    "name": "x",
                    "members": [
                        {"index": 0, "judge": "j1", "name": "Grammar!"},
                        {"index": 1, "judge": "j2", "name": "Grammar accuracy"},
                        {"index": 2, "judge": "j1", "name": "Vocabulary"},
                    ],
                }
            ],
            "name",
        ),
        (
            [{"name": "x", "members": [{"index": 7, "judge": "j1", "name": "Grammar"}]}],
            "invented index",
        ),
    ],
)
def test_validate_clusters_rejects_violations(clusters, msg):
    with pytest.raises(ValueError, match=msg):
        validate_clusters(clusters, ENTRIES)


def test_cluster_prompt_and_extract_json():
    out = cluster_prompt(
        "A [[CORPUS_FRAMING]] B\n[[INDEXED_CRITERION_LIST]]", ENTRIES[:1], "essays"
    )
    assert out == "A essays B\n[0] (j1) Grammar: d0"
    assert extract_json('```json\n{"clusters": []}\n```') == '{"clusters": []}'


# ---------------------------------------------------------------------------
# request specs
# ---------------------------------------------------------------------------

SCHEMA = {
    "name": "cmp",
    "schema": {
        "type": "object",
        "properties": {"b": {"type": "string"}, "a": {"type": "string"}},
        "required": ["b", "a"],
        "additionalProperties": False,
    },
}


def test_openai_spec_canonical_has_no_provider_only_fields():
    """Fields added for other providers enter ``canonical()`` only when set, so every
    OpenAI spec -- and every cached OpenAI response of Exps 1 and 3 -- keeps its key."""
    spec = build_scoring_spec(
        "p0", "gpt-5.4-mini", "sys", "usr", SCHEMA, 100, reasoning_effort="none"
    )
    assert set(spec.canonical()) == {
        "endpoint",
        "model",
        "system",
        "user",
        "max_output_tokens",
        "temperature",
        "seed",
        "json_schema",
        "reasoning_effort",
        "cache_version",
    }


def test_provider_specs_never_collide_and_carry_the_schema():
    args = ("p0", "m", "sys", "usr", SCHEMA, 100)
    specs = [
        build_scoring_spec(*args),
        build_anthropic_scoring_spec(*args),
        build_gemini_scoring_spec(*args),
        build_deepseek_scoring_spec(*args),
    ]
    assert len({s.cache_key() for s in specs}) == 4
    _, anth, gem, ds = specs
    assert anth.endpoint == MESSAGES_ENDPOINT
    assert anth.body()["tool_choice"] == {"type": "tool", "name": "cmp"}
    assert gem.endpoint == GENERATE_ENDPOINT
    gcfg = gem.body()["generationConfig"]
    assert gcfg["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
    # Gemini: additionalProperties dropped, declared order kept as generation order.
    assert gcfg["responseSchema"]["propertyOrdering"] == ["b", "a"]
    assert "additionalProperties" not in gcfg["responseSchema"]
    assert ds.endpoint == DEEPSEEK_ENDPOINT
    assert ds.body()["thinking"] == {"type": "disabled"}
    assert ds.body()["tools"][0]["function"]["strict"] is True
