"""Step 6d: solve for the scoring reps that cost-match the taxonomy arm to "Ours".

    python scripts/06d_cost_match.py --config config.yaml

Every cost here is on a **uniform Batch-API basis** (the paper reports batch prices throughout),
even for the phases that were actually billed sync — the rubric elicitations and the taxonomy
call. What was submitted to which API is recorded per phase in the ``mode`` field so the two are
never conflated.

  target  C*     = comparative judging + taxonomy            (the "Ours" arm)
  fixed          = E elicitations + 1 taxonomy fusion        (the taxonomy arm, before scoring)
  per rep        = 200 letters x (a + b K) $/spec            (a, b fit on M3a/b/c)
  n*             = floor((C* - fixed) / per rep)

The per-spec cost is fit on the M3 arms rather than the M1/M2 arm on purpose: pooled rubrics
carry long per-level descriptors, so M3a runs ~3.5k input tokens/spec at K=27 while the compact
whole-dataset rubric runs ~3.0k at K=49. The taxonomy criteria inherit their medoid's levels, so
they look like the pooled rubrics.

Writes results/rubric_stability/taxo_pool/cost_match.json, read by 07_score_pointwise.py and by
the table steps (the Cost column of tab:main).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.judging.pricing import actual_usd

N_LETTERS = 200


def _cache_records(root: Path, cache_rel: str) -> list[dict]:
    out = []
    for p in glob.glob(str(root / cache_rel / "**" / "*.json"), recursive=True):
        try:
            out.append(json.loads(Path(p).read_text()))
        except (ValueError, OSError):
            pass
    return out


def _by_prefix(recs, prefix):
    return [r for r in recs if (r.get("custom_id") or "").startswith(prefix)]


def _batch_usd(recs, pricing):
    """Batch-basis cost of these records, regardless of how they were actually billed."""
    return actual_usd(recs, pricing, batch=True)


def _per_spec_fit(root: Path, pricing) -> tuple[float, float, list]:
    """Fit $/spec = a + b*K on the M3a/b/c scoring batches (batch basis)."""
    pts = []
    for letter, dims_file in (("a", "3a"), ("b", "3b"), ("c", "3c")):
        bo = root / "batch" / f"pointwise_{dims_file}" / "batch_output.jsonl"
        rub = root / "results" / "rubric_stability" / f"pooled_rubric_{dims_file}.json"
        if not (bo.exists() and rub.exists()):
            continue
        K = json.loads(rub.read_text())["n_features"]
        n = ti = to = 0
        model = ""
        for line in bo.read_text().splitlines():
            if not line.strip():
                continue
            body = (json.loads(line).get("response") or {}).get("body") or {}
            u = body.get("usage") or {}
            if not u:
                continue
            model = body.get("model", model)
            n += 1
            ti += u.get("input_tokens", 0)
            to += u.get("output_tokens", 0)
        if not n:
            continue
        usd = _batch_usd(
            [{"model": model, "usage": {"input_tokens": ti, "output_tokens": to}}], pricing
        )
        pts.append({"arm": f"M3{letter}", "K": K, "n_specs": n, "usd_per_spec": usd / n})
    if len(pts) < 2:
        raise SystemExit("need >=2 M3 scoring batches to fit the per-spec cost model")
    K = np.array([p["K"] for p in pts], float)
    y = np.array([p["usd_per_spec"] for p in pts], float)
    b, a = np.polyfit(K, y, 1)
    return float(a), float(b), pts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    j = cfg["judge"]
    root = Path(args.config).parent
    pool = root / "results" / "rubric_stability" / "taxo_pool"
    pricing = cfg.get("pricing")
    E = int(j.get("m3_elicitations", 27))

    recs = _cache_records(root, j["cache_dir"])

    # --- target: the "Ours" arm, on a batch basis ------------------------- #
    comp = _by_prefix(recs, "cmp:")
    taxo_ours = _by_prefix(recs, "taxonomy")
    usd_comp = _batch_usd(comp, pricing)
    usd_taxo_ours = _batch_usd(taxo_ours, pricing)
    target = usd_comp + usd_taxo_ours

    # --- fixed cost of the taxonomy arm ----------------------------------- #
    # Per-elicitation cost comes from the shipped ledger for the 50 reps that actually feed the
    # pool. Do NOT prefix-match "rubric_stab" against the cache: it also catches 100 records
    # from an unrelated (non-open) elicitation variant and skews the mean.
    elic_ledger = root / "results" / "rubric_stability" / "cost_ledger.json"
    if elic_ledger.exists():
        led_e = json.loads(elic_ledger.read_text())
        # the ledger is recorded sync (batch_discount_applied=false); halve for the batch basis
        usd_elic_per = led_e["mean_usd_per_call"] * (
            0.5 if not led_e.get("batch_discount_applied") else 1.0
        )
        n_elic_src = f"ledger ({led_e['n_reps']} reps, {led_e.get('mode')})"
    else:
        elic = _by_prefix(recs, "rubric_stab_open:")
        if not elic:
            raise SystemExit("no elicitation records in the cache")
        usd_elic_per = _batch_usd(elic, pricing) / len(elic)
        n_elic_src = f"cache ({len(elic)} records)"
    ledger_f = pool / "cost_ledger.json"
    if ledger_f.exists():
        led = json.loads(ledger_f.read_text())
        usd_taxo_per = led["usd_batch_per_run"]
        K = int(round(led["K_mean"]))
        K_run0 = None
        r0 = pool / "run_000.json"
        if r0.exists():
            K_run0 = json.loads(r0.read_text())["n_features"]
        K = K_run0 or K
    else:
        raise SystemExit("no taxo_pool cost ledger — run 06b_pool_taxonomy.py collect first")
    fixed = E * usd_elic_per + usd_taxo_per

    # --- per-rep scoring cost at this K ----------------------------------- #
    a, b, pts = _per_spec_fit(root, pricing)
    per_spec = a + b * K
    per_rep = N_LETTERS * per_spec
    budget = target - fixed
    n_star = max(1, int(np.floor(budget / per_rep))) if per_rep > 0 else 1

    out = {
        "basis": "batch",
        "note": "all phases priced at batch rates; `mode` records what was actually submitted where",
        "modes": {
            "comparative_judging": "batch",
            "taxonomy_ours": "sync",
            "elicitations": "sync",
            "taxonomy_pool": "batch",
            "scoring": "batch",
        },
        "target_ours_usd": target,
        "target_breakdown": {
            "comparative_judging": usd_comp,
            "taxonomy": usd_taxo_ours,
            "n_comparative_calls": len(comp),
        },
        "fixed_usd": fixed,
        "fixed_breakdown": {
            "E": E,
            "usd_per_elicitation": usd_elic_per,
            "usd_per_elicitation_source": n_elic_src,
            "usd_elicitations": E * usd_elic_per,
            "usd_taxonomy_fusion": usd_taxo_per,
        },
        "K": K,
        "per_spec_model": {"a": a, "b": b, "points": pts},
        "usd_per_spec": per_spec,
        "usd_per_rep": per_rep,
        "budget_for_scoring_usd": budget,
        "n_star": n_star,
        "projected_total_usd": fixed + n_star * per_rep,
    }
    pool.mkdir(parents=True, exist_ok=True)
    (pool / "cost_match.json").write_text(json.dumps(out, indent=2))

    print(
        f"[cost-match] target  Ours          ${target:6.3f}  "
        f"(judging ${usd_comp:.3f} + taxonomy ${usd_taxo_ours:.3f})"
    )
    print(
        f"[cost-match] fixed   {E} elicitations + fusion  ${fixed:6.3f}  "
        f"(${usd_elic_per:.4f}/elicitation, ${usd_taxo_per:.3f} fusion)"
    )
    print(
        f"[cost-match] K = {K};  $/spec = {a:.6f} + {b:.6f}*K = {per_spec:.6f};  "
        f"$/rep = ${per_rep:.3f}"
    )
    print(
        f"[cost-match] budget  ${budget:6.3f}  ->  n* = {n_star}  "
        f"(projected total ${fixed + n_star * per_rep:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
