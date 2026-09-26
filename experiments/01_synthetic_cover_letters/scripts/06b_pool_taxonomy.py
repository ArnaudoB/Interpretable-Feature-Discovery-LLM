"""Step 6b: fuse the E subsample rubrics into one criterion set with an LLM taxonomy.

    python scripts/06b_pool_taxonomy.py submit  --config config.yaml
    python scripts/06b_pool_taxonomy.py collect --config config.yaml

The clustering arm of step 6 fuses the pooled features by agglomerative clustering at an
arbitrary cosine cutoff (0.55/0.60/0.65). That cutoff is doing real damage: the union of the
E=27 elicitations already *names* nearly every relevant dial, so the dials the clustering fusion
fails to recover are lost to the threshold, not to the elicitations (this is the bind that
figures/make_threshold_pair.py draws). This script replaces the threshold with the same
LLM-taxonomy fusion the comparative arm uses (``TAXONOMY_POOL_PROMPT``), which groups by MEANING
and has no knob. It is the fusion behind the paper's arms 2a and 2b (Local LLMs rows of
tab:main).

``judge.taxo_pool_runs`` (R, default 50) independent taxonomies are run, each over a random
E-subset of the ``rubric_stability/reps/`` elicitations, to get a distribution over recovered
dials (step 6c) rather than a single draw. Run 000 is fixed to the canonical first-E reps --
the same slice ``06_pool_rubrics.py`` takes -- so the scored arm is directly comparable to
M3a/b/c.

Writes, per run, ``results/rubric_stability/taxo_pool/run_XXX.json``: the criteria with their
definitions and members, plus a scorable ``features`` list (each criterion carries the levels of
its medoid member, the same medoid rule ``06_pool_rubrics.py`` uses). The "Other" catch-all is
kept in its own key and excluded from ``features`` -- it is a real bucket, not a scored column.

Batch API (gpt-5.4, high reasoning). Re-running ``submit`` after a partial batch resubmits only
the calls still missing from the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.embeddings import embed_texts
from core.judging import cache
from core.judging.batch import build_jsonl, collect_into_cache, poll_until_done, submit_batch
from core.judging.cache import partition_cached
from core.judging.pricing import actual_usd
from core.judging.requests import CHAT_ENDPOINT, build_continuation_spec
from core.openai_client import get_client
import prompts as ELIC

_SYS = "You organize evaluation dimensions into canonical criteria and output only JSON."
_OUT_SUB = "taxo_pool"


def _feature_key(f) -> str:
    """The embedded string for a feature — identical to 06_pool_rubrics.py."""
    return f"{f['name']}. {f.get('description', '')}"


def _load_reps(stab: Path) -> list[dict]:
    reps = [json.loads(p.read_text()) for p in sorted((stab / "reps").glob("rep_*.json"))]
    if not reps:
        raise SystemExit("no elicitations — run 05_elicit_rubrics.py reps first")
    return reps


def _subsets(n_reps: int, E: int, R: int, base_seed: int) -> list[list[int]]:
    """R subsets of E rep indices. Run 0 is the canonical first-E slice (matches 06_pool)."""
    if E > n_reps:
        raise SystemExit(f"need E={E} elicitations, found {n_reps}")
    out = [list(range(E))]
    for r in range(1, R):
        seed = int(hashlib.md5(f"{base_seed}:taxopool:{r}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out.append(sorted(rng.choice(n_reps, size=E, replace=False).tolist()))
    return out


def _pool(reps: list[dict], picked: list[int]) -> list[dict]:
    """Flatten the features of the picked elicitations, tagging each with its run."""
    return [{"rep": i, **f} for i in picked for f in reps[i]["features"]]


def _listing(feats: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {f['name']} - {f.get('description', '')} (run {f['rep']})"
        for i, f in enumerate(feats)
    )


def _specs(cfg, reps, subsets):
    j = cfg["judge"]
    specs = []
    for r, picked in enumerate(subsets):
        prompt = ELIC.TAXONOMY_POOL_PROMPT.replace(
            "[[INDEXED_DIMENSION_LIST]]", _listing(_pool(reps, picked))
        )
        specs.append(
            build_continuation_spec(
                f"taxo_pool:t{r:03d}",
                j["reasoning_model"],
                _SYS,
                prompt,
                int(j["max_out_taxonomy"]),
                reasoning_effort=j["high_effort"],
                meta={"run": r, "picked": picked},
            )
        )
    return specs


def _parse(text: str) -> list[dict]:
    txt = text.strip()
    if txt.startswith("```"):
        txt = txt[txt.find("{") : txt.rfind("}") + 1]
    return json.loads(txt)["criteria"]


def _validate(crit: list[dict], feats: list[dict]) -> tuple[dict, dict]:
    """Enforce the membership contract the prompt promises is checked programmatically.

    Returns ``(index -> criterion name, report)``. Duplicated indices keep their first
    placement; invented indices are dropped; unplaced indices go to "Other" (NOT to their own
    singleton criterion as 03_taxonomy.py does — a singleton per leftover would inflate K and
    manufacture recoveries that the taxonomy never actually made).
    """
    n = len(feats)
    placed: dict[int, str] = {}
    dup = invented = mismatched = 0
    for c in crit:
        for mem in c.get("members", []):
            try:
                i = int(mem["index"])
            except (KeyError, TypeError, ValueError):
                invented += 1
                continue
            if not (0 <= i < n):
                invented += 1
                continue
            if i in placed:
                dup += 1
                continue
            if (mem.get("name") or "").strip() != feats[i]["name"].strip():
                mismatched += 1  # recorded, not rejected: the index is authoritative
            placed[i] = c["name"]
    unplaced = [i for i in range(n) if i not in placed]
    for i in unplaced:
        placed[i] = ELIC.OTHER_CRITERION
    report = {
        "n_input": n,
        "n_placed": n - len(unplaced),
        "n_unplaced": len(unplaced),
        "n_duplicate_members": dup,
        "n_invented_indices": invented,
        "n_name_mismatches": mismatched,
    }
    return placed, report


def _medoid_levels(members: list[int], emb: np.ndarray, feats: list[dict]) -> dict:
    """The most-central member of a group — same rule as 06_pool_rubrics.py:64."""
    sub = emb[members]
    return feats[members[int(np.argmax((sub @ sub.T).sum(axis=1)))]]


def _assemble(run: int, picked: list[int], crit, feats, emb, usage) -> dict:
    placed, report = _validate(crit, feats)
    defs = {c["name"]: c.get("definition", "") for c in crit}
    groups: dict[str, list[int]] = {}
    for i, name in placed.items():
        groups.setdefault(name, []).append(i)

    features, other = [], None
    for name, members in sorted(groups.items()):
        med = _medoid_levels(sorted(members), emb, feats)
        entry = {
            "name": name,
            "description": defs.get(name, med.get("description", "")),
            "levels": med["levels"],
            "n_members": len(members),
            "n_runs": len({feats[i]["rep"] for i in members}),
            "medoid_name": med["name"],
        }
        if name == ELIC.OTHER_CRITERION:
            other = entry
        else:
            features.append(entry)
    return {
        "run": run,
        "picked_reps": picked,
        "n_pooled_features": len(feats),
        "n_features": len(features),
        "validation": report,
        "features": features,
        "other": other,
        "usage": usage,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["submit", "collect"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    j = cfg["judge"]
    root = Path(args.config).parent
    stab = root / "results" / "rubric_stability"
    cache_dir = root / j["cache_dir"]
    bd = root / "batch" / _OUT_SUB
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reps = _load_reps(stab)
    E = int(j.get("m3_elicitations", 27))
    R = int(j.get("taxo_pool_runs", 50))
    subsets = _subsets(len(reps), E, R, int(j.get("seed", 42)))
    specs = _specs(cfg, reps, subsets)
    client = get_client()

    if args.command == "collect":
        meta = json.loads((bd / "meta.json").read_text()) if (bd / "meta.json").exists() else {}
        if meta.get("batch_id"):
            batch = poll_until_done(
                client, meta["batch_id"], interval_s=60, meta_path=bd / "meta.json"
            )
            collect_into_cache(client, batch, {s.custom_id: s for s in specs}, cache_dir, bd)
    else:
        cached, todo = partition_cached(cache_dir, specs)
        print(
            f"[taxo_pool] {len(specs)} runs (E={E} of {len(reps)} elicitations) | "
            f"cached {len(cached)} | to submit {len(todo)}"
        )
        if todo:
            bd.mkdir(parents=True, exist_ok=True)
            build_jsonl(todo, bd / "requests.jsonl")
            submit_batch(client, bd / "requests.jsonl", CHAT_ENDPOINT, bd / "meta.json")
            print(
                "[taxo_pool] submitted a 24h batch — re-run `collect` when it resolves "
                "(re-run `submit` to resubmit any that fail)."
            )
            return 0

    # ---- assemble every run that came back -------------------------------- #
    out = stab / _OUT_SUB
    out.mkdir(parents=True, exist_ok=True)
    emb_cache = root / ".embed_cache"
    present, done, ledger = [], 0, []
    for spec, picked in zip(specs, subsets):
        rec = cache.get(cache_dir, spec.cache_key())
        if not (rec and (rec.get("text") or "").strip()):
            continue
        r = spec.meta["run"]
        feats = _pool(reps, picked)
        emb = embed_texts([_feature_key(f) for f in feats], cache_dir=emb_cache)
        try:
            crit = _parse(rec["text"])
        except (ValueError, KeyError) as e:
            print(f"[taxo_pool] run {r:03d}: unparseable output ({e}) — skipped")
            continue
        payload = _assemble(r, picked, crit, feats, emb, rec.get("usage"))
        (out / f"run_{r:03d}.json").write_text(json.dumps(payload, indent=2))
        present.append(rec)
        done += 1
        u = rec.get("usage") or {}
        ledger.append(
            {
                "run": r,
                "custom_id": spec.custom_id,
                "mode": rec.get("source", "batch"),
                "model": rec.get("model"),
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
                "n_features": payload["n_features"],
                "n_unplaced": payload["validation"]["n_unplaced"],
            }
        )

    if not done:
        print("[taxo_pool] nothing collected yet — run `collect` once the batch resolves")
        return 0

    usd_batch = actual_usd(present, cfg.get("pricing"), batch=True)
    usd_sync = actual_usd(present, cfg.get("pricing"), batch=False)
    modes = Counter(r["mode"] for r in ledger)
    ks = [r["n_features"] for r in ledger]
    (out / "cost_ledger.json").write_text(
        json.dumps(
            {
                "n_runs": done,
                "of_requested": R,
                "E": E,
                "modes": dict(modes),
                "model": j["reasoning_model"],
                "reasoning_effort": j["high_effort"],
                "usd_batch": usd_batch,
                "usd_sync": usd_sync,
                "usd_batch_per_run": usd_batch / done,
                "K_mean": float(np.mean(ks)),
                "K_min": int(min(ks)),
                "K_max": int(max(ks)),
                "calls": ledger,
            },
            indent=2,
        )
    )
    unplaced = sum(r["n_unplaced"] for r in ledger)
    print(
        f"[taxo_pool] {done}/{R} runs assembled | K mean {np.mean(ks):.1f} "
        f"[{min(ks)}, {max(ks)}] | modes {dict(modes)} | ${usd_batch:.2f} batch "
        f"(${usd_sync:.2f} sync)"
    )
    if unplaced:
        print(f"[taxo_pool] WARNING {unplaced} member slots unplaced across runs -> 'Other'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
