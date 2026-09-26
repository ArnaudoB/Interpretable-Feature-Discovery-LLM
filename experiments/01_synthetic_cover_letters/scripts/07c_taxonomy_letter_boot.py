"""Step 7c: one taxonomy per LETTER resample, for the paired Ours-vs-1a bootstrap.

    python scripts/07c_taxonomy_letter_boot.py submit  --config config.yaml
    python scripts/07c_taxonomy_letter_boot.py collect --config config.yaml

``08d`` letter-resamples the comparative arm but freezes the taxonomy, because the shipped
per-replicate groupings (``taxo_boot_groupings.json``) belong to the MATCHING resamples. Freezing
handicaps Ours: ``08b`` showed that letting discovery vary *raised* every metric. This script
removes that handicap for ~50 taxonomy calls and no new judging — the rationales of every
surviving pair are already in ``results/phrases.parquet``.

Per replicate b: take the letter resample from ``gated/m1_boot_resamples.json``, keep the pairs
whose BOTH endpoints were drawn (~1600 of 4000), aggregate their cited dimensions, trim to those
cited >= ``min_cite``, and send them through ``TAXONOMY_PROMPT`` (the comparative grouping prompt,
unchanged). ``min_cite`` is scaled from ``judge.taxonomy_min_cite`` by the surviving-pair fraction
so the trim is proportionally the same as the full-graph run.

Writes results/gated/letterboot_groupings.json, consumed by 08d (its default mode). Feeds
tab:paired (a) and fig:paired (a).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.judging import cache
from core.judging.batch import build_jsonl, collect_into_cache, poll_until_done, submit_batch
from core.judging.cache import partition_cached
from core.judging.pricing import actual_usd
from core.judging.requests import CHAT_ENDPOINT, build_continuation_spec
from core.openai_client import get_client
import prompts as ELIC

_SYS = "You organize evaluation dimensions into canonical criteria and output only JSON."
_OUT_SUB = "letterboot_taxo"


def _replicate_tables(root, cfg):
    """[(b, DataFrame of surviving cited dimensions, min_cite)] — one per letter resample."""
    res = root / "results"
    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    eidx = json.loads((res / "essay_index.json").read_text())
    n = len(eidx)
    resamp = json.loads((res / "gated/m1_boot_resamples.json").read_text())
    letter_ids, resamples = resamp["letter_ids"], resamp["resamples"]
    base_min = int(cfg["judge"].get("taxonomy_min_cite", 10))

    out = []
    for b, draw in enumerate(resamples):
        mult = np.bincount([eidx[letter_ids[i]] for i in draw], minlength=n)
        inset = mult > 0
        keep = inset[pairs[:, 0]] & inset[pairs[:, 1]]
        kept_pids = {str(i) for i in np.flatnonzero(keep)}
        sub = phrases[phrases["pair_id"].astype(str).isin(kept_pids)]
        g = (
            sub.groupby("phrase_norm")
            .agg(
                name=("phrase", "first"),
                description=("description", "first"),
                cnt=("phrase", "size"),
            )
            .reset_index()
            .sort_values("cnt", ascending=False)
            .reset_index(drop=True)
        )
        # scale the citation trim by the surviving-pair fraction
        min_cite = max(2, int(round(base_min * keep.sum() / len(pairs))))
        g = g[g["cnt"] >= min_cite].reset_index(drop=True)
        out.append((b, g, min_cite))
    return out


def _specs(cfg, tables):
    j = cfg["judge"]
    specs = []
    for b, g, _ in tables:
        listing = "\n".join(
            f"[{i}] {r.name} - {r.description} (cited {int(r.cnt)})"
            for i, r in enumerate(g.itertuples(index=False))
        )
        prompt = ELIC.TAXONOMY_PROMPT.replace("[[INDEXED_DIMENSION_LIST]]", listing)
        specs.append(
            build_continuation_spec(
                f"letterboot_taxo:b{b:03d}",
                j["reasoning_model"],
                _SYS,
                prompt,
                int(j["max_out_taxonomy"]),
                reasoning_effort=j["high_effort"],
                meta={"b": b},
            )
        )
    return specs


def _parse(text):
    txt = text.strip()
    if txt.startswith("```"):
        txt = txt[txt.find("{") : txt.rfind("}") + 1]
    return json.loads(txt)["criteria"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["submit", "collect"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    res = root / "results"
    cache_dir = root / cfg["judge"]["cache_dir"]
    bd = root / "batch" / _OUT_SUB
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tables = _replicate_tables(root, cfg)
    specs = _specs(cfg, tables)
    client = get_client()
    sizes = [len(g) for _, g, _ in tables]
    print(
        f"[letterboot-taxo] {len(specs)} replicates | dimensions/replicate "
        f"mean {np.mean(sizes):.0f} [{min(sizes)}, {max(sizes)}] | "
        f"min_cite {tables[0][2]}"
    )

    if args.command == "collect":
        meta = json.loads((bd / "meta.json").read_text()) if (bd / "meta.json").exists() else {}
        if meta.get("batch_id"):
            batch = poll_until_done(
                client, meta["batch_id"], interval_s=60, meta_path=bd / "meta.json"
            )
            collect_into_cache(client, batch, {s.custom_id: s for s in specs}, cache_dir, bd)
    else:
        cached, todo = partition_cached(cache_dir, specs)
        print(f"[letterboot-taxo] cached {len(cached)} | to submit {len(todo)}")
        if todo:
            bd.mkdir(parents=True, exist_ok=True)
            build_jsonl(todo, bd / "requests.jsonl")
            submit_batch(client, bd / "requests.jsonl", CHAT_ENDPOINT, bd / "meta.json")
            print(
                "[letterboot-taxo] submitted a 24h batch — re-run `collect` when it resolves "
                "(re-run `submit` to resubmit failures)."
            )
            return 0

    groupings, norms, unparseable, present = {}, {}, [], []
    for (b, g, _), spec in zip(tables, specs):
        rec = cache.get(cache_dir, spec.cache_key())
        if not (rec and (rec.get("text") or "").strip()):
            continue
        present.append(rec)
        try:
            groupings[b] = _parse(rec["text"])
            norms[b] = g["phrase_norm"].tolist()
        except (ValueError, KeyError):
            unparseable.append(b)

    if not groupings:
        print("[letterboot-taxo] nothing collected yet — run `collect` once the batch resolves")
        return 0

    usd = actual_usd(present, cfg.get("pricing"), batch=True)
    out = res / "gated" / "letterboot_groupings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "B": len(tables),
                "n_usable": len(groupings),
                "unparseable": unparseable,
                "source": "gpt-5.4 high-reasoning taxonomy, one call per LETTER resample "
                "(resamples shared with 1a via gated/m1_boot_resamples.json)",
                "usd_batch": usd,
                "norms": {str(k): v for k, v in norms.items()},
                "groupings": {str(k): v for k, v in groupings.items()},
            },
            indent=2,
        )
    )
    ks = [len(v) for v in groupings.values()]
    print(
        f"[letterboot-taxo] {len(groupings)}/{len(tables)} usable "
        f"(unparseable {unparseable}) | criteria mean {np.mean(ks):.1f} "
        f"[{min(ks)}, {max(ks)}] | ${usd:.2f} batch"
    )
    print(f"[letterboot-taxo] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
