"""Step 3: group the judge's raw dimension phrases into canonical criteria.

    python scripts/03_taxonomy.py --config config.yaml

Trims the cited dimensions to those named at least ``judge.taxonomy_min_cite`` times, then
sends them to a high-reasoning model (``judge.reasoning_model``) with TAXONOMY_PROMPT, which
groups phrasings of the SAME construct together (with an "Other" catch-all). Writes:

  * results/gated/llm_taxonomy/taxonomy.json  -- the criteria + members + usage
  * results/gated/llm_taxonomy/mapping.json   -- phrase_norm -> canonical criterion name

The mapping drives ``04_fit_comparative.py`` (raw phrase -> criterion column of the tensor).
Cached under ``cache/responses`` (shipped), so re-running does not re-bill. The 33 criteria
are the rows of tab:feats-ours.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.judging.batch import realtime_call
from core.judging.requests import build_continuation_spec
from core.openai_client import get_client
import prompts as ELIC


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    j = cfg["judge"]
    root = Path(args.config).parent
    res = root / "results"
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ph = pd.read_parquet(res / "phrases.parquet")
    g = (
        ph.groupby("phrase_norm")
        .agg(name=("phrase", "first"), description=("description", "first"), n=("phrase", "size"))
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    min_cite = int(j.get("taxonomy_min_cite", 1))
    total_cit = int(g["n"].sum())
    g = g[g["n"] >= min_cite].reset_index(drop=True)  # drop the idiosyncratic singleton tail
    norms = g["phrase_norm"].tolist()
    print(
        f"[taxonomy] grouping {len(norms)} dimensions cited >= {min_cite} "
        f"({100 * g['n'].sum() / total_cit:.1f}% of {total_cit} citations)"
    )

    listing = "\n".join(
        f"[{i}] {r.name} - {r.description} (cited {int(r.n)})"
        for i, r in enumerate(g.itertuples(index=False))
    )
    prompt = ELIC.TAXONOMY_PROMPT.replace("[[INDEXED_DIMENSION_LIST]]", listing)
    spec = build_continuation_spec(
        "taxonomy:build",
        j["reasoning_model"],
        "You organize evaluation dimensions into canonical criteria and output only JSON.",
        prompt,
        int(j["max_out_taxonomy"]),
        reasoning_effort=j["high_effort"],
    )
    client = get_client().with_options(timeout=float(j.get("taxonomy_timeout_s", 2400)))
    rec = realtime_call(client, spec)
    if not (rec.get("text") or "").strip():
        raise RuntimeError(
            f"taxonomy returned empty output (usage={rec.get('usage')}); "
            f"raise max_out_taxonomy or lower taxonomy_min_cite."
        )
    txt = rec["text"].strip()
    if txt.startswith("```"):
        txt = txt[txt.find("{") : txt.rfind("}") + 1]
    crit = json.loads(txt)["criteria"]

    mapping, placed = {}, set()
    for c in crit:
        for mem in c.get("members", []):
            i = int(mem["index"])
            if 0 <= i < len(norms):
                mapping[norms[i]] = c["name"]
                placed.add(i)
    for i, nm in enumerate(norms):  # leftovers -> own single-member criterion
        if i not in placed:
            mapping[nm] = g.iloc[i]["name"]

    out = res / "gated" / "llm_taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy.json").write_text(
        json.dumps({"criteria": crit, "usage": rec.get("usage")}, indent=2)
    )
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2))
    print(
        f"[taxonomy] {len(norms)} dimensions -> {len(set(mapping.values()))} criteria; "
        f"usage={rec.get('usage')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
