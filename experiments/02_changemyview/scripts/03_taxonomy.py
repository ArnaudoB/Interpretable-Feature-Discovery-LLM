"""Step 3: group the judge's cited dimensions into canonical criteria (one gpt-5.4 call).

    python scripts/03_taxonomy.py rederive --out-dir DIR       # offline (see below)
    python scripts/03_taxonomy.py collect --out-dir DIR        # replay from the cache
    python scripts/03_taxonomy.py run --in-place --yes         # the (billed, realtime) call

Takes step 2's cited dimensions, keeps those cited at least ``judge.taxonomy_min_cite`` (10)
times -- the idiosyncratic singleton tail is dropped -- and lists each once, most-cited first,
with its first-seen phrasing and description. gpt-5.4 (high reasoning) groups the phrasings of
one construct into a criterion (prompt: ``prompts/discovery.py`` TAXONOMY_PROMPT), with an
"Other" catch-all. A listed dimension the model leaves unplaced becomes its own criterion.

This is the single-call path: the discovery cell sets no ``judge.taxonomy_chunk_size``, so
no batched map -> reduce taxonomy is involved.

**The run's response was not cached.** The call was made with ``realtime_call`` and only the
parsed result was written, so ``collect`` misses on the shipped cell (the request itself is
rebuilt exactly: 664 dimensions, the count the run records). ``rederive`` is the offline route
instead: it takes the criteria the run saved in ``taxonomy.json`` as the response and re-derives
``mapping.json`` from them and the rebuilt dimension table, which checks everything downstream
of the call.

Writes <discovery cell>/results/gated/llm_taxonomy/{taxonomy.json, mapping.json}.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

import _paths  # noqa: F401
import _config as C
import _judge as J
from core.cmv.taxonomy import render_dimension_listing
from core.judging.batch import realtime_call
from core.judging.requests import build_continuation_spec

CELL = "discovery"
SYSTEM = "You organize evaluation dimensions into canonical criteria and output only JSON."


def dimension_table(phrases: pd.DataFrame, min_cite: int) -> pd.DataFrame:
    """Unique dimensions cited >= ``min_cite`` times, most-cited first.

    Sorted on the count alone (pandas' default, unstable quicksort, over the groupby's
    alphabetical order), with no tie-break, as the run sorted. The listing order is part of
    the prompt, so only this ordering reproduces the request the run sent.
    """
    g = (
        phrases.groupby("phrase_norm")
        .agg(name=("phrase", "first"), description=("description", "first"), n=("phrase", "size"))
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    return g[g["n"] >= min_cite].reset_index(drop=True)


def build(out_dir=None):
    """``(spec, dimension table)`` for the one taxonomy call."""
    cfg = C.cell_config(CELL)
    j = cfg["judge"]
    phrases = pd.read_parquet(C.input_path(C.cell(CELL) / "results" / "phrases.parquet", out_dir))
    g = dimension_table(phrases, int(j.get("taxonomy_min_cite", 1)))
    user = C.prompt("discovery").TAXONOMY_PROMPT.replace(
        "[[INDEXED_DIMENSION_LIST]]", render_dimension_listing(g)
    )
    spec = build_continuation_spec(
        "taxonomy:build",
        j["reasoning_model"],
        SYSTEM,
        user,
        int(j["max_out_taxonomy"]),
        reasoning_effort=j["high_effort"],
    )
    total = len(phrases)
    print(
        f"[taxonomy] {len(g)} dimensions cited >= {j.get('taxonomy_min_cite', 1)} "
        f"({100 * g['n'].sum() / total:.1f}% of {total} citations)"
    )
    return spec, g


def assemble(rec, g, out):
    txt = (rec.get("text") or "").strip()
    if not txt:
        raise SystemExit(f"taxonomy response is empty (usage={rec.get('usage')})")
    if txt.startswith("```"):
        txt = txt[txt.find("{") : txt.rfind("}") + 1]
    crit = json.loads(txt)["criteria"]
    norms = g["phrase_norm"].tolist()
    mapping, placed = {}, set()
    for c in crit:
        for mem in c.get("members", []):
            i = int(mem["index"])
            if 0 <= i < len(norms):
                mapping[norms[i]] = c["name"]
                placed.add(i)
    for i, nm in enumerate(norms):  # leftovers -> their own single-member criterion
        if i not in placed:
            mapping[nm] = g.iloc[i]["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy.json").write_text(
        json.dumps({"criteria": crit, "usage": rec.get("usage")}, indent=2)
    )
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2))
    print(
        f"[taxonomy] {len(norms)} dimensions -> {len(crit)} criteria listed, "
        f"{len(norms) - len(placed)} unplaced -> {len(set(mapping.values()))} in the mapping "
        f"-> {out}"
    )


def main() -> int:
    ap = J.add_stage_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        stages=("run", "collect", "rederive"),
    )
    a = ap.parse_args()
    J.guard(a.stage, a.yes)
    C.check_writable(a.out_dir, a.in_place)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec, g = build(a.out_dir)
    cache_dir = C.cell_cache(CELL)
    out = C.work_path(C.cell(CELL) / "results" / "gated" / "llm_taxonomy", a.out_dir)
    if a.stage == "run":
        from core.judging import cache

        rec = cache.get(cache_dir, spec.cache_key())
        if rec is None or not (rec.get("text") or "").strip():
            j = C.cell_config(CELL)["judge"]
            cl = J.client().with_options(timeout=float(j.get("taxonomy_timeout_s", 2400)))
            rec = realtime_call(cl, spec)
            if (rec.get("text") or "").strip():
                cache.put(cache_dir, spec.cache_key(), rec)
    elif a.stage == "rederive":
        shipped = json.loads(
            C.input_path(
                C.cell(CELL) / "results" / "gated" / "llm_taxonomy" / "taxonomy.json", a.out_dir
            ).read_text()
        )
        print(
            f"[taxonomy] rederive: response taken from the saved taxonomy.json "
            f"({len(shipped['criteria'])} criteria), not from the cache"
        )
        rec = {"text": json.dumps({"criteria": shipped["criteria"]}), "usage": shipped.get("usage")}
    else:
        rec = J.replay([spec], cache_dir, label="taxonomy")[spec.custom_id]
    assemble(rec, g, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
