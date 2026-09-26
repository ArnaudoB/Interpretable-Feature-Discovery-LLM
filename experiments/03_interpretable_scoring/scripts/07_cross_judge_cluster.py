"""Step 7: consolidate the four judges' reliable criteria into shared constructs, by MEANING.

    python scripts/07_cross_judge_cluster.py --corpus all              # offline replay
    python scripts/07_cross_judge_cluster.py --corpus asap2 --live --yes   # re-issue the call

One GPT-5.4 (high reasoning) call per corpus sees every judge's reliable criteria with their
definitions and groups them, at most one criterion per judge per construct; the grouping is
then checked programmatically (``core.analysis.cross_judge.validate_clusters``). The prompt
is ``prompts/cross_judge.py`` (paper Prompt ``prm:cross-judge``). No statistics are sent.

REPLAY (default, no API call). The exact request and the verbatim model output of both
paper calls are shipped in ``raw/cross_judge/<corpus>_{input,raw_response}.json`` (see that
directory's README). Replay

  1. rebuilds the criterion list from each run's ``pipeline_reliable/`` + ``taxonomy/`` and
     re-renders the prompt, asserting it is byte-identical to the saved request -- so the
     same call would be sent again;
  2. parses the saved output and re-validates it (every criterion placed exactly once,
     verbatim names, one criterion per judge per construct);
  3. writes the clusters, which the radar figures (``fig:cross_judge_radar``,
     ``fig:radar_ellipse``, ``fig:radar_asap``) read.

LIVE (``--live --yes``) sends the call, saving the request before it and the raw output
before parsing -- a response that fails validation is the one worth inspecting.

Writes ``results/cross_judge/<corpus>_{clusters.json,criterion_list.txt}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import (  # noqa: E402
    CORPORA,
    EXPERIMENT,
    JUDGE_LABEL,
    JUDGES,
    load_run,
    run_name,
)
from core.analysis.cross_judge import (  # noqa: E402
    cluster_prompt,
    extract_json,
    render_criterion_list,
    validate_clusters,
)
from core.judging.pricing import price_for  # noqa: E402
from core.openai_client import load_env  # noqa: E402

sys.path.insert(0, str(EXPERIMENT / "prompts"))
import cross_judge as P  # noqa: E402

RAW = EXPERIMENT / "raw" / "cross_judge"
OUT = EXPERIMENT / "results" / "cross_judge"


def collect_entries(corpus: str):
    """(index, judge label, criterion name, definition) over the four reliable panels."""
    entries = []
    for judge in JUDGES:
        res = load_run(run_name(judge, corpus)).results
        names = pd.read_parquet(res / "pipeline_reliable/active_criteria.parquet")["canonical_name"]
        defs = {
            c["name"]: c["definition"]
            for c in json.loads((res / "taxonomy/taxonomy.json").read_text())["criteria"]
        }
        for n in names:
            entries.append((len(entries), JUDGE_LABEL[judge], n, defs[n]))
    return entries


def _call(user: str, max_out: int, corpus: str):
    """Responses API in background mode + poll; persists request, then raw output."""
    load_env()
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    RAW.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": P.CLUSTER_SYSTEM}, {"role": "user", "content": user}]
    (RAW / f"{corpus}_input.json").write_text(
        json.dumps(
            {
                "model": P.MODEL,
                "reasoning_effort": P.EFFORT,
                "max_output_tokens": max_out,
                "messages": messages,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    resp = client.responses.create(
        model=P.MODEL,
        input=messages,
        max_output_tokens=max_out,
        background=True,
        reasoning={"effort": P.EFFORT},
    )
    rid = resp.id
    print(f"  [background response id: {rid}]", flush=True)
    while resp.status in ("queued", "in_progress"):
        time.sleep(10)
        resp = client.responses.retrieve(rid)
    u = resp.usage
    raw = {
        "response_id": rid,
        "model": P.MODEL,
        "status": resp.status,
        "usage": {"input_tokens": int(u.input_tokens), "output_tokens": int(u.output_tokens)},
        "output_text": (resp.output_text or "").strip(),
    }
    (RAW / f"{corpus}_raw_response.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    if resp.status != "completed" or not raw["output_text"]:
        raise RuntimeError(f"status={resp.status} (id={rid}); try a higher --max-out")
    return raw


def run_corpus(corpus: str, args) -> bool:
    entries = collect_entries(corpus)
    user = cluster_prompt(P.CLUSTER_PROMPT, entries, P.CORPUS_FRAMING[corpus])
    per_judge = {JUDGE_LABEL[j]: sum(1 for e in entries if e[1] == JUDGE_LABEL[j]) for j in JUDGES}
    print(f"\n=== {corpus}: {len(entries)} reliable criteria {per_judge}")

    if args.live:
        pin, pout = price_for(P.MODEL)
        print(f"projected cost <= ${(len(user) / 4 * pin + args.max_out * pout) / 1e6:.2f}")
        if not args.yes:
            print("ABORT: pass --yes to spend.", file=sys.stderr)
            return False
        raw = _call(user, args.max_out, corpus)
        prompt_ok = True
    else:
        sent = json.loads((RAW / f"{corpus}_input.json").read_text())["messages"]
        raw = json.loads((RAW / f"{corpus}_raw_response.json").read_text())
        prompt_ok = sent[0]["content"] == P.CLUSTER_SYSTEM and sent[1]["content"] == user
        print(f"prompt re-renders byte-identically to the saved request: {prompt_ok}")
        if raw["model"] != P.MODEL:
            raise SystemExit(f"saved call used {raw['model']}, prompts/ says {P.MODEL}")

    clusters = json.loads(extract_json(raw["output_text"]))["clusters"]
    stats = validate_clusters(clusters, entries)
    print(
        f"validated: {stats['n_placed']}/{len(entries)} placed, {stats['n_clusters']} "
        f"constructs, by size {stats['clusters_by_size']}"
    )

    pin, pout = price_for(P.MODEL)
    it, ot = raw["usage"]["input_tokens"], raw["usage"]["output_tokens"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{corpus}_criterion_list.txt").write_text(render_criterion_list(entries))
    (OUT / f"{corpus}_clusters.json").write_text(
        json.dumps(
            {
                "model": P.MODEL,
                "reasoning_effort": P.EFFORT,
                "variant": "reliable",
                "corpus": corpus,
                "response_id": raw["response_id"],
                "judges": [JUDGE_LABEL[j] for j in JUDGES],
                "n_criteria": len(entries),
                "input_tokens": it,
                "output_tokens": ot,
                "realized_usd": (it * pin + ot * pout) / 1e6,
                "validation": stats,
                "clusters": clusters,
            },
            indent=2,
        )
    )
    for c in sorted(clusters, key=lambda c: (-len(c["members"]), c["name"])):
        print(f"  [{len(c['members'])}] {c['name']}")
    return prompt_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", choices=[*CORPORA, "all"], default="all")
    ap.add_argument("--live", action="store_true", help="re-issue the call (spends money)")
    ap.add_argument("--max-out", type=int, default=P.MAX_OUT)
    ap.add_argument("--yes", action="store_true", help="confirm spending")
    args = ap.parse_args()
    corpora = CORPORA if args.corpus == "all" else (args.corpus,)
    if args.live and len(corpora) != 1:
        ap.error("--live spends money: name exactly one corpus")
    return 0 if all([run_corpus(c, args) for c in corpora]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
