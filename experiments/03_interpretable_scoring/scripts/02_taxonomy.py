"""Step 2: canonicalize one run's cited qualities into criteria.

    python scripts/02_taxonomy.py --run <run> --yes    # live: one GPT-5.4 call (+ repair)
    python scripts/02_taxonomy.py --run all --replay   # offline: re-derive and check

ONE GPT-5.4 high-reasoning call groups the judge's raw quality phrases into canonical
criteria by MEANING. There is deliberately NO embedding clustering anywhere in this
experiment — the taxonomy is the only canonicalization step (App. ``sec:extension-appendix``,
"Canonicalization and reliability gates"). Implementation:
  * the indexed-listing format ``[i] name (cited Nx): description`` with the MODAL
    description per normalized phrase;
  * ``call_llm`` in Responses-API BACKGROUND mode + poll (a synchronous call read-times-out
    on high reasoning), persisting the response id for recovery;
  * programmatic membership validation (every index placed exactly once, verbatim-name
    cross-check) with a REPAIR follow-up call for any leftovers.

Prompts come from the run's ``prompts/<corpus>.py``.

REPLAY. The call's verbatim output is shipped (``raw_response.json``). ``--replay`` rebuilds the phrase listing from ``phrases.parquet``
and asserts it equals the shipped ``dimension_list.txt`` (so the same prompt would be sent),
then re-runs the parse / validate / repair-merge below on the shipped raw text(s) and
asserts the result equals the shipped ``taxonomy.json`` criteria and ``mapping.json``.

Writes under ``runs/<run>/results/taxonomy/``:
  taxonomy.json        {model, effort, criteria:[{name, definition, members:[{index,name}]}]}
  mapping.json         flat {phrase_norm: canonical_criterion_name}   <- consumed downstream
  dimension_list.txt   the exact listing sent to the model
  raw_response.json    verbatim model output(s)
  usage.json           token counts + realized USD
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)

from _config import load_prompts, resolve  # noqa: E402
from core.judging.pricing import price_for  # noqa: E402
from core.openai_client import get_client, load_env  # noqa: E402

ELIC = None


def _client():
    """The repository's shared client; reads ``OPENAI_API_KEY`` from ``.env`` at the repository root."""
    load_env()
    return get_client()


def _extract_json(txt: str) -> str:
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1].lstrip("json").strip() if t.count("```") >= 2 else t
    i, j = t.find("{"), t.rfind("}")
    return t[i : j + 1] if i != -1 and j != -1 else t


def call_llm(
    client,
    model,
    system,
    user,
    max_tokens,
    effort,
    *,
    poll_s=10,
    max_wait_s=3600,
    id_log: Path | None = None,
):
    """Responses API in BACKGROUND mode + poll (see module docstring)."""
    body = dict(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_output_tokens=int(max_tokens),
        background=True,
    )
    if effort:
        body["reasoning"] = {"effort": effort}
    resp = client.responses.create(**body)
    rid = getattr(resp, "id", None)
    print(f"  [background response id: {rid}]", flush=True)
    if id_log is not None:
        id_log.write_text(str(rid))
    waited = 0
    while getattr(resp, "status", None) in ("queued", "in_progress"):
        time.sleep(poll_s)
        waited += poll_s
        if waited > max_wait_s:
            raise RuntimeError(
                f"taxonomy call exceeded {max_wait_s}s (status={resp.status}, id={rid})"
            )
        resp = client.responses.retrieve(resp.id)
    status = getattr(resp, "status", None)
    txt = (resp.output_text or "").strip()
    u = resp.usage
    it = int(getattr(u, "input_tokens", 0))
    ot = int(getattr(u, "output_tokens", 0))
    if status != "completed" or not txt:
        raise RuntimeError(
            f"taxonomy call status={status} out_tokens={ot} (id={rid}) — "
            f"try lower reasoning or a higher max-output-tokens"
        )
    print(f"  [call: status={status} out_tokens={ot} waited={waited}s]", flush=True)
    return txt, (it, ot)


def build_dimensions(phrases: pd.DataFrame, min_cite: int):
    """-> (entries, dropped_frac). entries = [(index, name, citations, modal_description)]."""
    cnt = phrases["phrase_norm"].value_counts()
    keep = cnt[cnt >= min_cite]
    desc_by_norm: dict[str, Counter] = defaultdict(Counter)
    for r in phrases.itertuples(index=False):
        if r.phrase_norm in keep.index and isinstance(r.description, str) and r.description:
            desc_by_norm[r.phrase_norm][r.description] += 1
    entries = []
    for i, norm in enumerate(sorted(keep.index)):
        modal = desc_by_norm[norm].most_common(1)
        entries.append((i, norm, int(keep[norm]), modal[0][0] if modal else norm))
    retained = float(keep.sum() / cnt.sum())
    return entries, retained


def _placement(criteria, name_at):
    placed: dict[int, str] = {}
    dups, bad_names = [], []
    for g in criteria:
        for m in g.get("members", []):
            idx = int(m["index"])
            if idx in placed:
                dups.append(idx)
            placed[idx] = g["name"]
            if name_at.get(idx) != m.get("name"):
                bad_names.append(idx)
    return placed, dups, bad_names


def _merge_repair(criteria, rtxt):
    """Fold a repair response into ``criteria`` in place: extend same-named groups."""
    by_name = {g["name"]: g for g in criteria}
    for g in json.loads(_extract_json(rtxt))["criteria"]:
        if g["name"] in by_name:
            by_name[g["name"]]["members"].extend(g.get("members", []))
        else:
            criteria.append(g)
            by_name[g["name"]] = g


def derive(raw_texts, entries):
    """Raw model output(s) -> (criteria, mapping). The one derivation, live and replay."""
    name_at = {i: nm for i, nm, _, _ in entries}
    criteria = json.loads(_extract_json(raw_texts[0]))["criteria"]
    for rtxt in raw_texts[1:]:
        _merge_repair(criteria, rtxt)
    placed, _, _ = _placement(criteria, name_at)
    mapping = {name_at[i]: c for i, c in placed.items() if i in name_at}
    return criteria, mapping


def replay(run) -> bool:
    global ELIC
    ELIC = load_prompts(run)
    jc = run.cfg["judge"]
    out = run.results / "taxonomy"
    phrases = pd.read_parquet(run.results / "phrases.parquet")
    entries, retained = build_dimensions(phrases, int(jc["taxonomy_min_cite"]))
    listing_ok = ELIC.render_dimension_list(entries) == (out / "dimension_list.txt").read_text()
    raw = json.loads((out / "raw_response.json").read_text())
    criteria, mapping = derive(raw, entries)
    shipped = json.loads((out / "taxonomy.json").read_text())
    tax_ok = criteria == shipped["criteria"] and shipped["n_phrases"] == len(entries)
    map_ok = mapping == json.loads((out / "mapping.json").read_text())
    usage = json.loads((out / "usage.json").read_text())
    ret_ok = abs(usage["citation_share_retained"] - retained) < 1e-12
    ok = listing_ok and tax_ok and map_ok and ret_ok
    print(
        f"{run.name:<28} {len(entries)} phrases, {len(raw)} call(s), "
        f"{len(set(mapping.values()))} criteria  listing={listing_ok} taxonomy={tax_ok} "
        f"mapping={map_ok} retained={ret_ok}  {'OK' if ok else 'MISMATCH'}"
    )
    return ok


def main() -> int:
    global ELIC
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run name; --replay also takes 'all'")
    ap.add_argument("--yes", action="store_true", help="confirm spending")
    ap.add_argument(
        "--replay",
        action="store_true",
        help="offline: re-derive from the shipped raw response and check",
    )
    ap.add_argument("--min-cite", type=int, default=None)
    args = ap.parse_args()

    runs = resolve(args.run)
    if args.replay:
        return 0 if all([replay(r) for r in runs]) else 1
    if len(runs) != 1:
        ap.error("a live taxonomy call spends money: name exactly one run")
    run = runs[0]
    cfg = run.cfg
    ELIC = load_prompts(run)

    jc = cfg["judge"]
    res = run.results
    out = res / "taxonomy"
    out.mkdir(parents=True, exist_ok=True)

    phrases = pd.read_parquet(res / "phrases.parquet")
    min_cite = int(args.min_cite if args.min_cite is not None else jc["taxonomy_min_cite"])
    entries, retained = build_dimensions(phrases, min_cite)

    model = jc["reasoning_model"]
    effort = jc.get("high_effort", "high")
    max_out = int(jc.get("max_out_taxonomy", 24000))
    user = ELIC.taxonomy_prompt(entries)
    (out / "dimension_list.txt").write_text(ELIC.render_dimension_list(entries))

    pin, pout = price_for(model)
    approx_in = len(ELIC.TAXONOMY_SYSTEM + user) / 4.0
    proj = (approx_in * pin + max_out * pout) / 1e6  # worst case: full output budget spent
    print(
        f"taxonomy input : {len(entries)} phrases (min_cite={min_cite}), "
        f"{100 * retained:.1f}% of citations retained"
    )
    print(f"model          : {model} (effort={effort}, max_output_tokens={max_out})")
    print(f"projected cost : <= ${proj:.2f} (worst case, full output budget)")
    if not args.yes:
        print("ABORT: pass --yes to spend.", file=sys.stderr)
        return 3

    client = _client()
    print("\ncalling taxonomy (background mode + poll) ...")
    txt, (it, ot) = call_llm(
        client,
        model,
        ELIC.TAXONOMY_SYSTEM,
        user,
        max_out,
        effort,
        max_wait_s=int(jc.get("taxonomy_timeout_s", 3600)),
        id_log=out / "pending_response.txt",
    )
    raw = [txt]
    criteria = json.loads(_extract_json(txt))["criteria"]

    # --- validate membership -----------------------------------------------------
    name_at = {i: nm for i, nm, _, _ in entries}
    placed, dups, bad_names = _placement(criteria, name_at)
    missing = sorted(set(name_at) - set(placed))
    print(
        f"\nplaced {len(placed)}/{len(name_at)} | groups {len(criteria)} | "
        f"missing {len(missing)} | dups {len(dups)} | name-mismatches {len(bad_names)}"
    )

    if missing:
        print("repair call for leftovers ...")
        left = [(i, name_at[i], c, d) for i, nm, c, d in entries if i in set(missing)]
        rtxt, (it2, ot2) = call_llm(
            client,
            model,
            ELIC.TAXONOMY_SYSTEM,
            ELIC.taxonomy_repair_prompt(criteria, left),
            max_out,
            effort,
            max_wait_s=int(jc.get("taxonomy_timeout_s", 3600)),
        )
        raw.append(rtxt)
        it += it2
        ot += ot2
        _merge_repair(criteria, rtxt)
        placed = {int(m["index"]): g["name"] for g in criteria for m in g.get("members", [])}
        missing = sorted(set(name_at) - set(placed))
        print(f"after repair: placed {len(placed)}/{len(name_at)} | missing {len(missing)}")

    # --- write --------------------------------------------------------------------
    mapping = {name_at[i]: c for i, c in placed.items() if i in name_at}
    usd = (it * pin + ot * pout) / 1e6
    (out / "taxonomy.json").write_text(
        json.dumps(
            {
                "model": model,
                "reasoning_effort": effort,
                "min_cite": min_cite,
                "n_phrases": len(entries),
                "criteria": criteria,
            },
            indent=2,
        )
    )
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True))
    (out / "raw_response.json").write_text(json.dumps(raw, indent=2))
    (out / "usage.json").write_text(
        json.dumps(
            {
                "model": model,
                "reasoning_effort": effort,
                "input_tokens": it,
                "output_tokens": ot,
                "realized_usd_sync": usd,
                "n_phrases_in": len(entries),
                "n_criteria": len(criteria),
                "citation_share_retained": retained,
            },
            indent=2,
        )
    )

    sizes = Counter(placed.values())
    print(f"\ncriteria ({len(criteria)}):")
    for nm, k in sizes.most_common():
        d = next((g.get("definition", "") for g in criteria if g["name"] == nm), "")
        print(f"  {k:>3} members  {nm}: {d[:96]}")
    print(f"\nrealized cost: ${usd:.2f}  ({it} in / {ot} out)")
    print(f"wrote → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
