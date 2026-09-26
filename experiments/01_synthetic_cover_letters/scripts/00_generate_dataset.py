"""Step 0: generate + verify the synthetic cover-letter corpus.

    python scripts/00_generate_dataset.py --config config.yaml

Builds the corpus by PURE TEMPLATING + a deterministic, length-preserving fluency
mistake-injector (no LLM, runs in seconds). The corpus responds to the frozen
Google InfoSec-SWE job ad; every letter carries a known level for each engineered
dial (the answer key). Enforces six gates and exits nonzero if any fails:

  1. realized == assigned  -- detectors re-read every clean letter and match the key
  2. validate              -- every degraded letter passes templates.validate
  3. fluency-count         -- injected mistakes == the fluency dial's target count
  4. orthogonality         -- max non-structural |corr| between dials < gate (balanced)
  5. length-decorrelated   -- |corr(word_count, each primary dial)| < gate (balanced)
  6. balance               -- each primary dial's 3 levels are balanced (balanced set)

Writes ``results/{letters.parquet, answer_key.parquet}`` and
``reports/{verification.md, orthogonality_corr.csv}``. No API key, no network.
Feeds App. app:dataset (the corpus, tab:dials, the orthogonality and length gates).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from dataset import generate_dataset, run_gates
from dataset import templates, job_ad


def _build_report(meta: dict, res: dict, df: pd.DataFrame) -> str:
    g = res["gates"]
    d = res["details"]
    L: list[str] = []
    L.append("# Cover-letter corpus — dataset verification\n")
    L.append(
        f"Synthetic corpus: **N = {meta['n_letters']}** ({meta['n_balanced']} balanced + "
        f"{meta['n_anchors']} extreme anchors), seed {meta['seed']}. Pure templating + "
        f"deterministic fluency degradation, no LLM.\n"
    )
    status = "ALL GATES PASS" if res["all_ok"] else "GATE FAILURE"
    L.append(f"## Gates: {status}\n")
    L.append("| gate | result |")
    L.append("|---|---|")
    L.append(
        f"| 1. realized == assigned | {'pass' if g['realized_equals_assigned'] else 'FAIL'} ({len(d['mismatches'])} shown) |"
    )
    L.append(
        f"| 2. validate | {'pass' if g['validate'] else 'FAIL'} ({len(d['validate_failures'])} shown) |"
    )
    L.append(f"| 3. fluency-count | {'pass' if g['fluency_count'] else 'FAIL'} |")
    L.append(
        f"| 4. orthogonality | {'pass' if g['orthogonality'] else 'FAIL'} (max |corr| = {d['max_nonstructural_abs_corr']:.3f} < {d['corr_gate']}) |"
    )
    L.append(
        f"| 5. length-decorrelated | {'pass' if g['length_decorrelated'] else 'FAIL'} (max |corr| = {d['max_length_corr']:.3f} < {d['len_gate']}) |"
    )
    L.append(
        f"\nOver the full corpus (anchors included; reported, not gated): max non-structural "
        f"|corr| = {d['full_corpus_max_nonstructural_abs_corr']:.3f}, max length |corr| = "
        f"{d['full_corpus_max_length_corr']:.3f}.\n"
    )
    L.append(f"| 6. balance | {'pass' if g['balance'] else 'FAIL'} |")
    L.append("")
    L.append("## Dial balance (balanced subset)\n")
    L.append("| dial | level counts |")
    L.append("|---|---|")
    for c, counts in d["balance"].items():
        L.append(f"| {c} | {counts} |")
    L.append("")
    L.append("## Orthogonality — top non-structural pairs\n")
    L.append("| dial_a | dial_b | corr |")
    L.append("|---|---|---|")
    for a, b, c in d["top_corr_pairs"]:
        L.append(f"| {a} | {b} | {c:+.3f} |")
    L.append(
        "\n_Structural correlations (skill-vs-k, skill-vs-skill, niche-vs-niche) are "
        "excluded by design; full matrix in `orthogonality_corr.csv`._\n"
    )
    L.append("## Length vs dials (balanced subset)\n")
    L.append("| dial | corr(word_count, level) |")
    L.append("|---|---|")
    for c, v in d["length_corrs"].items():
        L.append(f"| {c} | {v:+.3f} |")
    L.append(
        f"\nWord-count range: {int(df['n_words'].min())}-{int(df['n_words'].max())} "
        f"(mean {df['n_words'].mean():.0f}).\n"
    )
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    print(
        f"[dataset] generating N={cfg['cell']['n_letters']} letters (seed={cfg['cell']['seed']})..."
    )
    df, meta = generate_dataset(templates, job_ad, cfg)
    res = run_gates(df, meta, templates, job_ad, cfg)

    # determinism check
    df2, _ = generate_dataset(templates, job_ad, cfg)
    determinism = df.drop(columns=["_profile"]).equals(df2.drop(columns=["_profile"]))
    res["gates"]["determinism"] = determinism
    res["all_ok"] = res["all_ok"] and determinism

    results_dir = cfg_path.parent / "results"
    reports_dir = cfg_path.parent / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    save = df.drop(columns=["_profile"])
    save[["letter_id", "template_id", "is_anchor", "anchor_type", "n_words", "text"]].to_parquet(
        results_dir / "letters.parquet", index=False
    )
    key_cols = [c for c in save.columns if c not in ("text",)]
    save[key_cols].to_parquet(results_dir / "answer_key.parquet", index=False)
    res["details"]["corr_df"].to_csv(reports_dir / "orthogonality_corr.csv")
    (reports_dir / "verification.md").write_text(_build_report(meta, res, df))
    d = res["details"]
    (results_dir / "dataset_checks.json").write_text(
        json.dumps(
            {
                "n_letters": int(len(df)),
                "n_anchors": int(df["is_anchor"].sum()),
                "balanced": {
                    "max_nonstructural_abs_corr": d["max_nonstructural_abs_corr"],
                    "max_length_corr": d["max_length_corr"],
                },
                "full_corpus": {
                    "max_nonstructural_abs_corr": d["full_corpus_max_nonstructural_abs_corr"],
                    "max_length_corr": d["full_corpus_max_length_corr"],
                },
                "gates": {k: bool(v) for k, v in res["gates"].items()},
            },
            indent=2,
        )
    )

    print(
        f"[dataset] wrote {results_dir / 'letters.parquet'} and answer_key.parquet ({len(df)} rows)"
    )
    print(f"[dataset] gates: {res['gates']}")
    print(
        f"[dataset] max non-structural |corr| = {res['details']['max_nonstructural_abs_corr']:.3f} "
        f"(gate < {res['details']['corr_gate']}); max length |corr| = "
        f"{res['details']['max_length_corr']:.3f}"
    )
    print(
        "[dataset] ALL GATES PASS" if res["all_ok"] else "[dataset] GATE FAILURE — fix and re-run."
    )
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
