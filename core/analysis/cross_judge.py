"""Consolidation of criteria discovered independently by several judges, by MEANING.

Each judge's criteria come out of its own canonicalization step, so a construct two judges
both found appears twice under different names. This module groups them across judges the
same way the per-judge taxonomy groups phrases: a single high-reasoning call receives every
judge's criteria with their definitions, and its grouping is then checked programmatically.

The prompt text itself is a frozen paper artifact and lives with the experiment that uses it
(``experiments/03_interpretable_scoring/prompts/cross_judge.py``); this module only renders
it and validates the answer.

  * :func:`render_criterion_list` — the ``[i] (judge) name: definition`` listing.
  * :func:`cluster_prompt` — substitute the corpus framing and the listing into a template.
  * :func:`validate_clusters` — every index placed exactly once, no invented index, verbatim
    judge and name at each index, at most one criterion per judge per cluster.
  * :func:`extract_json` — the JSON object inside a model response (tolerates code fences).
"""

from __future__ import annotations

__all__ = ["render_criterion_list", "cluster_prompt", "validate_clusters", "extract_json"]


def render_criterion_list(entries) -> str:
    """Format ``(index, judge, name, definition)`` rows for the prompt."""
    return "\n".join(
        f"[{i}] ({judge}) {name}: {definition}" for i, judge, name, definition in entries
    )


def cluster_prompt(template: str, entries, corpus_framing: str) -> str:
    """Fill ``[[CORPUS_FRAMING]]`` and ``[[INDEXED_CRITERION_LIST]]`` in ``template``."""
    return template.replace("[[CORPUS_FRAMING]]", corpus_framing).replace(
        "[[INDEXED_CRITERION_LIST]]", render_criterion_list(entries)
    )


def validate_clusters(clusters, entries) -> dict:
    """Check the model's grouping against the input. Raises ValueError on any violation.

    Verifies exactly what the prompt promises is checked: every index placed once, no
    invented indices, verbatim judge/name at each index, and at most one criterion per
    judge per cluster.
    """
    by_index = {i: (judge, name) for i, judge, name, _ in entries}
    seen: dict[int, int] = {}
    problems: list[str] = []

    for ci, c in enumerate(clusters):
        judges_here: list[str] = []
        for m in c.get("members", []):
            idx = m.get("index")
            if idx not in by_index:
                problems.append(f"cluster {ci}: invented index {idx!r}")
                continue
            if idx in seen:
                problems.append(f"index {idx} appears in clusters {seen[idx]} and {ci}")
                continue
            seen[idx] = ci
            exp_judge, exp_name = by_index[idx]
            if m.get("name") != exp_name:
                problems.append(f"index {idx}: name {m.get('name')!r} != {exp_name!r}")
            if m.get("judge") != exp_judge:
                problems.append(f"index {idx}: judge {m.get('judge')!r} != {exp_judge!r}")
            judges_here.append(exp_judge)
        dupes = {j for j in judges_here if judges_here.count(j) > 1}
        if dupes:
            problems.append(f"cluster {ci} ({c.get('name')!r}) has two criteria from {dupes}")

    missing = sorted(set(by_index) - set(seen))
    if missing:
        problems.append(f"{len(missing)} index(es) never placed: {missing}")
    if problems:
        raise ValueError("cluster validation failed:\n  " + "\n  ".join(problems))

    sizes: dict[int, int] = {}
    for c in clusters:
        sizes[len(c["members"])] = sizes.get(len(c["members"]), 0) + 1
    return {"n_clusters": len(clusters), "n_placed": len(seen), "clusters_by_size": sizes}


def extract_json(txt: str) -> str:
    """The outermost ``{...}`` of a model response, after stripping a code fence if any."""
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1].lstrip("json").strip() if t.count("```") >= 2 else t
    i, j = t.find("{"), t.rfind("}")
    return t[i : j + 1] if i != -1 and j != -1 else t
