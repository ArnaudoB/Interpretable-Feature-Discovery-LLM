"""Resolve a run: ``config.yaml`` (shared) + its entry in ``runs.yaml`` → one config.

Every script takes ``--run <name>`` (or ``--run all``) and calls :func:`load_run`, which
returns a :class:`Run` whose ``cfg`` has the same sections the stages always read
(``cell``, ``source``, ``graph``, ``judge``, ``clustering``, ``model``, ``diagnostics``,
``validation``) and whose paths are absolute. No stage builds a path of its own, so the
directory layout is defined here and nowhere else:

    runs/<name>/results/            everything a run writes
    runs/<name>/results/graph/      step 00
    raw/cache/<name>/               its content-addressed response cache
    prompts/<corpus>.py             its frozen prompt module
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

import _paths  # noqa: F401  (sys.path bootstrap)

# Derived from __file__, not taken from ``_paths``: every experiment ships a module of that
# name, so in a full test run ``_paths`` may already be another experiment's, and then every
# path built here would silently point at it.
EXPERIMENT = Path(__file__).resolve().parents[1]
RUNS_FILE = EXPERIMENT / "runs.yaml"
CONFIG_FILE = EXPERIMENT / "config.yaml"
CORPORA = ("ellipse", "asap2")
JUDGES = ("gpt_5_4_mini", "claude_haiku_4_5", "gemini_3_6_flash", "deepseek_v4_flash")
#: Display names used in every table and figure, in this order.
JUDGE_LABEL = {
    "gpt_5_4_mini": "gpt-5.4-mini",
    "claude_haiku_4_5": "claude-haiku-4.5",
    "gemini_3_6_flash": "gemini-3.6-flash",
    "deepseek_v4_flash": "deepseek-v4-flash",
}
CORPUS_LABEL = {"ellipse": "ELLIPSE", "asap2": "ASAP-2"}  # dataset labels in results/calibration/


@dataclass(frozen=True)
class Run:
    name: str
    corpus: str
    judge: str
    cfg: dict
    results: Path
    graph: Path
    cache: Path
    prompts: Path
    source_path: Path


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def run_names() -> list[str]:
    return list(yaml.safe_load(RUNS_FILE.read_text()))


def run_name(judge: str, corpus: str) -> str:
    return f"{judge}__{corpus}"


def load_run(name: str) -> Run:
    runs = yaml.safe_load(RUNS_FILE.read_text())
    if name not in runs:
        raise SystemExit(f"unknown run {name!r}; known: {', '.join(runs)}")
    shared = yaml.safe_load(CONFIG_FILE.read_text())
    entry = runs[name]
    corpus = entry["corpus"]
    cdef = shared.pop("corpora")[corpus]

    cfg = _merge(shared, {"cell": cdef["cell"], "source": cdef["source"]})
    cfg = _merge(cfg, {k: v for k, v in entry.items() if k in ("cell", "judge")})
    cfg["cell"]["label"] = f"interpretable_scoring_{name}"
    cfg["run"] = {"name": name, "corpus": corpus}

    results = EXPERIMENT / "runs" / name / "results"
    return Run(
        name=name,
        corpus=corpus,
        judge=cfg["cell"]["judge"],
        cfg=cfg,
        results=results,
        graph=results / "graph",
        cache=EXPERIMENT / "raw" / "cache" / name,
        prompts=EXPERIMENT / cdef["prompts"],
        source_path=EXPERIMENT / cdef["source"]["path"],
    )


def resolve(arg: str) -> list[Run]:
    """``--run`` value → runs. ``all`` means all eight, in runs.yaml order."""
    names = run_names() if arg == "all" else [a for a in arg.split(",") if a]
    return [load_run(n) for n in names]


def load_prompts(run: Run):
    """Import the run's frozen prompt module (``prompts/<corpus>.py``) by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"prompts_{run.corpus}", run.prompts)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
