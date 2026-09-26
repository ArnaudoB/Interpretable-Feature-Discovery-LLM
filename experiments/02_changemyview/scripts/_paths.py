"""Import-path bootstrap. Imported for its side effect, before any local import.

Running ``python scripts/NN_something.py`` puts ``scripts/`` on ``sys.path`` (Python does
that for the script's own directory), which is how this module becomes importable. It then
adds the two directories the scripts import from:

* the repository root, for the shared library ``core`` (comparison graph, gated model,
  judging/caching, metrics) used by every experiment;
* this experiment's directory, for its ``prompts`` modules.

Nothing else in the tree manipulates ``sys.path``; a script that needs local imports starts
with ``import _paths  # noqa: F401`` and everything resolves from there. No installation step
and no ``PYTHONPATH`` are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]  # experiments/02_changemyview
ROOT = EXPERIMENT.parents[1]  # repository root

for _p in (str(EXPERIMENT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
