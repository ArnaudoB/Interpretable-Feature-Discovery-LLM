"""Put the repository root on ``sys.path`` so ``core`` imports without installation.

Mirrors what ``experiments/*/scripts/_paths.py`` does for the experiment scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
