"""Shared OpenAI client and ``.env`` loader for the pipeline scripts and embeddings.

API keys are read from the environment; a ``.env`` file at the repository root (next to
``pyproject.toml``) fills in any key not already set. Reproducing the paper's tables and
figures from the shipped ``results/`` needs no key; only (re-)running the LLM pipeline or
embedding new text calls out to the API.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The repository-root ``.env`` (``<repo>/.env``), independent of the working directory.
DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: str | Path | None = None) -> None:
    """Populate ``os.environ`` from a simple ``KEY=value`` ``.env`` file if present.

    ``path`` defaults to the repository-root ``.env`` (:data:`DEFAULT_ENV`); an explicit
    path is used as given. Variables already set in the environment take precedence and
    are never overwritten. A missing file is a no-op.
    """
    p = Path(path) if path is not None else DEFAULT_ENV
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_client():
    """An ``openai.OpenAI`` client authenticated from ``OPENAI_API_KEY``.

    The key comes from the environment, falling back to the repository-root ``.env``.
    Raises ``KeyError`` if it is set in neither.
    """
    load_env()
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
