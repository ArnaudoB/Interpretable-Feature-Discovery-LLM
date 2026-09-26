"""ASAP 2.0 candidate-pool loader, for the four ``*__asap2`` runs.

Returns the same schema as :func:`dataset.ellipse.load_candidates`, minus the six
ELLIPSE trait columns (ASAP 2.0 has no trait breakdown -- it ships one resolved holistic
score per essay)::

    essay_id (str), full_text (str), prompt (str), overall (float),
    word_count (int64), band (int = round(overall))

``overall`` is the ASAP 2.0 holistic ``score`` (integer 1..6), so ``band == score`` and
``score_gap`` downstream spans 0..5 rather than ELLIPSE's 0..4.

Only ``essay_id``/``full_text`` are used downstream of ``essays.parquet``; ``prompt`` and
``band`` exist so ``select_balanced_essays`` can be reused unmodified. With a single-prompt
ASAP selection that selector degenerates to pure band balancing, which is the intent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_asap2_candidates(
    path: str,
    *,
    prompt_name: str | None = None,
    score_col: str = "score",
    prompt_col: str = "prompt_name",
) -> pd.DataFrame:
    """Read an ASAP 2.0 essay table (``.csv`` or ``.parquet``) -> candidate pool.

    Args:
        path: an ASAP 2.0 table (``.csv`` or ``.parquet``) carrying essay_id / full_text /
            score / prompt_name -- here always the 500-essay subset shipped in
            ``raw/corpus/asap2_driverless_selected_items.parquet``.
        prompt_name: keep only rows with this ``prompt_col`` value. ``None`` keeps all.
        score_col: holistic score column, used as ``overall`` and ``band``.
        prompt_col: writing-prompt column, used as ``prompt``.

    Returns:
        DataFrame sorted arbitrarily, deduplicated on ``essay_id``.
    """
    raw = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    required = ["essay_id", "full_text", score_col, prompt_col]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"ASAP2 source missing columns {missing}; got {list(raw.columns)}")
    raw = raw.dropna(subset=required).copy()
    if prompt_name is not None:
        raw = raw[raw[prompt_col].astype(str) == prompt_name]
        if raw.empty:
            raise ValueError(f"no rows with {prompt_col} == {prompt_name!r}")

    out = pd.DataFrame(
        {
            "essay_id": raw["essay_id"].astype(str).to_numpy(),
            "full_text": raw["full_text"].astype(str).to_numpy(),
            "prompt": raw[prompt_col].astype(str).to_numpy(),
            "overall": raw[score_col].astype(np.float64).to_numpy(),
        }
    )
    out["word_count"] = out["full_text"].str.split().str.len().astype(np.int64)
    out["band"] = out["overall"].round().astype(int)
    return out.drop_duplicates(subset="essay_id").reset_index(drop=True)
