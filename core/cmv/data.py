"""Load the ConvoKit Winning-Args corpus and build the paired-unit table.

Each of the 4,263 matched pairs links, for a single OP thread, a *positive* unit
(the reply/path that won a delta, ``success==1``) and a *negative* unit
(``success==0``). We expose three text conditions per unit (Tan et al. 2016):

- ``root_reply``     — the root-level reply text alone (direct reply to the OP).
- ``full_path``      — the *challenger's* whole argument thread: all same-side
                       utterances authored by the root-reply author, concatenated in
                       timestamp order. The OP's own interjections on that side are
                       excluded — folding them in inflates length (72% vs the paper's
                       ~66% for ``#words``) and pollutes the interplay signal.
- ``root_truncated`` — the root reply truncated to the pair's *minimum* root word
                       count, so ``#words`` becomes uninformative by construction.

``op_text`` is the OP's title + self-text (the view is stated in the title), which
is the ``O`` used by the interplay features.

The corpus (~350 MB) is cached once, outside the repo, in the standard ConvoKit
location. Derived tables are cached as parquet under ``derived_dir``.
"""

from __future__ import annotations

import html
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

CORPUS_DIR = Path.home() / ".convokit" / "downloads" / "winning-args-corpus"
CORPUS_URL = (
    "https://zissou.infosci.cornell.edu/convokit/datasets/"
    "winning-args-corpus/winning-args-corpus.zip"
)
#: The derived tables ship with the experiment, so nothing here touches ConvoKit on a normal
#: run: ``build_pair_units`` returns the cached ``pair_units.parquet`` and never downloads the
#: ~350 MB corpus. ``ensure_corpus`` is kept for a genuine from-scratch rebuild.
DEFAULT_DERIVED = (
    Path(__file__).resolve().parents[2] / "experiments" / "02_changemyview" / "raw" / "data"
)


def ensure_corpus(corpus_dir: Path = CORPUS_DIR) -> Path:
    """Download + unzip the corpus into ``corpus_dir`` if not already present."""
    corpus_dir = Path(corpus_dir)
    if (corpus_dir / "utterances.jsonl").exists():
        return corpus_dir
    corpus_dir.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CORPUS_URL, timeout=180) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as z:
        for member in z.namelist():
            if member.startswith("winning-args-corpus/") and not member.startswith("__MACOSX"):
                z.extract(member, corpus_dir.parent)
    return corpus_dir


def _clean(text) -> str:
    """Unescape Reddit HTML entities (``&gt;`` etc.); ``None`` -> ``''`` (deleted)."""
    return html.unescape(text) if isinstance(text, str) else ""


def _truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def build_pair_units(
    corpus_dir: Path = CORPUS_DIR,
    derived_dir: Path = DEFAULT_DERIVED,
    force: bool = False,
) -> pd.DataFrame:
    """Return one row per matched pair with the three text conditions per side.

    Columns: ``pair_id, root_id, op_author, train, op_text``;
    ``pos_root, neg_root, pos_path, neg_path, pos_root_trunc, neg_root_trunc``;
    plus ``*_wc`` word counts. ``op_author`` falls back to ``root_id`` when the OP
    account is missing (so it is a usable CV grouping key).
    """
    corpus_dir = Path(corpus_dir)
    derived_dir = Path(derived_dir)
    cache = derived_dir / "pair_units.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    ensure_corpus(corpus_dir)
    conv = json.load(open(corpus_dir / "conversations.json"))

    # pair_id -> side(0/1) -> {'root', 'root_author', 'utts': [(ts, author, text)], 'root_id'}
    acc: dict = defaultdict(lambda: {0: None, 1: None})
    with open(corpus_dir / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line)
            m = u["meta"]
            s = m.get("success")
            if s not in (0, 1) or not m.get("pair_ids"):
                continue
            rt = u.get("reply-to")
            is_root = rt is not None and rt.startswith("t3_")
            text = _clean(u.get("text"))
            ts = u.get("timestamp") or 0
            for pid in m["pair_ids"]:
                side = acc[pid][s]
                if side is None:
                    side = acc[pid][s] = {
                        "root": "",
                        "root_author": None,
                        "utts": [],
                        "root_id": u["root"],
                    }
                side["utts"].append((ts, u["user"], text))
                if is_root:
                    side["root"] = text
                    side["root_author"] = u["user"]

    def _challenger_path(side):
        author = side["root_author"]
        return "\n".join(t for _, a, t in sorted(side["utts"]) if a == author)

    rows = []
    for pid, sides in acc.items():
        pos, neg = sides[1], sides[0]
        if pos is None or neg is None:
            continue  # need both sides
        root_id = pos["root_id"]
        c = conv.get(root_id, {})
        op_author = c.get("op-userID") or root_id
        pos_root, neg_root = pos["root"], neg["root"]
        min_wc = min(len(pos_root.split()), len(neg_root.split()))
        rows.append(
            {
                "pair_id": pid,
                "root_id": root_id,
                "op_author": op_author,
                "train": int(c.get("train", 1)),
                # O = title + self-text (the view is stated in the title)
                "op_text": (_clean(c.get("op-title")) + "\n" + _clean(c.get("op-text-body"))),
                "op_title": _clean(c.get("op-title")),
                "op_body": _clean(c.get("op-text-body")),
                "pos_root": pos_root,
                "neg_root": neg_root,
                "pos_path": _challenger_path(pos),  # challenger-only
                "neg_path": _challenger_path(neg),
                "pos_root_trunc": _truncate_words(pos_root, min_wc),
                "neg_root_trunc": _truncate_words(neg_root, min_wc),
            }
        )
    df = pd.DataFrame(rows)
    for c in ["pos_root", "neg_root", "pos_path", "neg_path"]:
        df[c + "_wc"] = df[c].str.split().str.len()
    derived_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# The unit-text column pair for each text condition.
CONDITIONS = {
    "root_reply": ("pos_root", "neg_root"),
    "full_path": ("pos_path", "neg_path"),
    "root_truncated": ("pos_root_trunc", "neg_root_trunc"),
}
