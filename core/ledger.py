"""Paper-number records: every number an experiment's section quotes, recomputed from artifacts.

Each experiment's last step writes ``results/paper_numbers.json`` with one record per quoted
number (or group of numbers that appear together)::

    {"id": ..., "where": <section, table or figure label>, "what": <description>, "value": ...}

``tests/test_paper_numbers.py`` checks the schema and compares the values with a stored snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path


class Ledger:
    def __init__(self, experiment: str):
        self.experiment = experiment
        self.records: list[dict] = []

    def record(self, rid: str, where: str, what: str, value) -> None:
        assert rid not in {r["id"] for r in self.records}, f"duplicate record id {rid}"
        self.records.append({"id": rid, "where": where, "what": what, "value": value})
        print(f"[{where}] {rid}: {value}")

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"experiment": self.experiment, "n": len(self.records), "numbers": self.records},
                indent=2,
                default=_jsonable,
            )
        )
        print(f"\n[paper-numbers] {len(self.records)} records; wrote {path.name}")


def _jsonable(x):
    try:
        return x.item()  # numpy scalars
    except AttributeError:
        return str(x)
