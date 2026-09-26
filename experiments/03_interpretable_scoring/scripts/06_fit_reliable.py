"""Step 6: refit the model on the criteria that passed the reliability screen.

    python scripts/06_fit_reliable.py --run all

Identical to step 4 in every respect except which criteria it is given: step 4 fits all K
criteria the taxonomy produced, step 5 screens them on (rho, kappa), and this step refits on
the survivors alone. Every number of App. ``app:verdict-extension`` -- the radars' sigma_k,
the total scores, beta, the calibration -- comes from *this* fit, not from step 4's.

A thin wrapper rather than a copy: it re-invokes ``04_fit.py --reliable``, so there is
exactly one implementation of the fit.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run name, a comma list, or 'all'")
    args = ap.parse_args()

    sys.argv = [str(HERE / "04_fit.py"), "--run", args.run, "--reliable"]
    runpy.run_path(str(HERE / "04_fit.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
