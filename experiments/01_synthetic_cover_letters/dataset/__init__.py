"""Synthetic cover-letter corpus generator (comparative-vs-pointwise halo study).

It plants a set of ORDERED, orthogonal dials (each with a known level per letter),
assembles letters from the frozen templates in ``dataset/templates.py``, and degrades
prose fluency by a deterministic, length-preserving mistake count. The ground-truth
level of every dial in every letter is the answer key that makes criterion-recovery
and score-vs-level analysis possible.

This package carries NO hardcoded experiment paths: ``generate_dataset`` takes the
``templates`` and ``job_ad`` modules as arguments (loaded by the entry script).
"""

from .generate import generate_dataset, run_gates  # noqa: F401
