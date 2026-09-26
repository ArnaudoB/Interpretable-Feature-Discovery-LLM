"""core — the machinery shared by every experiment in this repository.

The method is the same whatever the corpus: compare items in pairs along a balanced
comparison graph, let an LLM judge name in free text the dimensions on which each pair
differs, canonicalize those raw dimensions into criteria, and fit a gated symmetric
Bradley--Terry model that jointly explains *whether* a criterion is cited and *which* item
wins on it. What changes between experiments is the corpus, the prompts, and the baselines
they are compared against; those live under ``experiments/<NN>_<name>/``.

Subpackages and modules:
  * ``core.graph``      — nested balanced comparison graph (union of perfect matchings)
  * ``core.judging``    — response cache, request specs, realtime + batch orchestration
                          for OpenAI, Anthropic, Gemini and DeepSeek judges, cost accounting
  * ``core.model``      — gated symmetric Bradley--Terry fit, per-criterion
                          identification (kappa) and reliability (rho), sandwich covariance;
                          ``core.model.holistic`` is the holistic-verdict model of the
                          interpretable-scoring extension
  * ``core.cmv``        — ChangeMyView experiment (data, items, taxonomy, BT and feature baselines)
  * ``core.evaluation`` — calibration of fitted gates (ECE, Brier skill, pair bootstrap)
  * ``core.analysis``   — cross-judge consolidation of criteria (rendering + validation)
  * ``core.fit``        — gated-model design tensor + fit-diagnostics helpers
  * ``core.tensors``    — holistic-model observation tensors from canonical judgments
  * ``core.embeddings`` — text-embedding-3-large with an on-disk cache
  * ``core.metrics``    — Hungarian matching + recovery metrics (Exp 1)
  * ``core.canonical``  — normalization of free-text dimension names
  * ``core.ledger``     — per-experiment record of the numbers quoted in the paper
  * ``core.openai_client`` — the single OpenAI client constructor and ``.env`` loader

Each experiment's ``scripts/_paths.py`` puts the repository root on ``sys.path``, so
``import core...`` resolves with no installation step.
"""

__version__ = "0.1.0"
