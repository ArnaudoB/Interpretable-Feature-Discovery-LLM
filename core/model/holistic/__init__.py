"""Holistic verdict model — the variant used by the interpretable-scoring extension.

The main model (:mod:`core.model.gated`) treats every criterion symmetrically: the judge
reports dimensions of difference and no overall verdict is asked for. Here the judge is
asked *which item is better overall* and then *which qualities the winner has more of*, and
the statistical layer changes to match (App. ``app:verdict-extension``):

    P(w_p = 1)            = sigma( sum_k Delta_{p,k} + beta )
    P(c_{p,k} = 1 | w_p)  = sigma( (2 w_p - 1) Delta_{p,k} + gamma_k )

Two structural differences from the gated model. The holistic logit is *additive* in the
per-criterion gaps, which is what makes the overall verdict decomposable into named
criteria. And the citation link is *not* even in Delta: it models justification of a
declared winner, so it is driven by the signed margin in the winner's favour, where the
gated model's citation channel models salience and is driven by magnitude alone through the
even link ``log(2 cosh)``. The predictor is affine in every parameter, so unlike the gated
model this objective is convex.

``DERIVATION.md`` carries the full math and is the source of truth for the appendix.

  * :func:`fit` — ridge-penalized MLE of (s, gamma, beta) by L-BFGS-B.
  * :func:`sandwich_covariance`, :func:`std_errors`, :func:`per_criterion_se` — inference.
  * :mod:`core.model.holistic.diagnostics` — per-criterion rho and kappa, the reliability
    screen used to pick the retained panel, and :func:`score_spread` (the denoised
    per-criterion spread sigma_k with delta-method SEs).
  * :func:`gate_probabilities` — per-observation winner and citation probabilities, for
    calibration.
  * :mod:`core.model.holistic.synthetic` — generate/recover harness guarding the gradient
    algebra; the only correctness check this model has.
"""

from core.model.holistic.fit import fit, FitResult
from core.model.holistic.inference import (
    sandwich_covariance,
    std_errors,
    confidence_interval,
    per_criterion_se,
)
from core.model.holistic.likelihood import nll, grad
from core.model.holistic.hessian import hessian
from core.model.holistic.predict import gate_probabilities
from core.model.holistic.synthetic import generate, SyntheticData

__all__ = [
    "fit",
    "FitResult",
    "sandwich_covariance",
    "std_errors",
    "confidence_interval",
    "per_criterion_se",
    "nll",
    "grad",
    "hessian",
    "gate_probabilities",
    "generate",
    "SyntheticData",
]
