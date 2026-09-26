"""The two comparison models the paper fits.

**Gated symmetric Bradley--Terry** — the main model (Section ``sec:method``). The judge names
dimensions of difference with no overall verdict; citation is driven by the *magnitude* of
the score gap through an even link, and the direction channel by its sign.

  * :func:`core.model.gated.fit_model` — ridge-penalized MLE of the score matrix S,
    citation intercepts gamma, and global position bias beta.
  * :mod:`core.model.diagnostics` — per-criterion identification (condition number
    kappa) and reliability (signal fraction rho).
  * :mod:`core.model.sandwich` — cluster-robust sandwich covariance (standard errors, CIs).

**Holistic verdict** — the extension of Appendix ``app:verdict-extension``. The judge declares an overall winner
and justifies it; the verdict logit is additive in the per-criterion gaps and the citation
channel is driven by the *signed* margin in the winner's favour. Convex, unlike the gated
model.

  * :mod:`core.model.holistic` — the whole package, with its own ``DERIVATION.md``.

The two share nothing but conventions: both centre s per criterion, both screen criteria on
(kappa, rho), and both report sandwich standard errors, so the diagnostics are comparable
across experiments even though the likelihoods differ.
"""

from core.model.gated import FitResult, fit_model  # noqa: F401
from core.model import diagnostics  # noqa: F401
from core.model import holistic  # noqa: F401

__all__ = ["fit_model", "FitResult", "diagnostics", "holistic"]
