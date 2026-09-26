"""Token → USD cost projection for the budget gate.

Prices are USD per 1M tokens (input, output). The defaults are the rates used for the
costs reported in the paper; they can be overridden via the config ``pricing:`` section.
The Batch API is half price, so projections for batched phases apply
:data:`BATCH_DISCOUNT`.
"""

from __future__ import annotations

from collections import Counter

# (input_per_1M, output_per_1M) USD; override in config `pricing:`.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 10.00),
    # gpt-5.4-mini: the rate used for the costs reported in the paper (Exp 1 and the GPT
    # judge of Exp 3).
    "gpt-5.4-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Judges of the interpretable-scoring experiment (Exp 3).
    # Anthropic first-party list prices; the prefix match covers dated snapshots.
    "claude-haiku-4-5": (1.00, 5.00),
    # Google. Batch is half price, as for the other providers.
    "gemini-3.6-flash": (1.50, 7.50),
    # DeepSeek OFF-PEAK cache-miss rates; peak hours are 2x and cache-hit input is
    # $0.007/MTok, so realized cost normally lands BELOW this.
    "deepseek-v4-flash": (0.22, 0.66),
}
BATCH_DISCOUNT = 0.5


def price_for(model: str, overrides: dict | None = None) -> tuple[float, float]:
    """(input, output) per-1M price for a model id, honoring config overrides.

    Falls back to a family prefix match (``gpt-5.4-mini-2026-03-17`` → ``gpt-5.4-mini``)
    so dated snapshots resolve to their family price.
    """
    overrides = overrides or {}
    for table in (overrides, DEFAULT_PRICES):
        if model in table:
            return tuple(table[model])  # type: ignore[return-value]
    # longest matching family prefix
    for table in (overrides, DEFAULT_PRICES):
        cands = sorted((k for k in table if model.startswith(k)), key=len, reverse=True)
        if cands:
            return tuple(table[cands[0]])  # type: ignore[return-value]
    return (2.50, 10.00)


def estimate_usd(
    model_counts: dict[str, int],
    in_tok: float,
    out_tok: float,
    overrides: dict | None = None,
    batch: bool = True,
) -> float:
    """Projected USD for ``{model: n_requests}`` at avg ``in_tok``/``out_tok`` per request."""
    total = 0.0
    for model, n in model_counts.items():
        pin, pout = price_for(model, overrides)
        total += n * (in_tok * pin + out_tok * pout) / 1_000_000.0
    return total * (BATCH_DISCOUNT if batch else 1.0)


def actual_usd(records, overrides: dict | None = None, batch: bool = True) -> float:
    """Realized USD from cache records carrying ``model``/``usage`` fields."""
    total = 0.0
    for r in records:
        u = r.get("usage") or {}
        pin, pout = price_for(r.get("model", ""), overrides)
        total += (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1_000_000.0
    return total * (BATCH_DISCOUNT if batch else 1.0)


def model_counts(specs) -> dict[str, int]:
    return dict(Counter(s.model for s in specs))


def actual_usd_by_source(records, overrides: dict | None = None) -> float:
    """Realized USD applying the batch discount per record, from its ``source`` field.

    A cell that smoke-tests realtime and then runs the bulk through a batch API has a mix
    of both in one cache; pricing the whole set at one rate is wrong either way.
    """
    total = 0.0
    for r in records:
        u = r.get("usage") or {}
        pin, pout = price_for(r.get("model", ""), overrides)
        cost = (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1_000_000.0
        total += cost * (BATCH_DISCOUNT if r.get("source") == "batch" else 1.0)
    return total
