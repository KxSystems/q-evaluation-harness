"""List prices for agent backends whose CLI reports tokens but not dollars.

Claude Code reports an API-equivalent cost itself (``total_cost_usd``). The
Codex CLI reports only token counts, so its cost is computed here from the
provider's published list prices. Prices are per 1M tokens, standard tier.

Token semantics, as reported by ``codex exec --json`` (``turn.completed``):
cached input is a subset of ``input_tokens``, and reasoning tokens are a
subset of ``output_tokens``. Verified against Codex session logs, where
``total_tokens == input_tokens + output_tokens`` whenever reasoning > 0. So
reasoning is billed through output and needs no separate term.

A model missing from the table gets no cost (None), never a guess.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)

OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"


@dataclass(frozen=True)
class TokenPrices:
    """USD per 1M tokens."""

    input: float
    cached_input: float
    output: float
    source: str
    as_of: str


# Short-context tier (<=272K input tokens per request). The pricing page lists
# no long-context tier for these models.
OPENAI_PRICES: Dict[str, TokenPrices] = {
    "gpt-5.5": TokenPrices(5.00, 0.50, 30.00, OPENAI_PRICING_URL, "2026-10-01"),
    "gpt-5.4": TokenPrices(2.50, 0.25, 15.00, OPENAI_PRICING_URL, "2026-10-01"),
}

_warned: set = set()


def openai_prices(model: str) -> Optional[TokenPrices]:
    """Look up list prices by exact model id; warn once if unknown."""
    prices = OPENAI_PRICES.get(model)
    if prices is None and model not in _warned:
        _warned.add(model)
        logger.warning(
            f"No list price for OpenAI model '{model}'; cost will be omitted. "
            f"Add it to src/agents/pricing.py (source: {OPENAI_PRICING_URL})."
        )
    return prices


def openai_cost_usd(
    model: str,
    input_tokens: Optional[int],
    cached_input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """API-equivalent cost of one task from Codex token counts.

    cost = (input - cached) * input_price + cached * cached_price
           + output * output_price
    """
    prices = openai_prices(model)
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    cached = min(cached_input_tokens or 0, input_tokens)
    return (
        (input_tokens - cached) * prices.input
        + cached * prices.cached_input
        + output_tokens * prices.output
    ) / 1_000_000
