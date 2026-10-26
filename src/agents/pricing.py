"""List prices for agent backends, used to compute or audit task cost.

The Codex CLI reports only token counts, so its cost is computed here from
OpenAI's published list prices. Claude Code reports an API-equivalent cost
itself (``total_cost_usd``), but it is only as good as the CLI's price table:
for a model the CLI does not know (Haiku 5.5 on 2.1.287 reported ~32x the real
cost) or prices wrongly (Sonnet 5.5 cache reads at 2x the listed rate), it is
wrong. ``claude_cost_usd`` recomputes it from the usage the CLI records, so
sweeps can price runs from tokens. Prices are per 1M tokens, standard tier.

Token semantics, as reported by ``codex exec --json`` (``turn.completed``):
cached input is a subset of ``input_tokens``, and reasoning tokens are a
subset of ``output_tokens``. Verified against Codex session logs, where
``total_tokens == input_tokens + output_tokens`` whenever reasoning > 0. So
reasoning is billed through output and needs no separate term.

A model missing from the table gets no cost (None), never a guess.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

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


# Short-context tier (<=272K input tokens per request). 5.6/6.x models also have
# a long-context tier (2x input) that agent turns are not expected to reach.
OPENAI_PRICES: Dict[str, TokenPrices] = {
    "gpt-5.5": TokenPrices(5.00, 0.50, 30.00, OPENAI_PRICING_URL, "2026-10-01"),
    "gpt-5.4": TokenPrices(2.50, 0.25, 15.00, OPENAI_PRICING_URL, "2026-10-01"),
    # 5.6 Sol after the 2026-07-30 price cut; launch price was 5.00 / 30.00.
    "gpt-5.6-sol": TokenPrices(4.00, 0.40, 20.00, OPENAI_PRICING_URL, "2026-10-02"),
    "gpt-6-sol": TokenPrices(2.00, 0.20, 10.00, OPENAI_PRICING_URL, "2026-10-02"),
    "gpt-6-luna": TokenPrices(0.10, 0.01, 0.50, OPENAI_PRICING_URL, "2026-10-02"),
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


# --- Anthropic -------------------------------------------------------------

ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"


@dataclass(frozen=True)
class ClaudeTier:
    """USD per 1M tokens for one prompt-length tier."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float


@dataclass(frozen=True)
class ClaudePrices:
    base: ClaudeTier
    # Haiku 5.5 only: requests whose prompt exceeds ``long_threshold`` tokens
    # are billed at ``long``. Other models bill the whole window at ``base``.
    long: Optional[ClaudeTier] = None
    long_threshold: int = 100_000
    as_of: str = "2026-10-08"


CLAUDE_PRICES: Dict[str, ClaudePrices] = {
    "claude-fable-5-1": ClaudePrices(ClaudeTier(10.00, 50.00, 0.25, 12.50, 20.00)),
    "claude-opus-5-5": ClaudePrices(ClaudeTier(4.00, 20.00, 0.20, 5.00, 8.00)),
    "claude-opus-5": ClaudePrices(ClaudeTier(5.00, 25.00, 0.50, 6.25, 10.00)),
    "claude-opus-4-8": ClaudePrices(ClaudeTier(5.00, 25.00, 0.50, 6.25, 10.00)),
    # Sonnet 5.5 and Opus 5.5 cache hits are 0.05x input; earlier models 0.1x.
    "claude-sonnet-5-5": ClaudePrices(ClaudeTier(2.00, 10.00, 0.10, 2.50, 4.00)),
    "claude-sonnet-5": ClaudePrices(ClaudeTier(2.00, 10.00, 0.20, 2.50, 4.00)),
    "claude-haiku-5-5": ClaudePrices(
        ClaudeTier(0.10, 0.50, 0.01, 0.125, 0.20),
        long=ClaudeTier(0.50, 2.50, 0.05, 0.625, 1.00),
    ),
}


def claude_cost_usd(
    model: str, usage: Dict[str, Any], max_prompt_tokens: Optional[int] = None
) -> Optional[float]:
    """API-equivalent cost of one Claude Code task from its ``result`` usage.

    ``usage`` is the ``usage`` object of the stream's final result event: totals
    for the whole task. Cache writes use the per-TTL split when present, else
    are priced as 1-hour writes, which is what Claude Code issues.

    Totals cannot be split by request, so a length-tiered model (Haiku 5.5) is
    priced at the base tier only when the caller shows that no single request's
    prompt (input + cache read + cache write tokens) exceeded the threshold, by
    passing the largest one as ``max_prompt_tokens``. Otherwise, like an
    unpriced model, it returns None rather than guess.
    """
    prices = CLAUDE_PRICES.get(model)
    if prices is None:
        return None
    if prices.long is not None and (
        max_prompt_tokens is None or max_prompt_tokens > prices.long_threshold
    ):
        return None

    i = usage.get("input_tokens") or 0
    o = usage.get("output_tokens") or 0
    rd = usage.get("cache_read_input_tokens") or 0
    wr = usage.get("cache_creation_input_tokens") or 0
    split = usage.get("cache_creation") or {}
    w1h = split.get("ephemeral_1h_input_tokens", 0 if split else wr)
    w5m = split.get("ephemeral_5m_input_tokens", 0)
    t = prices.base
    return (
        i * t.input + o * t.output + rd * t.cache_read
        + w5m * t.cache_write_5m + w1h * t.cache_write_1h
    ) / 1_000_000
