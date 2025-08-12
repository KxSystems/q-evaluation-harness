"""Unified model configuration for LiteLLM models."""

import secrets
from dataclasses import dataclass
from typing import Dict, Any, List
from src.constants import (
    DEFAULT_MAX_THINKING_TOKENS,
    DEFAULT_REASONING_EFFORT,
)


@dataclass
class ModelConfig:
    """Unified configuration for model-specific behavior."""

    # Async configuration
    max_concurrent: int
    rate_limit_delay: float

    # Generation behavior
    requires_parallel_generation: bool  # n>1 not supported natively
    requires_seed_diversification: bool  # Need different seeds per request

    # Parameter modifications
    param_updates: Dict[str, Any]
    param_removals: List[str]


def _detect_model_type(model_name: str) -> str:
    """Detect model type from model name."""
    model_lower = model_name.lower()

    if "o1" in model_lower or "o3" in model_lower:
        return "o_series"
    elif "gpt-5" in model_lower:
        return "gpt5"
    elif "claude" in model_lower:
        return "claude"
    elif "grok" in model_lower:
        return "grok"
    elif "gemini" in model_lower:
        return "gemini"
    elif "gpt" in model_lower:
        return "openai"
    else:
        return "unknown"


# Model-specific configurations
MODEL_CONFIGS = {
    "o_series": ModelConfig(
        max_concurrent=50,
        rate_limit_delay=0.05,
        requires_parallel_generation=True,
        requires_seed_diversification=True,
        param_updates={
            "max_completion_tokens": DEFAULT_MAX_THINKING_TOKENS,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
        },
        param_removals=["temperature"]
    ),

    "gpt5": ModelConfig(
        max_concurrent=15,
        rate_limit_delay=0.1,
        requires_parallel_generation=True,
        requires_seed_diversification=True,
        param_updates={
            "max_completion_tokens": DEFAULT_MAX_THINKING_TOKENS,
        },
        param_removals=["max_tokens", "temperature", "top_p"]
    ),

    "claude": ModelConfig(
        max_concurrent=15,
        rate_limit_delay=0.1,
        requires_parallel_generation=True,
        requires_seed_diversification=True,
        param_updates={
            "max_tokens": DEFAULT_MAX_THINKING_TOKENS
        },
        param_removals=[]
    ),

    "grok": ModelConfig(
        max_concurrent=10,
        rate_limit_delay=0.2,
        requires_parallel_generation=True,
        requires_seed_diversification=True,
        param_updates={
            "max_tokens": DEFAULT_MAX_THINKING_TOKENS
        },
        param_removals=[]
    ),

    "gemini": ModelConfig(
        max_concurrent=15,
        rate_limit_delay=0.1,
        requires_parallel_generation=True,
        requires_seed_diversification=True,
        param_updates={
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_MAX_THINKING_TOKENS
            },
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
        },
        param_removals=["max_tokens"]
    ),

    "openai": ModelConfig(
        max_concurrent=50,
        rate_limit_delay=0.05,
        requires_parallel_generation=False,
        requires_seed_diversification=False,
        param_updates={},
        param_removals=[]
    ),

    "unknown": ModelConfig(
        max_concurrent=1,
        rate_limit_delay=0.2,
        requires_parallel_generation=False,
        requires_seed_diversification=False,
        param_updates={},
        param_removals=[]
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get unified configuration for a model.

    Args:
        model_name: Model identifier (e.g., 'gpt-4o', 'claude-3-sonnet')

    Returns:
        ModelConfig with all model-specific settings
    """
    model_type = _detect_model_type(model_name)
    return MODEL_CONFIGS[model_type]


def generate_unique_seed() -> int:
    """Generate a cryptographically secure random seed.

    Returns:
        Random integer in valid seed range
    """
    return secrets.randbelow(2**31)


def generate_unique_seeds(count: int) -> List[int]:
    """Generate multiple unique seeds for parallel requests.

    Args:
        count: Number of unique seeds to generate

    Returns:
        List of unique random seeds
    """
    if count <= 0:
        return []

    seeds: set[int] = set()
    max_attempts = count * 10  # Prevent infinite loops
    attempts = 0

    while len(seeds) < count and attempts < max_attempts:
        seeds.add(generate_unique_seed())
        attempts += 1

    # If we couldn't generate enough unique seeds, fill with randoms
    result = list(seeds)
    while len(result) < count:
        result.append(generate_unique_seed())

    return result[:count]
