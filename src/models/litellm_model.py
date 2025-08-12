"""LiteLLM model wrapper for unified API access."""

import litellm
import logging
import asyncio
from typing import List, Any
from .base import BaseModel
from .generation_strategy import (
    GenerationStrategy,
    AsyncGenerator,
    AsyncConfig
)
from .model_config import get_model_config, generate_unique_seeds
from src.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class LiteLLMModel(BaseModel, AsyncGenerator):
    """LiteLLM-based model for unified API access to various providers."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize LiteLLM model.

        Args:
            model_name: Model identifier (e.g., 'gpt-4o', 'claude-3-sonnet')
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)

        # Set LiteLLM to drop unsupported parameters
        litellm.drop_params = True

        # Default parameters
        self.default_params = {
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "seed": kwargs.get("seed", DEFAULT_SEED),
        }

        # Get unified model configuration
        self.config = get_model_config(model_name)

    def get_generation_strategy(self) -> GenerationStrategy:
        """LiteLLM models are optimized for async API calls."""
        return GenerationStrategy.ASYNC_OPTIMIZED

    def get_async_config(self) -> AsyncConfig:
        """Configure based on API provider limits and model type."""
        return AsyncConfig(
            max_concurrent=self.config.max_concurrent,
            rate_limit_delay=self.config.rate_limit_delay
        )

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions for single prompt (sync wrapper)."""
        return asyncio.run(self.generate_async(prompt, n, **kwargs))

    async def generate_async(
        self, prompt: str, n: int = 1, **kwargs: Any
    ) -> List[str]:
        """Generate n completions asynchronously using LiteLLM's native n.

        Args:
            prompt: Input prompt for code generation
            n: Number of completions to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated code completions
        """
        # Merge default parameters with provided ones
        params = {**self.default_params, **kwargs}

        # Apply model-specific parameter modifications
        params.update(self.config.param_updates)
        for key in self.config.param_removals:
            params.pop(key, None)

        # Use parallel generation if model requires it for n>1
        if self.config.requires_parallel_generation and n > 1:
            return await self._generate_parallel_async(prompt, n, **kwargs)

        # Build completion request with native n parameter
        completion_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,  # Use LiteLLM's native n parameter for efficiency
            **params,
        }

        try:
            # Single API call for all n completions
            response = await litellm.acompletion(**completion_params)

            # Extract all completions from the response
            completions = []

            for choice in response.choices:
                content = (
                    choice.message.content.strip()
                    if choice.message.content
                    else ""
                )
                completions.append(content)

            # Ensure we return exactly n completions (pad with empty if needed)
            while len(completions) < n:
                completions.append("")

            return completions[:n]  # Truncate if more than n

        except Exception as e:
            error_msg = f"Async generation failed for n={n}: {str(e)}"
            logger.warning(f"Warning: {error_msg}")
            # Return empty strings for all requested completions
            return [""] * n

    async def _generate_parallel_async(
        self, prompt: str, n: int, **kwargs: Any
    ) -> List[str]:
        """Generate completions in parallel using concurrent single calls.

        This method is used for models that don't support native n>1 parameter
        but can handle concurrent requests efficiently. Uses seed
        diversification for models that require it to ensure statistical
        independence.
        """
        async_config = self.get_async_config()
        semaphore = asyncio.Semaphore(async_config.max_concurrent)

        # Generate unique seeds if diversification is required
        unique_seeds = None
        if self.config.requires_seed_diversification:
            unique_seeds = generate_unique_seeds(n)
            logger.debug(f"Generated {n} unique seeds for parallel generation")

        async def single_generation(request_index: int) -> str:
            """Generate single completion with rate limiting and seed."""
            async with semaphore:
                if async_config.rate_limit_delay > 0:
                    await asyncio.sleep(async_config.rate_limit_delay)
                try:
                    # Create kwargs copy and add unique seed if needed
                    request_kwargs = dict(kwargs)
                    if unique_seeds is not None:
                        request_kwargs["seed"] = unique_seeds[request_index]
                        logger.debug(
                            f"Request {request_index} using seed "
                            f"{unique_seeds[request_index]}"
                        )

                    completions = await self.generate_async(
                        prompt, n=1, **request_kwargs
                    )
                    return completions[0] if completions else ""
                except Exception as e:
                    logger.warning(f"Parallel generation failed: {e}")
                    return ""

        # Execute all generations concurrently with indexed seeds
        tasks = [single_generation(i) for i in range(n)]
        return await asyncio.gather(*tasks)

    def __str__(self) -> str:
        """String representation of the model."""
        return f"LiteLLMModel({self.model_name})"
