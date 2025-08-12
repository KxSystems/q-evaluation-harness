"""Abstract base class for code generation models."""

from abc import ABC, abstractmethod
from typing import List, Any
import asyncio
from functools import partial

from .generation_strategy import GenerationStrategy


class BaseModel(ABC):
    """Abstract base class for code generation models."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            model_name: Name or identifier of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def get_generation_strategy(self) -> GenerationStrategy:
        """Declare the optimal generation strategy for this model type."""
        pass

    @abstractmethod
    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions for the given prompt.

        Args:
            prompt: Input prompt for code generation
            n: Number of completions to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated code completions
        """
        pass

    async def generate_async(
        self, prompt: str, n: int = 1, **kwargs: Any
    ) -> List[str]:
        """Async version of generate.
        Default implementation falls back to sync.
        
        NOTE: This method is deprecated for batch-optimized models.
        Use the generation orchestrator for optimal performance.

        Args:
            prompt: Input prompt for code generation
            n: Number of completions to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated code completions
        """
        # Default implementation uses asyncio executor to run sync generate
        loop = asyncio.get_event_loop()
        generate_func = partial(self.generate, prompt, n, **kwargs)
        return await loop.run_in_executor(None, generate_func)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"
