"""Generation strategy interfaces and configurations."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


class GenerationStrategy(Enum):
    """Defines how a model should be used for optimal performance."""
    BATCH_OPTIMIZED = "batch"      # VLLM, HuggingFace - batch all requests
    ASYNC_OPTIMIZED = "async"      # API models - concurrent requests
    SEQUENTIAL = "sequential"      # Fallback for resource-constrained


@dataclass
class BatchConfig:
    """Configuration for batch-optimized generation."""
    optimal_batch_size: int
    max_batch_size: int
    progress_callback: Optional[Callable] = None


@dataclass 
class AsyncConfig:
    """Configuration for async-optimized generation."""
    max_concurrent: int
    rate_limit_delay: float = 0.0
    progress_callback: Optional[Callable] = None


class BatchGenerator(ABC):
    """Interface for models that benefit from batch processing."""
    
    @abstractmethod
    def generate_batch(
        self, 
        prompts: List[str], 
        n_per_prompt: int = 1,
        **kwargs: Any
    ) -> List[List[str]]:
        """Generate completions for multiple prompts efficiently.
        
        Returns:
            List where each element is a list of n_per_prompt completions
            for the corresponding prompt.
        """
        pass
    
    @abstractmethod
    def get_batch_config(self) -> BatchConfig:
        """Return optimal batching configuration for this model."""
        pass


class AsyncGenerator(ABC):
    """Interface for models that benefit from async processing."""
    
    @abstractmethod
    async def generate_async(
        self, 
        prompt: str, 
        n: int = 1,
        **kwargs: Any
    ) -> List[str]:
        """Generate completions asynchronously."""
        pass
    
    @abstractmethod
    def get_async_config(self) -> AsyncConfig:
        """Return optimal async configuration for this model."""
        pass


class SequentialGenerator(ABC):
    """Interface for basic sequential generation."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        n: int = 1,
        **kwargs: Any
    ) -> List[str]:
        """Generate completions sequentially."""
        pass