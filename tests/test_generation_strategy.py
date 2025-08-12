"""Tests for generation strategy interfaces and orchestrator."""

import pytest
import asyncio
from unittest.mock import patch
from typing import List, Dict, Any

from src.models.generation_strategy import (
    GenerationStrategy,
    BatchConfig,
    AsyncConfig,
    BatchGenerator,
    AsyncGenerator,
    SequentialGenerator,
)
from src.models.generation_orchestrator import GenerationOrchestrator
from src.models.base import BaseModel

class MockBatchModel(BaseModel, BatchGenerator):
    """Mock model implementing BatchGenerator for testing."""

    def __init__(self) -> None:
        super().__init__("mock-batch-model")

    def get_generation_strategy(self) -> GenerationStrategy:
        return GenerationStrategy.BATCH_OPTIMIZED

    def get_batch_config(self) -> BatchConfig:
        return BatchConfig(optimal_batch_size=4, max_batch_size=8)

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return [f"batch_result_{i}" for i in range(n)]

    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs: Any
    ) -> List[List[str]]:
        results = []
        for i, prompt in enumerate(prompts):
            prompt_results = [f"batch_{i}_{j}" for j in range(n_per_prompt)]
            results.append(prompt_results)
        return results

class MockAsyncModel(BaseModel, AsyncGenerator):
    """Mock model implementing AsyncGenerator for testing."""

    def __init__(self):
        super().__init__("mock-async-model")

    def get_generation_strategy(self) -> GenerationStrategy:
        return GenerationStrategy.ASYNC_OPTIMIZED

    def get_async_config(self) -> AsyncConfig:
        return AsyncConfig(max_concurrent=5, rate_limit_delay=0.1)

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return [f"async_result_{i}" for i in range(n)]

    async def generate_async(
        self,
        prompt: str,
        n: int = 1,
        **kwargs: Any
    ) -> List[str]:
        await asyncio.sleep(0.01)  # Simulate async work
        return [f"async_{prompt}_{i}" for i in range(n)]

class MockSequentialModel(BaseModel, SequentialGenerator):
    """Mock model for sequential generation."""

    def __init__(self):
        super().__init__("mock-sequential-model")

    def get_generation_strategy(self) -> GenerationStrategy:
        return GenerationStrategy.SEQUENTIAL

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return [f"seq_result_{i}" for i in range(n)]

class MockPromptTemplate:
    """Mock prompt template for testing."""

    def format(self, problem: Dict[str, Any]) -> str:
        return f"prompt_for_{problem.get('task_id', 'unknown')}"

class TestGenerationStrategy:
    """Test generation strategy enums and configs."""

    def test_generation_strategy_values(self):
        """Test that strategy enum has expected values."""
        assert GenerationStrategy.BATCH_OPTIMIZED.value == "batch"
        assert GenerationStrategy.ASYNC_OPTIMIZED.value == "async"
        assert GenerationStrategy.SEQUENTIAL.value == "sequential"

    def test_batch_config(self):
        """Test BatchConfig creation and attributes."""
        config = BatchConfig(optimal_batch_size=4, max_batch_size=8)
        assert config.optimal_batch_size == 4
        assert config.max_batch_size == 8
        assert config.progress_callback is None

    def test_async_config(self):
        """Test AsyncConfig creation and attributes."""
        config = AsyncConfig(max_concurrent=10, rate_limit_delay=0.2)
        assert config.max_concurrent == 10
        assert config.rate_limit_delay == 0.2
        assert config.progress_callback is None

class TestMockModels:
    """Test that mock models implement interfaces correctly."""

    def test_batch_model_interface(self):
        """Test BatchGenerator interface implementation."""
        model = MockBatchModel()
        assert model.get_generation_strategy() == GenerationStrategy.BATCH_OPTIMIZED

        config = model.get_batch_config()
        assert config.optimal_batch_size == 4
        assert config.max_batch_size == 8

        # Test batch generation
        prompts = ["prompt1", "prompt2"]
        results = model.generate_batch(prompts, n_per_prompt=2)
        assert len(results) == 2
        assert len(results[0]) == 2
        assert results[0][0] == "batch_0_0"

    @pytest.mark.asyncio
    async def test_async_model_interface(self):
        """Test AsyncGenerator interface implementation."""
        model = MockAsyncModel()
        assert model.get_generation_strategy() == GenerationStrategy.ASYNC_OPTIMIZED

        config = model.get_async_config()
        assert config.max_concurrent == 5
        assert config.rate_limit_delay == 0.1

        # Test async generation
        results = await model.generate_async("test_prompt", n=2)
        assert len(results) == 2
        assert results[0] == "async_test_prompt_0"

class TestGenerationOrchestrator:
    """Test the generation orchestrator."""

    @pytest.mark.asyncio
    async def test_batch_strategy_execution(self):
        """Test orchestrator with batch-optimized model."""
        orchestrator = GenerationOrchestrator()
        model = MockBatchModel()
        template = MockPromptTemplate()

        problems = [
            {"task_id": "test1"},
            {"task_id": "test2"},
        ]

        with patch('src.utils.io.append_to_jsonl'):
            results = await orchestrator.generate_solutions(
                problems, model, template, num_samples=1
            )

        assert len(results) == 2
        assert results[0]["task_id"] == "test1"
        assert results[0]["sample_index"] == 0
        assert results[0]["model_name"] == "mock-batch-model"
        assert "completion" in results[0]

    @pytest.mark.asyncio
    async def test_async_strategy_execution(self):
        """Test orchestrator with async-optimized model."""
        orchestrator = GenerationOrchestrator()
        model = MockAsyncModel()
        template = MockPromptTemplate()

        problems = [{"task_id": "test1"}, {"task_id": "test2"}]

        with patch('src.utils.io.append_to_jsonl'):
            results = await orchestrator.generate_solutions(
                problems, model, template, num_samples=1
            )

        assert len(results) == 2
        assert results[0]["task_id"] == "test1"
        assert results[0]["model_name"] == "mock-async-model"

    @pytest.mark.asyncio
    async def test_sequential_strategy_execution(self):
        """Test orchestrator with sequential model."""
        orchestrator = GenerationOrchestrator()
        model = MockSequentialModel()
        template = MockPromptTemplate()

        problems = [{"task_id": "test1"}]

        with patch('src.utils.io.append_to_jsonl'):
            results = await orchestrator.generate_solutions(
                problems, model, template, num_samples=1
            )

        assert len(results) == 1
        assert results[0]["task_id"] == "test1"
        assert results[0]["model_name"] == "mock-sequential-model"

    @pytest.mark.asyncio
    async def test_multiple_samples(self):
        """Test generation with multiple samples per problem."""
        orchestrator = GenerationOrchestrator()
        model = MockBatchModel()
        template = MockPromptTemplate()

        problems = [{"task_id": "test1"}]

        with patch('src.utils.io.append_to_jsonl'):
            results = await orchestrator.generate_solutions(
                problems, model, template, num_samples=3
            )

        assert len(results) == 3  # 1 problem × 3 samples
        task_results = [r for r in results if r["task_id"] == "test1"]
        assert len(task_results) == 3

        # Check sample indices
        sample_indices = sorted([r["sample_index"] for r in task_results])
        assert sample_indices == [0, 1, 2]
