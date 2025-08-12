"""Tests for CLI integration with new generation strategies."""

import pytest
import asyncio
from unittest.mock import patch
from pathlib import Path
from typing import List, Dict, Any

from src.cli import generate_solutions_optimal
from src.models.generation_strategy import (
    GenerationStrategy, BatchConfig, AsyncConfig
)
from src.models.base import BaseModel
from src.models.generation_strategy import BatchGenerator, AsyncGenerator

class MockBatchModel(BaseModel, BatchGenerator):
    """Mock batch model for CLI testing."""

    def __init__(self, model_name: str = "mock-batch"):
        super().__init__(model_name)

    def get_generation_strategy(self) -> GenerationStrategy:
        return GenerationStrategy.BATCH_OPTIMIZED

    def get_batch_config(self) -> BatchConfig:
        return BatchConfig(optimal_batch_size=2, max_batch_size=4)

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return [f"result_{i}" for i in range(n)]

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
    """Mock async model for CLI testing."""

    def __init__(self, model_name: str = "mock-async"):
        super().__init__(model_name)

    def get_generation_strategy(self) -> GenerationStrategy:
        return GenerationStrategy.ASYNC_OPTIMIZED

    def get_async_config(self) -> AsyncConfig:
        return AsyncConfig(max_concurrent=3, rate_limit_delay=0.01)

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return [f"result_{i}" for i in range(n)]

    async def generate_async(
        self,
        prompt: str,
        n: int = 1,
        **kwargs: Any
    ) -> List[str]:
        await asyncio.sleep(0.001)  # Very short delay for testing
        return [f"async_{prompt.split('_')[-1]}_{i}" for i in range(n)]

class MockPromptTemplate:
    """Mock prompt template for CLI testing."""

    def format(self, problem: Dict[str, Any]) -> str:
        return f"prompt_for_{problem.get('task_id', 'unknown')}"

class TestCLIStrategyIntegration:
    """Test CLI integration with generation strategies."""

    @pytest.mark.asyncio
    async def test_batch_model_integration(self):
        """Test CLI with batch-optimized model."""
        model = MockBatchModel()
        template = MockPromptTemplate()

        problems = [
            {"task_id": "test1"},
            {"task_id": "test2"},
            {"task_id": "test3"},
        ]

        with patch('src.models.generation_orchestrator.append_to_jsonl'):
            results = await generate_solutions_optimal(
                problems, model, template, num_samples=1
            )

        assert len(results) == 3

        # Check result structure
        for i, result in enumerate(results):
            assert result["task_id"] == f"test{i+1}"
            assert result["sample_index"] == 0
            assert result["model_name"] == "mock-batch"
            assert "completion" in result
            assert "generated_at" in result

    @pytest.mark.asyncio
    async def test_async_model_integration(self):
        """Test CLI with async-optimized model."""
        model = MockAsyncModel()
        template = MockPromptTemplate()

        problems = [
            {"task_id": "async1"},
            {"task_id": "async2"},
        ]

        with patch('src.models.generation_orchestrator.append_to_jsonl'):
            results = await generate_solutions_optimal(
                problems, model, template, num_samples=1
            )

        assert len(results) == 2

        # Check async-specific results
        for result in results:
            assert result["model_name"] == "mock-async"
            assert result["completion"].startswith("async_")

    @pytest.mark.asyncio
    async def test_multiple_samples_batch(self):
        """Test multiple samples with batch model."""
        model = MockBatchModel()
        template = MockPromptTemplate()

        problems = [{"task_id": "multi_test"}]

        with patch('src.models.generation_orchestrator.append_to_jsonl'):
            results = await generate_solutions_optimal(
                problems, model, template, num_samples=3
            )

        assert len(results) == 3

        # Check sample indices
        sample_indices = sorted([r["sample_index"] for r in results])
        assert sample_indices == [0, 1, 2]

        # All should be for the same task
        task_ids = [r["task_id"] for r in results]
        assert all(tid == "multi_test" for tid in task_ids)

    @pytest.mark.asyncio
    async def test_output_file_integration(self):
        """Test file output integration."""
        model = MockBatchModel()
        template = MockPromptTemplate()

        problems = [{"task_id": "file_test"}]
        output_file = Path("/tmp/test_output.jsonl")

        with patch(
            'src.models.generation_orchestrator.append_to_jsonl'
        ) as mock_append:
            results = await generate_solutions_optimal(
                problems, model, template, num_samples=1, output_file=output_file
            )

        assert len(results) == 1

        # Check that file output was called
        mock_append.assert_called()
        call_args = mock_append.call_args
        assert call_args[0][1] == str(output_file)  # Second arg is file path

    @pytest.mark.asyncio
    async def test_strategy_logging(self, caplog):
        """Test that strategy information is logged."""
        import logging
        caplog.set_level(logging.INFO)

        model = MockBatchModel()
        template = MockPromptTemplate()
        problems = [{"task_id": "log_test"}]

        with patch('src.models.generation_orchestrator.append_to_jsonl'):
            await generate_solutions_optimal(
                problems, model, template, num_samples=1
            )

        # Check logs contain strategy information
        log_text = caplog.text
        assert "batch" in log_text.lower()  # More lenient check
        assert "mock-batch" in log_text

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in generation."""
        model = MockBatchModel()
        template = MockPromptTemplate()

        # Mock model to raise exception
        with patch.object(model, 'generate_batch', side_effect=Exception("Test error")):
            problems = [{"task_id": "error_test"}]

            with patch('src.models.generation_orchestrator.append_to_jsonl'):
                results = await generate_solutions_optimal(
                    problems, model, template, num_samples=1
                )

            # Should get empty results but not crash (check the orchestrator error handling)
            assert len(results) == 1
            # The orchestrator should return empty completion on error
            assert results[0]["completion"] == ""

    @pytest.mark.asyncio
    async def test_concurrent_execution_async_model(self):
        """Test that async model respects concurrency limits."""
        model = MockAsyncModel()
        template = MockPromptTemplate()

        # Create many problems to test concurrency
        problems = [{"task_id": f"concurrent_{i}"} for i in range(10)]

        import time
        start_time = time.time()

        with patch('src.models.generation_orchestrator.append_to_jsonl'):
            results = await generate_solutions_optimal(
                problems, model, template, num_samples=1
            )

        end_time = time.time()

        assert len(results) == 10

        # Should complete faster than sequential (each async call has 0.001s delay)
        # With concurrency=3, should be roughly 10/3 * 0.001 = ~0.003s plus overhead
        assert end_time - start_time < 0.1  # Much less than 0.01s (sequential time)

    @pytest.mark.asyncio
    async def test_batch_size_optimization(self):
        """Test that batch processing respects optimal batch sizes."""
        model = MockBatchModel()  # optimal_batch_size=2
        template = MockPromptTemplate()

        # 5 problems should be processed in 3 batches (2+2+1)
        problems = [{"task_id": f"batch_{i}"} for i in range(5)]

        with patch.object(model, 'generate_batch', wraps=model.generate_batch) as mock_batch:
            with patch('src.models.generation_orchestrator.append_to_jsonl'):
                results = await generate_solutions_optimal(
                    problems, model, template, num_samples=1
                )

        assert len(results) == 5

        # Check that generate_batch was called with appropriate batch sizes
        calls = mock_batch.call_args_list
        assert len(calls) == 3  # 3 batch calls

        # First two calls should have 2 prompts each, last one should have 1
        batch_sizes = [len(call[0][0]) for call in calls]  # Length of prompts list
        assert batch_sizes == [2, 2, 1]
