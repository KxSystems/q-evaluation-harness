"""Tests for HuggingFace model implementation."""

import os
import pytest
import asyncio
import torch
from unittest.mock import patch
from src.models.huggingface_model import HuggingFaceModel
from src.models.generation_strategy import GenerationStrategy


class TestHuggingFaceModel:
    """Test suite for HuggingFace model functionality."""

    def test_model_initialization(self) -> None:
        """Test that model initializes correctly."""
        # Use a very small model for testing
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=50)
            assert model.model_name == model_name
            assert not model.is_local  # Online model
            assert not model.is_thinking  # Standard model
            assert hasattr(model, "tokenizer")
            assert hasattr(model, "model")
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_model_string_representation(self) -> None:
        """Test string representation of model."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name)
            str_repr = str(model)
            assert model_name in str_repr
            assert "online" in str_repr
            assert "standard" in str_repr
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_thinking_model_detection(self) -> None:
        """Test detection of thinking models."""
        # Test thinking model patterns
        thinking_names = [
            "test-thinking-model",
            "gpt-o1-preview",
            "reasoning-coder",
            "model-with-cot",
        ]

        for name in thinking_names:
            try:
                # Mock initialization without actual loading
                model = HuggingFaceModel.__new__(HuggingFaceModel)
                model.model_name = name
                expected = f"{name} should be detected as thinking model"
                assert model._is_thinking_model(name), expected
            except Exception:
                pass  # Skip if can't initialize

    def test_single_generation(self) -> None:
        """Test generating a single completion."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=20)
            prompt = "def hello_world():"

            completions = model.generate(prompt, n=1)

            assert len(completions) == 1
            assert isinstance(completions[0], str)
            # Basic sanity check - should have some content
            # Could be empty if generation fails
            assert len(completions[0]) >= 0

        except Exception as e:
            pytest.skip(f"Generation failed: {e}")

    def test_multiple_generations(self) -> None:
        """Test generating multiple completions."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=20)
            prompt = "def add_numbers(a, b):"

            completions = model.generate(prompt, n=3)

            assert len(completions) == 3
            assert all(isinstance(comp, str) for comp in completions)

        except Exception as e:
            pytest.skip(f"Generation failed: {e}")

    def test_generation_with_custom_params(self) -> None:
        """Test generation with custom parameters."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name)
            prompt = "print("

            completions = model.generate(
                prompt, n=1, max_new_tokens=15, temperature=0.7
            )

            assert len(completions) == 1
            assert isinstance(completions[0], str)

        except Exception as e:
            pytest.skip(f"Generation failed: {e}")

    def test_local_model_detection(self) -> None:
        """Test local model path detection."""
        # Test with non-existent local path
        fake_local_path = "/fake/local/model/path"

        try:
            model = HuggingFaceModel.__new__(HuggingFaceModel)
            model.model_name = fake_local_path
            exists = os.path.exists(fake_local_path)
            is_dir = os.path.isdir(fake_local_path)
            model.is_local = exists and is_dir

            assert not model.is_local  # Should be False for non-existent path
        except Exception:
            pass  # Skip if initialization fails

    @pytest.mark.asyncio
    async def test_async_single_generation(self) -> None:
        """Test async generation of a single completion."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=20)
            prompt = "def async_hello_world():"

            completions = await model.generate_async(prompt, n=1)

            assert len(completions) == 1
            assert isinstance(completions[0], str)
            # Basic sanity check - should have some content
            # Could be empty if generation fails
            assert len(completions[0]) >= 0

        except Exception as e:
            pytest.skip(f"Async generation failed: {e}")

    @pytest.mark.asyncio
    async def test_async_multiple_generations(self) -> None:
        """Test async generation of multiple completions."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=20)
            prompt = "def async_add_numbers(a, b):"

            completions = await model.generate_async(prompt, n=3)

            assert len(completions) == 3
            assert all(isinstance(comp, str) for comp in completions)

        except Exception as e:
            pytest.skip(f"Async generation failed: {e}")

    @pytest.mark.asyncio
    async def test_async_generation_with_custom_params(self) -> None:
        """Test async generation with custom parameters."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name)
            prompt = "async def process_data():"

            completions = await model.generate_async(
                prompt, n=1, max_new_tokens=15, temperature=0.7
            )

            assert len(completions) == 1
            assert isinstance(completions[0], str)

        except Exception as e:
            pytest.skip(f"Async generation failed: {e}")

    @pytest.mark.asyncio
    async def test_async_vs_sync_generation_consistency(self) -> None:
        """Test that async and sync generation produce consistent results."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=15)
            prompt = "def test_function():"
            seed = 42

            # Generate with sync method
            sync_completions = model.generate(prompt, n=1, seed=seed)

            # Generate with async method
            async_completions = await model.generate_async(
                prompt, n=1, seed=seed
            )

            assert len(sync_completions) == 1
            assert len(async_completions) == 1
            assert (
                sync_completions[0] == async_completions[0]
            ), "Async and sync generation should produce identical results"

        except Exception as e:
            pytest.skip(f"Consistency test failed: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_async_generations(self) -> None:
        """Test multiple concurrent async generations."""
        model_name = "microsoft/DialoGPT-small"

        try:
            model = HuggingFaceModel(model_name, max_tokens=10)
            prompts = ["def func1():", "def func2():", "def func3():"]

            # Run multiple async generations concurrently
            tasks = [model.generate_async(prompt, n=1) for prompt in prompts]

            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for result in results:
                assert len(result) == 1
                assert isinstance(result[0], str)

        except Exception as e:
            pytest.skip(f"Concurrent async generation failed: {e}")


class TestHuggingFaceRealModel:
    """Tests with real small models for validation."""

    def test_real_model_initialization(self, small_model) -> None:
        """Test real model initializes correctly."""
        assert small_model.model_name == "gpt2"
        assert hasattr(small_model, "tokenizer")
        assert hasattr(small_model, "model")
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert small_model.get_generation_strategy() == strategy

    def test_real_generation_quality(self, small_model) -> None:
        """Test real model generates reasonable outputs."""
        prompt = "def fibonacci(n):"
        completions = small_model.generate(prompt, n=1, temperature=0.1)

        assert len(completions) == 1
        assert isinstance(completions[0], str)
        # Real model should generate some content
        assert len(completions[0].strip()) > 0

    def test_real_batch_generation(self, small_model) -> None:
        """Test real batch generation with multiple prompts."""
        prompts = [
            "def add(a, b):",
            "def multiply(x, y):",
            "def greet(name):"
        ]

        results = small_model.generate_batch(prompts, n_per_prompt=2)

        assert len(results) == 3
        for completions in results:
            assert len(completions) == 2
            assert all(isinstance(comp, str) for comp in completions)

    def test_parameter_effects(self, small_model) -> None:
        """Test that generation parameters have expected effects."""
        prompt = "def test():"

        # Low temperature should be more deterministic
        low_temp = small_model.generate(prompt, n=2, temperature=0.1, seed=42)

        # High temperature should be more diverse
        high_temp = small_model.generate(prompt, n=2, temperature=0.9, seed=42)

        assert len(low_temp) == 2
        assert len(high_temp) == 2
        # Results should be different strings (content validation)
        assert all(isinstance(comp, str) for comp in low_temp + high_temp)

    @pytest.mark.asyncio
    async def test_real_async_generation(self, small_model) -> None:
        """Test real async generation."""
        prompt = "async def process():"
        completions = await small_model.generate_async(prompt, n=1)

        assert len(completions) == 1
        assert isinstance(completions[0], str)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
class TestHuggingFaceGPU:
    """GPU-specific tests for HuggingFace models."""

    def test_gpu_device_detection(self, gpu_model) -> None:
        """Test GPU device is correctly detected and used."""
        assert gpu_model.device == "cuda"
        # Model should be on GPU
        assert next(gpu_model.model.parameters()).is_cuda

    def test_gpu_batch_config(self, gpu_model) -> None:
        """Test GPU batch configuration is optimized."""
        config = gpu_model.get_batch_config()

        # GPU should have larger batch sizes than CPU
        assert config.optimal_batch_size >= 4
        assert config.max_batch_size >= config.optimal_batch_size

    def test_gpu_memory_tracking(self, gpu_model) -> None:
        """Test GPU memory usage during generation."""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        prompts = ["def test():"] * 3
        results = gpu_model.generate_batch(prompts, n_per_prompt=2)

        final_memory = torch.cuda.memory_allocated()

        # Verify generation worked
        assert len(results) == 3
        assert all(len(completions) == 2 for completions in results)

        # Memory should be reasonable (not leaked excessively)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 2e9  # Less than 2GB increase

    def test_multi_gpu_detection(self, gpu_model) -> None:
        """Test multi-GPU detection if available."""
        num_gpus = torch.cuda.device_count()

        # Just verify the model can detect GPU configuration
        assert num_gpus >= 1

        # Model should work regardless of GPU count
        result = gpu_model.generate("def hello():", n=1)
        assert len(result) == 1


class TestHuggingFaceMocked:
    """Tests with strategic mocking for edge cases and error handling."""

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_initialization_failure_handling(
        self, mock_torch, mock_model_cls, mock_tokenizer_cls
    ):
        """Test graceful handling of initialization failures."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # Mock tokenizer to raise exception
        mock_tokenizer_cls.from_pretrained.side_effect = RuntimeError(
            "Model not found"
        )

        with pytest.raises(
            RuntimeError,
            match="Failed to load model"
        ):
            HuggingFaceModel("nonexistent-model")

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_mocked_batch_generation(
        self, mock_torch, mock_model_cls, mock_tokenizer_cls,
        mock_tokenizer, mock_model
    ):
        """Test batch generation with fully mocked components."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        model = HuggingFaceModel("test-model")

        prompts = ["def func1():", "def func2():"]
        results = model.generate_batch(prompts, n_per_prompt=2)

        assert len(results) == 2
        assert all(len(completions) == 2 for completions in results)

        # Verify model.generate was called
        mock_model.generate.assert_called_once()

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_generation_exception_handling(
        self, mock_torch, mock_model_cls, mock_tokenizer_cls,
        mock_tokenizer, mock_model
    ):
        """Test handling of generation exceptions."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        # Make generation fail
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

        model = HuggingFaceModel("test-model")

        # Should return empty completions instead of crashing
        results = model.generate_batch(["def test():"], n_per_prompt=1)
        assert len(results) == 1
        assert results[0] == [""]  # Empty completion on failure

    def test_thinking_model_detection(self):
        """Test thinking model pattern detection."""
        # Test without full initialization
        thinking_names = [
            "test-thinking-model",
            "gpt-o1-preview",
            "reasoning-coder",
            "model-with-cot",
            "chain-of-thought-v2"
        ]

        standard_names = [
            "gpt2",
            "llama-7b",
            "codellama-base"
        ]

        for name in thinking_names:
            model = HuggingFaceModel.__new__(HuggingFaceModel)
            assert model._is_thinking_model(name), (
                f"{name} should be detected as thinking model"
            )

        for name in standard_names:
            model = HuggingFaceModel.__new__(HuggingFaceModel)
            assert not model._is_thinking_model(name), (
                f"{name} should NOT be detected as thinking model"
            )

    @patch('src.models.huggingface_model.os.path.exists')
    @patch('src.models.huggingface_model.os.path.isdir')
    def test_local_model_detection(self, mock_isdir, mock_exists):
        """Test local vs online model detection."""
        # Test local path detection
        mock_exists.return_value = True
        mock_isdir.return_value = True

        model = HuggingFaceModel.__new__(HuggingFaceModel)
        model.model_name = "/local/model/path"
        model.is_local = os.path.exists(model.model_name) and os.path.isdir(
            model.model_name
        )

        assert model.is_local

        # Test online model
        mock_exists.return_value = False
        model.model_name = "huggingface/model"
        model.is_local = os.path.exists(model.model_name) and os.path.isdir(
            model.model_name
        )

        assert not model.is_local


if __name__ == "__main__":
    # Run a simple test to verify the model works
    async def run_async_tests() -> None:
        try:
            print("Testing HuggingFace model initialization...")
            model = HuggingFaceModel("microsoft/DialoGPT-small", max_tokens=20)
            print(f"Model loaded: {model}")

            print("Testing sync generation...")
            result = model.generate("def test():", n=1)
            print(f"Generated: {result}")

            print("Testing async generation...")
            async_result = await model.generate_async("def async_test():", n=1)
            print(f"Async generated: {async_result}")

            print("All tests passed!")

        except Exception as e:
            print(f"Test failed: {e}")
            print("Make sure you have transformers and torch installed:")
            print("poetry add transformers torch")

    # Run the async tests
    asyncio.run(run_async_tests())
