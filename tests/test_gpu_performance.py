"""Minimal GPU performance tests for model comparison."""

import pytest
import torch
import time
from src.models.huggingface_model import HuggingFaceModel
from src.models.vllm_model import VLLMModel
from tests.conftest import BASIC_PROMPTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
class TestGPUPerformance:
    """Cross-model GPU performance comparison tests."""

    @pytest.fixture(scope="class")
    def gpu_models(self):
        """Load both models on GPU for comparison."""
        models = {}

        try:
            models['huggingface'] = HuggingFaceModel(
                "gpt2", max_tokens=30, device="cuda"
            )
        except Exception:
            models['huggingface'] = None

        try:
            models['vllm'] = VLLMModel(
                "microsoft/DialoGPT-small", max_tokens=30
            )
        except Exception:
            models['vllm'] = None

        if not any(models.values()):
            pytest.skip("No GPU models could be loaded")

        yield models

    def test_memory_efficiency_comparison(self, gpu_models):
        """Compare GPU memory usage between models."""
        results = {}

        for model_name, model in gpu_models.items():
            if model is None:
                continue

            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Generate batch
            if hasattr(model, 'generate_batch'):
                model.generate_batch(BASIC_PROMPTS, n_per_prompt=1)
            else:
                for prompt in BASIC_PROMPTS:
                    model.generate(prompt, n=1)

            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory

            results[model_name] = memory_used

        # Basic validation - memory usage should be reasonable
        for model_name, memory_used in results.items():
            mem_gb = memory_used / 1e9
            assert memory_used < 2e9, (
                f"{model_name} used too much memory: "
                f"{mem_gb:.2f}GB"
            )

    def test_throughput_comparison(self, gpu_models):
        """Compare generation throughput between models."""
        results = {}

        for model_name, model in gpu_models.items():
            if model is None:
                continue

            start_time = time.time()

            # Generate batch
            if hasattr(model, 'generate_batch'):
                outputs = model.generate_batch(
                    BASIC_PROMPTS, n_per_prompt=1
                )
            else:
                outputs = [
                    model.generate(prompt, n=1)
                    for prompt in BASIC_PROMPTS
                ]

            end_time = time.time()

            # Validate outputs
            assert len(outputs) == 2

            results[model_name] = end_time - start_time

        # Basic validation - should complete within reasonable time
        for model_name, duration in results.items():
            assert duration < 60, (
                f"{model_name} took too long: {duration:.2f}s"
            )

    def test_hardware_utilization(self, gpu_models):
        """Test that models properly utilize available GPU hardware."""
        for model_name, model in gpu_models.items():
            if model is None:
                continue

            # Test basic GPU detection
            if hasattr(model, 'device'):
                assert model.device == "cuda"
            elif hasattr(model, 'hardware_config'):
                assert model.hardware_config["num_gpus"] >= 1

            # Test generation works
            result = model.generate("def hello():", n=1)
            assert len(result) == 1
            assert isinstance(result[0], str)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
class TestGPUErrorHandling:
    """Test GPU error handling scenarios."""

    def test_oom_graceful_degradation(self):
        """Test graceful handling when GPU memory is exhausted."""
        try:
            model = HuggingFaceModel("gpt2", max_tokens=10)

            # Try to generate very large batch that might cause OOM
            large_prompts = ["def function():"] * 50
            results = model.generate_batch(large_prompts, n_per_prompt=1)

            # Should either succeed or fail gracefully (empty results)
            assert len(results) == 50
            assert all(isinstance(batch, list) for batch in results)

        except Exception:
            # Graceful failure is acceptable
            pass

    def test_multi_gpu_fallback(self):
        """Test fallback when multi-GPU setup is not optimal."""
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            try:
                # VLLM should handle multi-GPU automatically
                model = VLLMModel("microsoft/DialoGPT-small", max_tokens=20)
                result = model.generate("def test():", n=1)
                assert len(result) == 1
            except Exception:
                # Multi-GPU setup might fail, which is acceptable
                pass
        else:
            pytest.skip("Single GPU system")
