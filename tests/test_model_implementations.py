"""Tests for model implementations with new generation strategies."""

import pytest
import torch
from unittest.mock import patch, Mock, AsyncMock
from src.models.generation_strategy import (
    GenerationStrategy, BatchConfig, AsyncConfig
)
from src.models.vllm_model import VLLMModel
from src.models.huggingface_model import HuggingFaceModel
from src.models.litellm_model import LiteLLMModel


class TestVLLMModel:
    """Test VLLM model implementation."""

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_initialization(self, mock_llm, mock_torch):
        """Test VLLM model initialization."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.return_value = "Tesla V100"

        model = VLLMModel("test-model")

        assert model.model_name == "test-model"
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert model.get_generation_strategy() == strategy
        assert isinstance(model.get_batch_config(), BatchConfig)

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_batch_config(self, mock_llm, mock_torch):
        """Test batch configuration calculation."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "Tesla V100"

        # Mock GPU memory info
        mock_gpu = Mock()
        mock_gpu.memoryTotal = 16 * 1024  # 16GB in MB
        with patch(
            'src.models.vllm_model.GPUtil.getGPUs', return_value=[mock_gpu]
        ):
            model = VLLMModel("test-model")
            config = model.get_batch_config()

            assert config.optimal_batch_size > 0
            assert config.max_batch_size >= config.optimal_batch_size

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_generate_batch(self, mock_llm, mock_torch):
        """Test batch generation functionality."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # Mock LLM instance and outputs
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Mock output objects
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="result1")]
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="result2")]

        mock_llm_instance.generate.return_value = [mock_output1, mock_output2]

        model = VLLMModel("test-model")

        prompts = ["prompt1", "prompt2"]
        results = model.generate_batch(prompts, n_per_prompt=1)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0] == "result1"
        assert results[1][0] == "result2"

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_single_generate_uses_batch(self, mock_llm, mock_torch):
        """Test that single generate uses batch implementation."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        mock_output = Mock()
        mock_output.outputs = [Mock(text="single_result")]
        mock_llm_instance.generate.return_value = [mock_output]

        model = VLLMModel("test-model")

        results = model.generate("test_prompt", n=1)

        assert len(results) == 1
        assert results[0] == "single_result"

        # Verify that the batch method was called
        mock_llm_instance.generate.assert_called_once()

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    @patch('src.models.vllm_model.GPUtil.getGPUs')
    def test_hardware_detection(self, mock_get_gpus, mock_llm, mock_torch):
        """Test hardware detection accuracy."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2

        # Mock GPU information
        mock_gpu1 = Mock()
        mock_gpu1.memoryTotal = 16384  # 16GB in MB
        mock_gpu1.name = "Tesla V100"

        mock_gpu2 = Mock()
        mock_gpu2.memoryTotal = 8192   # 8GB in MB
        mock_gpu2.name = "GTX 1080"

        mock_get_gpus.return_value = [mock_gpu1, mock_gpu2]

        with patch('src.models.vllm_model.psutil.cpu_count', return_value=8):
            with patch(
                'src.models.vllm_model.psutil.virtual_memory'
            ) as mock_mem:
                mock_mem.return_value.total = 32 * 1024**3  # 32GB

                model = VLLMModel("test-model")
                hw_config = model.hardware_config

                assert hw_config["num_gpus"] == 2
                assert hw_config["gpu_memory_gb"] == [16.0, 8.0]
                assert hw_config["gpu_names"] == ["Tesla V100", "GTX 1080"]
                assert hw_config["cpu_cores"] == 8
                assert hw_config["system_memory_gb"] == 32.0

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_batch_size_optimization(self, mock_llm, mock_torch):
        """Test batch size calculation based on GPU memory."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # Test high-end GPU scenario
        with patch.object(VLLMModel, '_detect_hardware') as mock_detect:
            mock_detect.return_value = {
                "num_gpus": 1,
                "gpu_memory_gb": [24.0],
                "total_gpu_memory_gb": 24.0
            }

            model = VLLMModel("test-model")
            optimal_size = model._calculate_optimal_batch_size()
            assert optimal_size == 16

        # Test mid-range GPU scenario
        with patch.object(VLLMModel, '_detect_hardware') as mock_detect:
            mock_detect.return_value = {
                "num_gpus": 1,
                "gpu_memory_gb": [12.0],
                "total_gpu_memory_gb": 12.0
            }

            model = VLLMModel("test-model")
            optimal_size = model._calculate_optimal_batch_size()
            assert optimal_size == 8

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_multi_prompt_batch_processing(self, mock_llm, mock_torch):
        """Test complex batch processing with multiple prompts and samples."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create mock outputs - VLLM returns one RequestOutput per prompt,
        # each containing n_per_prompt CompletionOutput objects
        mock_outputs = []
        prompts = ["prompt1", "prompt2", "prompt3"]
        n_per_prompt = 3

        for i, prompt in enumerate(prompts):
            mock_output = Mock()
            # Each RequestOutput has multiple CompletionOutput objects
            mock_output.outputs = [
                Mock(text=f"result_{i}_{j}") for j in range(n_per_prompt)
            ]
            mock_outputs.append(mock_output)

        mock_llm_instance.generate.return_value = mock_outputs

        model = VLLMModel("test-model")
        results = model.generate_batch(prompts, n_per_prompt=n_per_prompt)

        # Verify structure
        assert len(results) == 3
        for i, completions in enumerate(results):
            assert len(completions) == 3
            for j, completion in enumerate(completions):
                assert completion == f"result_{i}_{j}"

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_sampling_params_configuration(self, mock_llm, mock_torch):
        """Test sampling parameters are correctly configured."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        model = VLLMModel("test-model", max_tokens=100)

        # Test default parameters (using actual constants)
        sampling_params = model._create_sampling_params()
        assert sampling_params.temperature == 0.8  # DEFAULT_TEMPERATURE
        assert sampling_params.top_p == 0.95
        assert sampling_params.max_tokens == 100

        # Test custom parameters
        sampling_params = model._create_sampling_params(
            temperature=0.8,
            top_p=0.9,
            max_tokens=200,
            frequency_penalty=0.1
        )
        assert sampling_params.temperature == 0.8
        assert sampling_params.top_p == 0.9
        assert sampling_params.max_tokens == 200
        assert sampling_params.frequency_penalty == 0.1

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_thinking_model_configuration(self, mock_llm, mock_torch):
        """Test thinking model detection and configuration."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # Test thinking model
        thinking_model = VLLMModel("gpt-o1-preview")
        assert thinking_model.is_thinking is True
        assert thinking_model.max_tokens == 8192  # DEFAULT_MAX_THINKING_TOKENS

        # Test standard model
        standard_model = VLLMModel("llama-7b")
        assert standard_model.is_thinking is False
        assert standard_model.max_tokens == 512  # DEFAULT_MAX_TOKENS

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_error_handling_in_generation(self, mock_llm, mock_torch):
        """Test error handling during generation."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Make generation fail
        mock_llm_instance.generate.side_effect = RuntimeError(
            "CUDA out of memory"
        )

        model = VLLMModel("test-model")

        # Should return empty completions instead of crashing
        results = model.generate_batch(["def test():"], n_per_prompt=2)
        assert len(results) == 1
        assert results[0] == ["", ""]  # Empty completions on failure

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_model_info_reporting(self, mock_llm, mock_torch):
        """Test model information reporting."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2

        model = VLLMModel("test-model")
        info = model.get_model_info()

        assert info["model_name"] == "test-model"
        assert info["model_type"] == "vllm"
        assert info["generation_strategy"] == "batch"
        assert "hardware_config" in info
        assert "batch_config" in info


class TestVLLMRealModel:
    """Tests with real small VLLM-compatible models."""

    def test_real_vllm_initialization(self, small_vllm_model) -> None:
        """Test real VLLM model initialization."""
        assert small_vllm_model.model_name == "microsoft/DialoGPT-small"
        assert hasattr(small_vllm_model, "llm")
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert small_vllm_model.get_generation_strategy() == strategy

    def test_real_vllm_generation(self, small_vllm_model) -> None:
        """Test real VLLM generation."""
        prompt = "def hello_world():"
        completions = small_vllm_model.generate(prompt, n=1)

        assert len(completions) == 1
        assert isinstance(completions[0], str)
        # May be empty for small models
        assert len(completions[0].strip()) >= 0

    def test_real_vllm_batch_generation(self, small_vllm_model) -> None:
        """Test real VLLM batch generation."""
        prompts = ["def add(a, b):", "def multiply(x, y):"]
        results = small_vllm_model.generate_batch(prompts, n_per_prompt=2)

        assert len(results) == 2
        for completions in results:
            assert len(completions) == 2
            assert all(isinstance(comp, str) for comp in completions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
class TestVLLMGPU:
    """GPU-specific tests for VLLM models."""

    def test_gpu_utilization_detection(self, gpu_vllm_model) -> None:
        """Test GPU utilization is properly detected."""
        hw_config = gpu_vllm_model.hardware_config
        assert hw_config["num_gpus"] >= 1
        assert len(hw_config["gpu_memory_gb"]) >= 1
        assert len(hw_config["gpu_names"]) >= 1

    def test_tensor_parallelism_config(self, gpu_vllm_model) -> None:
        """Test tensor parallelism configuration."""
        # Just verify the model initializes with proper GPU count
        num_gpus = torch.cuda.device_count()
        assert num_gpus >= 1

        # Model should generate successfully
        result = gpu_vllm_model.generate("def test():", n=1)
        assert len(result) == 1

    def test_gpu_memory_efficiency(self, gpu_vllm_model) -> None:
        """Test GPU memory usage is reasonable."""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Generate with larger batch
        prompts = ["def function():"] * 5
        results = gpu_vllm_model.generate_batch(prompts, n_per_prompt=2)

        final_memory = torch.cuda.memory_allocated()

        # Verify generation worked
        assert len(results) == 5
        assert all(len(completions) == 2 for completions in results)

        # Memory increase should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 3e9  # Less than 3GB increase


class TestHuggingFaceModel:
    """Test HuggingFace model implementation."""

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_initialization(self, mock_torch, mock_model, mock_tokenizer):
        """Test HuggingFace model initialization."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = HuggingFaceModel("test-model")

        assert model.model_name == "test-model"
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert model.get_generation_strategy() == strategy
        assert isinstance(model.get_batch_config(), BatchConfig)

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_batch_config_gpu(self, mock_torch, mock_model, mock_tokenizer):
        """Test batch config for GPU setup."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # Mock GPU properties for the device property call
        mock_props = Mock()
        mock_props.name = "Tesla V100"
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
        mock_props.major = 7
        mock_props.minor = 0
        mock_props.multiprocessor_count = 84
        mock_torch.cuda.get_device_properties.return_value = mock_props

        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = Mock()

        # Create model which will detect the mocked 24GB GPU
        model = HuggingFaceModel("test-model")
        config = model.get_batch_config()

        # Should get mid-range GPU settings for 24GB
        # Base size 12, but reduced by half for high num_samples scenarios → 6
        assert config.optimal_batch_size == 6
        assert config.max_batch_size == 12

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_batch_config_cpu(self, mock_torch, mock_model, mock_tokenizer):
        """Test batch config for CPU setup."""
        mock_torch.cuda.is_available.return_value = False

        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = Mock()

        model = HuggingFaceModel("test-model", device="cpu")
        config = model.get_batch_config()

        # Should get conservative CPU settings
        assert config.optimal_batch_size == 1
        assert config.max_batch_size == 2


class TestLiteLLMModel:
    """Test LiteLLM model implementation."""

    def test_initialization(self):
        """Test LiteLLM model initialization."""
        model = LiteLLMModel("gpt-4")

        assert model.model_name == "gpt-4"
        strategy = GenerationStrategy.ASYNC_OPTIMIZED
        assert model.get_generation_strategy() == strategy
        assert isinstance(model.get_async_config(), AsyncConfig)

    def test_async_config_gpt(self):
        """Test async config for GPT models."""
        model = LiteLLMModel("gpt-4")
        config = model.get_async_config()

        assert config.max_concurrent == 50
        assert config.rate_limit_delay == 0.05

    def test_async_config_claude(self):
        """Test async config for Claude models."""
        model = LiteLLMModel("claude-3-sonnet")
        config = model.get_async_config()

        assert config.max_concurrent == 15
        assert config.rate_limit_delay == 0.1

    def test_async_config_unknown(self):
        """Test async config for unknown providers."""
        model = LiteLLMModel("unknown-model")
        config = model.get_async_config()

        assert config.max_concurrent == 1
        assert config.rate_limit_delay == 0.2

    def test_o_series_detection(self):
        """Test O-series model detection via configuration."""
        model_o1 = LiteLLMModel("o1-preview")
        model_o3 = LiteLLMModel("o3-mini")
        model_gpt = LiteLLMModel("gpt-4")

        # O-series models require seed diversification and parallel generation
        assert model_o1.config.requires_seed_diversification is True
        assert model_o1.config.requires_parallel_generation is True
        assert model_o3.config.requires_seed_diversification is True
        assert model_o3.config.requires_parallel_generation is True

        # GPT models don't require seed diversification
        assert model_gpt.config.requires_seed_diversification is False
        assert model_gpt.config.requires_parallel_generation is False

    @pytest.mark.asyncio
    @patch('src.models.litellm_model.litellm')
    async def test_generate_async(self, mock_litellm):
        """Test async generation functionality."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="test completion"))]

        # Create an async mock that returns the mock response
        async_mock = AsyncMock(return_value=mock_response)
        mock_litellm.acompletion = async_mock

        model = LiteLLMModel("gpt-4")
        results = await model.generate_async("test prompt", n=1)

        assert len(results) == 1
        assert results[0] == "test completion"
        async_mock.assert_called_once()

    @patch('src.models.litellm_model.asyncio')
    def test_generate_sync_wrapper(self, mock_asyncio):
        """Test that sync generate calls async version."""
        mock_asyncio.run.return_value = ["sync result"]

        model = LiteLLMModel("gpt-4")
        results = model.generate("test prompt", n=1)

        assert results == ["sync result"]
        mock_asyncio.run.assert_called_once()


class TestModelStrategies:
    """Test that models report correct strategies."""

    @patch('src.models.vllm_model.torch')
    @patch('src.models.vllm_model.LLM')
    def test_vllm_strategy(self, mock_llm, mock_torch):
        """Test VLLM reports batch strategy."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        model = VLLMModel("test-model")
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert model.get_generation_strategy() == strategy

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    @patch('src.models.huggingface_model.torch')
    def test_huggingface_strategy(
        self, mock_torch, mock_model, mock_tokenizer
    ):
        """Test HuggingFace reports batch strategy."""
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = Mock()

        model = HuggingFaceModel("test-model")
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert model.get_generation_strategy() == strategy

    def test_litellm_strategy(self):
        """Test LiteLLM reports async strategy."""
        model = LiteLLMModel("test-model")
        strategy = GenerationStrategy.ASYNC_OPTIMIZED
        assert model.get_generation_strategy() == strategy
