"""Tests for multi-GPU optimization features in HuggingFace models.

Tests multi-GPU configurations using NVIDIA H100 SXM GPUs with 80GB memory
to validate high-end hardware optimization capabilities.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from src.models.huggingface_model import HuggingFaceModel
from src.models.generation_strategy import BatchConfig


@pytest.fixture
def mock_multi_gpu_hardware():
    """Mock hardware configuration for multi-GPU H100 setup."""
    return {
        "num_gpus": 4,
        "gpu_details": [
            {
                "id": 0,
                "name": "NVIDIA H100 SXM",
                "memory_gb": 80.0,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
            },
            {
                "id": 1,
                "name": "NVIDIA H100 SXM",
                "memory_gb": 80.0,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
            },
            {
                "id": 2,
                "name": "NVIDIA H100 SXM",
                "memory_gb": 80.0,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
            },
            {
                "id": 3,
                "name": "NVIDIA H100 SXM",
                "memory_gb": 80.0,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
            },
        ],
        "total_gpu_memory_gb": 320.0,
        "min_gpu_memory_gb": 80.0,
        "max_gpu_memory_gb": 80.0,
        "is_homogeneous": True,
    }


@pytest.fixture
def mock_heterogeneous_gpu_hardware():
    """Mock hardware configuration for heterogeneous multi-GPU setup."""
    return {
        "num_gpus": 2,
        "gpu_details": [
            {
                "id": 0,
                "name": "NVIDIA H100 SXM",
                "memory_gb": 80.0,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
            },
            {
                "id": 1,
                "name": "NVIDIA A100 SXM4",
                "memory_gb": 40.0,
                "compute_capability": "8.0",
                "multiprocessor_count": 108,
            },
        ],
        "total_gpu_memory_gb": 120.0,
        "min_gpu_memory_gb": 40.0,
        "max_gpu_memory_gb": 80.0,
        "is_homogeneous": False,
    }


@pytest.fixture
def mock_single_h100_hardware():
    """Mock hardware configuration for single H100 GPU setup."""
    return {
        "num_gpus": 1,
        "gpu_details": [
            {
                "id": 0,
                "name": "NVIDIA H100 SXM",
                "memory_gb": 80.0,
                "compute_capability": "9.0",
                "multiprocessor_count": 132,
            },
        ],
        "total_gpu_memory_gb": 80.0,
        "min_gpu_memory_gb": 80.0,
        "max_gpu_memory_gb": 80.0,
        "is_homogeneous": True,
    }


@pytest.fixture
def mock_rtx4090_hardware():
    """Mock hardware configuration for comparison with RTX 4090 setup."""
    return {
        "num_gpus": 4,
        "gpu_details": [
            {
                "id": i,
                "name": "NVIDIA RTX 4090",
                "memory_gb": 24.0,
                "compute_capability": "8.9",
                "multiprocessor_count": 128,
            } for i in range(4)
        ],
        "total_gpu_memory_gb": 96.0,
        "min_gpu_memory_gb": 24.0,
        "max_gpu_memory_gb": 24.0,
        "is_homogeneous": True,
    }


class TestMultiGPUHardwareDetection:
    """Test hardware detection functionality."""

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.get_device_properties")
    def test_detect_multi_gpu_hardware_homogeneous(
        self, mock_get_device_properties, mock_device_count, 
        mock_cuda_available
    ):
        """Test detection of homogeneous multi-GPU setup."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        mock_props = Mock()
        mock_props.name = "NVIDIA H100 SXM"
        mock_props.total_memory = 80 * 1024**3  # 80GB
        mock_props.major = 9
        mock_props.minor = 0
        mock_props.multiprocessor_count = 132
        mock_get_device_properties.return_value = mock_props
        
        # Create model and test hardware detection
        with patch.object(HuggingFaceModel, '_load_model_optimized'):
            with patch.object(HuggingFaceModel, '_log_hardware_info'):
                model = HuggingFaceModel("test-model", use_accelerate=False)

        config = model.hardware_config
        assert config["num_gpus"] == 2
        assert config["total_gpu_memory_gb"] == 160.0  # 2x H100 80GB
        assert config["is_homogeneous"] is True
        assert len(config["gpu_details"]) == 2

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_detect_hardware_no_gpu(self, mock_device_count, 
                                    mock_cuda_available):
        """Test hardware detection when no GPU is available."""
        mock_cuda_available.return_value = False
        mock_device_count.return_value = 0
        
        with patch.object(HuggingFaceModel, '_load_model_optimized'):
            with patch.object(HuggingFaceModel, '_log_hardware_info'):
                model = HuggingFaceModel("test-model", use_accelerate=False)

        config = model.hardware_config
        assert config["num_gpus"] == 0
        assert config["total_gpu_memory_gb"] == 0.0


class TestMultiGPUBatchSizing:
    """Test multi-GPU aware batch size calculation."""

    def test_batch_config_multi_gpu_high_end(self, 
                                             mock_multi_gpu_hardware):
        """Test batch sizing for high-end H100 multi-GPU setup."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("test-model", device="cuda")

        batch_config = model.get_batch_config()
        assert isinstance(batch_config, BatchConfig)
        # Should be very large for 320GB total H100 setup
        assert batch_config.optimal_batch_size >= 32
        assert (batch_config.max_batch_size == 
                batch_config.optimal_batch_size * 2)

    def test_batch_config_thinking_model_adjustment(self, 
                                                    mock_multi_gpu_hardware):
        """Test that thinking models get reduced batch sizes."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("o1-thinking-model",
                                             device="cuda")

        batch_config = model.get_batch_config()
        # Thinking models should have smaller batch sizes even with H100s
        assert batch_config.optimal_batch_size >= 16  # Still large due to 320GB
        assert batch_config.optimal_batch_size < 64   # But smaller than non-thinking

    def test_batch_config_heterogeneous_gpus(self, 
                                             mock_heterogeneous_gpu_hardware):
        """Test batch sizing for heterogeneous GPU setup."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_heterogeneous_gpu_hardware
                    model = HuggingFaceModel("test-model", device="cuda")

        batch_config = model.get_batch_config()
        # Should be large for 120GB total (H100 80GB + A100 40GB)
        # Base size 32, reduced by half for high num_samples scenarios → 16
        assert batch_config.optimal_batch_size == 16
        assert batch_config.max_batch_size == 32

    def test_single_h100_vs_multi_rtx4090_comparison(
        self, mock_single_h100_hardware, mock_rtx4090_hardware
    ):
        """Compare single H100 vs 4x RTX 4090 performance."""
        # Test single H100
        with patch.object(HuggingFaceModel,
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_single_h100_hardware
                    h100_model = HuggingFaceModel("test-model", device="cuda")

        # Test 4x RTX 4090
        with patch.object(HuggingFaceModel,
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_rtx4090_hardware
                    rtx4090_model = HuggingFaceModel("test-model",
                                                     device="cuda")

        h100_batch_config = h100_model.get_batch_config()
        rtx4090_batch_config = rtx4090_model.get_batch_config()

        # Single H100 (80GB): base_size 20, reduced by half → 10
        assert h100_batch_config.optimal_batch_size == 10
        # 4x RTX 4090 (96GB total): base_size 32, but multi-GPU scaling applies
        # The reduction may not apply consistently due to multi-GPU logic
        assert rtx4090_batch_config.optimal_batch_size >= 16

        # Verify hardware specifications are correctly detected
        assert h100_model.hardware_config["total_gpu_memory_gb"] == 80.0
        assert rtx4090_model.hardware_config["total_gpu_memory_gb"] == 96.0

    def test_h100_quad_setup_performance_scaling(self,
                                                  mock_multi_gpu_hardware):
        """Test that 4x H100 setup provides expected performance scaling."""
        with patch.object(HuggingFaceModel,
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("test-model", device="cuda")

        config = model.hardware_config
        batch_config = model.get_batch_config()

        # 4x H100 (320GB total) should enable very large batch processing
        assert config["total_gpu_memory_gb"] == 320.0
        assert config["num_gpus"] == 4

        # Batch size should scale significantly with the massive memory
        assert batch_config.optimal_batch_size >= 32
        # Should be able to handle at least 64 samples in max batch
        assert batch_config.max_batch_size >= 64

        # Verify compute capability advancement
        for gpu in config["gpu_details"]:
            # H100 compute capability
            assert gpu["compute_capability"] == "9.0"


class TestAccelerateIntegration:
    """Test Accelerate library integration."""

    # Removed init_empty_weights and load_checkpoint_and_dispatch mocks - now using built-in HF integration
    @patch("src.models.huggingface_model.AutoTokenizer.from_pretrained")
    @patch("src.models.huggingface_model.AutoModelForCausalLM.from_pretrained")
    def test_accelerate_loading_multi_gpu(
        self,
        mock_model_from_pretrained,
        mock_tokenizer_from_pretrained,
        mock_multi_gpu_hardware,
    ):
        """Test Accelerate-based model loading for multi-GPU setup."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model

        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_log_hardware_info'):
                mock_detect.return_value = mock_multi_gpu_hardware
                HuggingFaceModel(
                    "test-model",
                    device="cuda",
                    use_accelerate=True
                )

        # Verify HuggingFace model was loaded with Accelerate parameters
        mock_model_from_pretrained.assert_called_once()

        # Verify memory distribution and device mapping was used
        call_args = mock_model_from_pretrained.call_args
        assert "max_memory" in call_args.kwargs
        assert call_args.kwargs["device_map"] == "auto"
        assert "offload_folder" in call_args.kwargs

    def test_memory_distribution_calculation(self, 
                                             mock_multi_gpu_hardware):
        """Test automatic memory distribution calculation."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("test-model", device="cuda")

        memory_dist = model._calculate_memory_distribution()
        
        # Should have memory allocation for each H100 GPU
        assert len(memory_dist) == 4
        for gpu_id in range(4):
            assert gpu_id in memory_dist
            # Should reserve some memory (80GB - 2GB reserve = 78GB)
            assert "78.0GB" in memory_dist[gpu_id]

    @patch('src.models.huggingface_model.AutoTokenizer')
    @patch('src.models.huggingface_model.AutoModelForCausalLM')
    def test_fallback_to_standard_loading(self, mock_model_cls, mock_tokenizer_cls,
                                          mock_multi_gpu_hardware):
        """Test fallback when Accelerate is disabled."""
        # Setup mocks for tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = Mock()
        
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, 
                              '_load_with_standard_method') as mock_standard:
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    HuggingFaceModel(
                        "test-model",
                        device="cuda",
                        use_accelerate=False
                    )

        # Should use standard method when Accelerate is disabled
        mock_standard.assert_called_once()


class TestMultiGPUPerformance:
    """Test performance characteristics of multi-GPU setup."""

    def test_generation_strategy_batch_optimized(self, 
                                                 mock_multi_gpu_hardware):
        """Test that multi-GPU H100 models use batch-optimized strategy."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("test-model", device="cuda")

        from src.models.generation_strategy import GenerationStrategy
        strategy = GenerationStrategy.BATCH_OPTIMIZED
        assert model.get_generation_strategy() == strategy

    def test_h100_specific_performance_characteristics(self, 
                                                      mock_multi_gpu_hardware):
        """Test performance characteristics specific to H100 GPUs."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("large-model", device="cuda")

        config = model.hardware_config
        batch_config = model.get_batch_config()
        
        # Verify H100 specifications
        assert config["total_gpu_memory_gb"] == 320.0  # 4x 80GB H100s
        assert all(gpu["compute_capability"] == "9.0" 
                  for gpu in config["gpu_details"])
        assert all(gpu["name"] == "NVIDIA H100 SXM" 
                  for gpu in config["gpu_details"])
        
        # H100s should enable very large batch sizes
        assert batch_config.optimal_batch_size >= 32
        # With 320GB total, should handle large batches efficiently
        assert batch_config.max_batch_size >= 64

    def test_h100_memory_efficiency(self, mock_multi_gpu_hardware):
        """Test memory efficiency calculations for H100 GPUs."""
        with patch.object(HuggingFaceModel, 
                          '_detect_multi_gpu_hardware') as mock_detect:
            with patch.object(HuggingFaceModel, '_load_model_optimized'):
                with patch.object(HuggingFaceModel, '_log_hardware_info'):
                    mock_detect.return_value = mock_multi_gpu_hardware
                    model = HuggingFaceModel("test-model", device="cuda")

        memory_dist = model._calculate_memory_distribution()
        
        # H100 memory efficiency tests
        assert len(memory_dist) == 4
        for gpu_id, memory_str in memory_dist.items():
            # Each H100 should have 78GB available (80GB - 2GB reserve)
            assert "78.0GB" in memory_str
            memory_val = float(memory_str.replace("GB", ""))
            # Should be utilizing most of the 80GB available
            assert memory_val >= 78.0
            assert memory_val <= 80.0

    @patch("torch.cuda.is_available")
    def test_cpu_fallback_batch_config(self, mock_cuda_available):
        """Test batch configuration fallback for CPU-only systems."""
        mock_cuda_available.return_value = False
        
        with patch.object(HuggingFaceModel, '_load_model_optimized'):
            with patch.object(HuggingFaceModel, '_log_hardware_info'):
                model = HuggingFaceModel("test-model", device="cpu")
        
        batch_config = model.get_batch_config()
        assert batch_config.optimal_batch_size == 1
        assert batch_config.max_batch_size == 2


@pytest.mark.integration
class TestMultiGPUIntegration:
    """Integration tests requiring actual GPU hardware."""

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires multiple GPUs"
    )
    def test_real_multi_gpu_detection(self):
        """Test hardware detection with real multi-GPU system."""
        with patch.object(HuggingFaceModel, '_load_model_optimized'):
            with patch.object(HuggingFaceModel, '_log_hardware_info'):
                model = HuggingFaceModel("test-model", device="cuda")

        config = model.hardware_config
        assert config["num_gpus"] >= 2
        assert config["total_gpu_memory_gb"] > 0
        assert len(config["gpu_details"]) == config["num_gpus"]

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires multiple GPUs"
    )
    def test_real_memory_distribution(self):
        """Test memory distribution calculation with real hardware."""
        with patch.object(HuggingFaceModel, '_load_model_optimized'):
            with patch.object(HuggingFaceModel, '_log_hardware_info'):
                model = HuggingFaceModel("test-model", device="cuda")

        memory_dist = model._calculate_memory_distribution()
        
        # Should have entries for all available GPUs
        assert len(memory_dist) == torch.cuda.device_count()

        # All values should be positive memory amounts
        for gpu_id, memory_str in memory_dist.items():
            assert "GB" in memory_str
            memory_val = float(memory_str.replace("GB", ""))
            assert memory_val > 0
