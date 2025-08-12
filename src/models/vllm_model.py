"""vLLM-based model wrapper for high-performance inference on multi-GPU."""

import logging
from typing import List, Any, Dict

import psutil
import torch
import GPUtil
from vllm import LLM, SamplingParams

from .base import BaseModel
from .chat_template_utils import ChatTemplateHandler
from .generation_strategy import (
    GenerationStrategy,
    BatchGenerator,
    BatchConfig
)
from ..constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_THINKING_TOKENS,
)
from ..utils.thinking_detector import OpenSourceThinkingDetector


class VLLMModel(BaseModel, BatchGenerator):
    """vLLM-based model for high-performance multi-GPU inference."""

    logger = logging.getLogger(__name__)

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize vLLM model with optimized multi-GPU settings.

        Args:
            model_name: Model identifier (HuggingFace model name or local path)
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)

        # Hardware detection and optimization
        self.hardware_config = self._detect_hardware()
        self.logger.info(f"Detected hardware: {self.hardware_config}")

        # Model configuration
        self.is_thinking = OpenSourceThinkingDetector.is_thinking_model(model_name)

        # Set max_tokens from kwargs or use default based on model type
        user_max_tokens = kwargs.get("max_tokens")
        self.max_tokens = OpenSourceThinkingDetector.get_default_max_tokens(
            model_name, user_override=user_max_tokens
        )

        # Initialize single optimized engine for batch processing
        self.llm = self._initialize_engine(**kwargs)

        # Initialize chat template handler
        self.chat_handler = ChatTemplateHandler(self.model_name, self.logger)

    def get_generation_strategy(self) -> GenerationStrategy:
        """VLLM is optimized for batch processing."""
        return GenerationStrategy.BATCH_OPTIMIZED

    def get_batch_config(self) -> BatchConfig:
        """Calculate optimal batch size based on GPU memory and model size."""
        optimal_size = self._calculate_optimal_batch_size()
        return BatchConfig(
            optimal_batch_size=optimal_size,
            max_batch_size=optimal_size * 2
        )

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        if self.hardware_config["num_gpus"] == 0:
            return 1  # CPU fallback

        try:
            # Use conservative estimates based on GPU memory
            gpu_memory_gb = self.hardware_config.get("total_gpu_memory_gb", 8)
            if gpu_memory_gb > 20:
                return 16
            elif gpu_memory_gb > 10:
                return 8
            else:
                return 4
        except Exception:
            return 4  # Safe fallback

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware resources for optimization."""
        config = {
            "num_gpus": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "gpu_memory_gb": [],
            "gpu_names": [],
            "cpu_cores": psutil.cpu_count(),
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
        }

        if config["num_gpus"] > 0:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                # Convert to GB
                config["gpu_memory_gb"].append(gpu.memoryTotal / 1024)
                config["gpu_names"].append(gpu.name)

        return config



    def _optimize_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU memory."""
        if not self.hardware_config["gpu_memory_gb"]:
            return 1

        # Conservative estimate: ~2GB per sequence for large models
        min_gpu_memory = min(self.hardware_config["gpu_memory_gb"])
        # Use 40% of memory
        estimated_batch_size = max(1, int(min_gpu_memory * 0.4))

        msg = f"Estimated optimal batch size: {estimated_batch_size}"
        self.logger.info(msg)
        return estimated_batch_size

    def _initialize_engine(self, **kwargs: Any) -> LLM:
        """Initialize single optimized vLLM engine for batch processing."""
        num_gpus = self.hardware_config["num_gpus"]

        # Tensor parallelism: use all available GPUs
        # vLLM supports up to 8-way TP
        tensor_parallel_size = min(num_gpus, 8)

        # Pipeline parallelism for very large models
        pipeline_parallel_size = 1
        if num_gpus > 8:
            pipeline_parallel_size = num_gpus // 8
            tensor_parallel_size = 8

        # GPU memory utilization
        gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.85)

        # Max model length optimization
        # max_model_len should be larger than max_tokens to account for input tokens
        # Rule: max_model_len = max_output_tokens + reasonable_input_buffer
        max_model_len = kwargs.get("max_model_len", None)
        if max_model_len is not None:
            self.logger.warning(f"max_model_len explicitly set to {max_model_len}, bypassing hardware-optimized defaults")
        elif self.is_thinking:
                # Thinking models: 8192 output + 4096 input buffer = 12288 total
                max_model_len = DEFAULT_MAX_THINKING_TOKENS + (DEFAULT_MAX_THINKING_TOKENS // 2)
        else:
            # Use hardware-appropriate context length for large GPU setups
            total_gpu_memory = sum(self.hardware_config.get("gpu_memory_gb", [8]))
            self.logger.info(f"Detected total GPU memory: {total_gpu_memory:.1f}GB")
            
            if total_gpu_memory > 500:  # 8x 80GB setup
                max_model_len = 16384  # Optimal for high-end multi-GPU setups
                self.logger.info(f"Using high-end GPU config: max_model_len={max_model_len}")
            elif total_gpu_memory > 200:  # 4x 80GB or 8x 40GB setup
                max_model_len = 8192   # Good for mid-range multi-GPU setups
                self.logger.info(f"Using mid-range GPU config: max_model_len={max_model_len}")
            else:
                # Standard models: 512 output + 1536 input buffer = 2048 total
                max_model_len = DEFAULT_MAX_TOKENS + (DEFAULT_MAX_TOKENS * 3)
                self.logger.info(f"Using standard GPU config: max_model_len={max_model_len}")

        try:
            # Configure download directory to avoid file locking issues
            download_dir = kwargs.get("download_dir", None)
            if download_dir is None:
                # Use persistent cache instead of /tmp to avoid multi-process conflicts
                import os
                download_dir = os.path.expanduser("~/.cache/huggingface")
            
            llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                # data_parallel_size=8,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                download_dir=download_dir,
                enforce_eager=kwargs.get("enforce_eager", False),
                enable_chunked_prefill=kwargs.get(
                    "enable_chunked_prefill", True
                ),
                max_num_batched_tokens=kwargs.get(
                    "max_num_batched_tokens", None
                ),
                enable_prefix_caching=kwargs.get(
                    "enable_prefix_caching", True
                ),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                dtype=kwargs.get("dtype", "auto"),
                quantization=kwargs.get("quantization", None),
                seed=kwargs.get("seed", DEFAULT_SEED),
            )

            msg = (
                f"vLLM engine initialized: TP={tensor_parallel_size}, "
                f"PP={pipeline_parallel_size}, "
                f"GPU_util={gpu_memory_utilization}, max_model_len={max_model_len}, "
                f"max_new_tokens={self.max_tokens}"
            )
            self.logger.info(msg)
            return llm

        except Exception as e:
            error_msg = (
                f"Failed to initialize vLLM engine for {self.model_name}: "
                f"{str(e)}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_sampling_params(self, n: int = 1, **kwargs: Any) -> SamplingParams:
        """Create vLLM sampling parameters.
        
        Args:
            n: Number of completions to generate per prompt
            **kwargs: Additional generation parameters
        """
        return SamplingParams(
            n=n,  # Use vLLM's native n parameter for multiple completions
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),  # Max NEW tokens to generate
            seed=kwargs.get("seed", DEFAULT_SEED),
            stop=kwargs.get("stop", None),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs: Any
    ) -> List[List[str]]:
        """Generate completions for multiple prompts efficiently.

        Uses vLLM's native batching and n parameter for diverse outputs.

        Args:
            prompts: List of input prompts
            n_per_prompt: Number of completions per prompt
            **kwargs: Additional generation parameters

        Returns:
            List where each element is a list of n_per_prompt
            completions for the corresponding prompt
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")

        try:
            # Format all prompts using chat template if supported
            formatted_prompts = [
                self.chat_handler.format_as_conversation(p) for p in prompts
            ]

            # Use vLLM's native n parameter for multiple diverse completions
            sampling_params = self._create_sampling_params(n=n_per_prompt, **kwargs)

            # Single efficient vLLM batch call using native n parameter
            outputs = self.llm.generate(formatted_prompts, sampling_params)

            # Extract results using vLLM's native multiple completion structure
            return self._extract_native_outputs(outputs, len(prompts), n_per_prompt)

        except Exception as e:
            self.logger.error(f"vLLM batch generation failed: {str(e)}")
            # Return empty completions to maintain structure
            return [[""] * n_per_prompt for _ in prompts]

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions for single prompt (fallback method)."""
        batch_results = self.generate_batch([prompt], n, **kwargs)
        return batch_results[0]

    def _extract_native_outputs(
        self,
        outputs: List[Any],
        num_prompts: int,
        n_per_prompt: int
    ) -> List[List[str]]:
        """Extract outputs from vLLM's native multiple completion structure.
        
        When using SamplingParams(n=N), vLLM returns one RequestOutput per prompt,
        but each RequestOutput contains N CompletionOutput objects.
        """
        results = []
        
        for i, output in enumerate(outputs):
            prompt_completions = []
            
            # Each output should have n_per_prompt completions
            if output.outputs:
                for completion_output in output.outputs:
                    completion_text = completion_output.text.strip()
                    prompt_completions.append(completion_text)
            
            # Ensure we have exactly n_per_prompt completions (pad with empty if needed)
            while len(prompt_completions) < n_per_prompt:
                prompt_completions.append("")
                
            # Truncate if we somehow got more (shouldn't happen with vLLM)
            prompt_completions = prompt_completions[:n_per_prompt]
            
            results.append(prompt_completions)
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model and hardware information."""
        return {
            "model_name": self.model_name,
            "model_type": "vllm",
            "generation_strategy": self.get_generation_strategy().value,
            "is_thinking": self.is_thinking,
            "supports_chat_template": self.chat_handler.supports_chat_template,
            "hardware_config": self.hardware_config,
            "batch_config": {
                "optimal_batch_size": (
                    self.get_batch_config().optimal_batch_size
                ),
                "max_batch_size": self.get_batch_config().max_batch_size
            }
        }

    def __str__(self) -> str:
        """String representation of the model."""
        hardware_info = f"{self.hardware_config['num_gpus']}xGPU"
        thinking_type = "thinking" if self.is_thinking else "standard"
        chat_type = (
            ("chat" if self.chat_handler.supports_chat_template
             else "completion")
        )
        batch_size = self.get_batch_config().optimal_batch_size
        return (
            f"VLLMModel({self.model_name}, {hardware_info}, "
            f"batch={batch_size}, {thinking_type}, {chat_type})"
        )

    def __del__(self) -> None:
        """Cleanup resources."""
        try:
            if hasattr(self, "llm") and self.llm:
                # VLLM handles cleanup automatically
                pass
        except Exception:
            pass  # Ignore cleanup errors
