"""HuggingFace model wrapper for local and online code generation models."""

import os
import tempfile
import torch
from typing import List, Any, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from accelerate.utils import get_balanced_memory, infer_auto_device_map
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    # Define dummy functions for when accelerate is not available
    get_balanced_memory = None
    infer_auto_device_map = None
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
import logging


class HuggingFaceModel(BaseModel, BatchGenerator):
    """HuggingFace-based model for local and online code generation."""

    logger = logging.getLogger(__name__)

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize HuggingFace model with multi-GPU optimization.

        Args:
            model_name: Model identifier (local path or HuggingFace model name)
            **kwargs: Additional model parameters including:
                - max_memory_per_gpu: Dict[int, str] - Memory limit per GPU
                - offload_folder: str - Folder for CPU offloading
                - low_cpu_mem_usage: bool - Use low CPU memory loading
        """
        super().__init__(model_name, **kwargs)

        # Check if model_name is a local path
        self.is_local = os.path.exists(model_name) and os.path.isdir(
            model_name
        )

        # Multi-GPU configuration
        self.hardware_config = self._detect_multi_gpu_hardware()
        self.use_accelerate = kwargs.get("use_accelerate", True)
        self.max_memory_per_gpu = kwargs.get("max_memory_per_gpu", None)
        self.offload_folder = kwargs.get("offload_folder", None)
        
        # Default parameters
        self.default_params = {
            "max_new_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "do_sample": (kwargs.get("temperature", DEFAULT_TEMPERATURE) > 0),
            "pad_token_id": None,  # Will be set after loading tokenizer
        }

        # Detect thinking models based on name patterns
        self.is_thinking = OpenSourceThinkingDetector.is_thinking_model(model_name)
        if self.is_thinking:
            max_tokens = OpenSourceThinkingDetector.get_default_max_tokens(
                model_name
            )
            self.default_params["max_new_tokens"] = max_tokens
            self.logger.info(f"Using max_tokens: {max_tokens} for thinking model")

        # Device selection
        device_choice = "cuda"
        self.device = kwargs.get("device", device_choice)

        # Log comprehensive hardware info
        self._log_hardware_info()

        # Load model and tokenizer with multi-GPU optimization
        self._load_model_optimized(**kwargs)

        # Initialize chat template handler
        self.chat_handler = ChatTemplateHandler(self.model_name, self.logger)

    def get_generation_strategy(self) -> GenerationStrategy:
        """HuggingFace models are optimized for batch processing."""
        return GenerationStrategy.BATCH_OPTIMIZED

    def _is_thinking_model(self, model_name: str) -> bool:
        """Check if a model name indicates a thinking/reasoning model.
        
        Args:
            model_name: The model name to check
            
        Returns:
            True if the model is a thinking model, False otherwise
        """
        return OpenSourceThinkingDetector.is_thinking_model(model_name)

    def get_batch_config(self) -> BatchConfig:
        """Calculate optimal batch size based on total GPU memory and model size."""
        if self.device == "cpu":
            return BatchConfig(optimal_batch_size=1, max_batch_size=2)

        # Multi-GPU aware batch sizing with large model considerations
        try:
            total_gpu_memory_gb = self.hardware_config["total_gpu_memory_gb"]
            num_gpus = self.hardware_config["num_gpus"]
            
            # Very conservative batch sizes for large models (20B+)
            if self._is_very_large_model():
                if total_gpu_memory_gb > 600:  # 8x80GB H100s
                    base_size = 4  # Very conservative for 20B+ models
                elif total_gpu_memory_gb > 160:  # 8x24GB or 4x40GB+
                    base_size = 2
                elif total_gpu_memory_gb > 80:  # 4x24GB or 2x40GB+
                    base_size = 2
                elif total_gpu_memory_gb > 40:  # 2x24GB
                    base_size = 1
                else:  # Single GPU
                    base_size = 1
                    
                self.logger.info(f"Using conservative batch size for large model (20B+): {base_size}")
            else:
                # Standard batch size calculation for smaller models
                if total_gpu_memory_gb > 80:  # Multi-GPU high-end
                    base_size = 32
                elif total_gpu_memory_gb > 40:  # Multi-GPU mid-range
                    base_size = 20
                elif total_gpu_memory_gb > 20:  # Single high-end or dual mid-range
                    base_size = 12
                elif total_gpu_memory_gb > 10:  # Single mid-range
                    base_size = 8
                else:  # Single low-end
                    base_size = 4
                
            # Adjust for thinking models (they need more memory per sample)
            if self.is_thinking:
                base_size = max(1, base_size // 2)
                
            # Scale properly for multi-GPU but cap for large models
            if num_gpus > 1 and not self._is_very_large_model():
                base_size = min(base_size * num_gpus // 2, base_size * 2)
                
            # Further reduce batch size for high num_samples scenarios
            # This is a conservative approach to handle cases like --num-samples 50
            # where total sequences = batch_size × num_samples could be very large
            if base_size > 2:
                self.logger.info("Applying additional batch size reduction for high num_samples scenarios")
                base_size = max(1, base_size // 2)
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal batch size: {e}")
            base_size = 1 if self._is_very_large_model() else 4  # Safe fallback

        return BatchConfig(
            optimal_batch_size=base_size,
            max_batch_size=base_size * 2  # Ensure at least 2x for max
        )

    def _detect_multi_gpu_hardware(self) -> Dict[str, Any]:
        """Detect comprehensive multi-GPU hardware configuration."""
        config = {
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_details": [],
            "total_gpu_memory_gb": 0.0,
            "min_gpu_memory_gb": float('inf'),
            "max_gpu_memory_gb": 0.0,
            "is_homogeneous": True,
        }
        
        if config["num_gpus"] > 0:
            try:
                first_gpu_name = None
                for i in range(config["num_gpus"]):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    
                    gpu_info = {
                        "id": i,
                        "name": props.name,
                        "memory_gb": memory_gb,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                    
                    # Add multiprocessor_count if available (newer PyTorch versions)
                    if hasattr(props, 'multiprocessor_count'):
                        gpu_info["multiprocessor_count"] = props.multiprocessor_count
                    config["gpu_details"].append(gpu_info)
                    
                    # Track memory statistics
                    config["total_gpu_memory_gb"] += memory_gb
                    config["min_gpu_memory_gb"] = min(config["min_gpu_memory_gb"], memory_gb)
                    config["max_gpu_memory_gb"] = max(config["max_gpu_memory_gb"], memory_gb)
                    
                    # Check homogeneity
                    if first_gpu_name is None:
                        first_gpu_name = props.name
                    elif first_gpu_name != props.name:
                        config["is_homogeneous"] = False
                        
                # Fix infinity value for single GPU
                if config["min_gpu_memory_gb"] == float('inf'):
                    config["min_gpu_memory_gb"] = 0.0
                    
            except Exception as e:
                self.logger.warning(f"Failed to detect GPU hardware: {e}")
                
        return config

    def _log_hardware_info(self) -> None:
        """Log comprehensive hardware information for debugging."""
        config = self.hardware_config
        
        if config["num_gpus"] == 0:
            self.logger.info("Running on CPU")
            return
        
        try:
            # Log summary - handle potential mock objects safely
            homogeneous = ("homogeneous" if config["is_homogeneous"]
                          else "heterogeneous")
            
            # Safe formatting for numeric values that might be mocks
            total_memory = config["total_gpu_memory_gb"]
            if hasattr(total_memory, '_mock_name'):
                # Handle mock objects
                memory_str = str(total_memory)
            else:
                memory_str = f"{float(total_memory):.1f}GB"
            
            msg = (f"Multi-GPU setup detected: {config['num_gpus']} GPUs "
                   f"({homogeneous}), Total memory: {memory_str}")
            self.logger.info(msg)
            
            # Log individual GPU details
            for gpu in config["gpu_details"]:
                try:
                    gpu_memory = gpu["memory_gb"]
                    if hasattr(gpu_memory, '_mock_name'):
                        memory_str = str(gpu_memory)
                    else:
                        memory_str = f"{float(gpu_memory):.1f}GB"
                    
                    mp_info = ""
                    if 'multiprocessor_count' in gpu:
                        mp_count = gpu['multiprocessor_count']
                        if hasattr(mp_count, '_mock_name'):
                            mp_info = f", MPs: {mp_count}"
                        else:
                            mp_info = f", MPs: {mp_count}"
                    
                    self.logger.info(
                        f"  GPU {gpu['id']}: {gpu['name']} - {memory_str} "
                        f"(CC {gpu['compute_capability']}{mp_info})"
                    )
                except (TypeError, ValueError) as e:
                    self.logger.debug(f"Skipping GPU {gpu.get('id', '?')} logging due to mock: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Hardware info logging failed (likely due to mocking): {e}")

    def _load_model_optimized(self, **kwargs: Any) -> None:
        """Load model with Accelerate-based multi-GPU optimization."""
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.is_local,
            )

            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Fix padding side for decoder-only models (prevents generation quality issues)
            # Most modern LLMs (GPT, LLaMA, Mistral, etc.) are decoder-only
            self.tokenizer.padding_side = "left"

            self.default_params["pad_token_id"] = self.tokenizer.pad_token_id

            # Choose loading strategy based on hardware and user preference
            self.logger.info(f"Model loading conditions: use_accelerate={self.use_accelerate}, "
                           f"ACCELERATE_AVAILABLE={ACCELERATE_AVAILABLE}, "
                           f"num_gpus={self.hardware_config['num_gpus']}, "
                           f"device={self.device}")
            
            if (self.use_accelerate and
                ACCELERATE_AVAILABLE and
                self.hardware_config["num_gpus"] > 1 and
                self.device == "cuda"):
                self.logger.info("Using Accelerate multi-GPU loading strategy")
                self._load_with_accelerate(**kwargs)
            else:
                self.logger.info("Using standard PyTorch loading strategy")
                self._load_with_standard_method(**kwargs)
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            raise RuntimeError(error_msg)

    def _load_with_accelerate(self, **kwargs: Any) -> None:
        """Load model using Accelerate for optimal multi-GPU distribution."""
        self.logger.info("Loading model with Accelerate multi-GPU optimization")
        
        # Set memory allocation strategy to avoid fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # Calculate optimal memory distribution for large models
        max_memory = self._calculate_memory_distribution(**kwargs)
        
        # Ensure offload folder exists when needed
        offload_folder = self.offload_folder
        if offload_folder is None:
            # Create temporary directory for offloading if none provided
            temp_dir = tempfile.mkdtemp(prefix="model_offload_")
            offload_folder = temp_dir
            self.logger.info(f"Created temporary offload directory: {offload_folder}")
            self.logger.warning("No offload folder provided. Using temporary directory. "
                              "For better performance, consider providing a persistent offload_folder.")
        
        # Detect optimal dtype for the model
        torch_dtype = self._get_optimal_dtype(**kwargs)
        
        # Load model with Accelerate's built-in HuggingFace integration
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.is_local,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "max_memory": max_memory,
            "offload_folder": offload_folder,
            "low_cpu_mem_usage": True,  # Essential for large models
        }
        
        self.logger.info(f"Loading model with kwargs: {model_kwargs}")
        self.logger.info(f"Max memory per GPU: {max_memory}")
        self.logger.info(f"Is very large model: {self._is_very_large_model()}")
        
        # Add specific optimizations for very large models (30B+)
        if self._is_very_large_model():
            model_kwargs.update({
                "load_in_8bit": kwargs.get("load_in_8bit", False),
                "load_in_4bit": kwargs.get("load_in_4bit", False),
            })
            self.logger.info("Applied large model optimizations (30B+ parameters)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        
        # Log actual device placement
        if hasattr(self.model, 'hf_device_map'):
            self.logger.info(f"Model device map: {self.model.hf_device_map}")
        else:
            self.logger.info("No device map attribute found on model")
            
        self.logger.info(f"Model loaded across {self.hardware_config['num_gpus']} GPUs with Accelerate")
        self.logger.info(f"Model's main device: {self.model.device}")

    def _is_very_large_model(self) -> bool:
        """Detect if this is a very large model (20B+ parameters) that needs special handling."""
        large_model_patterns = [
            "20b", "30b", "33b", "34b", "40b", "65b", "70b", "72b", "180b", "480b",
            "-20b-", "-30b-", "-33b-", "-34b-", "-40b-", "-65b-", "-70b-", "-72b-", "-180b-", "-480b-"
        ]
        model_lower = self.model_name.lower()
        return any(pattern in model_lower for pattern in large_model_patterns)
    
    def _get_optimal_dtype(self, **kwargs: Any) -> torch.dtype:
        """Get optimal dtype based on model characteristics and user preferences."""
        # Handle explicit torch_dtype parameter
        torch_dtype = kwargs.get("torch_dtype", "auto")
        return torch_dtype
        
    def _calculate_memory_distribution(self, **kwargs: Any) -> Optional[Dict[int, str]]:
        """Calculate optimal memory distribution across GPUs for large models."""
        if self.max_memory_per_gpu:
            return self.max_memory_per_gpu
            
        # Auto-calculate based on available memory with adjustments for large models
        max_memory = {}
        base_reserve = kwargs.get("reserve_memory_gb", 2.0)
        
        # Increase reserved memory for very large models
        if self._is_very_large_model():
            reserve_memory_gb = max(base_reserve, 8.0)  # Reserve more for large models (increased from 4.0 to 8.0)
            self.logger.info("Increased memory reservation for large model (20B+ parameters)")
        else:
            reserve_memory_gb = base_reserve
            
        # Check if GPU 0 is heavily used and increase reservation if needed
        try:
            if torch.cuda.is_available():
                gpu0_used = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                gpu0_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu0_usage_ratio = gpu0_used / gpu0_total
                
                if gpu0_usage_ratio > 0.7:  # If GPU 0 is >70% used
                    reserve_memory_gb = max(reserve_memory_gb, 12.0)  # Reserve even more
                    self.logger.warning(f"GPU 0 is heavily used ({gpu0_usage_ratio:.1%}), "
                                      f"increasing memory reservation to {reserve_memory_gb}GB")
        except Exception as e:
            self.logger.warning(f"Could not check GPU 0 usage: {e}")
            
        self.logger.info(f"Using {reserve_memory_gb}GB memory reservation per GPU")
        
        for gpu in self.hardware_config["gpu_details"]:
            available_memory = gpu["memory_gb"] - reserve_memory_gb
            max_memory[gpu["id"]] = f"{max(1.0, available_memory):.1f}GB"
            self.logger.info(f"GPU {gpu['id']}: {gpu['memory_gb']:.1f}GB total, "
                           f"{available_memory:.1f}GB available after {reserve_memory_gb}GB reservation")
            
        self.logger.info(f"Auto-calculated memory distribution: {max_memory}")
        return max_memory
    
    def _get_safe_max_length(self, **kwargs: Any) -> int:
        """Get safe max_length for tokenization based on model size and memory."""
        # Use user-provided max_length if available
        if "max_length" in kwargs:
            return kwargs["max_length"]
        
        # Conservative max_length for large models to prevent OOM
        if self._is_very_large_model():
            # Very large models need much shorter contexts
            return 1024
        elif self.is_thinking:
            # Thinking models typically need longer contexts
            return 4096
        else:
            # Standard models
            return 2048

    def _load_with_standard_method(self, **kwargs: Any) -> None:
        """Fallback to standard PyTorch model loading with large model support."""
        self.logger.info("Loading model with standard PyTorch method")
        
        # Get optimal dtype for this model
        torch_dtype = self._get_optimal_dtype(**kwargs)
        
        # Build model kwargs with large model optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.is_local,
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,  # Essential for large models
        }
        
        # Add specific optimizations for very large models
        if self._is_very_large_model() and self.device == "cuda":
            model_kwargs.update({
                "load_in_8bit": kwargs.get("load_in_8bit", False),
                "load_in_4bit": kwargs.get("load_in_4bit", False),
            })
            self.logger.info("Applied large model optimizations (30B+ parameters)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Move to device if not using device_map (mainly for CPU)
        if self.device != "cuda" or model_kwargs["device_map"] is None:
            self.model = self.model.to(self.device)

    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs: Any
    ) -> List[List[str]]:
        """Generate completions for multiple prompts efficiently.

        Uses batch processing.

        Args:
            prompts: List of input prompts
            n_per_prompt: Number of completions per prompt
            **kwargs: Additional generation parameters

        Returns:
            List where each element is a list of n_per_prompt
            completions for the corresponding prompt
        """
        # Format all prompts using chat template if supported
        formatted_prompts = [
            self.chat_handler.format_as_conversation(p) for p in prompts
        ]

        # Prepare generation parameters
        params = {**self.default_params, **kwargs}

        # Handle thinking model adjustments
        if self.is_thinking and "max_new_tokens" not in kwargs:
            params["max_new_tokens"] = DEFAULT_MAX_THINKING_TOKENS

        # Configure for multiple outputs if needed
        if n_per_prompt > 1:
            params["do_sample"] = True
            params["num_return_sequences"] = n_per_prompt
            
            # For high num_samples (>10), use multiple smaller batches to avoid OOM
            # This implements a "gradient accumulation" style approach for inference
            if n_per_prompt > 10:
                self.logger.warning(f"High num_samples ({n_per_prompt}) detected. "
                                  f"Consider reducing --num-samples or expect slower processing.")
                # Limit num_return_sequences to prevent massive memory usage
                max_sequences_per_call = min(n_per_prompt, 8)
                params["num_return_sequences"] = max_sequences_per_call

        try:
            # Use conservative max_length for large models to avoid OOM
            max_length = self._get_safe_max_length(**kwargs)
            
            # Tokenize all prompts with padding
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.model.device)

            input_length = inputs["input_ids"].shape[1]

            # Set seed for reproducible generation
            if "seed" in kwargs or n_per_prompt > 1:
                seed = kwargs.get("seed", DEFAULT_SEED)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            # Handle high num_samples by splitting into multiple generation calls
            if n_per_prompt > 10 and n_per_prompt > params.get("num_return_sequences", n_per_prompt):
                # Multiple smaller generation calls to avoid OOM
                max_sequences_per_call = params["num_return_sequences"]
                all_outputs = []
                
                num_calls_needed = (n_per_prompt + max_sequences_per_call - 1) // max_sequences_per_call
                self.logger.info(f"Splitting {n_per_prompt} samples into {num_calls_needed} calls "
                               f"of max {max_sequences_per_call} sequences each")
                
                for call_idx in range(num_calls_needed):
                    # Calculate how many sequences for this call
                    remaining_sequences = n_per_prompt - (call_idx * max_sequences_per_call)
                    current_sequences = min(max_sequences_per_call, remaining_sequences)
                    
                    # Update params for this call
                    call_params = params.copy()
                    call_params["num_return_sequences"] = current_sequences
                    
                    # Generate for this subset
                    with torch.inference_mode():
                        if self._is_very_large_model() and hasattr(self.model, 'gradient_checkpointing_enable'):
                            self.model.gradient_checkpointing_enable()
                        
                        call_outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **call_params
                        )
                        all_outputs.append(call_outputs)
                
                # Combine all outputs
                outputs = torch.cat(all_outputs, dim=0)
            else:
                # Standard single generation call
                with torch.inference_mode():
                    # Use gradient checkpointing for large models to save memory
                    if self._is_very_large_model() and hasattr(self.model, 'gradient_checkpointing_enable'):
                        self.model.gradient_checkpointing_enable()
                    
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **params
                    )

            # Decode and reshape results
            return self._decode_batch_outputs(
                outputs, formatted_prompts, n_per_prompt, input_length
            )

        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            # Return empty completions to maintain structure
            return [[""] * n_per_prompt for _ in prompts]

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions for single prompt (fallback method)."""
        batch_results = self.generate_batch([prompt], n, **kwargs)
        return batch_results[0]

    async def generate_async(
        self, prompt: str, n: int = 1, **kwargs: Any
    ) -> List[str]:
        """Generate n completions asynchronously.

        Args:
            prompt: Input prompt for code generation
            n: Number of completions to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated code completions
        """
        # HuggingFace transformers don't support native async,
        # so we use the base class implementation with executor
        return await super().generate_async(prompt, n, **kwargs)

    def _decode_batch_outputs(
        self,
        outputs: torch.Tensor,
        prompts: List[str],
        n_per_prompt: int,
        input_length: int
    ) -> List[List[str]]:
        """Decode batch outputs and reshape to [prompt][sample] structure."""
        results = []

        if n_per_prompt == 1:
            # Simple case: one completion per prompt
            for i, output in enumerate(outputs):
                generated_tokens = output[input_length:]
                completion = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                results.append([completion])
        else:
            # Multiple completions per prompt - outputs are interleaved
            for i in range(len(prompts)):
                prompt_completions = []
                for j in range(n_per_prompt):
                    idx = i * n_per_prompt + j
                    if idx < len(outputs):
                        generated_tokens = outputs[idx][input_length:]
                        completion = self.tokenizer.decode(
                            generated_tokens, skip_special_tokens=True
                        ).strip()
                        prompt_completions.append(completion)
                    else:
                        prompt_completions.append("")
                results.append(prompt_completions)

        return results

    def __str__(self) -> str:
        """String representation of the model."""
        model_type = "local" if self.is_local else "online"
        thinking_type = "thinking" if self.is_thinking else "standard"
        chat_type = (
            ("chat" if self.chat_handler.supports_chat_template
             else "completion")
        )
        return (
            f"HuggingFaceModel({self.model_name}, {model_type}, "
            f"{thinking_type}, {chat_type})"
        )
