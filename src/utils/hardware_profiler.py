"""Hardware profiling utilities for optimal model configuration."""

import logging
import time
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

import torch
import psutil
import GPUtil
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class HardwareProfiler:
    """Profile hardware to determine optimal inference parameters."""

    def __init__(self) -> None:
        """Initialize hardware profiler."""
        self.hardware_info = self._detect_hardware()

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect comprehensive hardware information."""
        info = {
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_details": [],
            "cpu_cores": psutil.cpu_count(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
            "system_memory_available_gb": psutil.virtual_memory().available
            / (1024**3),
        }

        if info["num_gpus"] > 0:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_info = {
                        "id": i,
                        "name": gpu.name,
                        "memory_total_gb": gpu.memoryTotal / 1024,
                        "memory_free_gb": gpu.memoryFree / 1024,
                        "memory_used_gb": gpu.memoryUsed / 1024,
                        "temperature": gpu.temperature,
                        "load": gpu.load,
                    }
                    info["gpu_details"].append(gpu_info)

                    # Add PyTorch GPU info
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(i)
                        gpu_pytorch_info = {
                            "compute_capability": f"{props.major}.{props.minor}",
                            "memory_total_pytorch_gb": props.total_memory / (1024**3),
                        }
                        
                        # Add properties that may not exist in older PyTorch versions
                        if hasattr(props, 'multiprocessor_count'):
                            gpu_pytorch_info["multiprocessor_count"] = props.multiprocessor_count
                        if hasattr(props, 'max_threads_per_multiprocessor'):
                            gpu_pytorch_info["max_threads_per_multiprocessor"] = props.max_threads_per_multiprocessor
                            
                        gpu_info.update(gpu_pytorch_info)

            except Exception as e:
                logger.warning(f"Failed to get detailed GPU info: {e}")

        return info

    def profile_model_performance(
        self,
        model_name: str,
        test_prompts: Optional[List[str]] = None,
        batch_sizes: Optional[List[int]] = None,
        max_tokens_options: Optional[List[int]] = None,
        tensor_parallel_sizes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Profile model performance across different configurations.

        Args:
            model_name: Model to profile
            test_prompts: List of test prompts (defaults to standard benchmarks)
            batch_sizes: Batch sizes to test (auto-generated if None)
            max_tokens_options: Max token options to test
            tensor_parallel_sizes: TP sizes to test

        Returns:
            Performance profile results
        """
        if test_prompts is None:
            test_prompts = self._get_default_test_prompts()

        if batch_sizes is None:
            batch_sizes = self._calculate_test_batch_sizes()

        if max_tokens_options is None:
            max_tokens_options = [256, 512, 1024]

        if tensor_parallel_sizes is None:
            tensor_parallel_sizes = self._calculate_test_tp_sizes()

        results = {
            "model_name": model_name,
            "hardware_info": self.hardware_info,
            "timestamp": time.time(),
            "configurations": [],
        }

        for tp_size in tensor_parallel_sizes:
            for max_tokens in max_tokens_options:
                for batch_size in batch_sizes:
                    try:
                        config_result = self._profile_configuration(
                            model_name, test_prompts, batch_size, max_tokens, tp_size
                        )
                        results["configurations"].append(config_result)

                    except Exception as e:
                        logger.error(
                            f"Failed to profile config TP={tp_size}, "
                            f"batch={batch_size}, max_tokens={max_tokens}: {e}"
                        )

        # Analyze results and provide recommendations
        results["recommendations"] = self._analyze_results(results["configurations"])
        return results

    def _get_default_test_prompts(self) -> List[str]:
        """Get default test prompts for profiling."""
        return [
            "Write a function to calculate the factorial of a number.",
            "Create a sorting algorithm for a list of integers.",
            "Implement a binary search function.",
            "Write code to reverse a string.",
            "Create a function to find the maximum element in an array.",
        ]

    def _calculate_test_batch_sizes(self) -> List[int]:
        """Calculate batch sizes to test based on available GPU memory."""
        if not self.hardware_info["gpu_details"]:
            return [1]

        min_gpu_memory = min(
            gpu["memory_total_gb"] for gpu in self.hardware_info["gpu_details"]
        )

        # Conservative batch size estimation
        max_batch = max(1, int(min_gpu_memory * 0.3))  # Use 30% of memory for batching

        batch_sizes = [1]
        if max_batch > 1:
            batch_sizes.extend([2, 4, 8, 16, 32])
            batch_sizes = [bs for bs in batch_sizes if bs <= max_batch]

        return batch_sizes

    def _calculate_test_tp_sizes(self) -> List[int]:
        """Calculate tensor parallel sizes to test."""
        num_gpus = self.hardware_info["num_gpus"]
        if num_gpus <= 1:
            return [1]

        # Test valid TP sizes (powers of 2, up to num_gpus)
        tp_sizes = [1]
        tp = 2
        while tp <= min(num_gpus, 8):  # vLLM supports up to 8-way TP
            tp_sizes.append(tp)
            tp *= 2

        return tp_sizes

    def _profile_configuration(
        self,
        model_name: str,
        test_prompts: List[str],
        batch_size: int,
        max_tokens: int,
        tp_size: int,
    ) -> Dict[str, Any]:
        """Profile a specific configuration."""
        logger.info(
            f"Profiling: TP={tp_size}, batch={batch_size}, max_tokens={max_tokens}"
        )

        config = {
            "tensor_parallel_size": tp_size,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "metrics": {},
            "error": None,
        }

        try:
            # Initialize model with this configuration
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=0.8,  # Conservative for profiling
                max_model_len=4096,
                trust_remote_code=True,
            )

            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=max_tokens,
            )

            # Warm up
            warmup_prompts = test_prompts[: min(2, len(test_prompts))]
            llm.generate(warmup_prompts, sampling_params)

            # Measure performance
            start_time = time.time()
            start_memory = self._get_gpu_memory_usage()

            # Run test batch
            test_batch = (test_prompts * batch_size)[:batch_size]
            outputs = llm.generate(test_batch, sampling_params)

            end_time = time.time()
            end_memory = self._get_gpu_memory_usage()

            # Calculate metrics
            total_time = end_time - start_time
            tokens_generated = sum(
                len(output.outputs[0].text.split()) for output in outputs
            )

            config["metrics"] = {
                "total_time_seconds": total_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / total_time
                if total_time > 0
                else 0,
                "requests_per_second": len(test_batch) / total_time
                if total_time > 0
                else 0,
                "memory_used_gb": end_memory - start_memory,
                "peak_memory_gb": max(end_memory, start_memory),
            }

            # Cleanup
            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            config["error"] = str(e)
            logger.error(f"Configuration failed: {e}")

        return config

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return 0.0

        total_memory = 0.0
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.memory_allocated(i) / (1024**3)

        return total_memory

    def _analyze_results(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze profiling results and provide recommendations."""
        valid_configs = [
            config for config in configurations if config.get("error") is None
        ]

        if not valid_configs:
            return {"error": "No valid configurations found"}

        # Find best configuration by tokens per second
        best_throughput = max(
            valid_configs, key=lambda x: x["metrics"]["tokens_per_second"]
        )

        # Find most memory efficient
        best_memory = min(valid_configs, key=lambda x: x["metrics"]["memory_used_gb"])

        # Find balanced recommendation (good throughput, reasonable memory)
        balanced_scores = []
        for config in valid_configs:
            # Normalize metrics (0-1 scale)
            max_throughput = max(
                c["metrics"]["tokens_per_second"] for c in valid_configs
            )
            max_memory = max(c["metrics"]["memory_used_gb"] for c in valid_configs)

            throughput_score = config["metrics"]["tokens_per_second"] / max_throughput
            memory_score = (
                1 - (config["metrics"]["memory_used_gb"] / max_memory)
                if max_memory > 0
                else 1
            )

            # Weighted score (favor throughput slightly)
            balanced_score = 0.7 * throughput_score + 0.3 * memory_score
            balanced_scores.append((balanced_score, config))

        best_balanced = max(balanced_scores, key=lambda x: x[0])[1]

        return {
            "best_throughput": best_throughput,
            "best_memory_efficiency": best_memory,
            "recommended_balanced": best_balanced,
            "summary": {
                "configurations_tested": len(configurations),
                "valid_configurations": len(valid_configs),
                "max_tokens_per_second": best_throughput["metrics"][
                    "tokens_per_second"
                ],
                "min_memory_usage_gb": best_memory["metrics"]["memory_used_gb"],
            },
        }

    def save_profile_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save profiling results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Profiling results saved to {output_path}")

    def load_profile_results(self, profile_path: Path) -> Dict[str, Any]:
        """Load profiling results from file."""
        with open(profile_path, "r") as f:
            return json.load(f)

    def get_recommended_config(self, model_name: str) -> Dict[str, Any]:
        """Get recommended configuration for a model (quick estimation)."""
        num_gpus = self.hardware_info["num_gpus"]

        if num_gpus == 0:
            return {
                "tensor_parallel_size": 1,
                "batch_size": 1,
                "gpu_memory_utilization": 0.8,
                "max_tokens": 512,
                "reasoning": "No GPUs detected, using CPU fallback",
            }

        # Quick estimation based on hardware
        gpu_memory_gb = min(
            gpu["memory_total_gb"] for gpu in self.hardware_info["gpu_details"]
        )

        # Tensor parallelism
        tp_size = min(num_gpus, 8)  # vLLM max TP
        if gpu_memory_gb < 40:  # For smaller GPUs, reduce TP
            tp_size = min(tp_size, 4)

        # Batch size estimation
        batch_size = max(1, int(gpu_memory_gb * 0.2))  # Conservative

        # Memory utilization
        gpu_util = 0.85 if gpu_memory_gb >= 80 else 0.8  # H100s can handle more

        return {
            "tensor_parallel_size": tp_size,
            "batch_size": batch_size,
            "gpu_memory_utilization": gpu_util,
            "max_tokens": 512,
            "reasoning": f"Estimated for {num_gpus}x{gpu_memory_gb:.0f}GB GPUs",
        }
