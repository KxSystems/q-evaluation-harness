"""Benchmarking utilities for model performance evaluation."""

import time
import logging
import statistics
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark model performance with various configurations."""

    def __init__(self, model: BaseModel) -> None:
        """Initialize benchmark with a model instance.

        Args:
            model: Model instance to benchmark
        """
        self.model = model
        self.results: List[Dict[str, Any]] = []

    def run_latency_benchmark(
        self,
        prompts: List[str],
        num_runs: int = 5,
        warmup_runs: int = 2,
        **generation_kwargs: Any,
    ) -> Dict[str, Any]:
        """Benchmark model latency with multiple runs.

        Args:
            prompts: List of test prompts
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not measured)
            **generation_kwargs: Parameters for generation

        Returns:
            Latency benchmark results
        """
        logger.info(
            f"Running latency benchmark with {len(prompts)} prompts, "
            f"{num_runs} runs, {warmup_runs} warmup runs"
        )

        # Warmup runs
        for i in range(warmup_runs):
            logger.info(f"Warmup run {i+1}/{warmup_runs}")
            for prompt in prompts:
                self.model.generate(prompt, n=1, **generation_kwargs)

        # Benchmark runs
        latencies = []
        for run in range(num_runs):
            logger.info(f"Benchmark run {run+1}/{num_runs}")
            run_start = time.time()

            for prompt in prompts:
                start_time = time.time()
                outputs = self.model.generate(prompt, n=1, **generation_kwargs)
                end_time = time.time()

                latency = end_time - start_time
                latencies.append(latency)

                # Log token count if available
                if outputs and outputs[0]:
                    token_count = len(outputs[0].split())
                    logger.debug(f"Generated {token_count} tokens in {latency:.3f}s")

            run_end = time.time()
            logger.info(f"Run {run+1} completed in {run_end - run_start:.3f}s")

        # Calculate statistics
        results = {
            "model_name": str(self.model),
            "num_prompts": len(prompts),
            "num_runs": num_runs,
            "generation_kwargs": generation_kwargs,
            "latency_stats": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "min": min(latencies),
                "max": max(latencies),
                "p95": self._percentile(latencies, 0.95),
                "p99": self._percentile(latencies, 0.99),
            },
            "raw_latencies": latencies,
            "timestamp": time.time(),
        }

        self.results.append(results)
        return results

    def run_throughput_benchmark(
        self,
        prompt: str,
        batch_sizes: List[int],
        num_samples_per_batch: int = 1,
        **generation_kwargs: Any,
    ) -> Dict[str, Any]:
        """Benchmark model throughput with different batch sizes.

        Args:
            prompt: Test prompt to use
            batch_sizes: List of batch sizes to test
            num_samples_per_batch: Number of samples per batch element
            **generation_kwargs: Parameters for generation

        Returns:
            Throughput benchmark results
        """
        logger.info(f"Running throughput benchmark with batch sizes: {batch_sizes}")

        batch_results = []

        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Create batch of prompts
            prompts = [prompt] * batch_size

            # Time the generation
            start_time = time.time()
            outputs = []

            for p in prompts:
                result = self.model.generate(
                    p, n=num_samples_per_batch, **generation_kwargs
                )
                outputs.extend(result)

            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            total_outputs = len(outputs)
            throughput = total_outputs / total_time if total_time > 0 else 0

            # Count tokens
            total_tokens = sum(len(output.split()) for output in outputs if output)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0

            batch_result = {
                "batch_size": batch_size,
                "num_samples_per_batch": num_samples_per_batch,
                "total_time": total_time,
                "total_outputs": total_outputs,
                "total_tokens": total_tokens,
                "throughput_requests_per_sec": throughput,
                "throughput_tokens_per_sec": tokens_per_second,
                "avg_time_per_request": total_time / batch_size
                if batch_size > 0
                else 0,
            }

            batch_results.append(batch_result)
            logger.info(
                f"Batch {batch_size}: {throughput:.2f} req/s, {tokens_per_second:.2f} tokens/s"
            )

        results = {
            "model_name": str(self.model),
            "prompt_length": len(prompt.split()),
            "generation_kwargs": generation_kwargs,
            "batch_results": batch_results,
            "timestamp": time.time(),
        }

        self.results.append(results)
        return results

    def run_memory_benchmark(
        self,
        prompts: List[str],
        max_tokens_options: List[int],
        **generation_kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Benchmark memory usage with different configurations.

        Args:
            prompts: List of test prompts
            max_tokens_options: Different max_tokens values to test
            **generation_kwargs: Parameters for generation

        Returns:
            Memory benchmark results (None if GPU not available)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping memory benchmark")
                return None

            logger.info(
                f"Running memory benchmark with max_tokens: {max_tokens_options}"
            )

            memory_results = []

            for max_tokens in max_tokens_options:
                logger.info(f"Testing max_tokens: {max_tokens}")

                # Clear cache before measurement
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Measure initial memory
                initial_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()

                # Run generation
                kwargs = {**generation_kwargs, "max_tokens": max_tokens}
                start_time = time.time()

                for prompt in prompts:
                    self.model.generate(prompt, n=1, **kwargs)

                end_time = time.time()

                # Measure final memory
                final_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()

                memory_result = {
                    "max_tokens": max_tokens,
                    "initial_memory_mb": initial_memory / (1024**2),
                    "final_memory_mb": final_memory / (1024**2),
                    "peak_memory_mb": peak_memory / (1024**2),
                    "memory_increase_mb": (final_memory - initial_memory) / (1024**2),
                    "total_time": end_time - start_time,
                    "num_prompts": len(prompts),
                }

                memory_results.append(memory_result)
                logger.info(
                    f"max_tokens {max_tokens}: Peak memory {peak_memory/(1024**2):.1f}MB"
                )

            results = {
                "model_name": str(self.model),
                "generation_kwargs": generation_kwargs,
                "memory_results": memory_results,
                "timestamp": time.time(),
            }

            self.results.append(results)
            return results

        except ImportError:
            logger.warning("PyTorch not available, skipping memory benchmark")
            return None

    def run_comprehensive_benchmark(
        self,
        test_prompts: Optional[List[str]] = None,
        batch_sizes: Optional[List[int]] = None,
        max_tokens_options: Optional[List[int]] = None,
        **generation_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a comprehensive benchmark suite.

        Args:
            test_prompts: Test prompts (defaults to standard set)
            batch_sizes: Batch sizes to test (defaults to [1, 2, 4, 8])
            max_tokens_options: Max tokens to test (defaults to [256, 512, 1024])
            **generation_kwargs: Parameters for generation

        Returns:
            Comprehensive benchmark results
        """
        if test_prompts is None:
            test_prompts = [
                "Write a function to calculate the factorial of a number.",
                "Create a sorting algorithm for a list of integers.",
                "Implement a binary search function.",
            ]

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        if max_tokens_options is None:
            max_tokens_options = [256, 512, 1024]

        logger.info("Starting comprehensive benchmark suite")
        start_time = time.time()

        # Run all benchmarks
        latency_results = self.run_latency_benchmark(test_prompts, **generation_kwargs)
        throughput_results = self.run_throughput_benchmark(
            test_prompts[0], batch_sizes, **generation_kwargs
        )
        memory_results = self.run_memory_benchmark(
            test_prompts, max_tokens_options, **generation_kwargs
        )

        total_time = time.time() - start_time

        comprehensive_results = {
            "model_name": str(self.model),
            "benchmark_duration_seconds": total_time,
            "latency_benchmark": latency_results,
            "throughput_benchmark": throughput_results,
            "memory_benchmark": memory_results,
            "test_configuration": {
                "test_prompts": test_prompts,
                "batch_sizes": batch_sizes,
                "max_tokens_options": max_tokens_options,
                "generation_kwargs": generation_kwargs,
            },
            "timestamp": time.time(),
        }

        logger.info(f"Comprehensive benchmark completed in {total_time:.1f}s")
        return comprehensive_results

    def save_results(self, output_path: Path) -> None:
        """Save benchmark results to file.

        Args:
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {output_path}")

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset.

        Args:
            data: List of values
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Percentile value
        """
        sorted_data = sorted(data)
        index = percentile * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
