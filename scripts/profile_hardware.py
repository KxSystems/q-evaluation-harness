#!/usr/bin/env python3
"""Hardware profiling script to determine optimal vLLM parameters for your setup."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.hardware_profiler import HardwareProfiler
from utils.benchmarking import ModelBenchmark
from models.factory import create_model


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """Main profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile hardware and determine optimal vLLM parameters"
    )
    parser.add_argument(
        "model_name",
        help="Model name to profile (e.g., 'meta-llama/CodeLlama-7b-Python-hf')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./hardware_profiles",
        help="Output directory for profile results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick estimation instead of full profiling",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks after profiling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting hardware profiling for model: {args.model_name}")

    # Initialize profiler
    profiler = HardwareProfiler()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print hardware information
    print("\n🔍 Hardware Detection Results:")
    print("=" * 50)
    hw_info = profiler.hardware_info
    print(f"GPUs: {hw_info['num_gpus']}")
    for gpu in hw_info.get("gpu_details", []):
        print(f"  - {gpu['name']}: {gpu['memory_total_gb']:.1f}GB")
    print(
        f"CPU Cores: {hw_info['cpu_cores']} ({hw_info['cpu_cores_physical']} physical)"
    )
    print(f"System RAM: {hw_info['system_memory_gb']:.1f}GB")
    print()

    if args.quick:
        # Quick estimation
        print("⚡ Quick Parameter Estimation:")
        print("=" * 50)
        config = profiler.get_recommended_config(args.model_name)
        print(f"Recommended Configuration:")
        print(f"  - Tensor Parallel Size: {config['tensor_parallel_size']}")
        print(f"  - Batch Size: {config['batch_size']}")
        print(f"  - GPU Memory Utilization: {config['gpu_memory_utilization']}")
        print(f"  - Max Tokens: {config['max_tokens']}")
        print(f"  - Reasoning: {config['reasoning']}")

        # Save quick results
        quick_results = {
            "model_name": args.model_name,
            "hardware_info": hw_info,
            "recommended_config": config,
            "profiling_type": "quick_estimation",
        }

        output_file = (
            output_dir / f"quick_profile_{args.model_name.replace('/', '_')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(quick_results, f, indent=2, default=str)
        print(f"\n📁 Quick profile saved to: {output_file}")

    else:
        # Full profiling
        print("🔬 Running Full Performance Profiling...")
        print("=" * 50)
        print("This may take 10-30 minutes depending on your hardware.")
        print(
            "The script will test different configurations to find optimal settings.\n"
        )

        try:
            # Run comprehensive profiling
            results = profiler.profile_model_performance(args.model_name)

            # Save detailed results
            output_file = (
                output_dir / f"full_profile_{args.model_name.replace('/', '_')}.json"
            )
            profiler.save_profile_results(results, output_file)

            # Print recommendations
            recommendations = results.get("recommendations", {})
            if "recommended_balanced" in recommendations:
                best_config = recommendations["recommended_balanced"]
                print("\n🎯 Optimal Configuration Found:")
                print("=" * 50)
                print(f"Tensor Parallel Size: {best_config['tensor_parallel_size']}")
                print(f"Batch Size: {best_config['batch_size']}")
                print(f"Max Tokens: {best_config['max_tokens']}")
                print(
                    f"Expected Throughput: {best_config['metrics']['tokens_per_second']:.1f} tokens/sec"
                )
                print(f"Memory Usage: {best_config['metrics']['memory_used_gb']:.1f}GB")

            print(f"\n📁 Full profile saved to: {output_file}")

        except Exception as e:
            logger.error(f"Full profiling failed: {e}")
            print(f"❌ Full profiling failed: {e}")
            print("💡 Try running with --quick for basic estimation")
            return

    # Optional benchmarking
    if args.benchmark:
        print("\n🏃 Running Performance Benchmarks...")
        print("=" * 50)

        try:
            # Create model with optimal settings
            if args.quick:
                model_config = config
            else:
                model_config = recommendations.get("recommended_balanced", {})

            # Create vLLM model
            model = create_model(
                args.model_name,
                model_type="vllm",
                tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
                gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.8),
            )

            # Run benchmarks
            benchmark = ModelBenchmark(model)
            benchmark_results = benchmark.run_comprehensive_benchmark()

            # Save benchmark results
            benchmark_file = (
                output_dir / f"benchmark_{args.model_name.replace('/', '_')}.json"
            )
            benchmark.save_results(benchmark_file)

            # Print summary
            latency_stats = benchmark_results["latency_benchmark"]["latency_stats"]
            print(f"Average Latency: {latency_stats['mean']:.3f}s")
            print(f"P95 Latency: {latency_stats['p95']:.3f}s")

            throughput_results = benchmark_results["throughput_benchmark"][
                "batch_results"
            ]
            best_throughput = max(
                throughput_results, key=lambda x: x["throughput_tokens_per_sec"]
            )
            print(
                f"Peak Throughput: {best_throughput['throughput_tokens_per_sec']:.1f} tokens/sec"
            )
            print(f"Optimal Batch Size: {best_throughput['batch_size']}")

            print(f"\n📁 Benchmark results saved to: {benchmark_file}")

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            print(f"❌ Benchmarking failed: {e}")

    print("\n✅ Hardware profiling completed!")
    print(f"📂 All results saved in: {output_dir}")

    print("\n💡 Next Steps:")
    print("1. Use the recommended configuration in your model initialization")
    print("2. Monitor GPU utilization during inference")
    print("3. Adjust batch size based on your workload requirements")
    print("4. Re-run profiling if you change hardware or model")


if __name__ == "__main__":
    main()
