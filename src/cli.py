"""Main CLI interface for the evaluation system."""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
from dotenv import load_dotenv

from .datasets.registry import get_dataset_config, list_datasets
from .datasets.loaders import get_loader
from .prompts.templates import get_prompt_template
from .models.factory import create_model
from .models.base import BaseModel
from .models.generation_orchestrator import GenerationOrchestrator
from .models.generation_strategy import BatchGenerator, AsyncGenerator
from .prompts.base import BasePromptTemplate
from .evaluation.base import BaseTestExecutor
from .evaluation.executor import get_executor
from .evaluation.metrics import calculate_pass_at_k
from .utils.io import append_to_jsonl, load_jsonl, save_json
from .utils.extraction import extract_function_from_content
from .utils.hardware_profiler import HardwareProfiler
from .utils.benchmarking import ModelBenchmark
from .constants import (
    DEFAULT_TIMEOUT,
    DEFAULT_TEMPERATURE,
    DEFAULT_SEED,
    DEFAULT_MAX_TOKENS,
)

# Module-level logger
logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = 0) -> None:
    """Setup logging configuration based on verbosity level.
    
    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG, 3+=TRACE-level debug
    """
    # Map verbosity to logging levels
    if verbosity == 0:
        level = logging.WARNING  # Quiet mode - errors/warnings only
    elif verbosity == 1:
        level = logging.INFO     # Normal mode - progress updates
    elif verbosity == 2:
        level = logging.DEBUG    # Debug mode - detailed info
    else:  # verbosity >= 3
        level = logging.DEBUG    # Ultra-verbose mode
        # For ultra-verbose, also enable debug for external libraries
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Enhanced format for better readability
    if verbosity >= 2:
        # Detailed format for debug levels
        fmt = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
    else:
        # Cleaner format for normal use
        fmt = "%(levelname)s: %(message)s"
        datefmt = None
    
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Override any existing configuration
    )
    
    # Suppress verbose debug logs from external libraries unless in ultra-verbose mode
    if verbosity < 3:
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        
        # Keep vLLM info logs but suppress debug unless explicitly requested
        if verbosity < 2:
            logging.getLogger("vllm").setLevel(logging.INFO)


async def generate_solutions_optimal(
    problems: List[Dict[str, Any]],
    model: BaseModel,
    template: BasePromptTemplate,
    num_samples: int = 1,
    output_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Generate solutions using model's optimal strategy."""
    orchestrator = GenerationOrchestrator()
    
    # Log the strategy being used
    strategy = model.get_generation_strategy()
    logger.info(f"Using {strategy.value} generation strategy for {model.model_name}")
    
    if isinstance(model, BatchGenerator):
        config = model.get_batch_config()
        logger.info(f"Batch config: optimal={config.optimal_batch_size}, max={config.max_batch_size}")
    elif isinstance(model, AsyncGenerator):
        config = model.get_async_config()
        logger.info(f"Async config: max_concurrent={config.max_concurrent}, rate_limit={config.rate_limit_delay}")
    
    return await orchestrator.generate_solutions(
        problems, model, template, num_samples, output_file
    )


def _load_and_validate_solutions(solutions_file: Path) -> List[Dict[str, Any]]:
    """Load and validate solutions from JSONL file."""
    
    try:
        solutions = load_jsonl(str(solutions_file))
        logger.info(f"Loaded {len(solutions)} solutions")
        return solutions
    except Exception as e:
        logger.error(f"Failed to load solutions: {e}")
        return []


def _create_problem_lookup(
    problems: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Create problem lookup dictionary by task_id."""
    return {
        p.get("task_id", f"task_{i}"): p
        for i, p in enumerate(problems)
    }


def _calculate_optimal_workers(max_workers: Optional[int]) -> int:
    """Calculate optimal worker count for execution."""
    if max_workers is not None:
        return max_workers
    
    # Use conservative threading approach: I/O bound tasks benefit from
    # more threads but we don't want to overwhelm system resources
    cpu_count = os.cpu_count() or 4
    # Common pattern for I/O bound work
    return min(32, (cpu_count * 2) + 1)


def _calculate_and_save_metrics(
    results: List[Dict[str, Any]],
    output_file: Path,
    max_workers: int
) -> None:
    """Calculate pass@k metrics and save results to file."""
    
    # Calculate metrics by grouping results by task_id
    task_results = defaultdict(list)
    for result in results:
        task_id = result.get("task_id", "unknown")
        task_results[task_id].append(result)
    
    # Prepare data for pass@k calculation
    pass_at_k_data = []
    for task_id, task_solutions in task_results.items():
        num_samples = len(task_solutions)
        num_correct = sum(1 for r in task_solutions if r.get("passed", False))
        pass_at_k_data.append({
            "task_id": task_id,
            "num_samples": num_samples,
            "num_correct": num_correct
        })
    
    # Calculate pass@k metrics
    total = len(results)
    passed = sum(1 for r in results if bool(r.get("passed", False)))
    pass_rate = passed / total if total > 0 else 0
    
    # Calculate pass@k for multiple k values
    k_values = [1, 5, 10, 20, 50, 100]
    pass_at_k_metrics = {}
    
    # Find minimum number of samples across all tasks
    min_samples = (
        min(task["num_samples"] for task in pass_at_k_data)
        if pass_at_k_data else 0
    )
    
    for k in k_values:
        # Only include k if all tasks have at least k samples
        if pass_at_k_data and min_samples >= k:
            pass_at_k_metrics[f"pass_at_{k}"] = calculate_pass_at_k(
                pass_at_k_data, k
            )
    
    # Save results with comprehensive metrics
    execution_method = "sequential" if max_workers == 1 else "multithreaded"
    summary = {
        "total_solutions": total,
        "passed_solutions": passed,
        "pass_rate": pass_rate,
        "total_problems": len(pass_at_k_data),
        "execution_method": execution_method,
        "max_workers": max_workers,
        **pass_at_k_metrics,
        "per_problem_results": pass_at_k_data,
        "results": results,
    }
    save_json(summary, str(output_file))
    logger.info(f"Results saved to {output_file}")
    
    # Log all pass@k metrics (only those that were calculated)
    metrics_str = ", ".join([
        f"Pass@{k}: {pass_at_k_metrics[f'pass_at_{k}']:.3f}"
        for k in k_values
        if f"pass_at_{k}" in pass_at_k_metrics
    ])
    if metrics_str:
        logger.info(f"Metrics - {metrics_str}")
    else:
        logger.info("No pass@k metrics calculated (insufficient samples)")
    
    logger.info(
        f"Execution completed: {passed}/{total} passed ({pass_rate:.1%})"
    )


def _execute_single_solution(
    solution: Dict[str, Any],
    problem_lookup: Dict[str, Dict[str, Any]],
    executor: BaseTestExecutor,
    solution_index: int
) -> Dict[str, Any]:
    """Execute a single solution in a thread-safe manner."""
    
    task_id = solution.get("task_id", f"task_{solution_index}")
    sample_index = solution.get("sample_index", 0)
    
    # Extract code during execution step
    raw_completion = solution.get(
        "completion", solution.get("extracted_code", "")
    )
    
    logger.debug(
        f"Executing {task_id} sample {sample_index} "
        f"(solution {solution_index + 1})"
    )
    
    problem = problem_lookup.get(task_id)
    if not problem:
        error_msg = f"No problem data found for {task_id}"
        logger.error(error_msg)
        return {
            "task_id": task_id,
            "sample_index": sample_index,
            "passed": False,
            "error": error_msg,
        }
    
    # Perform code extraction here
    entry_point = problem.get("entry_point")
    if not entry_point:
        error_msg = f"Missing entry_point for {task_id}"
        logger.error(error_msg)
        return {
            "task_id": task_id,
            "sample_index": sample_index,
            "passed": False,
            "error": error_msg,
        }
    
    if raw_completion.strip():
        try:
            extracted_code = extract_function_from_content(
                raw_completion, entry_point
            )
        except Exception as e:
            error_msg = f"Code extraction failed: {str(e)}"
            logger.error(error_msg)
            return {
                "task_id": task_id,
                "sample_index": sample_index,
                "passed": False,
                "error": error_msg,
            }
    else:
        extracted_code = raw_completion
    
    logger.debug(f"Extracted function: {extracted_code}")
    
    if not extracted_code.strip():
        logger.warning(f"Empty code for {task_id}")
        return {
            "task_id": task_id,
            "sample_index": sample_index,
            "passed": False,
            "error": "Empty code",
        }
    
    tests = problem.get("tests", "")
    setup_code = problem.get("test_setup_code", "")
    
    try:
        passed, info = executor.execute(
            extracted_code, tests, setup_code, timeout=DEFAULT_TIMEOUT
        )
        
        result = {
            "task_id": task_id,
            "sample_index": sample_index,
            "passed": passed,
            "info": info,
        }
        
        logger.debug(f"  Result: {'PASS' if passed else 'FAIL'}")
        return result
        
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        logger.debug(error_msg)
        return {
            "task_id": task_id,
            "sample_index": sample_index,
            "passed": False,
            "error": error_msg,
        }


def execute_solutions(
    solutions_file: Path,
    problems: List[Dict[str, Any]],
    executor: BaseTestExecutor,
    output_file: Path,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute solutions with automatic strategy selection.
    
    Uses sequential execution if max_workers=1, otherwise uses threading.
    """
    
    # Load and validate solutions
    solutions = _load_and_validate_solutions(solutions_file)
    if not solutions:
        logger.warning("No solutions to execute")
        return []
    
    # Setup execution environment
    problem_lookup = _create_problem_lookup(problems)
    max_workers = _calculate_optimal_workers(max_workers)
    
    # Check PYKX_THREADING for multithreaded execution
    if max_workers > 1 and os.environ.get("PYKX_THREADING") != "1":
        logger.warning(
            "Multithreaded execution requires PYKX_THREADING=1. "
            "Falling back to sequential execution."
        )
        max_workers = 1
    
    logger.info(f"Using {max_workers} workers for execution")
    strategy = 'sequential' if max_workers == 1 else 'multithreaded'
    logger.info(f"Execution strategy: {strategy}")
    
    # Execute based on strategy
    if max_workers == 1:
        results = _execute_sequential(solutions, problem_lookup, executor)
    else:
        results = _execute_threaded(
            solutions, problem_lookup, executor, max_workers
        )
    
    # Calculate and save metrics
    _calculate_and_save_metrics(results, output_file, max_workers)
    return results


def _execute_sequential(
    solutions: List[Dict[str, Any]],
    problem_lookup: Dict[str, Dict[str, Any]],
    executor: BaseTestExecutor,
) -> List[Dict[str, Any]]:
    """Execute solutions sequentially with progress tracking."""
    results = []
    
    with tqdm(
        total=len(solutions), desc="Executing solutions", unit="solution"
    ) as pbar:
        for i, solution in enumerate(solutions):
            result = _execute_single_solution(
                solution, problem_lookup, executor, i
            )
            results.append(result)
            
            # Update progress bar
            status = "PASS" if result.get("passed", False) else "FAIL"
            pass_rate = (
                sum(1 for r in results if r.get("passed", False)) /
                len(results)
            )
            pbar.update(1)
            pbar.set_postfix({
                "task": result.get("task_id", "unknown"),
                "status": status,
                "pass_rate": f"{pass_rate:.1%}"
            })
    
    return results


def _execute_threaded(
    solutions: List[Dict[str, Any]],
    problem_lookup: Dict[str, Dict[str, Any]],
    executor: BaseTestExecutor,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Execute solutions using ThreadPoolExecutor."""
    results = []
    results_lock = Lock()
    
    # Progress tracking variables (thread-safe)
    completed_count = 0
    completed_lock = Lock()
    
    def update_progress(pbar: tqdm, result: Dict[str, Any]) -> None:
        """Thread-safe progress update."""
        nonlocal completed_count
        with completed_lock:
            completed_count += 1
            with results_lock:
                results.append(result)
                passed_count = sum(
                    1 for r in results if r.get("passed", False)
                )
                pass_rate = passed_count / len(results)
            
            status = "PASS" if result.get("passed", False) else "FAIL"
            pbar.update(1)
            pbar.set_postfix({
                "task": result.get("task_id", "unknown"),
                "status": status,
                "pass_rate": f"{pass_rate:.1%}"
            })
    
    # Execute solutions using ThreadPoolExecutor
    with tqdm(
        total=len(solutions), desc="Executing solutions", unit="solution"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor_pool:
            # Submit all tasks
            future_to_index = {
                executor_pool.submit(
                    _execute_single_solution,
                    solution,
                    problem_lookup,
                    executor,
                    i
                ): i for i, solution in enumerate(solutions)
            }
            
            # Process completed futures as they finish
            try:
                timeout = DEFAULT_TIMEOUT * len(solutions)
                for future in as_completed(future_to_index, timeout=timeout):
                    try:
                        result = future.result(timeout=DEFAULT_TIMEOUT)
                        update_progress(pbar, result)
                    except Exception as e:
                        # Handle individual task failures gracefully
                        solution_index = future_to_index[future]
                        task_id = solutions[solution_index].get(
                            "task_id", f"task_{solution_index}"
                        )
                        error_result = {
                            "task_id": task_id,
                            "sample_index": solutions[solution_index].get(
                                "sample_index", 0
                            ),
                            "passed": False,
                            "error": f"Task execution failed: {str(e)}",
                        }
                        logger.debug(
                            f"Failed to execute solution {solution_index}: {e}"
                        )
                        update_progress(pbar, error_result)
                        
            except Exception as e:
                logger.error(f"Execution pool error: {e}")
    
    return results


def run_generate_command(
    dataset: str,
    model: str,
    num_samples: int = 1,
    output_file: Optional[str] = None,
    **model_kwargs: Any,
) -> None:
    """Generate solutions only."""

    # Load dataset
    try:
        config = get_dataset_config(dataset)
        loader = get_loader(config["format"])
        problems = loader.load(config["path"])
        template = get_prompt_template(config["prompt_template"])
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset}': {e}")
        return

    # Setup model
    try:
        model_instance = create_model(model, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to create model '{model}': {e}")
        return

    # Setup and validate output path
    if not output_file:
        output_file = f"solutions_{model.replace('/', '_')}.jsonl"
    output_path = Path(output_file)

    # Ensure output directory exists and is writable
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        logger.error(f"Permission denied writing to: {output_path.parent}")
        return
    except Exception as e:
        logger.error(f"Cannot write to output directory: {e}")
        return

    # Generate solutions
    try:
        solutions = asyncio.run(
            generate_solutions_optimal(
                problems,
                model_instance,
                template,
                num_samples,
                output_path,
            )
        )
        logger.info(f"Generation complete: {len(solutions)} solutions saved")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return


def run_execute_command(
    solutions_file: str,
    dataset: str,
    output_file: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Execute solutions only."""

    # Validate solutions file path
    solutions_path = Path(solutions_file)
    if not solutions_path.exists():
        logger.error(f"Solutions file not found: {solutions_file}")
        return

    if not solutions_path.is_file():
        logger.error(f"Path is not a file: {solutions_file}")
        return

    try:
        # Test if file is readable
        with open(solutions_path, "r") as f:
            f.read(1)  # Try to read one character
    except PermissionError:
        logger.error(f"Permission denied reading file: {solutions_file}")
        return
    except Exception as e:
        logger.error(f"Cannot read solutions file: {e}")
        return

    try:
        config = get_dataset_config(dataset)
        loader = get_loader(config["format"])
        problems = loader.load(config["path"])
        executor = get_executor(config["language"], config["test_language"])
    except Exception as e:
        logger.error(f"Failed to setup dataset/executor: {e}")
        return

    # Setup and validate output path
    if not output_file:
        base_name = solutions_path.stem
        output_file = f"results_{base_name}.json"
    output_path = Path(output_file)

    # Ensure output directory exists and is writable
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        logger.error(f"Permission denied writing to: {output_path.parent}")
        return
    except Exception as e:
        logger.error(f"Cannot write to output directory: {e}")
        return

    # Execute solutions with optimal strategy
    try:
        results = execute_solutions(
            solutions_path, problems, executor, output_path, max_workers
        )
        logger.info(f"Execution complete: {len(results)} results")
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Execution failed: {e}")
        return


def run_evaluation(
    dataset: str,
    model: str,
    num_samples: int = 1,
    output_dir: str = "./outputs",
    max_workers: Optional[int] = None,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """Run complete evaluation (generate + execute)."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate solutions
    solutions_file = output_path / f"solutions_{model.replace('/', '_')}.jsonl"
    run_generate_command(
        dataset, model, num_samples, str(solutions_file), **model_kwargs
    )

    # Execute solutions
    results_file = output_path / f"results_{model.replace('/', '_')}.json"
    run_execute_command(
        str(solutions_file), dataset, str(results_file), max_workers
    )

    # Load and return results with metrics
    try:
        if results_file.exists():
            # Validate file is readable
            if not results_file.is_file():
                logger.warning(f"Results path is not a file: {results_file}")
                return {"results": []}

            # Test file accessibility
            try:
                with open(results_file, "r") as f:
                    f.read(1)  # Try to read one character
            except PermissionError:
                logger.error(f"Permission denied reading: {results_file}")
                return {"results": []}

            # Load and parse JSON
            with open(results_file, "r") as f:
                results_data: Dict[str, Any] = json.load(f)
            return results_data
        else:
            logger.warning(f"Results file not found: {results_file}")
            return {"results": []}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in results file: {e}")
        return {"results": []}
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return {"results": []}


def run_command(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    setup_logging(args.verbose)

    model_kwargs = {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "model_type": args.backend,
    }

    try:
        run_evaluation(
            dataset=args.dataset,
            model=args.model,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            **model_kwargs,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def generate_command(args: argparse.Namespace) -> None:
    """Handle the 'generate' subcommand."""
    setup_logging(args.verbose)

    model_kwargs = {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "model_type": args.backend,
    }

    try:
        run_generate_command(
            dataset=args.dataset,
            model=args.model,
            num_samples=args.num_samples,
            output_file=args.output,
            **model_kwargs,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


def execute_command(args: argparse.Namespace) -> None:
    """Handle the 'execute' subcommand."""
    setup_logging(args.verbose)

    try:
        run_execute_command(
            solutions_file=args.solutions_file,
            dataset=args.dataset,
            output_file=args.output,
            max_workers=args.max_workers,
        )
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


def profile_command(args: argparse.Namespace) -> None:
    """Handle the 'profile' subcommand."""
    try:
        from pathlib import Path
        import json
        
        logger.info(f"Starting hardware profiling for model: {args.model}")
        
        # Initialize profiler
        profiler = HardwareProfiler()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Log hardware information
        hw_info = profiler.hardware_info
        logger.info(
            f"Hardware: {hw_info['num_gpus']} GPUs, "
            f"{hw_info['cpu_cores']} CPUs"
        )
        for gpu in hw_info.get("gpu_details", []):
            logger.info(f"  - {gpu['name']}: {gpu['memory_total_gb']:.1f}GB")

        if args.quick:
            # Quick estimation
            logger.info("Running quick parameter estimation...")
            config = profiler.get_recommended_config(args.model)
            logger.info(
                f"Recommended: TP={config['tensor_parallel_size']}, "
                f"Batch={config['batch_size']}, "
                f"GPU_util={config['gpu_memory_utilization']}"
            )
            
            # Save results
            model_safe_name = args.model.replace('/', '_')
            output_file = output_dir / f"quick_profile_{model_safe_name}.json"
            quick_results = {
                "model_name": args.model,
                "hardware_info": hw_info,
                "recommended_config": config,
                "profiling_type": "quick_estimation",
            }
            with open(output_file, "w") as f:
                json.dump(quick_results, f, indent=2, default=str)
            logger.info(f"Profile saved to: {output_file}")
            
        else:
            # Full profiling
            logger.info(
                "Running full performance profiling "
                "(this may take 15-30 minutes)..."
            )
            results = profiler.profile_model_performance(args.model)
            
            # Save results
            model_safe_name = args.model.replace('/', '_')
            output_file = output_dir / f"full_profile_{model_safe_name}.json"
            profiler.save_profile_results(results, output_file)
            
            # Log recommendations
            recommendations = results.get("recommendations", {})
            if "recommended_balanced" in recommendations:
                best_config = recommendations["recommended_balanced"]
                tp_size = best_config['tensor_parallel_size']
                batch_size = best_config['batch_size']
                throughput = best_config['metrics']['tokens_per_second']
                logger.info(
                    f"Optimal Config: TP={tp_size}, Batch={batch_size}, "
                    f"Throughput={throughput:.1f} tokens/sec"
                )
            
            logger.info(f"Full profile saved to: {output_file}")

        # Optional benchmarking
        if args.benchmark:
            logger.info("Running performance benchmarks...")
            try:
                if args.quick:
                    model_config = config
                else:
                    model_config = recommendations.get(
                        "recommended_balanced", {}
                    )
                
                tp_size = model_config.get("tensor_parallel_size", 1)
                gpu_util = model_config.get("gpu_memory_utilization", 0.8)
                model = create_model(
                    args.model,
                    model_type="vllm",
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                )
                
                benchmark = ModelBenchmark(model)
                benchmark_results = benchmark.run_comprehensive_benchmark()
                
                model_safe_name = args.model.replace('/', '_')
                benchmark_filename = f"benchmark_{model_safe_name}.json"
                benchmark_file = output_dir / benchmark_filename
                benchmark.save_results(benchmark_file)
                
                latency_benchmark = benchmark_results["latency_benchmark"]
                latency_stats = latency_benchmark["latency_stats"]
                mean_latency = latency_stats['mean']
                p95_latency = latency_stats['p95']
                logger.info(
                    f"Latency: {mean_latency:.3f}s avg, {p95_latency:.3f}s p95"
                )
                logger.info(f"Benchmark saved to: {benchmark_file}")
                
            except Exception as e:
                logger.error(f"Benchmarking failed: {e}")

        logger.info("Hardware profiling completed!")
        
    except Exception as e:
        logger.error(f"Hardware profiling failed: {e}")
        logger.info("Ensure vLLM is installed: pip install vllm")


def list_command(args: argparse.Namespace) -> None:
    """Handle the 'list' subcommand."""
    setup_logging(args.verbose)

    try:
        datasets = list_datasets()

        if not datasets:
            logger.info("No datasets available.")
            return

        logger.info("Available datasets:")

        # Calculate max width for alignment
        max_name_width = max(len(name) for name in datasets.keys())

        for name, description in datasets.items():
            logger.info(f"  {name:<{max_name_width}}  {description}")

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Q code generation model evaluation harness", prog="qeval"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v: DEBUG, -vv: DEBUG + external libs)"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # 'run' subcommand (combined generation + execution)
    run_parser = subparsers.add_parser(
        "run",
        help="Run complete evaluation (generate + execute)"
    )
    run_parser.add_argument("dataset", help="Dataset name")
    run_parser.add_argument("model", help="Model name")
    run_parser.add_argument("--num-samples", "-n", type=int, default=50)
    run_parser.add_argument("--output-dir", "-o", default="./outputs")
    run_parser.add_argument("--max-workers", type=int, default=None,
                            help="Max workers for execution (default: auto)")
    run_parser.add_argument("--backend", choices=["auto", "litellm", "huggingface", "vllm"], 
                            default="auto", help="Model backend (default: auto-detect)")
    run_parser.set_defaults(func=run_command)

    # 'generate' subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Generate solutions only"
    )
    gen_parser.add_argument("dataset", help="Dataset name")
    gen_parser.add_argument("model", help="Model name")
    gen_parser.add_argument("--num-samples", "-n", type=int, default=1)
    gen_parser.add_argument("--output", "-o", help="Output JSONL file")
    gen_parser.add_argument("--backend", choices=["auto", "litellm", "huggingface", "vllm"], 
                            default="auto", help="Model backend (default: auto-detect)")
    gen_parser.set_defaults(func=generate_command)

    # 'execute' subcommand
    exec_parser = subparsers.add_parser(
        "execute", help="Execute solutions only"
    )
    exec_parser.add_argument("solutions_file", help="Solutions JSONL file")
    exec_parser.add_argument("dataset", help="Dataset name")
    exec_parser.add_argument("--output", "-o", help="Output JSON file")
    exec_parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max workers for execution (default: auto)"
    )
    exec_parser.set_defaults(func=execute_command)

    # 'profile' subcommand
    profile_parser = subparsers.add_parser(
        "profile", help="Profile hardware for optimal model parameters"
    )
    profile_parser.add_argument("model", help="Model name to profile")
    profile_parser.add_argument(
        "--output-dir", "-o", default="./hardware_profiles",
        help="Output directory for profile results"
    )
    profile_parser.add_argument(
        "--quick", action="store_true",
        help="Run quick estimation instead of full profiling"
    )
    profile_parser.add_argument(
        "--benchmark", action="store_true",
        help="Run performance benchmarks after profiling"
    )
    profile_parser.set_defaults(func=profile_command)

    # 'list' subcommand
    list_parser = subparsers.add_parser(
        "list", help="List available datasets"
    )
    list_parser.set_defaults(func=list_command)

    # Parse and dispatch
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
