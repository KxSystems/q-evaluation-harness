"""Agent evaluation runner — orchestrates the agent-run pipeline."""

import asyncio
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .base import AgentBackend, AgentResult
from ..evaluation.metrics import calculate_pass_at_k
from ..utils.io import append_to_jsonl, save_json
from ..constants import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.Handler):
    """Route log output through tqdm.write() to avoid corrupting progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


async def run_agent_evaluation(
    dataset: str,
    backend: AgentBackend,
    output_dir: str = "./outputs",
    concurrency: int = 4,
    keep_workspaces: bool = False,
    problem_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run a full agent evaluation: scaffold -> invoke -> score.

    Args:
        dataset: Dataset name (e.g., 'q-humaneval').
        backend: Configured AgentBackend instance.
        output_dir: Directory for output files.
        concurrency: Maximum parallel agent invocations.
        keep_workspaces: If True, preserve workspace directories for debugging.
        problem_ids: Optional subset of task IDs to evaluate.

    Returns:
        Results summary dict with standard + agent-specific metrics.
    """
    # --- Phase 0: Load dataset and setup (lazy imports to avoid QHOME issues) ---
    from ..datasets.registry import get_dataset_config
    from ..datasets.loaders import get_loader
    from ..prompts.templates import get_prompt_template
    from ..evaluation.executor import get_executor

    config = get_dataset_config(dataset)
    loader = get_loader(config["format"])
    problems = loader.load(config["path"])
    template = get_prompt_template(config["prompt_template"])
    executor = get_executor(config["language"], config["test_language"])

    if problem_ids is not None:
        problems = [p for p in problems if p.get("task_id") in problem_ids]
        logger.info(f"Filtered to {len(problems)} problems by ID")

    logger.info(
        f"Agent evaluation: {len(problems)} problems, "
        f"backend={backend.name}, model={backend.model}, "
        f"concurrency={concurrency}"
    )

    # Setup output paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_safe = backend.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"agent_{backend.name}_{model_safe}_{timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    workspaces_dir = run_dir / "workspaces"
    workspaces_dir.mkdir(exist_ok=True)

    solutions_file = run_dir / "solutions.jsonl"
    results_file = run_dir / "results.json"

    # --- Phase 1: Invoke agents in parallel ---
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    ok_count = 0
    no_solution_count = 0
    total_cost = 0.0

    # Install tqdm-safe logging so log lines don't corrupt the progress bar
    tqdm_handler = _TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.handlers = [tqdm_handler]

    pbar = tqdm(
        total=len(problems),
        desc="Agent solving",
        unit="task",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    )

    async def solve_problem(problem: Dict[str, Any]) -> AgentResult:
        nonlocal completed_count, ok_count, no_solution_count, total_cost
        async with semaphore:
            task_id = problem.get("task_id", "unknown")
            entry_point = problem.get("entry_point", "")

            prompt_text = _build_agent_prompt(problem, template)

            workspace = backend.prepare_workspace(
                problem, workspaces_dir, prompt_text
            )

            agent_result = await backend.invoke(prompt_text, workspace)

            solution_code = backend.extract_solution(workspace, entry_point)
            agent_result.solution_code = solution_code

            # Stream result to solutions JSONL
            solution_record = {
                "task_id": task_id,
                "sample_index": 0,
                "completion": solution_code,
                "model_name": f"{backend.name}/{backend.model}",
                "generated_at": datetime.now().isoformat(),
                "agent_wall_time": agent_result.wall_time_seconds,
                "agent_num_turns": agent_result.num_turns,
                "agent_cost_usd": agent_result.cost_usd,
            }
            append_to_jsonl(solution_record, str(solutions_file))

            # Update progress bar
            completed_count += 1
            has_solution = bool(solution_code.strip())
            if has_solution:
                ok_count += 1
            else:
                no_solution_count += 1
            if agent_result.cost_usd:
                total_cost += agent_result.cost_usd

            status = "OK" if has_solution else "NO_SOLUTION"
            postfix = {"ok": ok_count, "empty": no_solution_count}
            if total_cost > 0:
                postfix["cost"] = f"${total_cost:.2f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

            logger.info(
                f"Task {task_id}: {status} "
                f"({agent_result.wall_time_seconds:.1f}s)"
            )

            return agent_result

    logger.info("Phase 1: Invoking agents...")
    tasks = [solve_problem(p) for p in problems]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()

    # Restore original logging handlers
    root_logger.handlers = original_handlers

    # Filter out exceptions
    valid_results: List[AgentResult] = []
    for i, result in enumerate(gathered):
        if isinstance(result, Exception):
            logger.error(f"Problem {i} raised exception: {result}")
            valid_results.append(
                AgentResult(
                    task_id=problems[i].get("task_id", i),
                    success=False,
                    solution_code="",
                    wall_time_seconds=0.0,
                    error=str(result),
                )
            )
        else:
            valid_results.append(result)

    # --- Phase 2: Score solutions ---
    logger.info("Phase 2: Scoring solutions...")
    problem_lookup = {p.get("task_id", i): p for i, p in enumerate(problems)}

    execution_results = []
    passed_count = 0

    for agent_result in tqdm(
        valid_results, desc="Scoring", unit="task",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}] {postfix}"
        ),
    ):
        task_id = agent_result.task_id
        problem = problem_lookup.get(task_id)

        if not problem or not agent_result.solution_code.strip():
            execution_results.append(
                {
                    "task_id": task_id,
                    "sample_index": 0,
                    "passed": False,
                    "error": agent_result.error or "No solution produced",
                    "agent_wall_time": agent_result.wall_time_seconds,
                    "agent_num_turns": agent_result.num_turns,
                    "agent_cost_usd": agent_result.cost_usd,
                }
            )
            continue

        tests = problem.get("tests", "")
        setup_code = problem.get("test_setup_code", "")

        try:
            passed, info = executor.execute(
                agent_result.solution_code,
                tests,
                setup_code,
                timeout=DEFAULT_TIMEOUT,
            )
            if passed:
                passed_count += 1
            execution_results.append(
                {
                    "task_id": task_id,
                    "sample_index": 0,
                    "passed": passed,
                    "info": info,
                    "agent_wall_time": agent_result.wall_time_seconds,
                    "agent_num_turns": agent_result.num_turns,
                    "agent_cost_usd": agent_result.cost_usd,
                    "agent_input_tokens": agent_result.input_tokens,
                    "agent_output_tokens": agent_result.output_tokens,
                }
            )
        except Exception as e:
            execution_results.append(
                {
                    "task_id": task_id,
                    "sample_index": 0,
                    "passed": False,
                    "error": f"Execution error: {str(e)}",
                    "agent_wall_time": agent_result.wall_time_seconds,
                    "agent_num_turns": agent_result.num_turns,
                    "agent_cost_usd": agent_result.cost_usd,
                }
            )

    # --- Phase 3: Calculate metrics ---
    summary = _calculate_agent_metrics(
        execution_results, valid_results, backend, dataset
    )
    summary["results"] = execution_results

    save_json(summary, str(results_file))
    logger.info(f"Results saved to {results_file}")

    # --- Cleanup ---
    if not keep_workspaces:
        shutil.rmtree(workspaces_dir, ignore_errors=True)
        logger.info("Cleaned up workspaces")
    else:
        logger.info(f"Workspaces preserved at {workspaces_dir}")

    _log_summary(summary)

    return summary


def _build_agent_prompt(
    problem: Dict[str, Any],
    template: Any,
) -> str:
    """Build the full prompt text for an agent."""
    base_prompt = template.format(problem)
    entry_point = problem.get("entry_point", "solution")

    agent_prompt = (
        f"{base_prompt}\n\n"
        f"Write your complete Q function implementation in `solution.q` in the "
        f"current directory. The function should be named `{entry_point}`. "
        f"The file should contain ONLY the Q function definition.\n"
    )

    return agent_prompt


def _calculate_agent_metrics(
    execution_results: List[Dict[str, Any]],
    agent_results: List[AgentResult],
    backend: AgentBackend,
    dataset: str,
) -> Dict[str, Any]:
    """Calculate standard + agent-specific metrics."""
    total = len(execution_results)
    passed = sum(1 for r in execution_results if r.get("passed", False))
    pass_rate = passed / total if total > 0 else 0.0

    # Standard pass@k (with n=1 per problem, pass@1 equals pass_rate)
    per_problem_results = []
    for result in execution_results:
        per_problem_results.append(
            {
                "task_id": result["task_id"],
                "num_samples": 1,
                "num_correct": 1 if result.get("passed", False) else 0,
            }
        )

    pass_at_1 = (
        calculate_pass_at_k(per_problem_results, 1)
        if per_problem_results
        else 0.0
    )

    # Agent-specific metrics
    wall_times = [
        r.wall_time_seconds for r in agent_results if r.wall_time_seconds > 0
    ]
    costs = [r.cost_usd for r in agent_results if r.cost_usd is not None]
    turns = [r.num_turns for r in agent_results if r.num_turns is not None]
    input_tokens_list = [
        r.input_tokens for r in agent_results if r.input_tokens is not None
    ]
    output_tokens_list = [
        r.output_tokens for r in agent_results if r.output_tokens is not None
    ]

    passed_times = [
        r.wall_time_seconds
        for r, er in zip(agent_results, execution_results)
        if er.get("passed", False) and r.wall_time_seconds > 0
    ]
    failed_times = [
        r.wall_time_seconds
        for r, er in zip(agent_results, execution_results)
        if not er.get("passed", False) and r.wall_time_seconds > 0
    ]

    no_solution_count = sum(
        1 for r in agent_results if not r.solution_code.strip()
    )

    summary: Dict[str, Any] = {
        # Standard metrics (compatible with existing format)
        "total_solutions": total,
        "passed_solutions": passed,
        "pass_rate": pass_rate,
        "total_problems": total,
        "pass_at_1": pass_at_1,
        "per_problem_results": per_problem_results,
        # Agent evaluation metadata
        "evaluation_type": "agent",
        "agent_backend": backend.name,
        "agent_model": backend.model,
        "agent_max_turns": backend.max_turns,
        "dataset": dataset,
        # Agent-specific metrics
        "agent_metrics": {
            "total_wall_time_seconds": sum(wall_times) if wall_times else 0,
            "mean_wall_time_seconds": _safe_mean(wall_times),
            "median_wall_time_seconds": _safe_median(wall_times),
            "p90_wall_time_seconds": _safe_percentile(wall_times, 90),
            "mean_wall_time_passed": _safe_mean(passed_times),
            "mean_wall_time_failed": _safe_mean(failed_times),
            "total_cost_usd": sum(costs) if costs else None,
            "mean_cost_usd": _safe_mean(costs),
            "mean_turns": _safe_mean(turns),
            "median_turns": _safe_median(turns),
            "mean_input_tokens": _safe_mean(input_tokens_list),
            "mean_output_tokens": _safe_mean(output_tokens_list),
            "no_solution_count": no_solution_count,
            "timeout_count": sum(
                1
                for r in agent_results
                if r.error and "Timed out" in r.error
            ),
        },
    }

    return summary


def _log_summary(summary: Dict[str, Any]) -> None:
    """Log a human-readable summary of agent evaluation results."""
    total = summary.get("total_solutions", 0)
    passed = summary.get("passed_solutions", 0)
    pass_rate = summary.get("pass_rate", 0)
    backend = summary.get("agent_backend", "unknown")
    model = summary.get("agent_model", "unknown")
    logger.info(
        f"\n{'=' * 60}\n"
        f"Agent Evaluation Results: {backend} / {model}\n"
        f"{'=' * 60}\n"
        f"Pass@1: {pass_rate:.1%} ({passed}/{total})"
    )

    metrics = summary.get("agent_metrics", {})
    if metrics.get("mean_wall_time_seconds") is not None:
        logger.info(
            f"Mean wall time: {metrics['mean_wall_time_seconds']:.1f}s"
        )
    if metrics.get("median_wall_time_seconds") is not None:
        logger.info(
            f"Median wall time: {metrics['median_wall_time_seconds']:.1f}s"
        )
    if metrics.get("total_cost_usd") is not None:
        logger.info(f"Total cost: ${metrics['total_cost_usd']:.2f}")
    if metrics.get("mean_turns") is not None:
        logger.info(f"Mean turns: {metrics['mean_turns']:.1f}")
    if metrics.get("no_solution_count", 0) > 0:
        logger.info(
            f"No solution produced: {metrics['no_solution_count']}"
        )
    if metrics.get("timeout_count", 0) > 0:
        logger.info(f"Timeouts: {metrics['timeout_count']}")


def _safe_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _safe_median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_v[mid - 1] + sorted_v[mid]) / 2
    return sorted_v[mid]


def _safe_percentile(values: List[float], pct: int) -> Optional[float]:
    if not values:
        return None
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]
