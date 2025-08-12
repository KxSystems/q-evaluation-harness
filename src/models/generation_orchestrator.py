"""Generation orchestrator that dispatches based on model strategy."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from .base import BaseModel
from .generation_strategy import (
    GenerationStrategy, 
    BatchGenerator, 
    AsyncGenerator,
    SequentialGenerator
)
from ..prompts.base import BasePromptTemplate
from ..utils.io import append_to_jsonl


class GenerationOrchestrator:
    """Orchestrates generation based on model's optimal strategy."""
    
    def __init__(self):
        self._strategies = {
            GenerationStrategy.BATCH_OPTIMIZED: BatchGenerationStrategy(),
            GenerationStrategy.ASYNC_OPTIMIZED: AsyncGenerationStrategy(),
            GenerationStrategy.SEQUENTIAL: SequentialGenerationStrategy(),
        }
    
    async def generate_solutions(
        self,
        problems: List[Dict[str, Any]],
        model: BaseModel,
        template: BasePromptTemplate,
        num_samples: int = 1,
        output_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate solutions using the model's optimal strategy."""
        strategy = self._strategies[model.get_generation_strategy()]
        return await strategy.execute(
            problems, model, template, num_samples, output_file
        )


class GenerationStrategyBase(ABC):
    """Base class for generation strategies."""
    
    @abstractmethod
    async def execute(
        self,
        problems: List[Dict[str, Any]],
        model: BaseModel,
        template: BasePromptTemplate,
        num_samples: int,
        output_file: Optional[Path],
    ) -> List[Dict[str, Any]]:
        """Execute generation strategy."""
        pass

    def _create_result(
        self,
        task_id: str,
        sample_index: int,
        completion: str,
        model_name: str
    ) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            "task_id": task_id,
            "sample_index": sample_index,
            "completion": completion,
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
        }


class BatchGenerationStrategy(GenerationStrategyBase):
    """Strategy for batch-optimized models like VLLM and HuggingFace."""
    
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
    
    async def execute(
        self,
        problems: List[Dict[str, Any]],
        model: BaseModel,
        template: BasePromptTemplate,
        num_samples: int,
        output_file: Optional[Path],
    ) -> List[Dict[str, Any]]:
        """Process problems in optimally-sized batches."""
        if not isinstance(model, BatchGenerator):
            raise ValueError(f"Model {model} does not support batch generation")
        
        batch_config = model.get_batch_config()
        all_results = []
        
        with tqdm(total=len(problems), desc="Generating solutions") as pbar:
            for batch_start in range(0, len(problems), batch_config.optimal_batch_size):
                batch_end = min(batch_start + batch_config.optimal_batch_size, len(problems))
                batch_problems = problems[batch_start:batch_end]
                
                # Prepare batch prompts
                batch_prompts = [template.format(p) for p in batch_problems]
                
                try:
                    # Single efficient batch call
                    batch_results = model.generate_batch(batch_prompts, num_samples)
                except Exception as e:
                    # Handle batch generation errors gracefully
                    self.logger.error(f"Batch generation failed: {e}")
                    batch_results = [[""] * num_samples for _ in batch_problems]
                
                # Process and save results
                for prob, completions in zip(batch_problems, batch_results):
                    task_id = prob.get("task_id", "unknown")
                    for idx, completion in enumerate(completions):
                        result = self._create_result(
                            task_id, idx, completion, model.model_name
                        )
                        all_results.append(result)
                        
                        if output_file:
                            append_to_jsonl(result, str(output_file))
                
                pbar.update(len(batch_problems))
                pbar.set_postfix({
                    "batch_size": len(batch_problems),
                    "total_completions": len(batch_results) * num_samples
                })
        
        return all_results


class AsyncGenerationStrategy(GenerationStrategyBase):
    """Strategy for async-optimized models (APIs)."""
    
    async def execute(
        self,
        problems: List[Dict[str, Any]],
        model: BaseModel,
        template: BasePromptTemplate,
        num_samples: int,
        output_file: Optional[Path],
    ) -> List[Dict[str, Any]]:
        """Process problems with controlled concurrency."""
        if not isinstance(model, AsyncGenerator):
            raise ValueError(f"Model {model} does not support async generation")
        
        async_config = model.get_async_config()
        semaphore = asyncio.Semaphore(async_config.max_concurrent)
        all_results = []
        
        async def process_problem(problem: Dict[str, Any], pbar: atqdm) -> List[Dict[str, Any]]:
            async with semaphore:
                if async_config.rate_limit_delay > 0:
                    await asyncio.sleep(async_config.rate_limit_delay)
                
                task_id = problem.get("task_id", "unknown")
                prompt = template.format(problem)
                completions = await model.generate_async(prompt, num_samples)
                
                problem_results = []
                for idx, completion in enumerate(completions):
                    result = self._create_result(
                        task_id, idx, completion, model.model_name
                    )
                    problem_results.append(result)
                    
                    if output_file:
                        append_to_jsonl(result, str(output_file))
                
                pbar.update(1)
                pbar.set_postfix({
                    "task": task_id,
                    "samples": len(completions)
                })
                return problem_results
        
        with atqdm(total=len(problems), desc="Generating solutions") as pbar:
            tasks = [process_problem(p, pbar) for p in problems]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for problem_results in results:
                if isinstance(problem_results, list):
                    all_results.extend(problem_results)
        
        return all_results


class SequentialGenerationStrategy(GenerationStrategyBase):
    """Strategy for sequential processing (fallback)."""
    
    async def execute(
        self,
        problems: List[Dict[str, Any]],
        model: BaseModel,
        template: BasePromptTemplate,
        num_samples: int,
        output_file: Optional[Path],
    ) -> List[Dict[str, Any]]:
        """Process problems sequentially."""
        all_results = []
        
        with tqdm(total=len(problems), desc="Generating solutions") as pbar:
            for problem in problems:
                task_id = problem.get("task_id", "unknown")
                prompt = template.format(problem)
                
                try:
                    completions = model.generate(prompt, num_samples)
                    
                    for idx, completion in enumerate(completions):
                        result = self._create_result(
                            task_id, idx, completion, model.model_name
                        )
                        all_results.append(result)
                        
                        if output_file:
                            append_to_jsonl(result, str(output_file))
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "task": task_id,
                        "samples": len(completions)
                    })
                    
                except Exception as e:
                    # Log error but continue
                    pbar.set_postfix({"task": task_id, "status": f"failed: {e}"})
                    pbar.update(1)
        
        return all_results