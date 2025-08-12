"""Metrics calculation for code evaluation."""

import itertools
import numpy as np
from typing import Union, List, Iterator


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array.

    Args:
        num_samples: Number of samples per problem (int or array)
        num_correct: Number of correct samples per problem
        k: k value for pass@k calculation

    Returns:
        Array of pass@k estimates for each problem
    """

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it: Iterator[int] = itertools.repeat(
            num_samples, len(num_correct)
        )
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    results: np.ndarray = np.array(
        [
            estimator(int(n), int(c), k)
            for n, c in zip(num_samples_it, num_correct)
        ]
    )
    return results


def calculate_pass_at_k(results: List[dict], k: int) -> float:
    """Calculate overall pass@k from evaluation results.

    Args:
        results: List of result dicts with 'num_samples' and 'num_correct'
        k: k value for pass@k calculation

    Returns:
        Overall pass@k score
    """
    if not results:
        return 0.0

    num_samples = [r["num_samples"] for r in results]
    num_correct = [r["num_correct"] for r in results]

    pass_at_k_scores = estimate_pass_at_k(num_samples, num_correct, k)
    return float(np.mean(pass_at_k_scores))
