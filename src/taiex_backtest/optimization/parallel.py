"""Parallel execution utilities for optimization."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypeVar
import os

T = TypeVar("T")
R = TypeVar("R")


def get_max_workers(n_jobs: int = -1) -> int:
    """Resolve n_jobs to actual worker count.
    
    Args:
        n_jobs: Number of workers. -1 means all CPUs, 0 or 1 means sequential.
    
    Returns:
        Actual number of workers to use.
    """
    if n_jobs <= 0:
        cpu_count = os.cpu_count() or 1
        if n_jobs == -1:
            return cpu_count
        return 1
    return n_jobs


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    n_jobs: int = -1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Apply func to each item in parallel using threads.
    
    Uses ThreadPoolExecutor (thread-safe for Python backtests since the GIL
    is released during I/O, and our CPU-bound work benefits from avoiding
    pickle overhead of ProcessPoolExecutor).
    
    Args:
        func: Function to apply to each item.
        items: List of inputs.
        n_jobs: Number of parallel workers (-1 = all CPUs).
        progress_callback: Optional callback(completed, total).
    
    Returns:
        List of results in the SAME ORDER as input items.
    """
    if not items:
        return []
    
    max_workers = get_max_workers(n_jobs)
    
    if max_workers <= 1:
        # Sequential fallback
        results = []
        for i, item in enumerate(items):
            results.append(func(item))
            if progress_callback is not None:
                progress_callback(i + 1, len(items))
        return results
    
    # Parallel execution preserving order
    total = len(items)
    results: list[R | None] = [None] * total
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, item): i
            for i, item in enumerate(items)
        }
        
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
    
    return results  # type: ignore[return-value]


def chunked(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of given size.
    
    Args:
        items: List to split.
        chunk_size: Maximum size of each chunk.
    
    Returns:
        List of lists, each of size <= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive: {chunk_size}")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
