from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Native thread pools inside each worker would oversubscribe the CPU when
# several workers run at once, so every worker is pinned to one thread.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _init_worker(init_fn: Callable[..., None], init_args: Sequence[Any]) -> None:
    import torch

    torch.set_num_threads(1)
    init_fn(*init_args)


def run_ordered(
    items: Sequence[T],
    process_fn: Callable[[T], R],
    *,
    init_fn: Callable[..., None],
    init_args: Sequence[Any] = (),
    num_workers: int = 1,
    chunksize: int = 8,
) -> Iterator[R]:
    """Yield process_fn(item) for every item, in input order.

    Queries are independent, so results are identical to a sequential loop;
    only the wall-clock time changes. With num_workers <= 1 the loop runs in
    the current process exactly as before.
    """
    if num_workers <= 1:
        init_fn(*init_args)
        for item in items:
            yield process_fn(item)
        return

    saved = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    for name in _THREAD_ENV_VARS:
        os.environ[name] = "1"
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers, initializer=_init_worker, initargs=(init_fn, tuple(init_args))) as pool:
            yield from pool.imap(process_fn, items, chunksize=chunksize)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def recommended_workers(reserve_cores: int = 4, per_worker_gb: float = 1.0, reserve_ram_gb: float = 4.0) -> int:
    """Conservative worker count: leave cores and RAM for the OS and editor."""
    cores = os.cpu_count() or 1
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or cores
        total_gb = psutil.virtual_memory().total / 1024**3
    except ImportError:  # pragma: no cover
        physical, total_gb = cores, float("inf")
    by_cpu = max(1, physical - reserve_cores)
    by_ram = max(1, int((total_gb - reserve_ram_gb) // per_worker_gb))
    return min(by_cpu, by_ram)

