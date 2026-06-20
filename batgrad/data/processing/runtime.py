from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from batgrad.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

logger = get_logger(__name__)
_PROGRESS_TIME_INTERVAL_S = 30.0


class StageRuntimeConfig(Protocol):
    n_jobs: int
    worker_polars_max_threads: int | None
    chunk_rows: int
    row_group_size: int
    max_shard_size_bytes: int


@dataclass(frozen=True)
class ProcessTaskSpec[T]:
    task_index: int
    task_id: str
    arg: T


@dataclass(frozen=True)
class ProcessTaskResult[T]:
    task_index: int
    task_id: str
    value: T | None = None
    error: str | None = None


def validate_stage_runtime_config(config: StageRuntimeConfig) -> None:
    if config.n_jobs < -1 or config.n_jobs == 0:
        raise ValueError(f"n_jobs must be -1 or >= 1, got {config.n_jobs}")
    if config.worker_polars_max_threads is not None and config.worker_polars_max_threads < -1:
        raise ValueError(
            "worker_polars_max_threads must be -1, >= 1, or None, "
            f"got {config.worker_polars_max_threads}",
        )
    if config.chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {config.chunk_rows}")
    if config.row_group_size < 1:
        raise ValueError(f"row_group_size must be >= 1, got {config.row_group_size}")
    if config.max_shard_size_bytes < 0:
        raise ValueError(f"max_shard_size_bytes must be >= 0, got {config.max_shard_size_bytes}")


def available_cpu_count() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        return process_cpu_count() or 1
    return os.cpu_count() or 1


def resolve_worker_count(n_jobs: int, task_count: int) -> int:
    if task_count < 1:
        return 0
    if n_jobs == 1:
        return 1
    if n_jobs == -1:
        return min(task_count, max(1, available_cpu_count() - 1))
    if n_jobs > 1:
        return min(task_count, n_jobs)
    raise ValueError(f"n_jobs must be -1 or >= 1, got {n_jobs}")


def resolve_worker_polars_max_threads(
    worker_polars_max_threads: int | None,
    worker_count: int,
) -> int | None:
    if worker_polars_max_threads is None:
        return None
    if worker_polars_max_threads > 0:
        return worker_polars_max_threads
    if worker_polars_max_threads == -1:
        return max(1, (available_cpu_count() - 1) // max(1, worker_count))
    raise ValueError(
        f"worker_polars_max_threads must be -1, >= 1, or None, got {worker_polars_max_threads}",
    )


def iter_process_task_results[ArgT, ResultT](  # noqa: C901, PLR0912
    worker: Callable[[ArgT], ResultT],
    specs: Sequence[ProcessTaskSpec[ArgT]],
    config: StageRuntimeConfig,
    *,
    ordered: bool = True,
) -> Iterator[ProcessTaskResult[ResultT]]:
    worker_count = resolve_worker_count(config.n_jobs, len(specs))
    if worker_count <= 1:
        _log_runtime_settings(
            task_count=len(specs),
            requested_n_jobs=config.n_jobs,
            worker_count=worker_count,
            requested_worker_polars_max_threads=config.worker_polars_max_threads,
            resolved_worker_polars_max_threads=None,
        )
        for spec in specs:
            yield _run_task(worker, spec)
        return

    worker_polars_threads = resolve_worker_polars_max_threads(
        config.worker_polars_max_threads,
        worker_count,
    )
    _log_runtime_settings(
        task_count=len(specs),
        requested_n_jobs=config.n_jobs,
        worker_count=worker_count,
        requested_worker_polars_max_threads=config.worker_polars_max_threads,
        resolved_worker_polars_max_threads=worker_polars_threads,
    )
    pending_results: dict[int, ProcessTaskResult[ResultT]] = {}
    next_yield_idx = 1
    next_submit_idx = 0
    aborted = False
    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=init_process_worker,
            initargs=(worker_polars_threads,),
        )
        future_to_index = {}
        for _ in range(min(worker_count, len(specs))):
            spec = specs[next_submit_idx]
            future_to_index[executor.submit(_run_task, worker, spec)] = spec.task_index
            next_submit_idx += 1

        while future_to_index:
            done, _not_done = wait(future_to_index, return_when=FIRST_COMPLETED)
            for future in done:
                future_to_index.pop(future)
                result = cast("ProcessTaskResult[ResultT]", future.result())
                if next_submit_idx < len(specs):
                    spec = specs[next_submit_idx]
                    future_to_index[executor.submit(_run_task, worker, spec)] = spec.task_index
                    next_submit_idx += 1
                if ordered:
                    pending_results[result.task_index] = result
                    while next_yield_idx in pending_results:
                        yield pending_results.pop(next_yield_idx)
                        next_yield_idx += 1
                else:
                    yield result
    except (GeneratorExit, KeyboardInterrupt):
        aborted = True
        if executor is not None:
            abort_process_pool(executor)
        raise
    finally:
        if not aborted and executor is not None:
            executor.shutdown(wait=True)


def _log_runtime_settings(
    *,
    task_count: int,
    requested_n_jobs: int,
    worker_count: int,
    requested_worker_polars_max_threads: int | None,
    resolved_worker_polars_max_threads: int | None,
) -> None:
    available_cpus = available_cpu_count()
    if worker_count <= 1 and requested_worker_polars_max_threads is not None:
        logger.warning(
            "worker_polars_max_threads=%s is ignored because n_jobs=1 or only one task is run; "
            "set POLARS_MAX_THREADS before process start to control sequential Polars threads",
            requested_worker_polars_max_threads,
        )
    if worker_count > 1 and requested_worker_polars_max_threads is None:
        logger.warning(
            "n_jobs resolved to %d workers with worker_polars_max_threads=None; "
            "Polars thread pools are unrestricted and may oversubscribe CPU cores",
            worker_count,
        )
    logger.info(
        "processing runtime tasks=%d n_jobs=%d workers=%d available_cpus=%d "
        "worker_polars_max_threads=%s resolved_worker_polars_max_threads=%s",
        task_count,
        requested_n_jobs,
        worker_count,
        available_cpus,
        requested_worker_polars_max_threads,
        "unused" if worker_count <= 1 else resolved_worker_polars_max_threads,
    )


def log_task_progress_if_due(
    stage: str,
    dataset_id: str,
    task_index: int,
    task_count: int,
    succeeded: int,
    failed: int,
    warnings: int,
    last_progress_at: float,
    *,
    force: bool = False,
    interval_s: float = _PROGRESS_TIME_INTERVAL_S,
) -> float:
    now = time.perf_counter()
    if not force and now - last_progress_at < interval_s:
        return last_progress_at
    logger.info(
        "%s dataset=%s task=%d/%d succeeded=%d failed=%d warnings=%d",
        stage,
        dataset_id,
        task_index,
        task_count,
        succeeded,
        failed,
        warnings,
    )
    return now


def init_process_worker(polars_max_threads: int | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if polars_max_threads is not None:
        os.environ["POLARS_MAX_THREADS"] = str(polars_max_threads)


def abort_process_pool(executor: object) -> None:
    process_map = getattr(executor, "_processes", {}) or {}
    processes = (
        tuple(process_map.values()) if hasattr(process_map, "values") else tuple(process_map)
    )
    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):
        terminate_workers()
        return
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=1.0)
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=True, cancel_futures=True)


def _run_task[ArgT, ResultT](
    worker: Callable[[ArgT], ResultT],
    spec: ProcessTaskSpec[ArgT],
) -> ProcessTaskResult[ResultT]:
    try:
        return ProcessTaskResult(
            spec.task_index,
            spec.task_id,
            value=worker(spec.arg),
        )
    except Exception as exc:  # noqa: BLE001 - processing stages continue and log task errors.
        return ProcessTaskResult(
            spec.task_index,
            spec.task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
