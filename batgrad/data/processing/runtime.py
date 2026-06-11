from __future__ import annotations

import multiprocessing as mp
import os
import resource
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

ArgT = TypeVar("ArgT")
ResultT = TypeVar("ResultT")


class StageRuntimeConfig(Protocol):
    n_jobs: int
    worker_polars_max_threads: int | None
    chunk_rows: int
    row_group_size: int
    max_shard_size_bytes: int


@dataclass(frozen=True, slots=True)
class WorkerMetrics:
    worker_pid: int
    worker_peak_rss_mb: float
    worker_elapsed_s: float


@dataclass(frozen=True, slots=True)
class ProcessTaskSpec[ArgT]:
    task_index: int
    task_id: str
    arg: ArgT


@dataclass(frozen=True, slots=True)
class ProcessTaskResult[ResultT]:
    task_index: int
    task_id: str
    success: bool
    result: ResultT | None
    error_type: str
    error: str
    metrics: WorkerMetrics


def resolve_process_count(n_jobs: int, task_count: int) -> int:
    if task_count < 1:
        return 0
    if n_jobs == -1:
        requested = os.cpu_count() or 1
    elif n_jobs >= 1:
        requested = n_jobs
    else:
        raise ValueError(f"n_jobs must be -1, 0, or >= 1, got {n_jobs}")
    return min(requested, task_count)


def validate_stage_runtime_config(config: StageRuntimeConfig) -> None:
    if config.chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {config.chunk_rows}")
    if config.row_group_size < 1:
        raise ValueError(f"row_group_size must be >= 1, got {config.row_group_size}")
    if config.max_shard_size_bytes < 0:
        raise ValueError(
            f"max_shard_size_bytes must be >= 0, got {config.max_shard_size_bytes}",
        )
    if config.n_jobs < -1:
        raise ValueError(f"n_jobs must be -1, 0, or >= 1, got {config.n_jobs}")
    if (
        config.worker_polars_max_threads is not None
        and config.worker_polars_max_threads != -1
        and config.worker_polars_max_threads < 1
    ):
        raise ValueError(
            "worker_polars_max_threads must be >= 1, -1, or None, "
            f"got {config.worker_polars_max_threads}",
        )

    max_batch_rows = getattr(config, "max_batch_rows", None)
    if max_batch_rows is not None and max_batch_rows < 1:
        raise ValueError(f"max_batch_rows must be >= 1, got {max_batch_rows}")


def resolve_stage_worker_count(n_jobs: int, task_count: int) -> int:
    if task_count < 1:
        return 0
    if n_jobs in (0, 1) or task_count <= 1:
        return 1
    return resolve_process_count(n_jobs, task_count)


def iter_stage_process_results[ArgT, ResultT](
    worker: Callable[[ArgT], ResultT],
    specs: Sequence[ProcessTaskSpec[ArgT]],
    config: StageRuntimeConfig,
) -> Iterator[ProcessTaskResult[ResultT]]:
    worker_count = resolve_stage_worker_count(config.n_jobs, len(specs))
    if worker_count < 1:
        return
    yield from iter_process_results(
        worker,
        specs,
        max_workers=worker_count,
        polars_max_threads=config.worker_polars_max_threads,
    )


def resolve_worker_polars_max_threads(
    requested: int | None,
    worker_count: int,
    *,
    reserve_cores: int = 1,
) -> int | None:
    if requested is None:
        return None
    if requested >= 1:
        return requested
    if requested != -1:
        raise ValueError(f"worker_polars_max_threads must be None, -1, or >= 1, got {requested}")
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")

    process_cpu_count = getattr(os, "process_cpu_count", None)
    available_cores = process_cpu_count() if callable(process_cpu_count) else None
    cores = available_cores or os.cpu_count() or 1
    return max(1, (cores - reserve_cores) // worker_count)


def read_peak_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


def init_process_worker(polars_max_threads: int | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if polars_max_threads is not None:
        os.environ["POLARS_MAX_THREADS"] = str(polars_max_threads)


def iter_process_results[ArgT, ResultT](  # noqa: C901, PLR0912
    worker: Callable[[ArgT], ResultT],
    specs: Sequence[ProcessTaskSpec[ArgT]],
    *,
    max_workers: int,
    polars_max_threads: int | None = None,
) -> Iterator[ProcessTaskResult[ResultT]]:
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if not specs:
        return

    if max_workers == 1:
        for spec in specs:
            yield _run_process_task(worker, spec)
        return

    next_submit_idx = 0

    resolved_polars_max_threads = resolve_worker_polars_max_threads(
        polars_max_threads,
        max_workers,
    )
    previous_polars_max_threads = os.environ.get("POLARS_MAX_THREADS")
    if resolved_polars_max_threads is not None:
        os.environ["POLARS_MAX_THREADS"] = str(resolved_polars_max_threads)

    aborted = False
    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=init_process_worker,
            initargs=(resolved_polars_max_threads,),
        )
        future_to_spec = {}
        for _ in range(min(max_workers, len(specs))):
            spec = specs[next_submit_idx]
            future = executor.submit(_run_process_task, worker, spec)
            future_to_spec[future] = spec
            next_submit_idx += 1

        while future_to_spec:
            done, _pending = wait(future_to_spec, return_when=FIRST_COMPLETED)
            for future in done:
                spec = future_to_spec.pop(future)
                try:
                    result = future.result()
                except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    result = _failed_process_result(spec, exc)
                if next_submit_idx < len(specs):
                    next_spec = specs[next_submit_idx]
                    next_future = executor.submit(_run_process_task, worker, next_spec)
                    future_to_spec[next_future] = next_spec
                    next_submit_idx += 1
                yield result
    except (GeneratorExit, KeyboardInterrupt):
        aborted = True
        if executor is not None:
            abort_process_pool(executor)
        raise
    finally:
        if not aborted and executor is not None:
            executor.shutdown(wait=True)
        if resolved_polars_max_threads is not None:
            if previous_polars_max_threads is None:
                os.environ.pop("POLARS_MAX_THREADS", None)
            else:
                os.environ["POLARS_MAX_THREADS"] = previous_polars_max_threads


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


def _run_process_task[ArgT, ResultT](
    worker: Callable[[ArgT], ResultT],
    spec: ProcessTaskSpec[ArgT],
) -> ProcessTaskResult[ResultT]:
    started_at = time.perf_counter()
    worker_pid = os.getpid()
    try:
        result = worker(spec.arg)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _failed_process_result(
            spec,
            exc,
            worker_pid=worker_pid,
            started_at=started_at,
        )
    return ProcessTaskResult(
        task_index=spec.task_index,
        task_id=spec.task_id,
        success=True,
        result=result,
        error_type="",
        error="",
        metrics=WorkerMetrics(
            worker_pid=worker_pid,
            worker_peak_rss_mb=read_peak_rss_mb(),
            worker_elapsed_s=time.perf_counter() - started_at,
        ),
    )


def _failed_process_result[ArgT, ResultT](
    spec: ProcessTaskSpec[ArgT],
    exc: Exception,
    *,
    worker_pid: int | None = None,
    started_at: float | None = None,
) -> ProcessTaskResult[ResultT]:
    elapsed_s = 0.0 if started_at is None else time.perf_counter() - started_at
    return ProcessTaskResult(
        task_index=spec.task_index,
        task_id=spec.task_id,
        success=False,
        result=None,
        error_type=type(exc).__name__,
        error=str(exc),
        metrics=WorkerMetrics(
            worker_pid=os.getpid() if worker_pid is None else worker_pid,
            worker_peak_rss_mb=read_peak_rss_mb(),
            worker_elapsed_s=elapsed_s,
        ),
    )
