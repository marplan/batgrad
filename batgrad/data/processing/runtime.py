from __future__ import annotations

import multiprocessing as mp
import os
import resource
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

ArgT = TypeVar("ArgT")
ResultT = TypeVar("ResultT")


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


def read_peak_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


def init_process_worker(polars_max_threads: int | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if polars_max_threads is not None:
        os.environ["POLARS_MAX_THREADS"] = str(polars_max_threads)


def iter_ordered_process_results[ArgT, ResultT](  # noqa: C901
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

    next_submit_idx = 0
    next_emit_index = specs[0].task_index
    buffered: dict[int, ProcessTaskResult[ResultT]] = {}

    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_process_worker,
        initargs=(polars_max_threads,),
    )
    aborted = False
    try:
        future_to_spec = {}
        for _ in range(min(max_workers, len(specs))):
            spec = specs[next_submit_idx]
            future = executor.submit(_run_process_task, worker, spec)
            future_to_spec[future] = spec
            next_submit_idx += 1

        while future_to_spec or buffered:
            if next_emit_index in buffered:
                result = buffered.pop(next_emit_index)
                next_emit_index += 1
                if next_submit_idx < len(specs):
                    spec = specs[next_submit_idx]
                    future = executor.submit(_run_process_task, worker, spec)
                    future_to_spec[future] = spec
                    next_submit_idx += 1
                yield result
                continue

            done, _pending = wait(future_to_spec, return_when=FIRST_COMPLETED)
            for future in done:
                spec = future_to_spec.pop(future)
                try:
                    result = future.result()
                except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    result = _failed_process_result(spec, exc)
                buffered[result.task_index] = result
    except (GeneratorExit, KeyboardInterrupt):
        aborted = True
        abort_process_pool(executor)
        raise
    finally:
        if not aborted:
            executor.shutdown(wait=True)


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
