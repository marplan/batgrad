from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from batgrad.data.processing.runtime import (
    ProcessTaskSpec,
    init_process_worker,
    iter_process_task_results,
    resolve_worker_count,
    resolve_worker_polars_max_threads,
    validate_stage_runtime_config,
)


@dataclass(frozen=True)
class RuntimeConfig:
    n_jobs: int = 1
    worker_polars_max_threads: int | None = -1
    chunk_rows: int = 2
    row_group_size: int = 2
    max_shard_size_bytes: int = 0


def _double(value: int) -> int:
    return value * 2


def _fail_on_two(value: int) -> int:
    if value == 2:
        raise ValueError("bad two")
    return value


def test_validate_stage_runtime_config_rejects_invalid_values() -> None:
    validate_stage_runtime_config(RuntimeConfig())
    for config in (
        RuntimeConfig(n_jobs=0),
        RuntimeConfig(worker_polars_max_threads=-2),
        RuntimeConfig(chunk_rows=0),
        RuntimeConfig(row_group_size=0),
        RuntimeConfig(max_shard_size_bytes=-1),
    ):
        with pytest.raises(ValueError):
            validate_stage_runtime_config(config)


def test_worker_count_and_thread_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("batgrad.data.processing.runtime.available_cpu_count", lambda: 8)
    assert resolve_worker_count(1, 3) == 1
    assert resolve_worker_count(4, 3) == 3
    assert resolve_worker_count(-1, 20) == 7
    assert resolve_worker_count(-1, 0) == 0
    assert resolve_worker_polars_max_threads(None, 3) is None
    assert resolve_worker_polars_max_threads(2, 3) == 2
    assert resolve_worker_polars_max_threads(-1, 3) == 2
    with pytest.raises(ValueError):
        resolve_worker_count(0, 1)
    with pytest.raises(ValueError):
        resolve_worker_polars_max_threads(-2, 3)


def test_init_process_worker_sets_polars_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    init_process_worker(3)
    assert os.environ["POLARS_MAX_THREADS"] == "3"


def test_iter_process_task_results_sequential_success_and_error() -> None:
    specs = tuple(ProcessTaskSpec(idx, f"task-{idx}", idx) for idx in range(1, 4))
    results = list(iter_process_task_results(_double, specs, RuntimeConfig(n_jobs=1)))
    assert [result.value for result in results] == [2, 4, 6]

    failed = list(iter_process_task_results(_fail_on_two, specs, RuntimeConfig(n_jobs=1)))
    assert failed[1].value is None
    assert failed[1].error is not None
    assert "ValueError" in failed[1].error


def test_iter_process_task_results_multiprocess_smoke() -> None:
    specs = tuple(ProcessTaskSpec(idx, f"task-{idx}", idx) for idx in range(1, 4))
    results = list(iter_process_task_results(_double, specs, RuntimeConfig(n_jobs=2)))
    assert [result.value for result in results] == [2, 4, 6]
