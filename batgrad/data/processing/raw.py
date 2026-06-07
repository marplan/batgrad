from __future__ import annotations

import time
from dataclasses import dataclass, field as dataclass_field
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Protocol

import polars as pl

from batgrad import _loggers
from batgrad.contracts.columns import (
    BaseColumns,
    ColumnSpec,
    MetadataColumns,
    collect_column_specs,
)
from batgrad.data.processing.config import FailureMode
from batgrad.data.processing.runtime import (
    ProcessTaskResult,
    ProcessTaskSpec,
    WorkerMetrics,
    iter_ordered_process_results,
    read_peak_rss_mb,
    resolve_process_count,
)
from batgrad.data.processing.sharding import RawProtocolShardWriter

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.data.datasets.specs import DatasetSpec, RawIngestSpec, RawProtocolSchema
    from batgrad.data.processing.config import RawStageConfig
    from batgrad.storage.store import DataStore

logger = _loggers.get_logger(__name__)
_PROGRESS_TIME_INTERVAL_S = 30.0


def is_excluded_raw_file(path: str, raw_spec: RawIngestSpec) -> bool:
    return any(fnmatch(path, pattern) for pattern in raw_spec.excluded_file_patterns)


@dataclass(frozen=True, slots=True)
class RawIngestIssue:
    kind: str
    source_paths: tuple[str, ...]
    message: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RawTaskStats:
    dropped_unknown_columns: int = 0
    dropped_declared_columns: int = 0
    duplicate_columns: int = 0
    dropped_mapped_columns: int = 0
    issues: tuple[RawIngestIssue, ...] = ()


@dataclass(slots=True)
class RawRunStats:
    tasks_total: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    batches_written: int = 0
    chunks_written: int = 0
    rows_written: int = 0
    warnings: int = 0
    dropped_unknown_columns: int = 0
    dropped_declared_columns: int = 0
    duplicate_columns: int = 0
    dropped_mapped_columns: int = 0
    workers: set[int] = dataclass_field(default_factory=set)
    worker_peak_rss_mb_max: float = 0.0

    def add_task_stats(self, stats: RawTaskStats) -> None:
        self.warnings += len(stats.issues)
        self.dropped_unknown_columns += stats.dropped_unknown_columns
        self.dropped_declared_columns += stats.dropped_declared_columns
        self.duplicate_columns += stats.duplicate_columns
        self.dropped_mapped_columns += stats.dropped_mapped_columns

    def add_issue(self, issue: RawIngestIssue) -> None:
        self.warnings += 1
        if issue.kind == "dropped_unknown_columns":
            self.dropped_unknown_columns += len(issue.columns)
        elif issue.kind == "dropped_declared_columns":
            self.dropped_declared_columns += len(issue.columns)
        elif issue.kind == "duplicate_columns":
            self.duplicate_columns += 1
        elif issue.kind == "dropped_mapped_columns":
            self.dropped_mapped_columns += len(issue.columns)

    def add_worker_metrics(self, metrics: WorkerMetrics) -> None:
        self.workers.add(metrics.worker_pid)
        self.worker_peak_rss_mb_max = max(
            self.worker_peak_rss_mb_max,
            metrics.worker_peak_rss_mb,
        )

    @property
    def warning_count(self) -> int:
        return self.warnings


@dataclass(frozen=True, slots=True)
class RawTask:
    task_id: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawBatch:
    data: pl.DataFrame | pl.LazyFrame
    stream_id: str
    source_paths: tuple[str, ...]
    metadata: dict[ColumnSpec, object]
    stats: RawTaskStats = RawTaskStats()


@dataclass(frozen=True, slots=True)
class PreparedRawBatch:
    data: pl.DataFrame
    stream_id: str
    source_paths: tuple[str, ...]
    metadata: dict[ColumnSpec, object]
    stats: RawTaskStats = RawTaskStats()


@dataclass(frozen=True, slots=True)
class PreparedRawTaskResult:
    task_index: int
    task: RawTask
    batches: tuple[PreparedRawBatch, ...]
    stats: RawTaskStats


@dataclass(frozen=True, slots=True)
class RawWorkerPayload:
    task_index: int
    task: RawTask
    adapter: RawDatasetAdapter
    input_store: DataStore
    failure_mode: FailureMode


class RawDatasetAdapter(Protocol):
    spec: DatasetSpec

    def plan_raw_tasks(self, input_store: DataStore) -> tuple[RawTask, ...]: ...

    def load_raw_task(
        self,
        task: RawTask,
        input_store: DataStore,
        failure_mode: FailureMode,
    ) -> Iterator[RawBatch]: ...


class RawProcessor:
    def __init__(self, adapter: RawDatasetAdapter) -> None:
        self.adapter = adapter

    def run(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: RawStageConfig,
    ) -> None:
        raw_spec = self._raw_spec()
        self._validate_run_config(config)

        writer = RawProtocolShardWriter(output_store, self.adapter.spec, raw_spec)
        failed_any = False
        started_at = time.perf_counter()
        last_update_at = started_at
        run_stats = RawRunStats()
        try:
            tasks = self.adapter.plan_raw_tasks(input_store)
            run_stats.tasks_total = len(tasks)
            if run_stats.tasks_total > 0:
                logger.info(
                    "raw_to_parquet dataset=%s task=%d/%d succeeded=%d failed=%d warnings=%d",
                    self.adapter.spec.dataset_id,
                    1,
                    run_stats.tasks_total,
                    run_stats.tasks_succeeded,
                    run_stats.tasks_failed,
                    run_stats.warning_count,
                )
            if config.n_jobs in (0, 1) or run_stats.tasks_total <= 1:
                for task_idx, task in enumerate(tasks, start=1):
                    try:
                        result = self.prepare_task_result(
                            task_index=task_idx,
                            task=task,
                            input_store=input_store,
                            failure_mode=config.failure_mode,
                        )
                        self._consume_prepared_task_result(
                            writer,
                            result,
                            config.chunk_rows,
                            run_stats,
                        )
                    except Exception as exc:
                        failed_any = True
                        self._record_task_failure(task_idx, task, exc, run_stats)
                        if config.failure_mode == FailureMode.STRICT:
                            raise
                    last_update_at = self._log_status_if_due(task_idx, last_update_at, run_stats)
            else:
                failed_any = self._run_parallel_tasks(
                    tasks=tasks,
                    input_store=input_store,
                    writer=writer,
                    config=config,
                    run_stats=run_stats,
                    last_update_at=last_update_at,
                )
        except BaseException:
            writer.close(manifest="skip")
            raise
        else:
            manifest = "skip" if failed_any else "write"
            writer.close(manifest=manifest)
            logger.info(
                "raw_to_parquet finished dataset=%s tasks=%d succeeded=%d failed=%d "
                "batches=%d chunks=%d rows=%d warnings=%d dropped_unknown_columns=%d "
                "dropped_declared_columns=%d duplicate_columns=%d dropped_mapped_columns=%d "
                "manifest=%s elapsed_s=%.1f workers=%d worker_peak_rss_mb_max=%.1f "
                "main_peak_rss_mb=%.1f",
                self.adapter.spec.dataset_id,
                run_stats.tasks_total,
                run_stats.tasks_succeeded,
                run_stats.tasks_failed,
                run_stats.batches_written,
                run_stats.chunks_written,
                run_stats.rows_written,
                run_stats.warning_count,
                run_stats.dropped_unknown_columns,
                run_stats.dropped_declared_columns,
                run_stats.duplicate_columns,
                run_stats.dropped_mapped_columns,
                "skipped" if manifest == "skip" else "written",
                time.perf_counter() - started_at,
                len(run_stats.workers),
                run_stats.worker_peak_rss_mb_max,
                read_peak_rss_mb(),
            )

    @staticmethod
    def _validate_run_config(config: RawStageConfig) -> None:
        if config.chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {config.chunk_rows}")
        if config.n_jobs < -1:
            raise ValueError(f"n_jobs must be -1, 0, or >= 1, got {config.n_jobs}")

    def _run_parallel_tasks(
        self,
        tasks: tuple[RawTask, ...],
        input_store: DataStore,
        writer: RawProtocolShardWriter,
        config: RawStageConfig,
        run_stats: RawRunStats,
        last_update_at: float,
    ) -> bool:
        failed_any = False
        max_workers = resolve_process_count(config.n_jobs, len(tasks))
        specs = tuple(
            ProcessTaskSpec(
                task_index=task_idx,
                task_id=task.task_id,
                arg=RawWorkerPayload(
                    task_index=task_idx,
                    task=task,
                    adapter=self.adapter,
                    input_store=input_store,
                    failure_mode=config.failure_mode,
                ),
            )
            for task_idx, task in enumerate(tasks, start=1)
        )

        for process_result in iter_ordered_process_results(
            prepare_raw_task_worker,
            specs,
            max_workers=max_workers,
            polars_max_threads=config.polars_max_threads,
        ):
            run_stats.add_worker_metrics(process_result.metrics)
            task = tasks[process_result.task_index - 1]
            if process_result.success and process_result.result is not None:
                self._consume_prepared_task_result(
                    writer,
                    process_result.result,
                    config.chunk_rows,
                    run_stats,
                )
            else:
                failed_any = True
                self._record_worker_failure(process_result, task, run_stats)
                if config.failure_mode == FailureMode.STRICT:
                    raise RuntimeError(
                        f"Raw task {process_result.task_id!r} failed in worker: "
                        f"{process_result.error_type}: {process_result.error}",
                    )
            last_update_at = self._log_status_if_due(
                process_result.task_index,
                last_update_at,
                run_stats,
            )
        return failed_any

    def prepare_task_result(
        self,
        task_index: int,
        task: RawTask,
        input_store: DataStore,
        failure_mode: FailureMode,
    ) -> PreparedRawTaskResult:
        logger.debug(
            "raw task started task=%d task_id=%s source_paths=%s",
            task_index,
            task.task_id,
            task.source_paths,
        )
        batches: list[PreparedRawBatch] = []
        stats = RawTaskStats()
        for batch in self.adapter.load_raw_task(task, input_store, failure_mode):
            stats = self._merge_task_stats(stats, batch.stats)
            data, align_stats = self._prepare_batch_data(batch, failure_mode)
            stats = self._merge_task_stats(stats, align_stats)
            batches.append(
                PreparedRawBatch(
                    data=data,
                    stream_id=batch.stream_id,
                    source_paths=batch.source_paths,
                    metadata=batch.metadata,
                    stats=batch.stats,
                ),
            )
        return PreparedRawTaskResult(
            task_index=task_index,
            task=task,
            batches=tuple(batches),
            stats=stats,
        )

    def _consume_prepared_task_result(
        self,
        writer: RawProtocolShardWriter,
        result: PreparedRawTaskResult,
        chunk_rows: int,
        run_stats: RawRunStats,
    ) -> None:
        run_stats.add_task_stats(result.stats)
        self._log_issues(result.stats.issues)
        task_batches, task_chunks, task_rows = self._append_prepared_batches(
            writer,
            iter(result.batches),
            chunk_rows,
        )
        run_stats.tasks_succeeded += 1
        run_stats.batches_written += task_batches
        run_stats.chunks_written += task_chunks
        run_stats.rows_written += task_rows
        logger.debug(
            "raw task finished task=%d/%d task_id=%s batches=%d chunks=%d rows=%d",
            result.task_index,
            run_stats.tasks_total,
            result.task.task_id,
            task_batches,
            task_chunks,
            task_rows,
        )

    def _record_task_failure(
        self,
        task_idx: int,
        task: RawTask,
        exc: Exception,
        run_stats: RawRunStats,
    ) -> None:
        run_stats.tasks_failed += 1
        logger.exception(
            "raw task failed task=%d/%d task_id=%s source_paths=%s error_type=%s",
            task_idx,
            run_stats.tasks_total,
            task.task_id,
            task.source_paths,
            type(exc).__name__,
        )

    def _record_worker_failure(
        self,
        process_result: ProcessTaskResult[PreparedRawTaskResult],
        task: RawTask,
        run_stats: RawRunStats,
    ) -> None:
        run_stats.tasks_failed += 1
        logger.error(
            "raw task failed task=%d/%d task_id=%s source_paths=%s error_type=%s error=%s",
            process_result.task_index,
            run_stats.tasks_total,
            task.task_id,
            task.source_paths,
            process_result.error_type,
            process_result.error,
        )

    def _log_status_if_due(
        self,
        task_idx: int,
        last_update_at: float,
        run_stats: RawRunStats,
    ) -> float:
        now = time.perf_counter()
        if now - last_update_at < _PROGRESS_TIME_INTERVAL_S:
            return last_update_at
        logger.info(
            "raw_to_parquet dataset=%s task=%d/%d succeeded=%d failed=%d warnings=%d",
            self.adapter.spec.dataset_id,
            task_idx,
            run_stats.tasks_total,
            run_stats.tasks_succeeded,
            run_stats.tasks_failed,
            run_stats.warning_count,
        )
        return now

    def _append_prepared_batches(
        self,
        writer: RawProtocolShardWriter,
        batches: Iterator[PreparedRawBatch],
        chunk_rows: int,
    ) -> tuple[int, int, int]:
        batch_count = 0
        chunk_count = 0
        row_count = 0
        for batch in batches:
            batch_count += 1
            for chunk in self._iter_data_chunks(batch.data, chunk_rows):
                writer.append(chunk, batch.metadata, batch.source_paths)
                chunk_count += 1
                row_count += chunk.height
        return batch_count, chunk_count, row_count

    def _raw_spec(self) -> RawIngestSpec:
        raw_spec = self.adapter.spec.raw
        if raw_spec is None:
            raise ValueError(
                f"Dataset {self.adapter.spec.dataset_id!r} does not support raw_to_parquet",
            )
        return raw_spec

    def _prepare_batch_data(
        self,
        batch: RawBatch,
        failure_mode: FailureMode,
    ) -> tuple[pl.DataFrame, RawTaskStats]:
        if isinstance(batch.data, pl.LazyFrame):
            collected = batch.data.collect()
            if not isinstance(collected, pl.DataFrame):
                raise TypeError(
                    "Expected LazyFrame.collect() to return DataFrame, "
                    f"got {type(collected).__name__}",
                )
            data = collected
        else:
            data = batch.data
        protocol_schema = self._validate_protocol_metadata(batch.metadata)

        exprs: list[pl.Expr] = []
        for column, value in batch.metadata.items():
            dtype = column.dtype
            expr = pl.lit(value)
            if dtype is not None:
                expr = expr.cast(dtype, strict=False)
            exprs.append(expr.alias(column))

        if exprs:
            data = data.with_columns(exprs)
        return self._align_to_protocol_schema(data, protocol_schema, failure_mode, batch)

    def _validate_protocol_metadata(
        self,
        metadata: dict[ColumnSpec, object],
    ) -> RawProtocolSchema:
        protocol_value = metadata.get(MetadataColumns.protocol)
        if protocol_value is None:
            raise ValueError("Raw batch metadata is missing protocol")

        raw_spec = self._raw_spec()
        try:
            protocol_schema = raw_spec.protocol_schema(protocol_value)
        except ValueError as exc:
            raise ValueError(
                f"Protocol {protocol_value!r} is not declared in dataset "
                f"{self.adapter.spec.dataset_id!r}",
            ) from exc

        missing = [column for column in protocol_schema.metadata if column not in metadata]
        if missing:
            raise ValueError(
                f"Protocol {protocol_value!r} metadata is missing required columns: {missing}",
            )

        if BaseColumns.cycle_index in metadata and metadata[BaseColumns.cycle_index] is None:
            raise ValueError(f"Protocol {protocol_value!r} metadata has null cycle_index")

        return protocol_schema

    def _align_to_protocol_schema(
        self,
        data: pl.DataFrame,
        protocol_schema: RawProtocolSchema,
        failure_mode: FailureMode,
        batch: RawBatch,
    ) -> tuple[pl.DataFrame, RawTaskStats]:
        output_columns = protocol_schema.output_columns
        output_set = set(output_columns)
        mapped_columns = set(collect_column_specs(self.adapter.spec.cols).values())
        extra_mapped = sorted(
            column for column in data.columns if column in mapped_columns - output_set
        )
        extra_source_columns = [
            column
            for column in data.columns
            if column not in output_set and column not in mapped_columns
        ]
        if extra_mapped:
            message = (
                f"Raw batch {batch.stream_id!r} for protocol {protocol_schema.protocol!r} has "
                f"mapped columns outside the declared protocol schema: {extra_mapped}"
            )
            issue = RawIngestIssue(
                kind="dropped_mapped_columns",
                source_paths=batch.source_paths,
                message=f"{message}; dropping columns",
                columns=tuple(extra_mapped),
            )
            if failure_mode == FailureMode.STRICT:
                raise ValueError(message)
            stats = RawTaskStats(
                dropped_mapped_columns=len(extra_mapped),
                issues=(issue,),
            )
        else:
            stats = RawTaskStats()

        exprs: list[pl.Expr] = []
        for column in output_columns:
            expr = pl.col(column) if column in data.columns else pl.lit(None)
            if column.dtype is not None:
                expr = expr.cast(column.dtype, strict=False)
            exprs.append(expr.alias(column))
        exprs.extend(pl.col(column) for column in extra_source_columns)

        return data.select(exprs), stats

    @staticmethod
    def _merge_task_stats(left: RawTaskStats, right: RawTaskStats) -> RawTaskStats:
        return RawTaskStats(
            dropped_unknown_columns=left.dropped_unknown_columns + right.dropped_unknown_columns,
            dropped_declared_columns=left.dropped_declared_columns + right.dropped_declared_columns,
            duplicate_columns=left.duplicate_columns + right.duplicate_columns,
            dropped_mapped_columns=left.dropped_mapped_columns + right.dropped_mapped_columns,
            issues=(*left.issues, *right.issues),
        )

    @staticmethod
    def _log_issues(issues: tuple[RawIngestIssue, ...]) -> None:
        for issue in issues:
            logger.warning(
                "raw ingest issue kind=%s source_paths=%s columns=%s message=%s",
                issue.kind,
                issue.source_paths,
                issue.columns,
                issue.message,
            )

    @staticmethod
    def _iter_data_chunks(data: pl.DataFrame, chunk_rows: int) -> Iterator[pl.DataFrame]:
        if data.height <= chunk_rows:
            yield data
            return

        for offset in range(0, data.height, chunk_rows):
            chunk = data.slice(offset, chunk_rows)
            if chunk.height > 0:
                yield chunk


def prepare_raw_task_worker(payload: RawWorkerPayload) -> PreparedRawTaskResult:
    return RawProcessor(payload.adapter).prepare_task_result(
        task_index=payload.task_index,
        task=payload.task,
        input_store=payload.input_store,
        failure_mode=payload.failure_mode,
    )
