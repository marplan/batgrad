from __future__ import annotations

import hashlib
import math
import os
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from batgrad import _loggers
from batgrad.contracts.columns import BaseColumns, BatteryColumns, ColumnSpec, MetadataColumns
from batgrad.data.processing.config import (
    PROCESSING_STAGE_SPECS,
    FailureMode,
    ProcessingStage,
)
from batgrad.data.processing.metadata import (
    extend_layout_with_group_columns,
    group_task_id,
    manifest_segment_file_paths,
    merge_manifest_raw_file_paths,
    merge_manifest_segments,
)
from batgrad.data.processing.runtime import (
    ProcessTaskResult,
    ProcessTaskSpec,
    WorkerMetrics,
    iter_stage_process_results,
    read_peak_rss_mb,
    resolve_stage_worker_count,
    validate_stage_runtime_config,
)
from batgrad.data.processing.schema import (
    coalesce_frames,
    collect_frame,
    iter_data_chunks,
    iter_parquet_chunks,
    select_and_cast_columns,
)
from batgrad.data.processing.sharding import ProtocolShardWriter
from batgrad.data.transforms.checks import (
    BoundedCheckState,
    CheckSpecBase,
    ColumnBoundsCheckSpec,
    MissingCheckSpec,
    apply_checks_bounded_chunk,
    apply_checks_full_task,
    collect_check_failures_bounded_chunk,
    collect_check_failures_full_task,
)
from batgrad.data.transforms.resampling import (
    DOWNSAMPLE_OVERSAMPLE_FACTOR,
    LinearResamplingSpec,
    MinMaxLTTBResamplingSpec,
    ResamplingSpecBase,
    apply_physics_preserving_downsampling,
    downsample_min_max_lttb_frame,
    resampling_metadata_values,
    resolve_downsampling_signal_col,
    resolve_min_max_lttb_budget,
    run_resampling,
    select_min_max_lttb_row_ids,
)
from batgrad.data.transforms.transforms import (
    apply_transforms_bounded_chunk,
    apply_transforms_full_task,
    transform_source_columns,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.data.datasets.specs import DatasetSpec
    from batgrad.data.processing.config import NormalizeStageConfig
    from batgrad.data.processing.normalize_spec import NormalizeSpec, ProtocolNormalizeSpec
    from batgrad.storage.store import DataStore

logger = _loggers.get_logger(__name__)
_PROGRESS_TIME_INTERVAL_S = 30.0
_DOWNSAMPLE_ROW_ID = "__normalize_downsample_row_id"
_ANNOTATION_CHECK_TYPES = (MissingCheckSpec, ColumnBoundsCheckSpec)
_ANNOTATION_ITEM_SEPARATOR = "\x1f"
_MAX_FLAT_ANNOTATIONS = 16
_MAX_CHECK_FAILURES_IN_ERROR = 5


@dataclass(frozen=True, slots=True)
class NormalizeTask:
    task_id: str
    protocol: str
    group_values: dict[ColumnSpec, object]
    file_paths: tuple[str, ...]
    raw_file_paths: tuple[str, ...]
    parquet_segments: tuple[dict[str, object], ...]
    row_count: int


@dataclass(slots=True)
class NormalizeRunStats:
    tasks_total: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    chunks_written: int = 0
    rows_written: int = 0
    worker_peak_rss_mb_max: float = 0.0

    def add_worker_metrics(self, metrics: WorkerMetrics) -> None:
        self.worker_peak_rss_mb_max = max(
            self.worker_peak_rss_mb_max,
            metrics.worker_peak_rss_mb,
        )


@dataclass(frozen=True, slots=True)
class NormalizeWorkerPayload:
    spec: DatasetSpec
    task_index: int
    task: NormalizeTask
    input_store: DataStore
    config: NormalizeStageConfig
    temp_root: str
    temp_run_id: str
    annotate: bool = False


@dataclass(frozen=True, slots=True)
class PreparedNormalizeTaskResult:
    task: NormalizeTask
    metadata: dict[ColumnSpec, object]
    raw_file_paths: tuple[str, ...]
    temp_output_path: str | None
    chunks_produced: int = 0
    rows_produced: int = 0


@dataclass(frozen=True, slots=True)
class NormalizeInteractiveRun:
    output_store: DataStore
    run_root: str
    manifest_path: str
    task_ids: tuple[str, ...]
    rows_written: int
    chunks_written: int

    def manifest(self) -> pl.DataFrame:
        return collect_frame(self.output_store.scan_table(self.manifest_path))

    def scan(self) -> pl.LazyFrame:
        manifest = self.manifest()
        paths: list[str] = []
        for segments in manifest[MetadataColumns.normalized_segments].to_list():
            for segment in segments or ():
                path = str(segment[BaseColumns.file_path])
                if path not in paths:
                    paths.append(path)
        if not paths:
            raise FileNotFoundError("Interactive normalize run has no normalized output segments")
        return self.output_store.scan_table(tuple(paths))

    def clean(self) -> None:
        parts = PurePosixPath(self.run_root).parts
        if "scratch" not in parts:
            raise ValueError(f"Refusing to clean non-scratch normalize run: {self.run_root}")
        scratch_index = parts.index("scratch")
        if len(parts) != scratch_index + 2:
            raise ValueError(f"Refusing to clean non-run scratch path: {self.run_root}")
        self.output_store.delete_dir(self.run_root, missing_ok=True)


def prepare_normalize_task_worker(
    payload: NormalizeWorkerPayload,
) -> PreparedNormalizeTaskResult:
    return NormalizeProcessor(payload.spec).prepare_task_result_to_temp(payload)


class NormalizeProcessor:
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    def run(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: NormalizeStageConfig,
    ) -> None:
        normalize_spec = self._normalize_spec()
        validate_stage_runtime_config(config)
        self._validate_normalize_spec(normalize_spec)

        stage_spec = PROCESSING_STAGE_SPECS[ProcessingStage.NORMALIZE]
        if config.failure_mode == FailureMode.DRY_RUN:
            temp_root = self._temp_normalize_root_path(output_store)
        else:
            temp_root = self._temp_normalize_root(output_store)
        temp_run_id = self._temp_normalize_run_id()
        writer = ProtocolShardWriter(
            output_store=output_store,
            spec=self.spec,
            stage_spec=stage_spec,
            config=config,
            manifest_layout=extend_layout_with_group_columns(
                stage_spec.manifest_layout,
                normalize_spec.protocol_specs,
            ),
            footer_layout=stage_spec.footer_layout,
            footer_metadata={MetadataColumns.time_convention: normalize_spec.time_convention},
            footer_metadata_columns=(
                MetadataColumns.resampling_method,
                MetadataColumns.resampling_params,
            ),
        )
        failed_any = False
        started_at = time.perf_counter()
        run_stats = NormalizeRunStats()
        try:
            tasks = self.plan_tasks(input_store)
            run_stats.tasks_total = len(tasks)
            if run_stats.tasks_total > 0:
                logger.info(
                    "normalize dataset=%s task=%d/%d succeeded=%d failed=%d",
                    self.spec.dataset_id,
                    1,
                    run_stats.tasks_total,
                    run_stats.tasks_succeeded,
                    run_stats.tasks_failed,
                )
            failed_any = self._run_tasks(
                tasks,
                input_store,
                writer,
                config,
                run_stats,
                temp_root,
                temp_run_id,
                annotate=False,
            )
        except BaseException:
            writer.close(manifest="skip")
            raise
        else:
            manifest = "skip" if config.failure_mode == FailureMode.DRY_RUN else "write"
            if failed_any and config.failure_mode == FailureMode.STRICT:
                manifest = "skip"
            writer.close(manifest=manifest)
            if manifest == "write" or config.failure_mode == FailureMode.DRY_RUN:
                self._cleanup_temp_run_dir(temp_root, temp_run_id)
            logger.info(
                "normalize finished dataset=%s tasks=%d succeeded=%d failed=%d chunks=%d "
                "rows=%d manifest=%s failure_mode=%s elapsed_s=%.1f worker_peak_rss_mb=%.1f "
                "main_peak_rss_mb=%.1f",
                self.spec.dataset_id,
                run_stats.tasks_total,
                run_stats.tasks_succeeded,
                run_stats.tasks_failed,
                run_stats.chunks_written,
                run_stats.rows_written,
                "skipped" if manifest == "skip" else "written",
                config.failure_mode,
                time.perf_counter() - started_at,
                run_stats.worker_peak_rss_mb_max,
                read_peak_rss_mb(),
            )

    def run_interactive(
        self,
        input_store: DataStore,
        scratch_store: DataStore,
        protocol: str,
        group_values: dict[ColumnSpec, object],
        config: NormalizeStageConfig,
        *,
        annotate: bool = True,
    ) -> NormalizeInteractiveRun:
        if config.failure_mode == FailureMode.DRY_RUN:
            raise ValueError("normalize_interactive requires a materializing failure_mode")

        normalize_spec = self._normalize_spec()
        validate_stage_runtime_config(config)
        self._validate_normalize_spec(normalize_spec)
        tasks = self._select_interactive_tasks(input_store, protocol, group_values)
        run_root = self._interactive_run_root()

        stage_spec = PROCESSING_STAGE_SPECS[ProcessingStage.NORMALIZE]
        temp_root = self._temp_normalize_root(scratch_store, output_root=run_root)
        temp_run_id = self._temp_normalize_run_id()
        writer = ProtocolShardWriter(
            output_store=scratch_store,
            spec=self.spec,
            stage_spec=stage_spec,
            config=config,
            manifest_layout=extend_layout_with_group_columns(
                stage_spec.manifest_layout,
                normalize_spec.protocol_specs,
            ),
            footer_layout=stage_spec.footer_layout,
            footer_metadata={MetadataColumns.time_convention: normalize_spec.time_convention},
            footer_metadata_columns=(
                MetadataColumns.resampling_method,
                MetadataColumns.resampling_params,
            ),
            output_root=run_root,
        )
        run_stats = NormalizeRunStats(tasks_total=len(tasks))
        failed_any = False
        try:
            failed_any = self._run_tasks(
                tasks,
                input_store,
                writer,
                config,
                run_stats,
                temp_root,
                temp_run_id,
                annotate=annotate,
            )
        except BaseException:
            writer.close(manifest="skip")
            raise
        manifest = "skip" if failed_any and config.failure_mode == FailureMode.STRICT else "write"
        writer.close(manifest=manifest)
        if manifest == "write":
            self._cleanup_temp_run_dir(temp_root, temp_run_id)
        return NormalizeInteractiveRun(
            output_store=scratch_store,
            run_root=run_root,
            manifest_path=f"{run_root}/manifest.parquet",
            task_ids=tuple(task.task_id for task in tasks),
            rows_written=run_stats.rows_written,
            chunks_written=run_stats.chunks_written,
        )

    def plan_tasks(self, input_store: DataStore) -> tuple[NormalizeTask, ...]:
        manifest_path = self.spec.location.manifest("parquet")
        if not input_store.exists(manifest_path):
            raise FileNotFoundError(f"Raw parquet manifest does not exist: {manifest_path}")

        manifest = collect_frame(input_store.scan_table(manifest_path))
        normalize_spec = self._normalize_spec()
        tasks: list[NormalizeTask] = []
        for protocol, protocol_spec in normalize_spec.protocol_specs.items():
            self._validate_manifest_columns(manifest, protocol, protocol_spec)
            protocol_rows = manifest.filter(pl.col(MetadataColumns.protocol) == protocol)
            groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
            for row in protocol_rows.iter_rows(named=True):
                key = tuple(row[column] for column in protocol_spec.group_by)
                groups.setdefault(key, []).append(row)

            for group_key, rows in groups.items():
                group_values = dict(zip(protocol_spec.group_by, group_key, strict=True))
                parquet_segments = merge_manifest_segments(rows, MetadataColumns.parquet_segments)
                file_paths = manifest_segment_file_paths(parquet_segments)
                raw_file_paths = merge_manifest_raw_file_paths(rows)
                row_count = sum(int(cast("str | int", row[BaseColumns.row_count])) for row in rows)
                tasks.append(
                    NormalizeTask(
                        task_id=group_task_id(protocol, group_values),
                        protocol=protocol,
                        group_values=group_values,
                        file_paths=file_paths,
                        raw_file_paths=raw_file_paths,
                        parquet_segments=parquet_segments,
                        row_count=row_count,
                    ),
                )
        return tuple(tasks)

    def _select_interactive_tasks(
        self,
        input_store: DataStore,
        protocol: str,
        group_values: dict[ColumnSpec, object],
    ) -> tuple[NormalizeTask, ...]:
        tasks = tuple(
            task
            for task in self.plan_tasks(input_store)
            if task.protocol == protocol
            and all(
                task.group_values.get(column) == value for column, value in group_values.items()
            )
        )
        if not tasks:
            raise ValueError(
                f"No normalize task matched protocol={protocol!r} group_values={group_values!r}",
            )
        if len(tasks) > 1:
            task_ids = [task.task_id for task in tasks]
            raise ValueError(
                f"Interactive normalize matched multiple tasks; narrow group_values: {task_ids}",
            )
        return tasks

    def _run_tasks(
        self,
        tasks: tuple[NormalizeTask, ...],
        input_store: DataStore,
        writer: ProtocolShardWriter,
        config: NormalizeStageConfig,
        run_stats: NormalizeRunStats,
        temp_root: Path,
        temp_run_id: str,
        *,
        annotate: bool,
    ) -> bool:
        if resolve_stage_worker_count(config.n_jobs, len(tasks)) == 1:
            return self._run_sequential_tasks(
                tasks,
                input_store,
                writer,
                config,
                run_stats,
                temp_root,
                temp_run_id,
                annotate=annotate,
            )

        failed_any = False
        last_update_at = time.perf_counter()
        specs = tuple(
            ProcessTaskSpec(
                task_index=task_index,
                task_id=task.task_id,
                arg=NormalizeWorkerPayload(
                    spec=self.spec,
                    task_index=task_index,
                    task=task,
                    input_store=input_store,
                    config=config,
                    temp_root=str(temp_root),
                    temp_run_id=temp_run_id,
                    annotate=annotate,
                ),
            )
            for task_index, task in enumerate(tasks, start=1)
        )
        for process_result in iter_stage_process_results(
            prepare_normalize_task_worker,
            specs,
            config,
        ):
            run_stats.add_worker_metrics(process_result.metrics)
            task = tasks[process_result.task_index - 1]
            if process_result.success and process_result.result is not None:
                chunks, rows = self._consume_prepared_task_result(
                    writer,
                    process_result.result,
                    config,
                )
                run_stats.tasks_succeeded += 1
                run_stats.chunks_written += chunks
                run_stats.rows_written += rows
            else:
                failed_any = True
                self._record_worker_failure(process_result, task, run_stats)
                if config.failure_mode == FailureMode.STRICT:
                    raise RuntimeError(
                        f"Normalize task {process_result.task_id!r} failed in worker: "
                        f"{process_result.error_type}: {process_result.error}",
                    )
            last_update_at = self._log_status_if_due(last_update_at, run_stats)
        return failed_any

    def _run_sequential_tasks(
        self,
        tasks: tuple[NormalizeTask, ...],
        input_store: DataStore,
        writer: ProtocolShardWriter,
        config: NormalizeStageConfig,
        run_stats: NormalizeRunStats,
        temp_root: Path,
        temp_run_id: str,
        *,
        annotate: bool,
    ) -> bool:
        failed_any = False
        last_update_at = time.perf_counter()
        for task_index, task in enumerate(tasks, start=1):
            try:
                chunks, rows = self._write_task_to_final(
                    task,
                    input_store,
                    writer,
                    config,
                    temp_root,
                    temp_run_id,
                    annotate=annotate,
                )
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failed_any = True
                self._record_task_failure(task_index, task, exc, run_stats)
                if config.failure_mode == FailureMode.STRICT:
                    raise
            else:
                run_stats.tasks_succeeded += 1
                run_stats.chunks_written += chunks
                run_stats.rows_written += rows
            last_update_at = self._log_status_if_due(last_update_at, run_stats)
        return failed_any

    def _write_task_to_final(
        self,
        task: NormalizeTask,
        input_store: DataStore,
        writer: ProtocolShardWriter,
        config: NormalizeStageConfig,
        temp_root: Path,
        temp_run_id: str,
        *,
        annotate: bool,
    ) -> tuple[int, int]:
        protocol_spec = self._normalize_spec().protocol_specs[task.protocol]
        if config.failure_mode == FailureMode.DRY_RUN:
            return self._dry_run_task_counts(task, protocol_spec, input_store, config)

        metadata = self._task_metadata(
            task,
            protocol_spec,
            apply_resampling=config.apply_resampling,
        )
        chunks_written = 0
        rows_written = 0
        for chunk in self._iter_task_output_chunks(
            task,
            input_store,
            config,
            temp_root,
            temp_run_id,
            annotate=annotate,
        ):
            if config.failure_mode != FailureMode.DRY_RUN:
                writer.append(chunk, metadata, task.raw_file_paths)
            chunks_written += 1
            rows_written += chunk.height
        return chunks_written, rows_written

    def _iter_task_output_chunks(
        self,
        task: NormalizeTask,
        input_store: DataStore,
        config: NormalizeStageConfig,
        temp_root: Path,
        temp_run_id: str,
        *,
        annotate: bool,
    ) -> Iterator[pl.DataFrame]:
        protocol_spec = self._normalize_spec().protocol_specs[task.protocol]
        resampling = protocol_spec.resampling
        requires_bounded = self._requires_bounded_task(task, config)
        if (
            config.apply_resampling
            and isinstance(resampling, MinMaxLTTBResamplingSpec)
            and requires_bounded
        ):
            yield from self._iter_task_resampling_bounded_temp_chunks(
                task,
                protocol_spec,
                input_store,
                resampling,
                config,
                temp_root,
                temp_run_id,
                annotate=annotate,
            )
            return

        if requires_bounded:
            if config.apply_resampling and resampling is not None:
                raise RuntimeError(
                    f"Normalize task {task.task_id!r} requires bounded execution, but "
                    f"resampling {type(resampling).__name__} has no bounded path",
                )
            yield from self._iter_task_bounded_checked_chunks(
                task,
                protocol_spec,
                input_store,
                config,
                annotate=annotate,
            )
            return

        input_columns, resolved_columns = self._task_input_resolution(
            task,
            protocol_spec,
            input_store,
        )
        data = input_store.scan_table(
            task.file_paths,
            columns=input_columns,
            filters=self._task_filter(task),
        )
        sort_columns = (*protocol_spec.group_by, *protocol_spec.order_by)
        if sort_columns:
            data = data.sort(list(sort_columns))
        data = apply_transforms_full_task(data, protocol_spec.transforms)
        data = apply_checks_full_task(
            data,
            self.spec,
            protocol_spec.group_by,
            self._transform_checks(protocol_spec.checks),
        )
        data = select_and_cast_columns(
            data,
            self._internal_output_columns(protocol_spec),
            resolved_columns=resolved_columns,
        )
        if not isinstance(data, pl.LazyFrame):
            raise TypeError(f"Expected LazyFrame, got {type(data).__name__}")
        if annotate:
            data = apply_checks_full_task(
                data,
                self.spec,
                protocol_spec.group_by,
                self._annotation_checks(protocol_spec.checks),
            )
        else:
            self._raise_for_check_failures(
                task,
                collect_check_failures_full_task(
                    data,
                    self.spec,
                    protocol_spec.group_by,
                    self._annotation_checks(protocol_spec.checks),
                ),
            )

        if config.apply_resampling and resampling is not None:
            task_data = self._apply_task_resampling(task, data, resampling, config)
        else:
            task_data = collect_frame(data)
        if task_data.height == 0:
            return
        task_data = self._finalize_output_columns(
            task_data,
            protocol_spec,
            include_annotations=annotate,
        )
        yield from iter_data_chunks(task_data, config.chunk_rows)

    @staticmethod
    def _requires_bounded_task(task: NormalizeTask, config: NormalizeStageConfig) -> bool:
        return config.max_batch_rows is not None and task.row_count > config.max_batch_rows

    @staticmethod
    def _transform_checks(checks: tuple[CheckSpecBase, ...]) -> tuple[CheckSpecBase, ...]:
        return tuple(check for check in checks if not isinstance(check, _ANNOTATION_CHECK_TYPES))

    @staticmethod
    def _annotation_checks(checks: tuple[CheckSpecBase, ...]) -> tuple[CheckSpecBase, ...]:
        return tuple(check for check in checks if isinstance(check, _ANNOTATION_CHECK_TYPES))

    @staticmethod
    def _raise_for_check_failures(task: NormalizeTask, failures: tuple[object, ...]) -> None:
        if not failures:
            return
        details = "; ".join(str(failure) for failure in failures[:_MAX_CHECK_FAILURES_IN_ERROR])
        extra = (
            ""
            if len(failures) <= _MAX_CHECK_FAILURES_IN_ERROR
            else f"; ... {len(failures) - _MAX_CHECK_FAILURES_IN_ERROR} more"
        )
        raise RuntimeError(f"Normalize task {task.task_id!r} failed checks: {details}{extra}")

    def _finalize_output_columns(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        protocol_spec: ProtocolNormalizeSpec,
        *,
        include_annotations: bool,
    ) -> pl.DataFrame:
        frame = collect_frame(
            self._convert_flat_annotations(data) if include_annotations else data,
        )
        return collect_frame(
            select_and_cast_columns(
                frame,
                self._output_columns(protocol_spec, include_annotations=include_annotations),
            ),
        )

    @staticmethod
    def _convert_flat_annotations(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        columns = data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.columns
        if (
            BaseColumns.annotation_columns not in columns
            or BaseColumns.annotation_reasons not in columns
        ):
            return data

        if isinstance(data, pl.LazyFrame):
            return NormalizeProcessor._convert_flat_annotations_generic(data)

        has_annotations = data.select(
            pl.col(BaseColumns.annotation_columns).is_not_null().any(),
        ).item()
        if not has_annotations:
            return data.with_columns(
                pl.lit(None, dtype=NormalizeProcessor._annotation_dtype()).alias(
                    BaseColumns.annotations,
                ),
            ).drop(BaseColumns.annotation_columns, BaseColumns.annotation_reasons)

        has_multi_annotations = data.select(
            pl.col(BaseColumns.annotation_columns)
            .str.contains(_ANNOTATION_ITEM_SEPARATOR, literal=True)
            .fill_null(value=False)
            .any(),
        ).item()
        if not has_multi_annotations:
            return data.with_columns(
                NormalizeProcessor._single_flat_annotation_expr(),
            ).drop(BaseColumns.annotation_columns, BaseColumns.annotation_reasons)

        row_idx = "__annotation_row_idx"
        indexed = data.with_row_index(row_idx)
        is_annotated = pl.col(BaseColumns.annotation_columns).is_not_null()
        is_multi = (
            pl.col(BaseColumns.annotation_columns)
            .str.contains(_ANNOTATION_ITEM_SEPARATOR, literal=True)
            .fill_null(value=False)
        )
        annotation_parts = [
            indexed.filter(~is_annotated).select(
                row_idx,
                pl.lit(None, dtype=NormalizeProcessor._annotation_dtype()).alias(
                    BaseColumns.annotations,
                ),
            ),
            indexed.filter(is_annotated & ~is_multi).select(
                row_idx,
                NormalizeProcessor._single_flat_annotation_expr(),
            ),
        ]
        multi = indexed.filter(is_multi)
        if multi.height > 0:
            multi_annotations = cast(
                "pl.DataFrame",
                NormalizeProcessor._convert_flat_annotations_generic(multi),
            )
            annotation_parts.append(
                multi_annotations.select(
                    row_idx,
                    BaseColumns.annotations,
                ),
            )
        annotations = (
            pl.concat(annotation_parts, how="vertical")
            .sort(row_idx)
            .select(BaseColumns.annotations)
        )
        return data.with_columns(annotations[BaseColumns.annotations]).drop(
            BaseColumns.annotation_columns,
            BaseColumns.annotation_reasons,
        )

    @staticmethod
    def _single_flat_annotation_expr() -> pl.Expr:
        annotation = pl.struct(
            pl.col(BaseColumns.annotation_columns).alias("column"),
            pl.col(BaseColumns.annotation_reasons).alias("reason"),
        )
        return (
            pl.when(pl.col(BaseColumns.annotation_columns).is_not_null())
            .then(pl.concat_list(annotation))
            .otherwise(None)
            .cast(NormalizeProcessor._annotation_dtype())
            .alias(BaseColumns.annotations)
        )

    @staticmethod
    def _annotation_dtype() -> pl.DataType:
        dtype = BaseColumns.annotations.dtype
        if dtype is None:
            raise ValueError(f"Column {BaseColumns.annotations!r} has no dtype")
        return dtype

    @staticmethod
    def _convert_flat_annotations_generic(
        data: pl.DataFrame | pl.LazyFrame,
    ) -> pl.DataFrame | pl.LazyFrame:
        annotation_columns = pl.col(BaseColumns.annotation_columns).str.split(
            _ANNOTATION_ITEM_SEPARATOR,
        )
        annotation_reasons = pl.col(BaseColumns.annotation_reasons).str.split(
            _ANNOTATION_ITEM_SEPARATOR,
        )
        structs: list[pl.Expr] = []
        for idx in range(_MAX_FLAT_ANNOTATIONS):
            column_value = annotation_columns.list.get(idx, null_on_oob=True)
            reason_value = annotation_reasons.list.get(idx, null_on_oob=True)
            structs.append(
                pl.when(column_value.is_not_null() & reason_value.is_not_null())
                .then(pl.struct(column_value.alias("column"), reason_value.alias("reason")))
                .otherwise(None),
            )
        return data.with_columns(
            pl.when(pl.col(BaseColumns.annotation_columns).is_not_null())
            .then(pl.concat_list(structs).list.drop_nulls())
            .otherwise(None)
            .alias(BaseColumns.annotations),
        ).drop(BaseColumns.annotation_columns, BaseColumns.annotation_reasons)

    def prepare_task_result_to_temp(
        self,
        payload: NormalizeWorkerPayload,
    ) -> PreparedNormalizeTaskResult:
        protocol_spec = self._normalize_spec().protocol_specs[payload.task.protocol]
        metadata = self._task_metadata(
            payload.task,
            protocol_spec,
            apply_resampling=payload.config.apply_resampling,
        )
        if payload.config.failure_mode == FailureMode.DRY_RUN:
            chunks_produced, rows_produced = self._dry_run_task_counts(
                payload.task,
                protocol_spec,
                payload.input_store,
                payload.config,
            )
            return PreparedNormalizeTaskResult(
                task=payload.task,
                metadata=metadata,
                raw_file_paths=payload.task.raw_file_paths,
                temp_output_path=None,
                chunks_produced=chunks_produced,
                rows_produced=rows_produced,
            )

        temp_root = Path(payload.temp_root)
        temp_output_path = self._temp_normalize_path(
            payload.task,
            temp_root,
            payload.temp_run_id,
            kind="output",
        )
        writer: pq.ParquetWriter | None = None
        rows_written = 0
        try:
            for chunk in self._iter_task_output_chunks(
                payload.task,
                payload.input_store,
                payload.config,
                temp_root,
                payload.temp_run_id,
                annotate=payload.annotate,
            ):
                table = chunk.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_output_path,
                        table.schema,
                        compression=payload.config.compression,
                    )
                writer.write_table(table, row_group_size=payload.config.row_group_size)
                rows_written += chunk.height
        finally:
            if writer is not None:
                writer.close()
        if rows_written == 0:
            temp_output_path.unlink(missing_ok=True)
            temp_output_path_value = None
        else:
            temp_output_path_value = str(temp_output_path)
        return PreparedNormalizeTaskResult(
            task=payload.task,
            metadata=metadata,
            raw_file_paths=payload.task.raw_file_paths,
            temp_output_path=temp_output_path_value,
        )

    def _consume_prepared_task_result(
        self,
        writer: ProtocolShardWriter,
        result: PreparedNormalizeTaskResult,
        config: NormalizeStageConfig,
    ) -> tuple[int, int]:
        if result.temp_output_path is None:
            return result.chunks_produced, result.rows_produced
        temp_output_path = Path(result.temp_output_path)
        chunks_written = 0
        rows_written = 0
        for chunk in iter_parquet_chunks(temp_output_path, config.chunk_rows):
            if config.failure_mode != FailureMode.DRY_RUN:
                writer.append(chunk, result.metadata, result.raw_file_paths)
            chunks_written += 1
            rows_written += chunk.height
        temp_output_path.unlink(missing_ok=True)
        return chunks_written, rows_written

    @staticmethod
    def _task_filter(task: NormalizeTask) -> pl.Expr:
        filters = pl.col(MetadataColumns.protocol) == task.protocol
        for column, value in task.group_values.items():
            filters &= pl.col(column) == value
        return filters

    def _iter_task_segment_frames(
        self,
        task: NormalizeTask,
        input_store: DataStore,
        columns: tuple[str, ...],
        chunk_rows: int,
    ) -> Iterator[pl.DataFrame]:
        segments_by_file: dict[str, list[tuple[int, int]]] = {}
        for segment in task.parquet_segments:
            file_path = str(segment[BaseColumns.file_path])
            row_start = int(cast("str | int", segment[MetadataColumns.row_start]))
            row_count = int(cast("str | int", segment[BaseColumns.row_count]))
            if row_count > 0:
                segments_by_file.setdefault(file_path, []).append((row_start, row_count))

        for file_path, segments in segments_by_file.items():
            parquet_file = pq.ParquetFile(input_store.resolve(file_path))
            row_group_starts: list[int] = []
            cursor = 0
            for idx in range(parquet_file.metadata.num_row_groups):
                row_group_starts.append(cursor)
                cursor += parquet_file.metadata.row_group(idx).num_rows

            for row_start, row_count in segments:
                row_end = row_start + row_count
                for row_group_idx, group_start in enumerate(row_group_starts):
                    group_rows = parquet_file.metadata.row_group(row_group_idx).num_rows
                    group_end = group_start + group_rows
                    if group_end <= row_start:
                        continue
                    if group_start >= row_end:
                        break
                    table = parquet_file.read_row_group(
                        row_group_idx,
                        columns=[str(column) for column in columns],
                    )
                    frame = pl.from_arrow(table)
                    if not isinstance(frame, pl.DataFrame):
                        raise TypeError(
                            "Expected row-group conversion to return DataFrame, "
                            f"got {type(frame).__name__}",
                        )
                    local_start = max(0, row_start - group_start)
                    local_end = min(group_rows, row_end - group_start)
                    yield from iter_data_chunks(
                        frame.slice(local_start, local_end - local_start),
                        chunk_rows,
                    )

    def _apply_task_resampling(
        self,
        task: NormalizeTask,
        data: pl.LazyFrame,
        resampling: ResamplingSpecBase,
        config: NormalizeStageConfig,
    ) -> pl.DataFrame:
        if isinstance(resampling, LinearResamplingSpec):
            return run_resampling(collect_frame(data), resampling)

        if not isinstance(resampling, MinMaxLTTBResamplingSpec):
            raise TypeError(f"Normalize resampling {type(resampling).__name__} is not supported")

        if config.max_batch_rows is not None and task.row_count > config.max_batch_rows:
            raise RuntimeError(
                "large downsampled tasks must use the bounded task chunk path",
            )

        row_count = int(collect_frame(data.select(pl.len())).item())
        if row_count == 0:
            return collect_frame(data.limit(0))
        budget = resolve_min_max_lttb_budget(resampling, row_count)
        if row_count <= budget:
            return collect_frame(data)

        source = data.with_row_index(_DOWNSAMPLE_ROW_ID)
        sampled = self._downsample_min_max_lttb(
            source,
            row_count=row_count,
            budget=budget,
            spec=resampling,
            max_chunk_rows=config.max_batch_rows,
        )
        if sampled.height == 0:
            return sampled.drop(_DOWNSAMPLE_ROW_ID)

        signal_col = resolve_downsampling_signal_col(source.collect_schema().names())
        if signal_col is None:
            return sampled.drop(_DOWNSAMPLE_ROW_ID)

        compensated = apply_physics_preserving_downsampling(
            source_lf=source,
            sampled_lf=sampled.with_row_index("__physics_sample_pos").lazy(),
            row_count=row_count,
            row_id_col=_DOWNSAMPLE_ROW_ID,
            dt_col=BatteryColumns.dt,
            time_col=BatteryColumns.time,
            signal_col=signal_col,
        )
        logger.debug(
            "normalize downsampled dataset=%s protocol=%s task_id=%s rows_in=%d rows_out=%d",
            self.spec.dataset_id,
            task.protocol,
            task.task_id,
            row_count,
            sampled.height,
        )
        return collect_frame(compensated)

    def _dry_run_task_counts(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        config: NormalizeStageConfig,
    ) -> tuple[int, int]:
        rows_produced = (
            self._dry_run_bounded_task_rows(task, protocol_spec, input_store, config)
            if self._requires_bounded_task(task, config)
            else self._dry_run_full_task_rows(task, protocol_spec, input_store, config)
        )
        if rows_produced == 0:
            return 0, 0
        return math.ceil(rows_produced / config.chunk_rows), rows_produced

    def _dry_run_full_task_rows(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        config: NormalizeStageConfig,
    ) -> int:
        input_columns, resolved_columns = self._task_input_resolution(
            task,
            protocol_spec,
            input_store,
        )
        data = input_store.scan_table(
            task.file_paths,
            columns=input_columns,
            filters=self._task_filter(task),
        )
        sort_columns = (*protocol_spec.group_by, *protocol_spec.order_by)
        if sort_columns:
            data = data.sort(list(sort_columns))
        data = apply_transforms_full_task(data, protocol_spec.transforms)
        data = apply_checks_full_task(
            data,
            self.spec,
            protocol_spec.group_by,
            self._transform_checks(protocol_spec.checks),
        )
        data = select_and_cast_columns(
            data,
            self._internal_output_columns(protocol_spec),
            resolved_columns=resolved_columns,
        )
        if not isinstance(data, pl.LazyFrame):
            raise TypeError(f"Expected LazyFrame, got {type(data).__name__}")
        self._raise_for_check_failures(
            task,
            collect_check_failures_full_task(
                data,
                self.spec,
                protocol_spec.group_by,
                self._annotation_checks(protocol_spec.checks),
            ),
        )

        row_count = int(collect_frame(data.select(pl.len())).item())
        resampling = protocol_spec.resampling if config.apply_resampling else None
        if row_count == 0 or resampling is None:
            return row_count
        if isinstance(resampling, MinMaxLTTBResamplingSpec):
            return min(row_count, resolve_min_max_lttb_budget(resampling, row_count))
        if isinstance(resampling, LinearResamplingSpec):
            return resampling.points
        raise TypeError(f"Normalize resampling {type(resampling).__name__} is not supported")

    def _dry_run_bounded_task_rows(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        config: NormalizeStageConfig,
    ) -> int:
        chunk_rows = config.max_batch_rows
        if chunk_rows is None or chunk_rows < 1:
            raise ValueError(f"max_batch_rows must be >= 1, got {chunk_rows}")

        input_columns, resolved_columns = self._task_input_resolution(
            task,
            protocol_spec,
            input_store,
        )
        check_state = BoundedCheckState()
        transform_checks = self._transform_checks(protocol_spec.checks)
        annotation_checks = self._annotation_checks(protocol_spec.checks)
        rows_produced = 0
        for input_frame in self._iter_task_segment_frames(
            task,
            input_store,
            input_columns,
            chunk_rows,
        ):
            frame = input_frame.filter(self._task_filter(task))
            if frame.height == 0:
                continue
            frame = apply_transforms_bounded_chunk(frame, protocol_spec.transforms)
            frame = apply_checks_bounded_chunk(
                frame,
                self.spec,
                transform_checks,
                check_state,
            )
            if frame.height == 0:
                continue
            frame = collect_frame(
                select_and_cast_columns(
                    frame,
                    self._internal_output_columns(protocol_spec),
                    resolved_columns=resolved_columns,
                ),
            )
            self._raise_for_check_failures(
                task,
                collect_check_failures_bounded_chunk(
                    frame,
                    self.spec,
                    annotation_checks,
                ),
            )
            rows_produced += frame.height

        resampling = protocol_spec.resampling if config.apply_resampling else None
        if rows_produced == 0 or resampling is None:
            return rows_produced
        if isinstance(resampling, MinMaxLTTBResamplingSpec):
            return min(rows_produced, resolve_min_max_lttb_budget(resampling, rows_produced))
        raise RuntimeError(
            f"Normalize task {task.task_id!r} requires bounded execution, but "
            f"resampling {type(resampling).__name__} has no bounded dry-run path",
        )

    @staticmethod
    def _temp_normalize_run_id() -> str:
        value = f"{time.time_ns()}-{os.getpid()}".encode()
        digest = hashlib.blake2s(value, digest_size=5).hexdigest()
        return f"{os.getpid():x}-{digest}"

    def _interactive_run_root(self) -> str:
        run_id = NormalizeProcessor._temp_normalize_run_id()
        return f"{self.spec.location.source_root('normalized')}/scratch/{run_id}"

    def _temp_normalize_root_path(
        self,
        output_store: DataStore,
        *,
        output_root: str | None = None,
    ) -> Path:
        stage_spec = PROCESSING_STAGE_SPECS[ProcessingStage.NORMALIZE]
        root = output_root or self.spec.location.source_root(stage_spec.output_source)
        path = Path(output_store.resolve(root))
        return path / "_tmp"

    def _temp_normalize_root(
        self,
        output_store: DataStore,
        *,
        output_root: str | None = None,
    ) -> Path:
        temp_root = self._temp_normalize_root_path(output_store, output_root=output_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root

    @staticmethod
    def _cleanup_temp_run_dir(temp_root: Path, temp_run_id: str) -> None:
        run_root = temp_root / temp_run_id
        if run_root.exists():
            try:
                shutil.rmtree(run_root)
            except OSError as exc:
                logger.warning(
                    "failed to remove normalize temp run dir path=%s error_type=%s error=%s",
                    run_root,
                    type(exc).__name__,
                    exc,
                )
                return
        with suppress(OSError):
            temp_root.rmdir()

    @staticmethod
    def _temp_normalize_path(
        task: NormalizeTask,
        temp_root: Path,
        temp_run_id: str,
        *,
        kind: str,
    ) -> Path:
        safe_id = "".join(char if char.isalnum() else "_" for char in task.task_id)[:96]
        digest = hashlib.blake2s(task.task_id.encode(), digest_size=5).hexdigest()
        run_root = temp_root / temp_run_id
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root / f"{kind}-{safe_id}-{digest}.parquet.tmp"

    def _iter_task_resampling_bounded_temp_chunks(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        resampling: MinMaxLTTBResamplingSpec,
        config: NormalizeStageConfig,
        temp_root: Path,
        temp_run_id: str,
        *,
        annotate: bool,
    ) -> Iterator[pl.DataFrame]:
        chunk_rows = config.max_batch_rows
        if chunk_rows is None or chunk_rows < 1:
            raise ValueError(f"max_batch_rows must be >= 1, got {chunk_rows}")

        if not annotate:
            yield from self._iter_task_resampling_bounded_two_pass_chunks(
                task,
                protocol_spec,
                input_store,
                resampling,
                config,
            )
            return

        temp_path = self._temp_normalize_path(task, temp_root, temp_run_id, kind="checked")
        try:
            rows_in = self._materialize_checked_task_bounded(
                task,
                protocol_spec,
                input_store,
                temp_path,
                chunk_rows,
                config.compression,
                annotate=annotate,
            )
            if rows_in == 0:
                return

            budget = resolve_min_max_lttb_budget(resampling, rows_in)
            if rows_in <= budget:
                yield from coalesce_frames(
                    (
                        collect_frame(
                            self._finalize_output_columns(
                                temp_chunk.drop(_DOWNSAMPLE_ROW_ID),
                                protocol_spec,
                                include_annotations=annotate,
                            ),
                        )
                        for temp_chunk in iter_parquet_chunks(temp_path, config.chunk_rows)
                    ),
                    config.chunk_rows,
                )
                return

            selected_ids = select_min_max_lttb_row_ids(
                self._iter_parquet_column_chunks(
                    temp_path,
                    chunk_rows,
                    [_DOWNSAMPLE_ROW_ID, str(resampling.x_col), str(resampling.y_col)],
                ),
                row_id_col=_DOWNSAMPLE_ROW_ID,
                row_count=rows_in,
                budget=budget,
                spec=resampling,
                chunk_rows=chunk_rows,
            )
            if selected_ids.size == 0:
                return

            signal_col = resolve_downsampling_signal_col(
                pl.scan_parquet(temp_path).collect_schema().names(),
            )
            if signal_col is not None and config.apply_physics_compensation:
                yield from self._iter_selected_physics_output_chunks_from_temp(
                    temp_path,
                    selected_ids,
                    row_count=rows_in,
                    protocol_spec=protocol_spec,
                    read_chunk_rows=chunk_rows,
                    output_chunk_rows=config.chunk_rows,
                    include_annotations=annotate,
                    signal_col=signal_col,
                )
                logger.debug(
                    "normalize downsampled bounded-temp dataset=%s protocol=%s task_id=%s "
                    "rows_in=%d rows_out=%d",
                    self.spec.dataset_id,
                    task.protocol,
                    task.task_id,
                    rows_in,
                    selected_ids.size,
                )
                return

            yield from coalesce_frames(
                self._iter_selected_output_chunks_from_temp(
                    temp_path,
                    selected_ids,
                    protocol_spec=protocol_spec,
                    chunk_rows=config.chunk_rows,
                    include_annotations=annotate,
                ),
                config.chunk_rows,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _iter_task_resampling_bounded_two_pass_chunks(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        resampling: MinMaxLTTBResamplingSpec,
        config: NormalizeStageConfig,
    ) -> Iterator[pl.DataFrame]:
        chunk_rows = config.max_batch_rows
        if chunk_rows is None or chunk_rows < 1:
            raise ValueError(f"max_batch_rows must be >= 1, got {chunk_rows}")

        chunk_count = math.ceil(task.row_count / chunk_rows)
        budget_estimate = resolve_min_max_lttb_budget(resampling, task.row_count)
        chunk_budget = max(
            resampling.min_points,
            math.ceil((budget_estimate * DOWNSAMPLE_OVERSAMPLE_FACTOR) / chunk_count),
        )
        sampled_chunks: list[pl.DataFrame] = []
        rows_in = 0
        signal_col = None
        for chunk in coalesce_frames(
            self._iter_checked_task_bounded_row_id_frames(
                task,
                protocol_spec,
                input_store,
                chunk_rows,
            ),
            chunk_rows,
        ):
            rows_in += chunk.height
            if signal_col is None:
                signal_col = resolve_downsampling_signal_col(chunk.columns)
            sampled_chunks.append(
                downsample_min_max_lttb_frame(
                    chunk.select(_DOWNSAMPLE_ROW_ID, resampling.x_col, resampling.y_col),
                    chunk_budget,
                    resampling,
                ),
            )
        if rows_in == 0 or not sampled_chunks:
            return

        budget = resolve_min_max_lttb_budget(resampling, rows_in)
        selected_ids = (
            downsample_min_max_lttb_frame(
                pl.concat(sampled_chunks, how="vertical"),
                budget,
                resampling,
            )
            .sort(_DOWNSAMPLE_ROW_ID)[_DOWNSAMPLE_ROW_ID]
            .to_numpy()
            .astype(np.int64)
        )
        if selected_ids.size == 0:
            return

        if signal_col is None or not config.apply_physics_compensation:
            for chunk in self._iter_selected_checked_frames_two_pass(
                task,
                protocol_spec,
                input_store,
                selected_ids,
                chunk_rows=chunk_rows,
                output_chunk_rows=config.chunk_rows,
            ):
                yield self._finalize_output_columns(
                    chunk.drop(_DOWNSAMPLE_ROW_ID),
                    protocol_spec,
                    include_annotations=False,
                )
            return

        yield from self._iter_selected_physics_checked_frames_two_pass(
            task,
            protocol_spec,
            input_store,
            selected_ids,
            row_count=rows_in,
            chunk_rows=chunk_rows,
            output_chunk_rows=config.chunk_rows,
            signal_col=signal_col,
        )

    def _iter_checked_task_bounded_row_id_frames(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        chunk_rows: int,
    ) -> Iterator[pl.DataFrame]:
        input_columns, resolved_columns = self._task_input_resolution(
            task,
            protocol_spec,
            input_store,
        )
        check_state = BoundedCheckState()
        transform_checks = self._transform_checks(protocol_spec.checks)
        annotation_checks = self._annotation_checks(protocol_spec.checks)
        output_row_id = 0
        for input_frame in self._iter_task_segment_frames(
            task,
            input_store,
            input_columns,
            chunk_rows,
        ):
            frame = input_frame.filter(self._task_filter(task))
            if frame.height == 0:
                continue
            frame = apply_transforms_bounded_chunk(frame, protocol_spec.transforms)
            frame = apply_checks_bounded_chunk(
                frame,
                self.spec,
                transform_checks,
                check_state,
            )
            if frame.height == 0:
                continue
            frame = collect_frame(
                select_and_cast_columns(
                    frame,
                    self._internal_output_columns(protocol_spec),
                    resolved_columns=resolved_columns,
                ),
            )
            self._raise_for_check_failures(
                task,
                collect_check_failures_bounded_chunk(frame, self.spec, annotation_checks),
            )
            frame = frame.with_columns(
                pl.Series(
                    _DOWNSAMPLE_ROW_ID,
                    np.arange(output_row_id, output_row_id + frame.height),
                ),
            )
            output_row_id += frame.height
            yield frame

    def _iter_selected_checked_frames_two_pass(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        selected_ids: np.ndarray,
        *,
        chunk_rows: int,
        output_chunk_rows: int,
    ) -> Iterator[pl.DataFrame]:
        pending: list[pl.DataFrame] = []
        pending_rows = 0
        for chunk in coalesce_frames(
            self._iter_checked_task_bounded_row_id_frames(
                task,
                protocol_spec,
                input_store,
                chunk_rows,
            ),
            chunk_rows,
        ):
            chunk_row_ids = chunk[_DOWNSAMPLE_ROW_ID].to_numpy().astype(np.int64)
            chunk_start = int(chunk_row_ids[0])
            chunk_end = int(chunk_row_ids[-1]) + 1
            selected_start = int(np.searchsorted(selected_ids, chunk_start, side="left"))
            selected_end = int(np.searchsorted(selected_ids, chunk_end, side="left"))
            if selected_end <= selected_start:
                continue
            local_positions = selected_ids[selected_start:selected_end] - chunk_start
            selected = chunk[local_positions.tolist()]
            pending.append(selected)
            pending_rows += selected.height
            if pending_rows >= output_chunk_rows:
                combined = pl.concat(pending, how="vertical")
                while combined.height >= output_chunk_rows:
                    yield combined.slice(0, output_chunk_rows)
                    combined = combined.slice(output_chunk_rows)
                pending = [combined] if combined.height > 0 else []
                pending_rows = combined.height
        if pending:
            yield from iter_data_chunks(pl.concat(pending, how="vertical"), output_chunk_rows)

    def _iter_selected_physics_checked_frames_two_pass(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        selected_ids: np.ndarray,
        *,
        row_count: int,
        chunk_rows: int,
        output_chunk_rows: int,
        signal_col: ColumnSpec,
    ) -> Iterator[pl.DataFrame]:
        ends = np.empty(selected_ids.size, dtype=np.int64)
        if selected_ids.size > 1:
            ends[:-1] = selected_ids[1:]
        if selected_ids.size:
            ends[-1] = row_count

        dt_sums = np.zeros(selected_ids.size, dtype=np.float64)
        signal_dt_sums = np.zeros(selected_ids.size, dtype=np.float64)
        pending_rows: list[pl.DataFrame] = []
        emit_idx = 0
        sample_pos_col = "__sample_pos"

        for chunk in coalesce_frames(
            self._iter_checked_task_bounded_row_id_frames(
                task,
                protocol_spec,
                input_store,
                chunk_rows,
            ),
            chunk_rows,
        ):
            chunk_row_ids = chunk[_DOWNSAMPLE_ROW_ID].to_numpy().astype(np.int64)
            chunk_start = int(chunk_row_ids[0])
            chunk_end = int(chunk_row_ids[-1]) + 1
            dt = chunk[BatteryColumns.dt].to_numpy().astype(np.float64)
            signal = chunk[signal_col].to_numpy().astype(np.float64)
            dt_prefix = np.concatenate(([0.0], np.cumsum(dt)))
            signal_prefix = np.concatenate(([0.0], np.cumsum(signal * dt)))

            active_start = int(np.searchsorted(ends, chunk_start, side="right"))
            active_end = int(np.searchsorted(selected_ids, chunk_end, side="left"))
            if active_end > active_start:
                active = np.arange(active_start, active_end, dtype=np.int64)
                local_starts = np.maximum(selected_ids[active], chunk_start) - chunk_start
                local_ends = np.minimum(ends[active], chunk_end) - chunk_start
                valid = local_ends > local_starts
                active_valid = active[valid]
                starts = local_starts[valid]
                stops = local_ends[valid]
                dt_sums[active_valid] += dt_prefix[stops] - dt_prefix[starts]
                signal_dt_sums[active_valid] += signal_prefix[stops] - signal_prefix[starts]

            selected_start = int(np.searchsorted(selected_ids, chunk_start, side="left"))
            selected_end = int(np.searchsorted(selected_ids, chunk_end, side="left"))
            if selected_end > selected_start:
                selected = np.arange(selected_start, selected_end, dtype=np.int64)
                local_positions = selected_ids[selected] - chunk_start
                pending_rows.append(
                    chunk[local_positions.tolist()]
                    .drop(_DOWNSAMPLE_ROW_ID)
                    .with_columns(pl.Series(sample_pos_col, selected)),
                )

            completed_end = int(np.searchsorted(ends, chunk_end, side="right"))
            while completed_end - emit_idx >= output_chunk_rows:
                yield self._emit_selected_physics_chunk(
                    pending_rows,
                    sample_pos_col,
                    emit_idx,
                    emit_idx + output_chunk_rows,
                    dt_sums,
                    signal_dt_sums,
                    signal_col,
                    protocol_spec,
                    include_annotations=False,
                )
                emit_idx += output_chunk_rows
                pending_rows = self._retain_pending_sample_rows(
                    pending_rows,
                    sample_pos_col,
                    emit_idx,
                )

        if emit_idx < selected_ids.size:
            yield self._emit_selected_physics_chunk(
                pending_rows,
                sample_pos_col,
                emit_idx,
                selected_ids.size,
                dt_sums,
                signal_dt_sums,
                signal_col,
                protocol_spec,
                include_annotations=False,
            )

    def _iter_selected_physics_output_chunks_from_temp(
        self,
        temp_path: Path,
        selected_ids: np.ndarray,
        *,
        row_count: int,
        protocol_spec: ProtocolNormalizeSpec,
        read_chunk_rows: int,
        output_chunk_rows: int,
        include_annotations: bool,
        signal_col: ColumnSpec,
    ) -> Iterator[pl.DataFrame]:
        ends = np.empty(selected_ids.size, dtype=np.int64)
        if selected_ids.size > 1:
            ends[:-1] = selected_ids[1:]
        if selected_ids.size:
            ends[-1] = row_count

        dt_sums = np.zeros(selected_ids.size, dtype=np.float64)
        signal_dt_sums = np.zeros(selected_ids.size, dtype=np.float64)
        pending_rows: list[pl.DataFrame] = []
        emit_idx = 0
        sample_pos_col = "__sample_pos"

        for chunk in iter_parquet_chunks(temp_path, read_chunk_rows):
            chunk_row_ids = chunk[_DOWNSAMPLE_ROW_ID].to_numpy().astype(np.int64)
            chunk_start = int(chunk_row_ids[0])
            chunk_end = int(chunk_row_ids[-1]) + 1
            dt = chunk[BatteryColumns.dt].to_numpy().astype(np.float64)
            signal = chunk[signal_col].to_numpy().astype(np.float64)
            dt_prefix = np.concatenate(([0.0], np.cumsum(dt)))
            signal_prefix = np.concatenate(([0.0], np.cumsum(signal * dt)))

            active_start = int(np.searchsorted(ends, chunk_start, side="right"))
            active_end = int(np.searchsorted(selected_ids, chunk_end, side="left"))
            if active_end > active_start:
                active = np.arange(active_start, active_end, dtype=np.int64)
                local_starts = np.maximum(selected_ids[active], chunk_start) - chunk_start
                local_ends = np.minimum(ends[active], chunk_end) - chunk_start
                valid = local_ends > local_starts
                active_valid = active[valid]
                starts = local_starts[valid]
                stops = local_ends[valid]
                dt_sums[active_valid] += dt_prefix[stops] - dt_prefix[starts]
                signal_dt_sums[active_valid] += signal_prefix[stops] - signal_prefix[starts]

            selected_start = int(np.searchsorted(selected_ids, chunk_start, side="left"))
            selected_end = int(np.searchsorted(selected_ids, chunk_end, side="left"))
            if selected_end > selected_start:
                selected = np.arange(selected_start, selected_end, dtype=np.int64)
                local_positions = selected_ids[selected] - chunk_start
                pending_rows.append(
                    chunk[local_positions.tolist()]
                    .drop(_DOWNSAMPLE_ROW_ID)
                    .with_columns(pl.Series(sample_pos_col, selected)),
                )

            completed_end = int(np.searchsorted(ends, chunk_end, side="right"))
            while completed_end - emit_idx >= output_chunk_rows:
                yield self._emit_selected_physics_chunk(
                    pending_rows,
                    sample_pos_col,
                    emit_idx,
                    emit_idx + output_chunk_rows,
                    dt_sums,
                    signal_dt_sums,
                    signal_col,
                    protocol_spec,
                    include_annotations=include_annotations,
                )
                emit_idx += output_chunk_rows
                pending_rows = self._retain_pending_sample_rows(
                    pending_rows,
                    sample_pos_col,
                    emit_idx,
                )

        if emit_idx < selected_ids.size:
            yield self._emit_selected_physics_chunk(
                pending_rows,
                sample_pos_col,
                emit_idx,
                selected_ids.size,
                dt_sums,
                signal_dt_sums,
                signal_col,
                protocol_spec,
                include_annotations=include_annotations,
            )

    def _emit_selected_physics_chunk(
        self,
        pending_rows: list[pl.DataFrame],
        sample_pos_col: str,
        start: int,
        end: int,
        dt_sums: np.ndarray,
        signal_dt_sums: np.ndarray,
        signal_col: ColumnSpec,
        protocol_spec: ProtocolNormalizeSpec,
        *,
        include_annotations: bool,
    ) -> pl.DataFrame:
        rows = (
            pl.concat(pending_rows, how="vertical")
            .filter((pl.col(sample_pos_col) >= start) & (pl.col(sample_pos_col) < end))
            .sort(sample_pos_col)
        )
        local = rows[sample_pos_col].to_numpy().astype(np.int64)
        cumulative_start = float(dt_sums[:start].sum())
        rebuilt_time = cumulative_start + np.cumsum(dt_sums[local]) - dt_sums[local]
        averaged_signal = np.divide(
            signal_dt_sums[local],
            dt_sums[local],
            out=np.zeros(local.size, dtype=np.float64),
            where=dt_sums[local] > 0.0,
        )
        rows = rows.drop(sample_pos_col).with_columns(
            pl.Series(BatteryColumns.dt, dt_sums[local]),
            pl.Series(BatteryColumns.time, rebuilt_time),
            pl.Series(signal_col, averaged_signal),
        )
        return self._finalize_output_columns(
            rows,
            protocol_spec,
            include_annotations=include_annotations,
        )

    @staticmethod
    def _retain_pending_sample_rows(
        pending_rows: list[pl.DataFrame],
        sample_pos_col: str,
        emit_idx: int,
    ) -> list[pl.DataFrame]:
        retained = []
        for rows in pending_rows:
            keep = rows.filter(pl.col(sample_pos_col) >= emit_idx)
            if keep.height > 0:
                retained.append(keep)
        return retained

    @staticmethod
    def _iter_parquet_column_chunks(
        path: Path,
        chunk_rows: int,
        columns: list[str],
    ) -> Iterator[pl.DataFrame]:
        parquet_file = pq.ParquetFile(path)
        for arrow_batch in parquet_file.iter_batches(batch_size=chunk_rows, columns=columns):
            frame = pl.from_arrow(arrow_batch)
            if not isinstance(frame, pl.DataFrame):
                raise TypeError(
                    f"Expected batch conversion to return DataFrame, got {type(frame).__name__}",
                )
            if frame.height > 0:
                yield frame

    def _materialize_checked_task_bounded(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        temp_path: Path,
        chunk_rows: int,
        compression: str,
        *,
        annotate: bool,
    ) -> int:
        input_columns, resolved_columns = self._task_input_resolution(
            task,
            protocol_spec,
            input_store,
        )
        writer: pq.ParquetWriter | None = None
        check_state = BoundedCheckState()
        transform_checks = self._transform_checks(protocol_spec.checks)
        annotation_checks = self._annotation_checks(protocol_spec.checks)
        output_row_id = 0
        try:
            for input_frame in self._iter_task_segment_frames(
                task,
                input_store,
                input_columns,
                chunk_rows,
            ):
                frame = input_frame.filter(self._task_filter(task))
                if frame.height == 0:
                    continue
                frame = apply_transforms_bounded_chunk(frame, protocol_spec.transforms)
                frame = apply_checks_bounded_chunk(
                    frame,
                    self.spec,
                    transform_checks,
                    check_state,
                )
                if frame.height == 0:
                    continue
                frame = collect_frame(
                    select_and_cast_columns(
                        frame,
                        self._internal_output_columns(protocol_spec),
                        resolved_columns=resolved_columns,
                    ),
                )
                if annotate:
                    frame = apply_checks_bounded_chunk(
                        frame,
                        self.spec,
                        annotation_checks,
                        BoundedCheckState(),
                    )
                else:
                    self._raise_for_check_failures(
                        task,
                        collect_check_failures_bounded_chunk(
                            frame,
                            self.spec,
                            annotation_checks,
                        ),
                    )
                frame = frame.with_columns(
                    pl.Series(
                        _DOWNSAMPLE_ROW_ID,
                        np.arange(output_row_id, output_row_id + frame.height),
                    ),
                )
                output_row_id += frame.height
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, table.schema, compression=compression)
                writer.write_table(table, row_group_size=chunk_rows)
        finally:
            if writer is not None:
                writer.close()
        return output_row_id

    def _iter_task_bounded_checked_chunks(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
        config: NormalizeStageConfig,
        *,
        annotate: bool,
    ) -> Iterator[pl.DataFrame]:
        chunk_rows = config.max_batch_rows
        if chunk_rows is None or chunk_rows < 1:
            raise ValueError(f"max_batch_rows must be >= 1, got {chunk_rows}")

        check_state = BoundedCheckState()
        transform_checks = self._transform_checks(protocol_spec.checks)
        annotation_checks = self._annotation_checks(protocol_spec.checks)

        def checked_chunks() -> Iterator[pl.DataFrame]:
            input_columns, resolved_columns = self._task_input_resolution(
                task,
                protocol_spec,
                input_store,
            )
            for input_frame in self._iter_task_segment_frames(
                task,
                input_store,
                input_columns,
                chunk_rows,
            ):
                frame = input_frame.filter(self._task_filter(task))
                if frame.height == 0:
                    continue
                frame = apply_transforms_bounded_chunk(frame, protocol_spec.transforms)
                frame = apply_checks_bounded_chunk(
                    frame,
                    self.spec,
                    transform_checks,
                    check_state,
                )
                if frame.height == 0:
                    continue
                frame = collect_frame(
                    select_and_cast_columns(
                        frame,
                        self._internal_output_columns(protocol_spec),
                        resolved_columns=resolved_columns,
                    ),
                )
                if annotate:
                    frame = apply_checks_bounded_chunk(
                        frame,
                        self.spec,
                        annotation_checks,
                        BoundedCheckState(),
                    )
                else:
                    self._raise_for_check_failures(
                        task,
                        collect_check_failures_bounded_chunk(
                            frame,
                            self.spec,
                            annotation_checks,
                        ),
                    )
                yield self._finalize_output_columns(
                    frame,
                    protocol_spec,
                    include_annotations=annotate,
                )

        yield from coalesce_frames(checked_chunks(), config.chunk_rows)

    def _iter_selected_output_chunks_from_temp(
        self,
        temp_path: Path,
        selected_ids: np.ndarray,
        *,
        protocol_spec: ProtocolNormalizeSpec,
        chunk_rows: int,
        include_annotations: bool,
        dt_sums: np.ndarray | None = None,
        rebuilt_time: np.ndarray | None = None,
        averaged_signal: np.ndarray | None = None,
        signal_col: ColumnSpec | None = None,
    ) -> Iterator[pl.DataFrame]:
        for chunk in iter_parquet_chunks(temp_path, chunk_rows):
            chunk_row_ids = chunk[_DOWNSAMPLE_ROW_ID].to_numpy().astype(np.int64)
            chunk_start = int(chunk_row_ids[0])
            chunk_end = int(chunk_row_ids[-1]) + 1
            global_start = int(np.searchsorted(selected_ids, chunk_start, side="left"))
            global_end = int(np.searchsorted(selected_ids, chunk_end, side="left"))
            if global_end <= global_start:
                continue

            local_positions = (selected_ids[global_start:global_end] - chunk_start).tolist()
            out = chunk[local_positions]
            if (
                dt_sums is not None
                and rebuilt_time is not None
                and averaged_signal is not None
                and signal_col is not None
            ):
                out = out.with_columns(
                    pl.Series(BatteryColumns.dt, dt_sums[global_start:global_end]),
                    pl.Series(BatteryColumns.time, rebuilt_time[global_start:global_end]),
                    pl.Series(signal_col, averaged_signal[global_start:global_end]),
                )
            out = out.drop(_DOWNSAMPLE_ROW_ID)
            yield self._finalize_output_columns(
                out,
                protocol_spec,
                include_annotations=include_annotations,
            )

    def _downsample_min_max_lttb(
        self,
        data: pl.LazyFrame,
        *,
        row_count: int,
        budget: int,
        spec: MinMaxLTTBResamplingSpec,
        max_chunk_rows: int | None,
    ) -> pl.DataFrame:
        chunk_rows = row_count if max_chunk_rows is None else max_chunk_rows
        if chunk_rows < 1:
            raise ValueError(f"max_batch_rows must be >= 1, got {max_chunk_rows}")
        if row_count <= chunk_rows:
            return downsample_min_max_lttb_frame(collect_frame(data), budget, spec)

        sampled_chunks: list[pl.DataFrame] = []
        chunk_count = math.ceil(row_count / chunk_rows)
        chunk_budget = max(
            spec.min_points,
            math.ceil((budget * DOWNSAMPLE_OVERSAMPLE_FACTOR) / chunk_count),
        )
        for offset in range(0, row_count, chunk_rows):
            chunk = collect_frame(data.slice(offset, chunk_rows))
            if chunk.height == 0:
                continue
            sampled_chunks.append(downsample_min_max_lttb_frame(chunk, chunk_budget, spec))

        if not sampled_chunks:
            return collect_frame(data.limit(0))
        sampled = pl.concat(sampled_chunks, how="diagonal_relaxed")
        return downsample_min_max_lttb_frame(sampled, budget, spec)

    def _normalize_spec(self) -> NormalizeSpec:
        normalize_spec = self.spec.normalize
        if normalize_spec is None:
            raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support normalize")
        return normalize_spec

    def _validate_normalize_spec(self, normalize_spec: NormalizeSpec) -> None:
        raw_spec = self.spec.raw
        if raw_spec is None:
            raise ValueError(
                f"Dataset {self.spec.dataset_id!r} has normalize spec but no raw spec",
            )
        for protocol, protocol_spec in normalize_spec.protocol_specs.items():
            raw_schema = raw_spec.protocol_schema(protocol)
            raw_columns = set(raw_schema.output_columns)
            missing_group = [
                column for column in protocol_spec.group_by if column not in raw_columns
            ]
            if missing_group:
                raise ValueError(
                    f"Normalize group_by columns for protocol {protocol!r} are not produced by "
                    f"raw protocol schema: {missing_group}",
                )

    def _validate_manifest_columns(
        self,
        manifest: pl.DataFrame,
        protocol: str,
        protocol_spec: ProtocolNormalizeSpec,
    ) -> None:
        required = (
            MetadataColumns.parquet_segments,
            BaseColumns.row_count,
            MetadataColumns.raw_file_paths,
            MetadataColumns.protocol,
            *protocol_spec.group_by,
        )
        missing = [column for column in required if column not in manifest.columns]
        if missing:
            raise ValueError(
                f"Raw manifest is missing columns required for normalize protocol {protocol!r}: "
                f"{missing}",
            )

    def _task_input_resolution(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        input_store: DataStore,
    ) -> tuple[tuple[str, ...], dict[ColumnSpec, str | None]]:
        raw_spec = self.spec.raw
        if raw_spec is None:
            raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support raw schemas")
        raw_columns = tuple(raw_spec.protocol_schema(task.protocol).output_columns)
        requested = [
            MetadataColumns.protocol,
            MetadataColumns.domain_id,
            *protocol_spec.group_by,
            protocol_spec.domain.axis_col,
            *protocol_spec.order_by,
            *transform_source_columns(protocol_spec.transforms),
            *protocol_spec.columns,
            BaseColumns.annotations,
        ]
        date_time_col = getattr(self.spec.cols, "date_time", None)
        for check in protocol_spec.checks:
            time_col = getattr(check, "time_col", None)
            dt_col = getattr(check, "dt_col", None)
            requested.extend(
                column
                for column in (time_col, dt_col, date_time_col)
                if isinstance(column, ColumnSpec)
            )
        input_columns: list[str] = []
        resolved_columns: dict[ColumnSpec, str | None] = {}
        alias_candidates = self._task_alias_candidates(requested, raw_columns)
        non_null_counts = self._task_alias_non_null_counts(task, input_store, alias_candidates)
        for column in dict.fromkeys(requested):
            candidates = alias_candidates.get(column)
            if candidates is None:
                source_column = column.matching_name(raw_columns)
            else:
                source_column = next(
                    (
                        candidate
                        for candidate in candidates
                        if non_null_counts.get(candidate, 0) > 0
                    ),
                    None,
                )
                resolved_columns[column] = source_column
            if source_column is None:
                continue
            input_columns.append(source_column)
            if source_column != str(column):
                resolved_columns[column] = source_column
        return tuple(dict.fromkeys(input_columns)), resolved_columns

    @staticmethod
    def _task_alias_candidates(
        requested: list[ColumnSpec],
        raw_columns: tuple[ColumnSpec, ...],
    ) -> dict[ColumnSpec, tuple[str, ...]]:
        raw_column_names = {str(column).casefold() for column in raw_columns}
        candidates: dict[ColumnSpec, tuple[str, ...]] = {}
        for column in dict.fromkeys(requested):
            matches = tuple(alias for alias in column.alias if alias.casefold() in raw_column_names)
            if len(matches) > 1:
                candidates[column] = matches
        return candidates

    def _task_alias_non_null_counts(
        self,
        task: NormalizeTask,
        input_store: DataStore,
        alias_candidates: dict[ColumnSpec, tuple[str, ...]],
    ) -> dict[str, int]:
        candidate_columns = tuple(
            dict.fromkeys(
                candidate for candidates in alias_candidates.values() for candidate in candidates
            ),
        )
        if not candidate_columns:
            return {}
        scan_columns = tuple(
            dict.fromkeys(
                (
                    str(MetadataColumns.protocol),
                    *(str(column) for column in task.group_values),
                    *candidate_columns,
                ),
            ),
        )
        counts = dict.fromkeys(candidate_columns, 0)
        for input_frame in self._iter_task_segment_frames(
            task,
            input_store,
            scan_columns,
            chunk_rows=500_000,
        ):
            frame = input_frame.filter(self._task_filter(task))
            if frame.height == 0:
                continue
            row = frame.select(
                pl.col(column).is_not_null().sum().alias(column) for column in candidate_columns
            ).row(0, named=True)
            for column in candidate_columns:
                counts[column] += int(row[column])
        return counts

    @staticmethod
    def _internal_output_columns(protocol_spec: ProtocolNormalizeSpec) -> tuple[ColumnSpec, ...]:
        return tuple(
            dict.fromkeys(
                (
                    protocol_spec.domain.axis_col,
                    *protocol_spec.columns,
                    BaseColumns.annotation_columns,
                    BaseColumns.annotation_reasons,
                ),
            ),
        )

    @staticmethod
    def _output_columns(
        protocol_spec: ProtocolNormalizeSpec,
        *,
        include_annotations: bool,
    ) -> tuple[ColumnSpec, ...]:
        columns = [protocol_spec.domain.axis_col, *protocol_spec.columns]
        if include_annotations:
            columns.append(BaseColumns.annotations)
        return tuple(dict.fromkeys(columns))

    def _task_metadata(
        self,
        task: NormalizeTask,
        protocol_spec: ProtocolNormalizeSpec,
        *,
        apply_resampling: bool,
    ) -> dict[ColumnSpec, object]:
        resampling = protocol_spec.resampling if apply_resampling else None
        resampling_method, resampling_params = resampling_metadata_values(resampling)
        return {
            BaseColumns.dataset_id: self.spec.dataset_id,
            MetadataColumns.protocol: task.protocol,
            MetadataColumns.domain_id: protocol_spec.domain.domain_id,
            MetadataColumns.parquet_segments: list(task.parquet_segments),
            MetadataColumns.resampling_method: resampling_method,
            MetadataColumns.resampling_params: resampling_params,
            MetadataColumns.time_convention: self._normalize_spec().time_convention,
            BaseColumns.axis_kind: protocol_spec.domain.domain_id,
            BaseColumns.axis_col: protocol_spec.domain.axis_col,
            **task.group_values,
        }

    def _record_worker_failure(
        self,
        result: ProcessTaskResult[PreparedNormalizeTaskResult],
        task: NormalizeTask,
        run_stats: NormalizeRunStats,
    ) -> None:
        run_stats.tasks_failed += 1
        logger.error(
            "normalize worker task failed task=%d/%d task_id=%s error_type=%s error=%s",
            result.task_index,
            run_stats.tasks_total,
            task.task_id,
            result.error_type,
            result.error,
        )

    def _record_task_failure(
        self,
        task_idx: int,
        task: NormalizeTask,
        exc: Exception,
        run_stats: NormalizeRunStats,
    ) -> None:
        run_stats.tasks_failed += 1
        logger.error(
            "normalize task failed task=%d/%d task_id=%s error_type=%s error=%s",
            task_idx,
            run_stats.tasks_total,
            task.task_id,
            type(exc).__name__,
            exc,
        )

    def _log_status_if_due(
        self,
        last_update_at: float,
        run_stats: NormalizeRunStats,
    ) -> float:
        now = time.perf_counter()
        if now - last_update_at < _PROGRESS_TIME_INTERVAL_S:
            return last_update_at
        completed_tasks = run_stats.tasks_succeeded + run_stats.tasks_failed
        logger.info(
            "normalize dataset=%s task=%d/%d succeeded=%d failed=%d",
            self.spec.dataset_id,
            completed_tasks,
            run_stats.tasks_total,
            run_stats.tasks_succeeded,
            run_stats.tasks_failed,
        )
        return now


def normalize_dataset(
    spec: DatasetSpec,
    input_store: DataStore,
    output_store: DataStore,
    config: NormalizeStageConfig,
) -> None:
    NormalizeProcessor(spec).run(input_store, output_store, config)
