from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Protocol

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetStageId
from batgrad.data.processing.io import (
    collect_frame,
    frame_columns,
)
from batgrad.data.processing.metadata import stage_layout_with_protocol_metadata
from batgrad.data.processing.runtime import (
    ProcessTaskResult,
    validate_stage_runtime_config,
)
from batgrad.data.processing.stage import (
    PreparedTable,
    TaskOutput,
    iter_stage_task_results,
    protocol_spec_by_id,
    run_stage_task_outputs,
    stage_output_spec,
    stage_temp_path,
)
from batgrad.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.contracts.mapping import DatasetProtocolId, MappingSpec
    from batgrad.contracts.metadata import MetadataLayout, ProtocolMetadata, StageLayout
    from batgrad.contracts.protocols import BatteryProtocolSpec
    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.data.processing.stage import StageOutputSpec
    from batgrad.storage.store import DataProcessingStore


@dataclass(frozen=True)
class IngestStageConfig:
    n_jobs: int = 1
    worker_polars_max_threads: int | None = -1
    chunk_rows: int = 256_000
    compression: str = "zstd"
    use_content_defined_chunking: bool = True
    row_group_size: int = 262_144
    max_shard_size_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class IngestProtocolSpec:
    protocol: BatteryProtocolSpec
    columns: tuple[MappingSpec, ...]
    metadata: ProtocolMetadata | None = None
    dropped_columns: tuple[MappingSpec, ...] = ()
    flip_current_sign: bool = False

    @property
    def protocol_id(self) -> DatasetProtocolId:
        return self.protocol.protocol_id

    @property
    def protocol_metadata(self) -> ProtocolMetadata:
        return self.metadata or self.protocol.metadata

    @property
    def output_columns(self) -> tuple[MappingSpec, ...]:
        return self.columns

    @property
    def manifest_columns(self) -> tuple[MappingSpec, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.protocol_metadata.task_key,
                    *self.protocol_metadata.manifest_extra.columns,
                ),
            ),
        )

    @property
    def required_metadata(self) -> tuple[MappingSpec, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.protocol_metadata.task_key,
                    *self.protocol_metadata.manifest_extra.required,
                ),
            ),
        )


@dataclass(frozen=True)
class IngestStageSpec:
    metadata: StageLayout
    included_file_patterns: tuple[str, ...]
    excluded_file_patterns: tuple[str, ...] = field(default_factory=tuple)
    protocol_specs: tuple[IngestProtocolSpec, ...] = field(default_factory=tuple)

    @property
    def footer_metadata(self) -> MetadataLayout:
        return self._expanded_metadata.footer

    @property
    def manifest_metadata(self) -> MetadataLayout:
        return self._expanded_metadata.manifest

    @property
    def _expanded_metadata(self) -> StageLayout:
        return stage_layout_with_protocol_metadata(
            self.metadata,
            (spec.protocol_metadata for spec in self.protocol_specs),
        )

    def protocol_spec(self, protocol: object) -> IngestProtocolSpec:
        return protocol_spec_by_id(self.protocol_specs, protocol)

    def output_columns(self, protocol: object) -> tuple[MappingSpec, ...]:
        return self.protocol_spec(protocol).output_columns

    def required_metadata(self, protocol: object) -> tuple[MappingSpec, ...]:
        return self.protocol_spec(protocol).required_metadata

    def manifest_columns(self, protocol: object) -> tuple[MappingSpec, ...]:
        return self.protocol_spec(protocol).manifest_columns

    def is_included_file(self, path: str) -> bool:
        if self.included_file_patterns and not any(
            fnmatch(path, pattern) for pattern in self.included_file_patterns
        ):
            return False
        return not any(fnmatch(path, pattern) for pattern in self.excluded_file_patterns)

    def output_spec(self, dataset_spec: DatasetSpec) -> StageOutputSpec:
        output_root = dataset_spec.source_root(self.metadata.stage_id)
        return stage_output_spec(
            dataset_spec=dataset_spec,
            stage_id=self.metadata.stage_id,
            output_root=output_root,
            manifest_path=dataset_spec.manifest(self.metadata.stage_id),
            manifest_metadata=self.manifest_metadata,
            footer_metadata=self.footer_metadata,
            shard_key_col=BaseColumns.proto,
            segment_col=BaseColumns.parq_segs,
            source_paths_col=BaseColumns.raw_paths,
        )


@dataclass(frozen=True)
class IngestTask:
    task_id: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class IngestBatch:
    data: pl.DataFrame | pl.LazyFrame
    protocol_id: DatasetProtocolId
    source_paths: tuple[str, ...]
    metadata: dict[MappingSpec, object]


@dataclass(frozen=True)
class _RawWorkerPayload:
    task_index: int
    task: IngestTask
    adapter: RawDatasetAdapter
    input_store: DataProcessingStore
    scratch_store: DataProcessingStore
    raw_spec: IngestStageSpec
    config: IngestStageConfig
    scratch_root: str


class RawDatasetAdapter(Protocol):
    spec: DatasetSpec

    def plan_raw_tasks(
        self,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> tuple[IngestTask, ...]: ...

    def load_raw_task(
        self,
        task: IngestTask,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> Iterator[IngestBatch]: ...


def run_ingest(
    adapter: RawDatasetAdapter,
    input_store: DataProcessingStore,
    output_store: DataProcessingStore,
    config: IngestStageConfig,
    *,
    scratch_store: DataProcessingStore | None = None,
    tasks: tuple[IngestTask, ...] | None = None,
) -> None:
    scratch_store = scratch_store or output_store
    raw_spec = _dataset_raw_spec(adapter.spec)
    validate_stage_runtime_config(config)
    tasks = adapter.plan_raw_tasks(input_store, raw_spec) if tasks is None else tasks

    output_spec = raw_spec.output_spec(adapter.spec)
    run_stage_task_outputs(
        output_store=output_store,
        scratch_store=scratch_store,
        output_spec=output_spec,
        config=config,
        tasks=tasks,
        iter_results=lambda scratch_root: _iter_raw_task_results(
            adapter,
            input_store,
            scratch_store,
            raw_spec,
            config,
            scratch_root,
            tasks,
        ),
        logger=logger,
        delete_output_root=True,
    )


def _process_raw_task(payload: _RawWorkerPayload) -> TaskOutput[IngestTask]:
    return _process_raw_task_output(
        payload.adapter,
        payload.input_store,
        payload.scratch_store,
        payload.raw_spec,
        payload.config,
        payload.scratch_root,
        payload.task_index,
        payload.task,
    )


def prepare_raw_batch(
    batch: IngestBatch, raw_spec: IngestStageSpec
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    protocol_spec = raw_spec.protocol_spec(batch.protocol_id)
    validate_raw_batch_metadata(batch, raw_spec)
    warnings: list[str] = []
    data = batch.data
    data, align_warnings = align_to_protocol_spec(data, protocol_spec, batch.source_paths)
    warnings.extend(align_warnings)
    if protocol_spec.flip_current_sign:
        data = _flip_current_sign(data)
    return collect_frame(data), tuple(warnings)


def validate_raw_batch_metadata(batch: IngestBatch, raw_spec: IngestStageSpec) -> None:
    protocol_value = batch.metadata.get(BaseColumns.proto)
    if protocol_value is None:
        raise ValueError(f"Raw batch metadata is missing {BaseColumns.proto!r}")
    if str(protocol_value) != str(batch.protocol_id):
        raise ValueError(
            f"Raw batch protocol {batch.protocol_id!r} does not match metadata "
            f"{BaseColumns.proto!r}={protocol_value!r}",
        )

    required = raw_spec.required_metadata(batch.protocol_id)
    missing = [column for column in required if batch.metadata.get(column) is None]
    if missing:
        raise ValueError(
            f"Raw batch metadata for protocol {batch.protocol_id!r} is missing required "
            f"columns: {missing}",
        )


def align_to_protocol_spec(  # noqa: C901
    data: pl.DataFrame | pl.LazyFrame,
    protocol_spec: IngestProtocolSpec,
    source_paths: tuple[str, ...],
) -> tuple[pl.DataFrame | pl.LazyFrame, tuple[str, ...]]:
    source_columns = frame_columns(data)
    select_exprs: list[pl.Expr] = []
    declared_sources: set[str] = set()
    output_counts: dict[str, int] = {}
    warnings: list[str] = []

    _validate_one_of_column_groups(source_columns, protocol_spec, source_paths)

    for spec in protocol_spec.columns:
        source_col = spec.matching_name(source_columns)
        output_col = _dedupe_output_column(spec, output_counts)
        if source_col is None:
            raise ValueError(
                f"Raw batch {source_paths} for protocol {protocol_spec.protocol_id!r} is missing "
                f"declared column {spec!r}. Dataset adapters must normalize optional raw "
                "columns before yielding IngestBatch.",
            )
        declared_sources.add(source_col)
        if output_col != str(spec):
            warnings.append(
                f"duplicate column {source_col!r} mapped to {spec!r}; using {output_col!r}"
            )
        if spec.parser is None:
            select_exprs.append(pl.col(source_col).cast(spec.dtype).alias(output_col))
        else:
            select_exprs.append(spec.parser(source_col).alias(output_col))

    dropped = []
    unknown = []
    for source_col in source_columns:
        if source_col in declared_sources:
            continue
        if any(spec.has_match(source_col) for spec in protocol_spec.manifest_columns):
            continue
        if any(spec.has_match(source_col) for spec in protocol_spec.dropped_columns):
            dropped.append(source_col)
        elif not any(spec.has_match(source_col) for spec in protocol_spec.columns):
            unknown.append(source_col)
    if dropped:
        warnings.append(f"dropped declared columns in {source_paths}: {dropped}")
    if unknown:
        expected = sorted(
            {
                alias
                for spec in (*protocol_spec.columns, *protocol_spec.dropped_columns)
                for alias in spec.alias
            },
        )
        raise ValueError(
            f"Raw batch {source_paths} for protocol {protocol_spec.protocol_id!r} has unknown "
            f"columns: {unknown}. Declare them in IngestProtocolSpec.columns or "
            f"dropped_columns. Expected aliases: {expected}",
        )
    return data.select(select_exprs), tuple(warnings)


def _iter_raw_task_results(
    adapter: RawDatasetAdapter,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    raw_spec: IngestStageSpec,
    config: IngestStageConfig,
    scratch_root: str,
    tasks: tuple[IngestTask, ...],
) -> Iterator[ProcessTaskResult[TaskOutput[IngestTask]]]:
    yield from iter_stage_task_results(
        tasks=tasks,
        task_id=lambda task: task.task_id,
        make_payload=lambda idx, task: _RawWorkerPayload(
            idx, task, adapter, input_store, scratch_store, raw_spec, config, scratch_root
        ),
        worker=_process_raw_task,
        config=config,
    )


def _process_raw_task_output(
    adapter: RawDatasetAdapter,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    raw_spec: IngestStageSpec,
    config: IngestStageConfig,
    scratch_root: str,
    task_index: int,
    task: IngestTask,
) -> TaskOutput[IngestTask]:
    tables: list[PreparedTable] = []
    warnings: list[str] = []
    for batch_index, batch in enumerate(adapter.load_raw_task(task, input_store, raw_spec)):
        data, batch_warnings = prepare_raw_batch(batch, raw_spec)
        warnings.extend(batch_warnings)
        if data.height == 0:
            continue
        temp_path = stage_temp_path(scratch_root, task_index, batch_index)
        scratch_store.write_table(data, temp_path, row_group_size=config.row_group_size)
        tables.append(
            PreparedTable(
                temp_path=temp_path,
                metadata=batch.metadata,
                source_paths=batch.source_paths,
            ),
        )
    return TaskOutput(task, tuple(tables), tuple(warnings))


def _flip_current_sign(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    if BaseColumns.curr not in frame_columns(data):
        return data
    return data.with_columns((pl.col(BaseColumns.curr) * -1).alias(BaseColumns.curr))


def _dedupe_output_column(spec: MappingSpec, output_counts: dict[str, int]) -> str:
    count = output_counts.get(str(spec), 0)
    output_counts[str(spec)] = count + 1
    if count == 0:
        return str(spec)
    return f"{spec} {count}"


def _validate_one_of_column_groups(
    source_columns: tuple[str, ...],
    protocol_spec: IngestProtocolSpec,
    source_paths: tuple[str, ...],
) -> None:
    if not protocol_spec.protocol.one_of_col_groups:
        return
    if any(
        all(column.has_match(source_columns) for column in group)
        for group in protocol_spec.protocol.one_of_col_groups
    ):
        return
    expected = [
        [str(column) for column in group] for group in protocol_spec.protocol.one_of_col_groups
    ]
    raise ValueError(
        f"Raw batch {source_paths} for protocol {protocol_spec.protocol_id!r} must include one "
        f"complete column group from {expected}; available columns: {sorted(source_columns)}"
    )


def _dataset_raw_spec(spec: DatasetSpec) -> IngestStageSpec:
    raw_spec = spec.processing_stages.get(DatasetStageId.ingested)
    if not isinstance(raw_spec, IngestStageSpec):
        raise TypeError(f"Dataset {spec.dataset_id!r} does not support ingest")
    return raw_spec
