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
    """Runtime and parquet-writing settings for the ingest stage.

    Attributes:
        n_jobs: Worker count. Use `1` for sequential execution, `-1` for
            available CPUs minus one, or a positive count capped by task count.
        worker_polars_max_threads: Polars threads per worker. `-1` divides CPUs
            across workers, `None` leaves Polars unrestricted, and a positive
            value sets an exact thread count.
        chunk_rows: Rows read from temporary task outputs at a time.
        compression: Parquet compression codec.
        use_content_defined_chunking: Whether table writers may use content
            defined chunking.
        row_group_size: Parquet row group size for written files.
        max_shard_size_bytes: Roll a protocol shard after this approximate size;
            `0` disables size-based rolling.

    Examples:
        >>> IngestStageConfig(n_jobs=-1, worker_polars_max_threads=-1)
        IngestStageConfig(...)
    """

    n_jobs: int = 1
    worker_polars_max_threads: int | None = -1
    chunk_rows: int = 256_000
    compression: str = "zstd"
    use_content_defined_chunking: bool = True
    row_group_size: int = 262_144
    max_shard_size_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class IngestProtocolSpec:
    """Raw-to-ingested mapping for one protocol.

    Dataset adapters yield raw batches for a protocol. The ingest stage aligns
    each batch to `columns` using `MappingSpec` aliases and parsers, validates
    protocol metadata, optionally flips current sign, and writes ingested parquet
    shards grouped by protocol.

    Attributes:
        protocol: Shared protocol definition.
        columns: Canonical output columns expected in adapter output.
        metadata: Optional protocol metadata override for this dataset.
        dropped_columns: Declared raw columns that may appear but are omitted.
        flip_current_sign: Whether to multiply canonical current by `-1`.

    Examples:
        >>> from batgrad.contracts.protocols import BatteryProtocols
        >>> IngestProtocolSpec(
        ...     protocol=BatteryProtocols.cyc,
        ...     columns=(BaseColumns.time, BaseColumns.curr, BaseColumns.volt),
        ...     flip_current_sign=True,
        ... )
        IngestProtocolSpec(...)
    """

    protocol: BatteryProtocolSpec
    columns: tuple[MappingSpec, ...]
    metadata: ProtocolMetadata | None = None
    dropped_columns: tuple[MappingSpec, ...] = ()
    flip_current_sign: bool = False

    @property
    def protocol_id(self) -> DatasetProtocolId:
        """Canonical protocol id for this ingest spec.

        Returns:
            Shared protocol id from `protocol`.
        """
        return self.protocol.protocol_id

    @property
    def protocol_metadata(self) -> ProtocolMetadata:
        """Protocol metadata override, or the shared protocol metadata.

        Returns:
            Metadata used to validate adapter batch metadata and expand manifests.
        """
        return self.metadata or self.protocol.metadata

    @property
    def output_columns(self) -> tuple[MappingSpec, ...]:
        """Canonical columns written for this protocol.

        Returns:
            Ingested parquet columns after raw alignment.
        """
        return self.columns

    @property
    def manifest_columns(self) -> tuple[MappingSpec, ...]:
        """Protocol task and manifest metadata columns expected from batches.

        Returns:
            Task-key and protocol manifest metadata columns. Adapter batches may
            carry these as metadata instead of data columns.
        """
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
        """Metadata columns that each adapter batch must provide.

        Returns:
            Task-key and required protocol manifest metadata columns.
        """
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
    """Dataset-level ingest configuration.

    Include/exclude patterns are used by raw adapters when planning source-file
    tasks. Stage metadata is expanded with each protocol's task keys and manifest
    extras before the output manifest is written.

    Attributes:
        metadata: Base stage metadata layout.
        included_file_patterns: Raw files an adapter should consider.
        excluded_file_patterns: Raw files an adapter should ignore.
        protocol_specs: Protocol-specific raw mappings.

    Examples:
        >>> from batgrad.contracts.metadata import INGEST_STAGE_METADATA
        >>> from batgrad.contracts.protocols import BatteryProtocols
        >>> protocol_spec = IngestProtocolSpec(
        ...     protocol=BatteryProtocols.cyc,
        ...     columns=(BaseColumns.time, BaseColumns.curr, BaseColumns.volt),
        ... )
        >>> spec = IngestStageSpec(
        ...     metadata=INGEST_STAGE_METADATA,
        ...     included_file_patterns=("*.xlsx",),
        ...     protocol_specs=(protocol_spec,),
        ... )
        >>> spec.is_included_file("raw/cell.xlsx")
        True
    """

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
        """Return the protocol spec matching a protocol id or value.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Matching ingest protocol spec.

        Raises:
            ValueError: If no protocol spec matches.
        """
        return protocol_spec_by_id(self.protocol_specs, protocol)

    def output_columns(self, protocol: object) -> tuple[MappingSpec, ...]:
        """Canonical output columns for a protocol.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Columns written to ingested parquet for the protocol.
        """
        return self.protocol_spec(protocol).output_columns

    def required_metadata(self, protocol: object) -> tuple[MappingSpec, ...]:
        """Batch metadata required for a protocol.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Metadata columns each adapter batch must provide.
        """
        return self.protocol_spec(protocol).required_metadata

    def manifest_columns(self, protocol: object) -> tuple[MappingSpec, ...]:
        """Protocol metadata columns written to the ingested manifest.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Metadata columns used by this protocol's manifest rows.
        """
        return self.protocol_spec(protocol).manifest_columns

    def is_included_file(self, path: str) -> bool:
        """Return whether a raw file path passes include/exclude patterns.

        Args:
            path: Raw source path relative to the store.

        Returns:
            `True` when `path` matches include patterns and no exclude pattern.
        """
        if self.included_file_patterns and not any(
            fnmatch(path, pattern) for pattern in self.included_file_patterns
        ):
            return False
        return not any(fnmatch(path, pattern) for pattern in self.excluded_file_patterns)

    def output_spec(self, dataset_spec: DatasetSpec) -> StageOutputSpec:
        """Build the output writer configuration for ingested shards.

        Args:
            dataset_spec: Dataset storage configuration.

        Returns:
            Stage writer configuration for protocol-sharded ingested output.
        """
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
    """One adapter-planned raw ingest unit, usually one source file.

    Attributes:
        task_id: Stable id used in logs and temporary output paths.
        source_paths: Raw source paths consumed by this task.
    """

    task_id: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class IngestBatch:
    """Raw data and metadata yielded by a raw dataset adapter.

    `metadata` must include `BaseColumns.proto` and all columns required by the
    selected protocol metadata. The ingest stage writes these values to manifest
    rows and parquet footer metadata where declared by the stage layout.

    Attributes:
        data: Raw frame yielded by the adapter.
        protocol_id: Protocol represented by this batch.
        source_paths: Raw source paths that produced the batch.
        metadata: Task and protocol metadata for manifest/footer writing.
    """

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
    """Interface implemented by dataset-specific raw loaders.

    Implementations discover raw source files in `plan_raw_tasks` and load them
    in `load_raw_task`. The loader is responsible for inferring protocol and
    task metadata; the ingest stage handles column alignment, validation, and
    parquet writing.
    """

    spec: DatasetSpec

    def plan_raw_tasks(
        self,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> tuple[IngestTask, ...]:
        """Discover raw source work units for the ingest stage.

        Args:
            input_store: Store containing raw files.
            raw_spec: Dataset ingest configuration.

        Returns:
            Planned ingest tasks. Implementations usually combine
            `input_store.list_files` with `raw_spec.is_included_file`.
        """
        ...

    def load_raw_task(
        self,
        task: IngestTask,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> Iterator[IngestBatch]:
        """Load a planned raw task and yield protocol batches.

        Args:
            task: Planned raw task.
            input_store: Store containing raw files.
            raw_spec: Dataset ingest configuration.

        Returns:
            Iterator of raw batches. A task may yield multiple batches when one
            source contains multiple protocols or logical groups.
        """
        ...


def run_ingest(
    adapter: RawDatasetAdapter,
    input_store: DataProcessingStore,
    output_store: DataProcessingStore,
    config: IngestStageConfig,
    *,
    scratch_store: DataProcessingStore | None = None,
    tasks: tuple[IngestTask, ...] | None = None,
) -> None:
    """Run raw-to-ingested parquet processing for a dataset adapter.

    The stage plans tasks with the adapter unless `tasks` is supplied, writes
    temporary task parquet to the scratch store, then appends results into
    protocol-sharded output files and an ingested manifest.

    Args:
        adapter: Dataset-specific raw loader.
        input_store: Store containing raw source files.
        output_store: Store receiving ingested shards and manifest.
        config: Runtime and parquet-writing settings.
        scratch_store: Optional store for temporary task outputs. Defaults to
            `output_store`.
        tasks: Optional task subset for retries or tests.

    Returns:
        `None`. Outputs are written to `output_store`.
    """
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
    """Validate, align, and materialize one adapter batch.

    Args:
        batch: Raw adapter batch.
        raw_spec: Dataset ingest configuration.

    Returns:
        Materialized canonical frame and non-fatal warnings, such as declared
        dropped columns or duplicate canonical output names.
    """
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
    """Validate protocol and required task metadata for an ingest batch.

    Args:
        batch: Raw adapter batch.
        raw_spec: Dataset ingest configuration.
    """
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
    """Select raw columns into canonical protocol columns.

    Source columns are matched through `MappingSpec` aliases. If a mapping has a
    parser, the parser builds the output expression; otherwise the source column
    is cast to the mapping dtype. Unknown raw columns are errors unless declared
    in protocol columns, manifest metadata columns, or `dropped_columns`. Missing
    declared columns are errors; adapters should add null optional columns before
    yielding batches. Duplicate canonical mappings are kept with suffixed output
    names such as `"column 1"` and returned as warnings.

    Returns:
        Canonically selected frame and non-fatal warnings.

    Raises:
        ValueError: If declared columns are missing, unknown raw columns remain,
            or protocol `one_of_col_groups` are not satisfied.
    """
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
