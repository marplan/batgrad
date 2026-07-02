from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, overload

import numpy as np
import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetStageId
from batgrad.contracts.metadata import NORMALIZE_STAGE_METADATA, MetadataLayout
from batgrad.contracts.segments import (
    ParquetSegment,
    normalize_segments,
    segment_manifest_dicts,
    segment_values,
)
from batgrad.data.processing.interactive import InteractiveStageRun
from batgrad.data.processing.io import (
    ResolvedColumns,
    add_metadata_columns,
    frame_columns,
    mapping_column_exprs,
    resolve_mapping_columns_for_segments,
    select_and_cast_columns,
)
from batgrad.data.processing.metadata import (
    as_int,
    group_task_id,
    merge_manifest_raw_paths,
    merge_manifest_segments,
    safe_name,
    stage_layout_with_protocol_metadata,
    stage_manifest_metadata,
    validate_metadata_columns,
)
from batgrad.data.processing.runtime import (
    ProcessTaskResult,
    resolve_worker_count,
    validate_stage_runtime_config,
)
from batgrad.data.processing.selection import (
    StageSelection,
    all_group_columns,
    normalize_selector_values,
)
from batgrad.data.processing.stage import (
    PreparedTable,
    StageOutputSpec,
    TaskOutput,
    close_stage_writer,
    create_stage_writer,
    interactive_run_root,
    iter_stage_task_results,
    protocol_spec_by_id,
    run_stage_task_outputs,
    stage_temp_path,
)
from batgrad.data.transforms.annotations import finalize_annotations
from batgrad.data.transforms.checks import (
    CheckSpec,
    apply_checks_bounded_chunk,
    apply_checks_full_task,
)
from batgrad.data.transforms.resampling import (
    ResamplingSpec,
    resampling_metadata_values,
    run_resampling,
)
from batgrad.logging import get_logger
from batgrad.storage.chunks import iter_data_chunks
from batgrad.storage.segments import (
    SegmentSource,
    coalesce_frames,
    collect_frame,
    iter_segment_frames,
    scan_segment_frames,
)

logger = get_logger(__name__)
_DOWNSAMPLE_ROW_ID = "__normalize_downsample_row_id"

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from batgrad.contracts.mapping import DatasetProtocolId, MappingSpec
    from batgrad.contracts.metadata import StageLayout
    from batgrad.contracts.protocols import BatteryProtocolSpec
    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.storage.store import DataProcessingStore


class NormalizeTransformSpec(Protocol):
    @property
    def input_columns(self) -> tuple[MappingSpec, ...]: ...

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]: ...

    @overload
    def apply(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def apply(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame: ...


@dataclass(frozen=True)
class NormalizeStageConfig:
    """Runtime, validation, and parquet-writing settings for normalization.

    Tasks are collected in memory when `max_batch_rows` is `None` or the task row
    count fits within that limit. Larger tasks use bounded chunk processing; if a
    protocol also has resampling, its resampling spec must support bounded
    execution.

    Attributes:
        n_jobs: Worker count. Use `1` for sequential execution, `-1` for
            available CPUs minus one, or a positive count capped by task count.
        worker_polars_max_threads: Polars threads per worker. `-1` divides CPUs
            across workers, `None` leaves Polars unrestricted, and a positive
            value sets an exact thread count.
        chunk_rows: Output chunk size written to final shards.
        compression: Parquet compression codec.
        use_content_defined_chunking: Whether table writers may use content
            defined chunking.
        row_group_size: Parquet row group size for written files.
        max_shard_size_bytes: Roll a protocol shard after this approximate size;
            `0` disables size-based rolling.
        max_batch_rows: Maximum rows processed in memory per bounded batch. Set
            to `None` to collect each task fully.
        apply_resampling: Whether protocol resampling specs are applied.
        apply_physics_compensation: Whether bounded MinMaxLTTB recomputes `dt`,
            rebuilt time, and averaged current/C-rate when possible.

    Examples:
        >>> NormalizeStageConfig(n_jobs=-1, max_batch_rows=500_000)
        NormalizeStageConfig(...)
    """

    n_jobs: int = 1
    worker_polars_max_threads: int | None = -1
    chunk_rows: int = 500_000
    compression: str = "zstd"
    use_content_defined_chunking: bool = False
    row_group_size: int = 262_144
    max_shard_size_bytes: int = 2 * 1024 * 1024 * 1024
    max_batch_rows: int | None = 500_000
    apply_resampling: bool = True
    apply_physics_compensation: bool = True


@dataclass(frozen=True)
class NormalizeProtocolSpec:
    """Normalization recipe for one protocol.

    Required input columns are inferred from order columns, transform inputs,
    output columns not produced by transforms/constants/checks, check inputs,
    and resampling inputs. When multiple aliases for a requested column are
    available in ingested shards, normalization coalesces them into the canonical
    output column.

    Resampling runs after transforms and checks. Large tasks only work with a
    resampler that implements bounded execution; otherwise keep `max_batch_rows`
    unset or high enough for full-task processing.

    Attributes:
        protocol: Shared protocol definition, including task grouping metadata.
        order_by: Columns used to sort each task. Defaults to `protocol.axis_col`.
        columns: Canonical output columns to write.
        constant_columns: Fixed columns added to every output row, such as
            ambient temperature when it is known from dataset metadata instead of
            raw measurements.
        transforms: Column derivations run before checks.
        checks: Validation or annotation checks run after transforms.
        resampling: Optional row reduction or interpolation step.

    Examples:
        >>> from batgrad.contracts.protocols import BatteryProtocols
        >>> from batgrad.data.transforms.checks import MissingCheckSpec, TimeCheckSpec
        >>> NormalizeProtocolSpec(
        ...     protocol=BatteryProtocols.cyc,
        ...     columns=(
        ...         BaseColumns.time,
        ...         BaseColumns.volt,
        ...         BaseColumns.curr,
        ...         BaseColumns.amb_temp,
        ...     ),
        ...     constant_columns={BaseColumns.amb_temp: 20.0},
        ...     checks=(MissingCheckSpec(), TimeCheckSpec(BaseColumns.time, BaseColumns.dt)),
        ... )
        NormalizeProtocolSpec(...)
    """

    protocol: BatteryProtocolSpec
    order_by: tuple[MappingSpec, ...] = ()
    columns: tuple[MappingSpec, ...] = ()
    constant_columns: dict[MappingSpec, object] = field(default_factory=dict)
    transforms: tuple[NormalizeTransformSpec, ...] = field(default_factory=tuple)
    checks: tuple[CheckSpec, ...] = field(default_factory=tuple)
    resampling: ResamplingSpec | None = None

    def __post_init__(self) -> None:
        if not self.order_by:
            object.__setattr__(self, "order_by", (self.protocol.axis_col,))

    @property
    def protocol_id(self) -> DatasetProtocolId:
        """Canonical protocol id for this normalize spec."""
        return self.protocol.protocol_id

    @property
    def group_by(self) -> tuple[MappingSpec, ...]:
        """Task grouping columns inherited from protocol metadata."""
        return self.protocol.task_key

    @property
    def output_columns(self) -> tuple[MappingSpec, ...]:
        """Canonical columns written for this protocol before annotations."""
        return tuple(dict.fromkeys((*self.order_by, *self.columns, *self.constant_columns)))

    @property
    def required_input_columns(self) -> tuple[MappingSpec, ...]:
        """Ingested columns needed to prepare this protocol's normalize tasks."""
        produced = set(self.produced_columns)
        return tuple(
            dict.fromkeys(
                (
                    *self.order_by,
                    *self.transform_input_columns,
                    *(column for column in self.columns if column not in produced),
                    *self._check_columns(),
                    *self.resampling_input_columns,
                ),
            ),
        )

    @property
    def transform_input_columns(self) -> tuple[MappingSpec, ...]:
        return tuple(
            dict.fromkeys(
                column for transform in self.transforms for column in transform.input_columns
            )
        )

    @property
    def transform_produced_columns(self) -> tuple[MappingSpec, ...]:
        return tuple(
            dict.fromkeys(
                column for transform in self.transforms for column in transform.produced_columns
            )
        )

    @property
    def resampling_input_columns(self) -> tuple[MappingSpec, ...]:
        return () if self.resampling is None else self.resampling.input_columns

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.constant_columns,
                    *(
                        column
                        for producer in (*self.transforms, *self.checks)
                        for column in producer.produced_columns
                    ),
                ),
            )
        )

    def _check_columns(self) -> tuple[MappingSpec, ...]:
        produced = {*self.constant_columns, *self.transform_produced_columns}
        columns: list[MappingSpec] = []
        for check in self.checks:
            columns.extend(column for column in check.input_columns if column not in produced)
            produced.update(check.produced_columns)
        return tuple(dict.fromkeys(columns))


@dataclass(frozen=True)
class NormalizeStageSpec:
    """Dataset-level normalization configuration.

    The stage expands manifest metadata with raw paths, ingested parquet segment
    references, resampling method/arguments, time convention, and protocol group
    keys. `time_convention` is also written to generated parquet footers.

    Attributes:
        metadata: Base normalized-stage metadata layout.
        protocol_specs: Protocol normalization recipes.
        time_convention: Label written to manifest/footer metadata describing
            how task time values are interpreted.

    Examples:
        >>> from batgrad.contracts.protocols import BatteryProtocols
        >>> NormalizeStageSpec(
        ...     protocol_specs=(NormalizeProtocolSpec(protocol=BatteryProtocols.cyc),),
        ...     time_convention="start_of_interval",
        ... )
        NormalizeStageSpec(...)
    """

    metadata: StageLayout = NORMALIZE_STAGE_METADATA
    protocol_specs: tuple[NormalizeProtocolSpec, ...] = field(default_factory=tuple)
    time_convention: str = "start_of_interval"

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
            (spec.protocol for spec in self.protocol_specs),
            manifest_extra=(
                BaseColumns.raw_paths,
                BaseColumns.ingest_segs,
                BaseColumns.resamp,
                BaseColumns.resamp_args,
                BaseColumns.time_conv,
                *(column for spec in self.protocol_specs for column in spec.group_by),
            ),
            footer_extra=(BaseColumns.time_conv,),
        ).with_footer(
            {BaseColumns.time_conv: self.time_convention},
        )

    def protocol_spec(self, protocol: object) -> NormalizeProtocolSpec:
        """Return the protocol spec matching a protocol id or value.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Matching normalize protocol spec.

        Raises:
            ValueError: If no protocol spec matches.
        """
        return protocol_spec_by_id(self.protocol_specs, protocol)

    def output_columns(self, protocol: object) -> tuple[MappingSpec, ...]:
        """Canonical output columns for a protocol.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Columns written to normalized parquet before annotations.
        """
        return self.protocol_spec(protocol).output_columns

    def required_input_columns(self, protocol: object) -> tuple[MappingSpec, ...]:
        """Ingested columns required for a protocol's normalize tasks.

        Args:
            protocol: Protocol id, enum value, or string-like value.

        Returns:
            Ingested columns needed to prepare and validate the protocol.
        """
        return self.protocol_spec(protocol).required_input_columns

    def task_metadata(self, dataset_id: str, task: NormalizeTask) -> dict[MappingSpec, object]:
        """Metadata written for one normalized task's manifest rows and footers.

        Args:
            dataset_id: Dataset id for generated output metadata.
            task: Planned normalization task.

        Returns:
            Metadata values for manifest rows and footer resolution.
        """
        return {
            BaseColumns.set_id: dataset_id,
            BaseColumns.proto: task.protocol_id,
            BaseColumns.raw_paths: list(task.raw_paths),
            BaseColumns.ingest_segs: list(segment_manifest_dicts(task.parquet_segments)),
            BaseColumns.time_conv: self.time_convention,
            **task.group_values,
        }

    def output_spec(
        self,
        dataset_spec: DatasetSpec,
        *,
        output_root: str | None = None,
        manifest_path: str | None = None,
    ) -> StageOutputSpec:
        """Build the output writer configuration for normalized shards.

        Args:
            dataset_spec: Dataset storage configuration.
            output_root: Optional output root override for interactive runs.
            manifest_path: Optional manifest path override for interactive runs.

        Returns:
            Stage writer configuration for protocol-sharded normalized output.
        """
        output_root = output_root or dataset_spec.source_root(self.metadata.stage_id)
        manifest_path = manifest_path or dataset_spec.manifest(self.metadata.stage_id)
        return StageOutputSpec(
            dataset_spec=dataset_spec,
            stage_id=self.metadata.stage_id,
            output_root=output_root,
            manifest_path=manifest_path,
            manifest_metadata=self.manifest_metadata,
            footer_metadata=self.footer_metadata,
            shard_key_col=BaseColumns.proto,
            segment_col=BaseColumns.norm_segs,
            source_paths_col=BaseColumns.raw_paths,
        )


def normalize_spec_with_resampling(
    normalize_spec: NormalizeStageSpec,
    resampling_by_protocol: dict[DatasetProtocolId, ResamplingSpec | None],
) -> NormalizeStageSpec:
    """Return a copy of a normalize spec with protocol resampling overrides.

    Args:
        normalize_spec: Base normalization spec.
        resampling_by_protocol: Resampling spec or `None` keyed by protocol id.

    Returns:
        A new stage spec with matching protocol specs replaced.

    Examples:
        >>> from batgrad.contracts.mapping import DatasetProtocolId
        >>> from batgrad.contracts.protocols import BatteryProtocols
        >>> from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec
        >>> base = NormalizeStageSpec(
        ...     protocol_specs=(NormalizeProtocolSpec(protocol=BatteryProtocols.cyc),),
        ... )
        >>> updated = normalize_spec_with_resampling(
        ...     base,
        ...     {
        ...         DatasetProtocolId.cycling: MinMaxLTTBResamplingSpec(
        ...             x_col=BaseColumns.time,
        ...             y_col=BaseColumns.volt,
        ...             points_ratio=0.1,
        ...         )
        ...     },
        ... )
    """
    return replace(
        normalize_spec,
        protocol_specs=tuple(
            replace(
                protocol_spec,
                resampling=resampling_by_protocol.get(protocol_spec.protocol_id),
            )
            for protocol_spec in normalize_spec.protocol_specs
        ),
    )


@dataclass(frozen=True)
class NormalizeTaskContext:
    task: NormalizeTask
    protocol_spec: NormalizeProtocolSpec
    input_columns: tuple[str, ...]
    resolved_columns: ResolvedColumns


@dataclass(frozen=True)
class NormalizeTask:
    """One protocol/group normalization unit planned from the ingested manifest.

    A task contains the ingested parquet segments for one protocol task key, such
    as one cell/cycle pair, plus source-path and row-count metadata used for the
    normalized manifest.

    Attributes:
        task_id: Stable id used in logs and temporary output paths.
        protocol_id: Protocol represented by the task.
        group_values: Task-key metadata values, such as cell and cycle.
        raw_paths: Raw source paths represented by the task.
        parquet_segments: Ingested parquet segments to read.
        row_count: Total input row count across segments.
    """

    task_id: str
    protocol_id: str
    group_values: dict[MappingSpec, object]
    raw_paths: tuple[str, ...]
    parquet_segments: tuple[ParquetSegment, ...]
    row_count: int


@dataclass(frozen=True)
class _NormalizeWorkerPayload:
    task_index: int
    task: NormalizeTask
    dataset_spec: DatasetSpec
    input_store: DataProcessingStore
    scratch_store: DataProcessingStore
    normalize_spec: NormalizeStageSpec
    config: NormalizeStageConfig
    scratch_root: str
    dry_run: bool = False
    annotate: bool = False


def run_normalize(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    output_store: DataProcessingStore,
    config: NormalizeStageConfig,
    *,
    scratch_store: DataProcessingStore | None = None,
    tasks: tuple[NormalizeTask, ...] | None = None,
    dry_run: bool = False,
) -> None:
    """Run ingested-to-normalized parquet processing for a dataset.

    The stage reads the ingested manifest, plans protocol/group tasks, applies
    configured transforms, checks, and resampling, then writes normalized shards
    and a normalized manifest. `dry_run=True` validates tasks and checks without
    writing outputs.

    Args:
        dataset_spec: Dataset configuration containing a normalized stage spec.
        input_store: Store containing ingested manifest and shards.
        output_store: Store receiving normalized shards and manifest.
        config: Runtime, validation, and parquet-writing settings.
        scratch_store: Optional store for temporary task outputs. Defaults to
            `output_store`.
        tasks: Optional task subset for retries or tests.
        dry_run: Validate and log check violations without writing output.

    Returns:
        `None`. Outputs are written to `output_store` unless `dry_run=True`.
    """
    normalize_spec = _dataset_normalize_spec(dataset_spec)
    output_root = dataset_spec.source_root(normalize_spec.metadata.stage_id)
    manifest_path = dataset_spec.manifest(normalize_spec.metadata.stage_id)
    _run_normalize_to_root(
        dataset_spec,
        input_store,
        output_store,
        normalize_spec,
        config,
        output_root=output_root,
        manifest_path=manifest_path,
        scratch_store=scratch_store,
        tasks=tasks,
        delete_output_root=True,
        dry_run=dry_run,
    )


def run_normalize_interactive(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    config: NormalizeStageConfig,
    *,
    protocols: object = None,
    group_values: object = None,
    annotate: bool = True,
    source_run: InteractiveStageRun | None = None,
    normalize_spec: NormalizeStageSpec | None = None,
) -> InteractiveStageRun:
    """Run selected normalization work into a scratch-backed interactive run.

    `protocols` and `group_values` filter planned tasks. Passing an unresampled
    normalized `source_run` lets developers try different resampling settings
    without repeating the full normalize preparation path.

    Args:
        dataset_spec: Dataset configuration containing a normalized stage spec.
        input_store: Store containing ingested manifest and shards, unless
            `source_run` is provided.
        scratch_store: Store receiving temporary interactive outputs.
        config: Normalize runtime, validation, and parquet-writing settings.
        protocols: Optional protocol selector, such as one id or a list of ids.
        group_values: Optional task group selector or list of selectors.
        annotate: Whether public annotation columns are included in outputs.
        source_run: Optional prior unresampled normalized run used as the source
            for resampling experiments.
        normalize_spec: Optional normalization spec override.

    Returns:
        Interactive run handle for scanning outputs and cleaning scratch files.
    """
    normalize_spec = normalize_spec or _dataset_normalize_spec(dataset_spec)
    if source_run is not None:
        return _run_normalize_interactive_from_source_run(
            dataset_spec,
            source_run,
            scratch_store,
            normalize_spec,
            config,
            protocols=protocols,
            group_values=group_values,
            annotate=annotate,
        )
    tasks = plan_normalize_tasks(
        dataset_spec,
        input_store,
        normalize_spec,
        protocols=protocols,
        group_values=group_values,
    )
    if not tasks:
        raise ValueError(
            f"No normalize task matched protocols={protocols!r} group_values={group_values!r}",
        )
    run_root = interactive_run_root(dataset_spec, normalize_spec.metadata.stage_id)
    manifest_path = f"{run_root}/manifest.parquet"
    _run_normalize_to_root(
        dataset_spec,
        input_store,
        scratch_store,
        normalize_spec,
        config,
        output_root=run_root,
        manifest_path=manifest_path,
        scratch_store=scratch_store,
        tasks=tasks,
        delete_output_root=True,
        dry_run=False,
        annotate=annotate,
    )
    return InteractiveStageRun(
        output_store=scratch_store,
        stage_spec=normalize_spec,
        segment_col=BaseColumns.norm_segs,
        protocol_order=_interactive_protocol_order(normalize_spec, protocols),
        dataset_spec=dataset_spec,
        input_store=input_store,
        protocols=protocols,
        group_values=group_values,
        source_stage=DatasetStageId.normalized,
        manifest_path=manifest_path,
        run_root=run_root,
    )


def _run_normalize_interactive_from_source_run(
    dataset_spec: DatasetSpec,
    source_run: InteractiveStageRun,
    scratch_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
    *,
    protocols: object,
    group_values: object,
    annotate: bool,
) -> InteractiveStageRun:
    _validate_source_run_compatible(source_run, normalize_spec, config)
    manifest = _select_source_run_manifest(source_run, normalize_spec, protocols, group_values)
    run_root = interactive_run_root(dataset_spec, normalize_spec.metadata.stage_id)
    manifest_path = f"{run_root}/manifest.parquet"
    _run_source_run_resampling_to_root(
        dataset_spec,
        source_run,
        scratch_store,
        normalize_spec,
        config,
        manifest,
        output_root=run_root,
        manifest_path=manifest_path,
        annotate=annotate,
    )
    return InteractiveStageRun(
        output_store=scratch_store,
        stage_spec=normalize_spec,
        segment_col=BaseColumns.norm_segs,
        protocol_order=_interactive_protocol_order(normalize_spec, protocols),
        dataset_spec=dataset_spec,
        input_store=source_run.input_store,
        protocols=protocols,
        group_values=group_values,
        source_stage=DatasetStageId.normalized,
        manifest_path=manifest_path,
        run_root=run_root,
    )


def _run_normalize_direct_to_root(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    output_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
    *,
    output_root: str,
    manifest_path: str,
    scratch_store: DataProcessingStore,
    tasks: tuple[NormalizeTask, ...],
    annotate: bool,
    delete_output_root: bool,
) -> int:
    validate_stage_runtime_config(config)
    if delete_output_root:
        output_store.delete_dir(output_root, missing_ok=True)
    output_store.create_dir(output_root)
    output_spec = normalize_spec.output_spec(
        dataset_spec,
        output_root=output_root,
        manifest_path=manifest_path,
    )
    writer = create_stage_writer(output_store, output_spec, config)
    temp_root = f"{output_root}/_tmp_direct"
    succeeded = failed = rows = 0
    aborted = False
    try:
        for task_index, task in enumerate(tasks, start=1):
            try:
                context = _normalize_task_context(task, input_store, normalize_spec)
                task_rows = 0
                task_stats: dict[str, tuple[float, float]] = {}
                task_manifest_rows: list[dict[MappingSpec, object]] = []
                metadata = _task_metadata(dataset_spec.dataset_id, normalize_spec, task, config)
                for chunk in coalesce_frames(
                    _iter_task_output_chunks(
                        context,
                        input_store,
                        scratch_store,
                        config,
                        temp_root,
                        annotate=annotate,
                    ),
                    config.max_batch_rows or config.chunk_rows,
                ):
                    if chunk.height == 0:
                        continue
                    _update_normalized_stats(task_stats, chunk)
                    row = writer.append(chunk, metadata, task.raw_paths)
                    if row is not None:
                        task_manifest_rows.append(row)
                    task_rows += chunk.height
                stats_value = _normalized_stats_value(task_stats)
                for row in task_manifest_rows:
                    row[BaseColumns.norm_stats] = stats_value
                if task_rows == 0:
                    logger.warning(
                        "%s task produced no rows task_id=%s",
                        normalize_spec.metadata.stage_id,
                        task.task_id,
                    )
                succeeded += 1
                rows += task_rows
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.error(  # noqa: TRY400
                    "%s task failed task=%d/%d task_id=%s error=%s: %s",
                    normalize_spec.metadata.stage_id,
                    task_index,
                    len(tasks),
                    task.task_id,
                    type(exc).__name__,
                    str(exc),
                )
    except BaseException:
        aborted = True
        raise
    finally:
        close_stage_writer(writer, succeeded=succeeded, failed=failed, aborted=aborted)
        scratch_store.delete_dir(temp_root, missing_ok=True)
    return rows


def _validate_source_run_compatible(
    source_run: InteractiveStageRun,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
) -> None:
    if source_run.source_stage != DatasetStageId.normalized:
        raise ValueError("source_run must be a normalized InteractiveStageRun")
    if not isinstance(source_run.stage_spec, NormalizeStageSpec):
        raise TypeError("source_run.stage_spec must be a NormalizeStageSpec")
    if not _normalize_specs_match_except_resampling(source_run.stage_spec, normalize_spec):
        raise ValueError("source_run is incompatible with normalize_spec except for resampling")
    if config.apply_resampling:
        methods = set(source_run.manifest()[str(BaseColumns.resamp)].to_list())
        if methods - {"none"}:
            raise ValueError("source_run is already resampled; provide an unresampled source_run")


def _normalize_specs_match_except_resampling(
    source_spec: NormalizeStageSpec,
    target_spec: NormalizeStageSpec,
) -> bool:
    if source_spec.time_convention != target_spec.time_convention:
        return False
    if len(source_spec.protocol_specs) != len(target_spec.protocol_specs):
        return False
    source_by_protocol = {str(spec.protocol_id): spec for spec in source_spec.protocol_specs}
    target_by_protocol = {str(spec.protocol_id): spec for spec in target_spec.protocol_specs}
    if source_by_protocol.keys() != target_by_protocol.keys():
        return False
    return all(
        replace(source_by_protocol[protocol], resampling=None)
        == replace(target_by_protocol[protocol], resampling=None)
        for protocol in source_by_protocol
    )


def _select_source_run_manifest(
    source_run: InteractiveStageRun,
    normalize_spec: NormalizeStageSpec,
    protocols: object,
    group_values: object,
) -> pl.DataFrame:
    selection = _normalize_selection(normalize_spec, protocols, group_values)
    filter_expr = selection.expr()
    manifest = source_run.manifest()
    selected = manifest.filter(filter_expr) if filter_expr is not None else manifest
    if selected.height == 0:
        raise ValueError(
            f"source_run does not cover protocols={protocols!r} group_values={group_values!r}"
        )
    return selected


def _run_source_run_resampling_to_root(
    dataset_spec: DatasetSpec,
    source_run: InteractiveStageRun,
    output_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
    manifest: pl.DataFrame,
    *,
    output_root: str,
    manifest_path: str,
    annotate: bool,
) -> int:
    validate_stage_runtime_config(config)
    output_store.delete_dir(output_root, missing_ok=True)
    output_store.create_dir(output_root)
    output_spec = normalize_spec.output_spec(
        dataset_spec,
        output_root=output_root,
        manifest_path=manifest_path,
    )
    writer = create_stage_writer(output_store, output_spec, config)
    rows = 0
    aborted = False
    try:
        for row in manifest.iter_rows(named=True):
            protocol_spec = normalize_spec.protocol_spec(row[str(BaseColumns.proto)])
            task = _source_manifest_row_task(row, protocol_spec)
            metadata = _task_metadata(dataset_spec.dataset_id, normalize_spec, task, config)
            task_rows = 0
            task_stats: dict[str, tuple[float, float]] = {}
            task_manifest_rows: list[dict[MappingSpec, object]] = []
            for chunk in coalesce_frames(
                _iter_source_run_output_chunks(
                    source_run,
                    row,
                    protocol_spec,
                    config,
                    annotate=annotate,
                ),
                config.max_batch_rows or config.chunk_rows,
            ):
                if chunk.height == 0:
                    continue
                _update_normalized_stats(task_stats, chunk)
                manifest_row = writer.append(chunk, metadata, task.raw_paths)
                if manifest_row is not None:
                    task_manifest_rows.append(manifest_row)
                task_rows += chunk.height
            stats_value = _normalized_stats_value(task_stats)
            for manifest_row in task_manifest_rows:
                manifest_row[BaseColumns.norm_stats] = stats_value
            rows += task_rows
    except BaseException:
        aborted = True
        raise
    finally:
        close_stage_writer(writer, succeeded=manifest.height, failed=0, aborted=aborted)
    return rows


def _source_manifest_row_task(
    row: dict[str, object],
    protocol_spec: NormalizeProtocolSpec,
) -> NormalizeTask:
    group_values = {column: row[str(column)] for column in protocol_spec.group_by}
    row_raw_paths = row.get(str(BaseColumns.raw_paths))
    raw_paths = (
        tuple(str(path) for path in row_raw_paths)
        if isinstance(row_raw_paths, list | tuple)
        else ()
    )
    row_segments = row.get(str(BaseColumns.norm_segs))
    segments = normalize_segments(segment_values(row_segments))
    return NormalizeTask(
        task_id=group_task_id(str(protocol_spec.protocol_id), group_values),
        protocol_id=str(protocol_spec.protocol_id),
        group_values=group_values,
        raw_paths=raw_paths,
        parquet_segments=segments,
        row_count=as_int(row[str(BaseColumns.row_n)]),
    )


def _iter_source_run_output_chunks(
    source_run: InteractiveStageRun,
    row: dict[str, object],
    protocol_spec: NormalizeProtocolSpec,
    config: NormalizeStageConfig,
    *,
    annotate: bool,
) -> Iterator[pl.DataFrame]:
    row_count = as_int(row[str(BaseColumns.row_n)])
    row_segments = row.get(str(BaseColumns.norm_segs))
    source = SegmentSource.from_values(
        source_run.output_store,
        segment_values(row_segments),
        row_count=row_count,
    )
    resampling = protocol_spec.resampling if config.apply_resampling else None

    def finalize(frame: pl.DataFrame) -> pl.DataFrame:
        return _finalize_output_columns(
            frame,
            protocol_spec,
            include_annotations=annotate,
        )

    if resampling is None:
        yield from coalesce_frames(
            (
                finalize(frame)
                for frame in source.iter_chunks(config.max_batch_rows or config.chunk_rows)
            ),
            config.chunk_rows,
        )
        return
    requires_bounded = config.max_batch_rows is not None and row_count > config.max_batch_rows
    if requires_bounded:
        chunk_rows = config.max_batch_rows or config.chunk_rows

        def chunks() -> Iterator[pl.DataFrame]:
            row_offset = 0
            for source_frame in source.iter_chunks(chunk_rows):
                frame = source_frame.with_columns(
                    pl.Series(
                        _DOWNSAMPLE_ROW_ID,
                        np.arange(row_offset, row_offset + source_frame.height),
                    )
                )
                row_offset += source_frame.height
                yield frame

        yield from coalesce_frames(
            (
                finalize(frame)
                for frame in _iter_resampled_bounded(
                    chunks,
                    row_count,
                    chunk_rows,
                    resampling,
                    config,
                )
            ),
            config.chunk_rows,
        )
        return
    frame = collect_frame(source.scan())
    yield from iter_data_chunks(
        resampling.apply_full(
            frame,
            apply_physics_compensation=config.apply_physics_compensation,
        ).pipe(finalize),
        config.chunk_rows,
    )


def plan_normalize_tasks(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    *,
    protocols: object = None,
    group_values: object = None,
) -> tuple[NormalizeTask, ...]:
    """Plan normalization tasks from the ingested manifest.

    The ingested manifest is optionally filtered by protocol and group values,
    then rows are grouped by each protocol's task key. Each output task merges
    raw source paths and parquet segment references for one group.

    Args:
        dataset_spec: Dataset configuration.
        input_store: Store containing the ingested manifest.
        normalize_spec: Normalization configuration.
        protocols: Optional protocol selector.
        group_values: Optional task group selector or list of selectors.

    Returns:
        Planned normalization tasks.

    Raises:
        ValueError: If required manifest metadata or group selectors are invalid.
    """
    manifest_path = dataset_spec.manifest(DatasetStageId.ingested)
    manifest_lf = input_store.scan_table(manifest_path)
    parquet_manifest_metadata = stage_manifest_metadata(dataset_spec, DatasetStageId.ingested)
    validate_metadata_columns(
        parquet_manifest_metadata,
        manifest_lf.collect_schema().names(),
        context=f"Parquet manifest {manifest_path!r}",
    )
    filter_expr = _normalize_selection(normalize_spec, protocols, group_values).expr()
    if filter_expr is not None:
        manifest_lf = manifest_lf.filter(filter_expr)
    manifest = collect_frame(manifest_lf)
    tasks: list[NormalizeTask] = []
    for protocol_spec in normalize_spec.protocol_specs:
        _validate_group_columns_declared(protocol_spec, parquet_manifest_metadata)
        protocol = str(protocol_spec.protocol_id)
        protocol_rows = manifest.filter(pl.col(BaseColumns.proto) == protocol)
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in protocol_rows.iter_rows(named=True):
            key = tuple(row[str(column)] for column in protocol_spec.group_by)
            groups.setdefault(key, []).append(row)

        for group_key, rows in groups.items():
            group_values = dict(zip(protocol_spec.group_by, group_key, strict=True))
            parquet_segments = normalize_segments(
                merge_manifest_segments(rows, BaseColumns.ingest_segs)
            )
            tasks.append(
                NormalizeTask(
                    task_id=group_task_id(protocol, group_values),
                    protocol_id=protocol,
                    group_values=group_values,
                    raw_paths=merge_manifest_raw_paths(rows),
                    parquet_segments=parquet_segments,
                    row_count=sum(as_int(row[str(BaseColumns.row_n)]) for row in rows),
                ),
            )
    return tuple(tasks)


def _run_normalize_to_root(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    output_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
    *,
    output_root: str,
    manifest_path: str,
    scratch_store: DataProcessingStore | None,
    tasks: tuple[NormalizeTask, ...] | None,
    delete_output_root: bool,
    dry_run: bool,
    annotate: bool = False,
) -> int:
    scratch_store = scratch_store or output_store
    validate_stage_runtime_config(config)
    if tasks is None:
        tasks = plan_normalize_tasks(dataset_spec, input_store, normalize_spec)

    if not dry_run and resolve_worker_count(config.n_jobs, len(tasks)) <= 1:
        return _run_normalize_direct_to_root(
            dataset_spec,
            input_store,
            output_store,
            normalize_spec,
            config,
            output_root=output_root,
            manifest_path=manifest_path,
            scratch_store=scratch_store,
            tasks=tasks,
            annotate=annotate,
            delete_output_root=delete_output_root,
        )

    output_spec = normalize_spec.output_spec(
        dataset_spec,
        output_root=output_root,
        manifest_path=manifest_path,
    )
    stats = run_stage_task_outputs(
        output_store=output_store,
        scratch_store=scratch_store,
        output_spec=output_spec,
        config=config,
        tasks=tasks,
        iter_results=lambda scratch_root: _iter_normalize_task_results(
            dataset_spec,
            input_store,
            scratch_store,
            normalize_spec,
            config,
            scratch_root,
            tasks,
            dry_run=dry_run,
            annotate=annotate,
        ),
        logger=logger,
        delete_output_root=delete_output_root,
        dry_run=dry_run,
    )
    return stats.rows


def _process_normalize_task(payload: _NormalizeWorkerPayload) -> TaskOutput[NormalizeTask]:
    return _process_normalize_task_output(
        payload.dataset_spec,
        payload.input_store,
        payload.scratch_store,
        payload.normalize_spec,
        payload.config,
        payload.scratch_root,
        payload.task_index,
        payload.task,
        dry_run=payload.dry_run,
        annotate=payload.annotate,
    )


def _update_normalized_stats(
    stats: dict[str, tuple[float, float]],
    frame: pl.DataFrame,
) -> None:
    numeric_columns = tuple(
        column
        for column, dtype in frame.schema.items()
        if _is_stats_numeric_dtype(dtype) and column != _DOWNSAMPLE_ROW_ID
    )
    if not numeric_columns:
        return
    observed = frame.select(
        *(pl.col(column).min().alias(f"{column}__min") for column in numeric_columns),
        *(pl.col(column).max().alias(f"{column}__max") for column in numeric_columns),
    ).row(0, named=True)
    for column in numeric_columns:
        min_value = observed[f"{column}__min"]
        max_value = observed[f"{column}__max"]
        if min_value is None or max_value is None:
            continue
        current = stats.get(column)
        next_min = float(min_value)
        next_max = float(max_value)
        stats[column] = (
            next_min if current is None else min(current[0], next_min),
            next_max if current is None else max(current[1], next_max),
        )


def _normalized_stats_value(stats: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    return [
        {"column": column, "min": values[0], "max": values[1]}
        for column, values in sorted(stats.items())
    ]


def _is_stats_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() and dtype != pl.Boolean


def _iter_normalize_task_results(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
    scratch_root: str,
    tasks: tuple[NormalizeTask, ...],
    *,
    dry_run: bool,
    annotate: bool,
) -> Iterator[ProcessTaskResult[TaskOutput[NormalizeTask]]]:
    yield from iter_stage_task_results(
        tasks=tasks,
        task_id=lambda task: task.task_id,
        make_payload=lambda idx, task: _NormalizeWorkerPayload(
            idx,
            task,
            dataset_spec,
            input_store,
            scratch_store,
            normalize_spec,
            config,
            scratch_root,
            dry_run,
            annotate,
        ),
        worker=_process_normalize_task,
        config=config,
        ordered=False,
    )


def _process_normalize_task_output(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
    config: NormalizeStageConfig,
    scratch_root: str,
    task_index: int,
    task: NormalizeTask,
    *,
    dry_run: bool,
    annotate: bool,
) -> TaskOutput[NormalizeTask]:
    context = _normalize_task_context(task, input_store, normalize_spec)
    if dry_run:
        return TaskOutput(
            task=task,
            tables=(),
            warnings=_dry_run_normalize_task(context, input_store, config),
        )

    temp_path = stage_temp_path(scratch_root, task_index)
    writer = None
    rows_written = 0
    stats: dict[str, tuple[float, float]] = {}
    try:
        for chunk in _iter_task_output_chunks(
            context,
            input_store,
            scratch_store,
            config,
            scratch_root,
            annotate=annotate,
        ):
            if writer is None:
                writer = scratch_store.open_table_writer(
                    temp_path,
                    chunk.to_arrow().schema,
                    config.compression,
                    use_content_defined_chunking=config.use_content_defined_chunking,
                )
            _update_normalized_stats(stats, chunk)
            writer.write_table(chunk, row_group_size=config.row_group_size)
            rows_written += chunk.height
    finally:
        if writer is not None:
            writer.close()
    if rows_written == 0:
        scratch_store.delete_file(temp_path, missing_ok=True)
        return TaskOutput(task=task, tables=())
    metadata = _task_metadata(dataset_spec.dataset_id, normalize_spec, task, config)
    metadata[BaseColumns.norm_stats] = _normalized_stats_value(stats)
    return TaskOutput(
        task=task,
        tables=(
            PreparedTable(
                temp_path=temp_path,
                metadata=metadata,
                source_paths=task.raw_paths,
            ),
        ),
    )


def _scan_task_segments(input_store: DataProcessingStore, task: NormalizeTask) -> pl.LazyFrame:
    if not task.parquet_segments:
        raise ValueError(f"Normalize task {task.task_id!r} has no parquet segments")
    return scan_segment_frames(input_store, task.parquet_segments)


def _normalize_task_context(
    task: NormalizeTask,
    input_store: DataProcessingStore,
    normalize_spec: NormalizeStageSpec,
) -> NormalizeTaskContext:
    protocol_spec = normalize_spec.protocol_spec(task.protocol_id)
    input_columns, resolved_columns = _task_input_resolution(task, input_store, protocol_spec)
    return NormalizeTaskContext(task, protocol_spec, input_columns, resolved_columns)


def _task_input_resolution(
    task: NormalizeTask,
    input_store: DataProcessingStore,
    protocol_spec: NormalizeProtocolSpec,
) -> tuple[tuple[str, ...], ResolvedColumns]:
    requested = tuple(
        dict.fromkeys(
            (
                *protocol_spec.required_input_columns,
                *protocol_spec.transform_input_columns,
                BaseColumns.ann_cols,
                BaseColumns.ann_reasons,
            )
        )
    )
    input_columns, resolved_columns = resolve_mapping_columns_for_segments(
        input_store,
        task.parquet_segments,
        requested,
        protocol_spec.required_input_columns,
        context=f"Normalize task {task.task_id!r} raw shards",
        one_of_col_groups=protocol_spec.protocol.one_of_col_groups,
    )
    _log_coalesced_aliases(task, resolved_columns)
    return input_columns, resolved_columns


def _log_coalesced_aliases(task: NormalizeTask, resolved_columns: ResolvedColumns) -> None:
    for column, sources in resolved_columns.items():
        if sources is not None and len(sources) > 1:
            logger.info(
                "Normalize task %s resolves %s by coalescing aliases: %s",
                task.task_id,
                column,
                ", ".join(sources),
            )


def _iter_task_output_chunks(
    context: NormalizeTaskContext,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    config: NormalizeStageConfig,
    scratch_root: str,
    *,
    annotate: bool,
) -> Iterator[pl.DataFrame]:
    task = context.task
    resampling = context.protocol_spec.resampling if config.apply_resampling else None
    requires_bounded = config.max_batch_rows is not None and task.row_count > config.max_batch_rows
    if resampling is not None and requires_bounded:
        yield from _iter_task_resampling_bounded_chunks(
            context,
            input_store,
            scratch_store,
            resampling,
            config,
            scratch_root,
            annotate=annotate,
        )
        return
    if requires_bounded:
        if resampling is not None:
            raise RuntimeError(
                f"Normalize task {task.task_id!r} requires bounded execution, but "
                f"resampling {type(resampling).__name__} has no bounded path",
            )
        yield from coalesce_frames(
            (
                _finalize_output_columns(frame, context.protocol_spec, include_annotations=annotate)
                for frame, _violations in _iter_bounded_frames(
                    context,
                    input_store,
                    config.max_batch_rows or config.chunk_rows,
                    annotate=annotate,
                    fail_on_violations=not annotate,
                )
            ),
            config.chunk_rows,
        )
        return

    data, violations = _prepare_full_task_frame(context, input_store, annotate=annotate)
    if not annotate:
        _raise_for_check_violations(task, violations)
    frame = (
        run_resampling(
            collect_frame(data),
            resampling,
            apply_physics_compensation=config.apply_physics_compensation,
        )
        if resampling is not None
        else collect_frame(data)
    )
    if frame.height:
        yield from iter_data_chunks(
            _finalize_output_columns(frame, context.protocol_spec, include_annotations=annotate),
            config.chunk_rows,
        )


def _prepare_full_task_frame(
    context: NormalizeTaskContext,
    input_store: DataProcessingStore,
    *,
    annotate: bool,
) -> tuple[pl.LazyFrame, tuple[tuple[str, str], ...]]:
    data = _prepare_lazy_task_frame(
        _scan_task_segments(input_store, context.task).select(list(context.input_columns)),
        context,
    )
    sort_columns = tuple(
        dict.fromkeys((*context.protocol_spec.group_by, *context.protocol_spec.order_by))
    )
    if sort_columns:
        data = data.sort([str(column) for column in sort_columns])
    data = _with_resolved_output_columns(data, context)
    data = _apply_task_transforms(data, context)
    data = _with_resolved_output_columns(data, context)
    data, violations = apply_checks_full_task(
        data,
        context.protocol_spec.group_by,
        context.protocol_spec.checks,
        annotate=annotate,
    )
    return _select_internal_output_columns(data, context), violations


def _iter_bounded_frames(
    context: NormalizeTaskContext,
    input_store: DataProcessingStore,
    chunk_rows: int,
    *,
    include_row_id: bool = False,
    annotate: bool = True,
    fail_on_violations: bool = False,
) -> Iterator[tuple[pl.DataFrame, tuple[tuple[str, str], ...]]]:
    check_states = tuple(check.init_state() for check in context.protocol_spec.checks)
    output_row_id = 0
    for input_frame in iter_segment_frames(
        input_store,
        context.task.parquet_segments,
        chunk_rows,
        columns=context.input_columns,
    ):
        frame = _apply_eager_task_transforms(
            _with_resolved_output_columns(_prepare_eager_task_frame(input_frame, context), context),
            context,
        )
        frame = _with_resolved_output_columns(frame, context)
        frame, violations = apply_checks_bounded_chunk(
            frame,
            context.protocol_spec.checks,
            check_states,
            annotate=annotate,
        )
        if frame.height == 0:
            continue
        frame = _select_internal_output_columns(frame, context)
        if fail_on_violations:
            _raise_for_check_violations(
                context.task,
                violations,
            )
        if include_row_id:
            frame = frame.with_columns(
                pl.Series(
                    _DOWNSAMPLE_ROW_ID, np.arange(output_row_id, output_row_id + frame.height)
                ),
            )
            output_row_id += frame.height
        yield frame, violations


def _prepare_lazy_task_frame(
    data: pl.LazyFrame,
    context: NormalizeTaskContext,
) -> pl.LazyFrame:
    return _with_constant_columns(
        add_metadata_columns(data, context.task.group_values),
        context.protocol_spec,
    )


def _prepare_eager_task_frame(
    data: pl.DataFrame,
    context: NormalizeTaskContext,
) -> pl.DataFrame:
    return _with_constant_columns(
        add_metadata_columns(data, context.task.group_values),
        context.protocol_spec,
    )


def _apply_task_transforms(data: pl.LazyFrame, context: NormalizeTaskContext) -> pl.LazyFrame:
    for transform in context.protocol_spec.transforms:
        data = transform.apply(data)
    return data


def _apply_eager_task_transforms(
    data: pl.DataFrame,
    context: NormalizeTaskContext,
) -> pl.DataFrame:
    for transform in context.protocol_spec.transforms:
        data = transform.apply(data)
    return data


@overload
def _with_resolved_output_columns(
    data: pl.DataFrame,
    context: NormalizeTaskContext,
) -> pl.DataFrame: ...


@overload
def _with_resolved_output_columns(
    data: pl.LazyFrame,
    context: NormalizeTaskContext,
) -> pl.LazyFrame: ...


def _with_resolved_output_columns(
    data: pl.DataFrame | pl.LazyFrame,
    context: NormalizeTaskContext,
) -> pl.DataFrame | pl.LazyFrame:
    exprs = mapping_column_exprs(
        context.protocol_spec.output_columns,
        set(frame_columns(data)),
        resolved_columns=context.resolved_columns,
    )
    if not exprs:
        return data
    if isinstance(data, pl.LazyFrame):
        return data.with_columns(exprs)
    return data.with_columns(exprs)


@overload
def _select_internal_output_columns(
    data: pl.DataFrame,
    context: NormalizeTaskContext,
) -> pl.DataFrame: ...


@overload
def _select_internal_output_columns(
    data: pl.LazyFrame,
    context: NormalizeTaskContext,
) -> pl.LazyFrame: ...


def _select_internal_output_columns(
    data: pl.DataFrame | pl.LazyFrame,
    context: NormalizeTaskContext,
) -> pl.DataFrame | pl.LazyFrame:
    return select_and_cast_columns(
        data,
        _internal_output_columns(context.protocol_spec),
        resolved_columns=_output_resolved_columns(context.resolved_columns),
    )


def _output_resolved_columns(resolved_columns: ResolvedColumns) -> ResolvedColumns:
    return {
        column: sources
        for column, sources in resolved_columns.items()
        if sources is not None and column not in (BaseColumns.ann_cols, BaseColumns.ann_reasons)
    }


@overload
def _with_constant_columns(
    data: pl.DataFrame,
    protocol_spec: NormalizeProtocolSpec,
) -> pl.DataFrame: ...


@overload
def _with_constant_columns(
    data: pl.LazyFrame,
    protocol_spec: NormalizeProtocolSpec,
) -> pl.LazyFrame: ...


def _with_constant_columns(
    data: pl.DataFrame | pl.LazyFrame,
    protocol_spec: NormalizeProtocolSpec,
) -> pl.DataFrame | pl.LazyFrame:
    if not protocol_spec.constant_columns:
        return data
    exprs = tuple(
        pl.lit(value).cast(column.dtype).alias(column)
        for column, value in protocol_spec.constant_columns.items()
    )
    if isinstance(data, pl.LazyFrame):
        lazy_data: pl.LazyFrame = data
        return lazy_data.with_columns(exprs)
    frame_data: pl.DataFrame = data
    return frame_data.with_columns(exprs)


def _iter_task_resampling_bounded_chunks(
    context: NormalizeTaskContext,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    resampling: ResamplingSpec,
    config: NormalizeStageConfig,
    scratch_root: str,
    *,
    annotate: bool,
) -> Iterator[pl.DataFrame]:
    chunk_rows = config.max_batch_rows
    if chunk_rows is None or chunk_rows < 1:
        raise ValueError(f"max_batch_rows must be >= 1, got {chunk_rows}")
    if annotate:
        temp_path = f"{scratch_root}/checked-{safe_name(context.task.task_id)}.parquet"
        try:
            rows_in = _materialize_checked_bounded(
                context, input_store, scratch_store, temp_path, chunk_rows, config
            )
            if rows_in == 0:
                return
            yield from coalesce_frames(
                (
                    _finalize_output_columns(
                        frame,
                        context.protocol_spec,
                        include_annotations=True,
                    )
                    for frame in _iter_resampled_bounded(
                        lambda: scratch_store.iter_table_chunks(temp_path, chunk_rows),
                        rows_in,
                        chunk_rows,
                        resampling,
                        config,
                    )
                ),
                config.chunk_rows,
            )
        finally:
            scratch_store.delete_file(temp_path, missing_ok=True)
        return

    def checked_chunks() -> Iterator[pl.DataFrame]:
        yield from coalesce_frames(
            (
                frame
                for frame, _violations in _iter_bounded_frames(
                    context,
                    input_store,
                    chunk_rows,
                    include_row_id=True,
                    annotate=False,
                    fail_on_violations=True,
                )
            ),
            chunk_rows,
        )

    # TODO: Avoid this second bounded pass if it becomes a bottleneck. The
    # current annotate=False path is correct and avoids scratch IO, but it
    # replays the checked stream once to count rows before bounded resampling.
    # Fix either by materializing checked chunks to scratch parquet, or by
    # teaching bounded resampling to own row counting for replayable streams.
    # Do not fix by collecting all chunks in memory.
    rows_in = sum(chunk.height for chunk in checked_chunks())
    if rows_in == 0:
        return
    yield from coalesce_frames(
        (
            _finalize_output_columns(frame, context.protocol_spec, include_annotations=False)
            for frame in _iter_resampled_bounded(
                checked_chunks,
                rows_in,
                chunk_rows,
                resampling,
                config,
            )
        ),
        config.chunk_rows,
    )


def _materialize_checked_bounded(
    context: NormalizeTaskContext,
    input_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    temp_path: str,
    chunk_rows: int,
    config: NormalizeStageConfig,
) -> int:
    writer = None
    rows = 0
    try:
        for frame, _violations in _iter_bounded_frames(
            context, input_store, chunk_rows, include_row_id=True
        ):
            if writer is None:
                writer = scratch_store.open_table_writer(
                    temp_path,
                    frame.to_arrow().schema,
                    config.compression,
                    use_content_defined_chunking=config.use_content_defined_chunking,
                )
            writer.write_table(frame, row_group_size=chunk_rows)
            rows += frame.height
    finally:
        if writer is not None:
            writer.close()
    return rows


def _iter_resampled_bounded(
    chunks: Callable[[], Iterator[pl.DataFrame]],
    row_count: int,
    chunk_rows: int,
    resampling: ResamplingSpec,
    config: NormalizeStageConfig,
) -> Iterator[pl.DataFrame]:
    yield from resampling.apply_bounded(
        chunks,
        row_count=row_count,
        max_batch_rows=chunk_rows,
        row_id_col=_DOWNSAMPLE_ROW_ID,
        apply_physics_compensation=config.apply_physics_compensation,
    )


def _dry_run_normalize_task(
    context: NormalizeTaskContext,
    input_store: DataProcessingStore,
    config: NormalizeStageConfig,
) -> tuple[str, ...]:
    requires_bounded = (
        config.max_batch_rows is not None and context.task.row_count > config.max_batch_rows
    )
    if requires_bounded:
        all_violations: list[tuple[str, str]] = []
        for _frame, frame_violations in _iter_bounded_frames(
            context,
            input_store,
            config.max_batch_rows or config.chunk_rows,
            annotate=False,
        ):
            all_violations.extend(frame_violations)
        return _check_violation_warnings(context.task, tuple(dict.fromkeys(all_violations)))
    _data, violations = _prepare_full_task_frame(context, input_store, annotate=False)
    return _check_violation_warnings(context.task, violations)


def _raise_for_check_violations(
    task: NormalizeTask,
    violations: tuple[tuple[str, str], ...],
) -> None:
    if not violations:
        return
    details = "; ".join(_format_check_violation(violation) for violation in violations)
    raise RuntimeError(f"Normalize task {task.task_id!r} failed checks; skipping task: {details}")


def _check_violation_warnings(
    task: NormalizeTask,
    violations: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    return tuple(
        f"dry_run check violation task_id={task.task_id!r} {_format_check_violation(violation)}"
        for violation in violations
    )


def _format_check_violation(violation: tuple[str, str]) -> str:
    column, reason = violation
    return f"column={column!r} reason={reason!r}"


def _finalize_output_columns(
    data: pl.DataFrame | pl.LazyFrame,
    protocol_spec: NormalizeProtocolSpec,
    *,
    include_annotations: bool,
) -> pl.DataFrame:
    frame = collect_frame(finalize_annotations(data, include_annotations=include_annotations))
    return collect_frame(
        select_and_cast_columns(
            frame, _output_columns(protocol_spec, include_annotations=include_annotations)
        )
    )


def _internal_output_columns(protocol_spec: NormalizeProtocolSpec) -> tuple[MappingSpec, ...]:
    return tuple(
        dict.fromkeys(
            (
                *protocol_spec.output_columns,
                BaseColumns.dt,
                BaseColumns.ann_cols,
                BaseColumns.ann_reasons,
            )
        )
    )


def _output_columns(
    protocol_spec: NormalizeProtocolSpec, *, include_annotations: bool
) -> tuple[MappingSpec, ...]:
    columns = [*protocol_spec.output_columns]
    if include_annotations:
        columns.append(BaseColumns.anns)
    return tuple(dict.fromkeys(columns))


def _task_metadata(
    dataset_id: str,
    normalize_spec: NormalizeStageSpec,
    task: NormalizeTask,
    config: NormalizeStageConfig,
) -> dict[MappingSpec, object]:
    protocol_spec = normalize_spec.protocol_spec(task.protocol_id)
    resampling = protocol_spec.resampling if config.apply_resampling else None
    method, params = resampling_metadata_values(resampling)
    values = normalize_spec.task_metadata(dataset_id, task)
    values[BaseColumns.resamp] = method
    values[BaseColumns.resamp_args] = params
    return values


def _normalize_selection(
    normalize_spec: NormalizeStageSpec,
    protocols: object,
    group_values: object,
) -> StageSelection:
    return StageSelection.from_values(
        protocols=protocols,
        group_values=group_values,
        group_columns=all_group_columns(
            tuple(spec.group_by for spec in normalize_spec.protocol_specs)
        ),
    )


def _interactive_protocol_order(
    normalize_spec: NormalizeStageSpec,
    protocols: object,
) -> tuple[str, ...]:
    protocol_values = normalize_selector_values(protocols)
    if protocol_values is not None:
        return tuple(str(protocol) for protocol in protocol_values)
    return tuple(str(spec.protocol_id) for spec in normalize_spec.protocol_specs)


def _dataset_normalize_spec(spec: DatasetSpec) -> NormalizeStageSpec:
    normalize_spec = spec.processing_stages.get(DatasetStageId.normalized)
    if not isinstance(normalize_spec, NormalizeStageSpec):
        raise TypeError(f"Dataset {spec.dataset_id!r} does not support normalize")
    return normalize_spec


def _validate_group_columns_declared(
    protocol_spec: NormalizeProtocolSpec,
    manifest_metadata: MetadataLayout,
) -> None:
    declared = set(manifest_metadata.columns)
    missing = [column for column in protocol_spec.group_by if column not in declared]
    if missing:
        raise ValueError(
            f"Normalize group_by columns for protocol {protocol_spec.protocol_id!r} are not "
            f"declared by the parquet manifest metadata: {missing}",
        )
