from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetStageId, MappingSpec
from batgrad.data.processing.io import (
    SegmentSource,
    collect_frame,
    scan_segment_frames,
    segment_values,
)
from batgrad.data.processing.selection import (
    StageSelection,
    all_group_columns,
    normalize_selector_values,
)
from batgrad.data.processing.stage import validate_scratch_run_root

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.contracts.protocols import BatteryProtocolSpec
    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.storage.store import DataProcessingStore


class InteractiveProtocolSpec(Protocol):
    protocol: BatteryProtocolSpec

    @property
    def protocol_id(self) -> object: ...

    @property
    def output_columns(self) -> tuple[MappingSpec, ...]: ...


class InteractiveStageSpec(Protocol):
    @property
    def protocol_specs(self) -> tuple[InteractiveProtocolSpec, ...]: ...

    def protocol_spec(self, protocol: object) -> InteractiveProtocolSpec: ...


@dataclass(frozen=True)
class InteractiveStageRun:
    """Handle for selected existing or scratch-generated stage outputs.

    Interactive runs are returned by `Dataset.normalize_interactive` and
    `Dataset.load_interactive`. They expose a selected manifest, lazy scanning of
    referenced data segments, and cleanup for scratch-backed runs.

    Attributes:
        output_store: Store containing the selected output segments.
        stage_spec: Stage configuration used to interpret protocols and outputs.
        segment_col: Manifest column containing parquet segment references.
        protocol_order: Preferred protocol ordering for UI or plotting callers.
        dataset_spec: Dataset configuration, when available.
        input_store: Source store used to create the run, when available.
        protocols: Protocol selector used to create the run.
        group_values: Group selector used to create the run.
        source_stage: Stage represented by the run.
        manifest_path: Manifest path for scratch-backed runs.
        manifest_frame: Already-loaded manifest for selected existing data.
        run_root: Scratch run root cleaned by `clean`.
    """

    output_store: DataProcessingStore
    stage_spec: InteractiveStageSpec
    segment_col: MappingSpec
    protocol_order: tuple[str, ...] = ()
    dataset_spec: DatasetSpec | None = None
    input_store: DataProcessingStore | None = None
    protocols: object = None
    group_values: object = None
    source_stage: DatasetStageId | None = None
    manifest_path: str | None = None
    manifest_frame: pl.DataFrame | None = None
    run_root: str | None = None

    def manifest(self) -> pl.DataFrame:
        """Load or return the manifest rows represented by this run.

        Returns:
            Manifest dataframe for the selected rows.
        """
        if self.manifest_frame is not None:
            return self.manifest_frame
        if self.manifest_path is None:
            raise ValueError("Interactive run has neither manifest_frame nor manifest_path")
        return collect_frame(self.output_store.scan_table(self.manifest_path))

    def protocol_spec(self, protocol: object) -> InteractiveProtocolSpec:
        """Return the stage protocol spec for a protocol id or value.

        Args:
            protocol: Protocol id or enum-like value.

        Returns:
            Matching protocol spec from the stage configuration.
        """
        return self.stage_spec.protocol_spec(protocol)

    def scan(self) -> pl.LazyFrame:
        """Scan all data segments referenced by the run manifest.

        Returns:
            Lazy frame over the selected output segments.

        Raises:
            FileNotFoundError: If the selected manifest contains no segments.
        """
        segments = tuple(
            segment
            for row_segments in self.manifest()[str(self.segment_col)].to_list()
            for segment in segment_values(row_segments)
        )
        if not segments:
            raise FileNotFoundError("Interactive run has no output segments")
        return scan_segment_frames(self.output_store, segments)

    def clean(self) -> None:
        """Delete this run's scratch root when it owns one."""
        if self.run_root is None:
            return
        validate_scratch_run_root(self.run_root)
        self.output_store.delete_dir(self.run_root, missing_ok=True)

    def iter_sources(self) -> Iterator[tuple[dict[str, object], SegmentSource]]:
        """Iterate manifest rows with segment readers for each row.

        Returns:
            Iterator of manifest row dictionaries and segment sources.
        """
        for row in self.manifest().iter_rows(named=True):
            segments = segment_values(row.get(str(self.segment_col)))
            if segments:
                row_count = row.get(str(BaseColumns.row_n))
                yield (
                    row,
                    SegmentSource.from_values(
                        self.output_store,
                        segments,
                        row_count=int(row_count) if row_count is not None else None,
                    ),
                )


def run_load_interactive(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    *,
    source: DatasetStageId | str,
    protocols: object = None,
    group_values: object = None,
) -> InteractiveStageRun:
    """Load selected rows from an existing stage manifest.

    Args:
        dataset_spec: Dataset configuration.
        input_store: Store containing the stage manifest and shards.
        source: Stage id to load.
        protocols: Optional protocol selector.
        group_values: Optional task group selector or list of selectors.

    Returns:
        Interactive run over selected existing stage rows.
    """
    source = DatasetStageId(str(source))
    raw_stage_spec = dataset_spec.processing_stages.get(source)
    if raw_stage_spec is None:
        raise TypeError(f"Dataset {dataset_spec.dataset_id!r} does not support stage {source!r}")
    stage_spec = cast("InteractiveStageSpec", raw_stage_spec)
    manifest_path = dataset_spec.manifest(source)
    manifest = collect_frame(input_store.scan_table(manifest_path))
    rows = _select_manifest_rows(manifest, stage_spec, protocols, group_values)
    return InteractiveStageRun(
        output_store=input_store,
        stage_spec=stage_spec,
        segment_col=_stage_segment_col(source),
        protocol_order=_stage_protocol_order(stage_spec, protocols),
        dataset_spec=dataset_spec,
        input_store=input_store,
        protocols=protocols,
        group_values=group_values,
        source_stage=source,
        manifest_frame=rows,
    )


def run_load_interactive_manifest(
    dataset_spec: DatasetSpec,
    input_store: DataProcessingStore,
    *,
    source: DatasetStageId | str,
    manifest: pl.DataFrame,
    protocols: object = None,
    group_values: object = None,
) -> InteractiveStageRun:
    """Wrap an already-loaded manifest as an interactive run.

    Args:
        dataset_spec: Dataset configuration.
        input_store: Store containing shards referenced by `manifest`.
        source: Stage id the manifest belongs to.
        manifest: Manifest rows to expose.
        protocols: Optional protocol selector retained on the run.
        group_values: Optional group selector retained on the run.

    Returns:
        Interactive run over the provided manifest rows.
    """
    source = DatasetStageId(str(source))
    raw_stage_spec = dataset_spec.processing_stages.get(source)
    if raw_stage_spec is None:
        raise TypeError(f"Dataset {dataset_spec.dataset_id!r} does not support stage {source!r}")
    stage_spec = cast("InteractiveStageSpec", raw_stage_spec)
    return InteractiveStageRun(
        output_store=input_store,
        stage_spec=stage_spec,
        segment_col=_stage_segment_col(source),
        protocol_order=_stage_protocol_order(stage_spec, protocols),
        dataset_spec=dataset_spec,
        input_store=input_store,
        protocols=protocols,
        group_values=group_values,
        source_stage=source,
        manifest_frame=manifest,
    )


def selected_manifest_inputs(
    manifest: pl.DataFrame,
    stage_spec: InteractiveStageSpec,
    protocol_order: tuple[object, ...],
) -> tuple[list[dict[MappingSpec, object]], tuple[object, ...]]:
    """Return group selectors and protocol ids represented by manifest rows."""
    selected_rows = list(manifest.iter_rows(named=True))
    specs_by_protocol = {str(spec.protocol_id): spec for spec in stage_spec.protocol_specs}
    fallback_group_columns = stage_group_columns(stage_spec)
    group_values = []
    selected_protocol_set = set()
    for row in selected_rows:
        protocol = row.get(str(BaseColumns.proto))
        if protocol is not None:
            selected_protocol_set.add(str(protocol))
        protocol_spec = specs_by_protocol.get(str(protocol))
        group_columns = (
            _protocol_group_columns(protocol_spec)
            if protocol_spec is not None
            else fallback_group_columns
        )
        group_values.append(
            {
                column: row[str(column)]
                for column in group_columns
                if str(column) in row and row[str(column)] is not None
            }
        )
    selected_protocols = tuple(
        protocol for protocol in protocol_order if protocol in selected_protocol_set
    )
    return group_values, selected_protocols


def _select_manifest_rows(
    manifest: pl.DataFrame,
    stage_spec: InteractiveStageSpec,
    protocols: object,
    group_values: object,
) -> pl.DataFrame:
    selection = StageSelection.from_values(
        protocols=protocols,
        group_values=group_values,
        group_columns=stage_group_columns(stage_spec),
    )
    rows = [row for row in manifest.iter_rows(named=True) if selection.matches_row(row)]
    if not rows:
        raise ValueError(
            f"No interactive row matched protocols={protocols!r} group_values={group_values!r}",
        )
    return pl.DataFrame(rows, schema=manifest.schema)


def stage_group_columns(stage_spec: InteractiveStageSpec) -> tuple[MappingSpec, ...]:
    return all_group_columns([_protocol_group_columns(spec) for spec in stage_spec.protocol_specs])


def _protocol_group_columns(spec: InteractiveProtocolSpec) -> tuple[MappingSpec, ...]:
    group_by = getattr(spec, "group_by", None)
    if group_by is None:
        group_by = spec.protocol.metadata.task_key
    return tuple(group_by)


def _stage_protocol_order(stage_spec: InteractiveStageSpec, protocols: object) -> tuple[str, ...]:
    requested = normalize_selector_values(protocols)
    if requested is not None:
        return tuple(str(protocol) for protocol in requested)
    return tuple(str(spec.protocol_id) for spec in stage_spec.protocol_specs)


def _stage_segment_col(source: DatasetStageId) -> MappingSpec:
    if source == DatasetStageId.ingested:
        return BaseColumns.parq_segs
    if source == DatasetStageId.normalized:
        return BaseColumns.norm_segs
    raise ValueError(f"Stage {source!r} does not have interactive parquet segments")
