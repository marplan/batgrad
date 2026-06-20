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
        if self.manifest_frame is not None:
            return self.manifest_frame
        if self.manifest_path is None:
            raise ValueError("Interactive run has neither manifest_frame nor manifest_path")
        return collect_frame(self.output_store.scan_table(self.manifest_path))

    def protocol_spec(self, protocol: object) -> InteractiveProtocolSpec:
        return self.stage_spec.protocol_spec(protocol)

    def scan(self) -> pl.LazyFrame:
        segments = tuple(
            segment
            for row_segments in self.manifest()[str(self.segment_col)].to_list()
            for segment in segment_values(row_segments)
        )
        if not segments:
            raise FileNotFoundError("Interactive run has no output segments")
        return scan_segment_frames(self.output_store, segments)

    def clean(self) -> None:
        if self.run_root is None:
            return
        validate_scratch_run_root(self.run_root)
        self.output_store.delete_dir(self.run_root, missing_ok=True)

    def iter_sources(self) -> Iterator[tuple[dict[str, object], SegmentSource]]:
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


def _select_manifest_rows(
    manifest: pl.DataFrame,
    stage_spec: InteractiveStageSpec,
    protocols: object,
    group_values: object,
) -> pl.DataFrame:
    selection = StageSelection.from_values(
        protocols=protocols,
        group_values=group_values,
        group_columns=_stage_group_columns(stage_spec),
    )
    rows = [row for row in manifest.iter_rows(named=True) if selection.matches_row(row)]
    if not rows:
        raise ValueError(
            f"No interactive row matched protocols={protocols!r} group_values={group_values!r}",
        )
    return pl.DataFrame(rows, schema=manifest.schema)


def _stage_group_columns(stage_spec: InteractiveStageSpec) -> tuple[MappingSpec, ...]:
    protocol_specs = stage_spec.protocol_specs
    columns = []
    for spec in protocol_specs:
        group_by = getattr(spec, "group_by", None)
        if group_by is None:
            group_by = spec.protocol.metadata.task_key
        columns.append(tuple(group_by))
    return all_group_columns(columns)


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
