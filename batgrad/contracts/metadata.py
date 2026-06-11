from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batgrad.contracts.columns import BaseColumns, ColumnSpec, MetadataColumns
from batgrad.contracts.domains import Domains
from batgrad.contracts.values import BaseValues

if TYPE_CHECKING:
    import polars as pl

    from batgrad.contracts.domains import DomainSpec
    from batgrad.contracts.values import ValueSpec


@dataclass(frozen=True, slots=True)
class ProtocolMetadataLayout:
    protocol: ValueSpec
    domain: DomainSpec
    metadata: tuple[ColumnSpec, ...]


@dataclass(frozen=True, slots=True)
class MetadataLayoutSpec:
    required: tuple[ColumnSpec, ...]
    optional: tuple[ColumnSpec, ...] = ()

    @property
    def columns(self) -> tuple[ColumnSpec, ...]:
        return (*self.required, *self.optional)


@dataclass(frozen=True, slots=True)
class MetadataLayout:
    cycling: tuple[ColumnSpec, ...] = (
        MetadataColumns.protocol,
        MetadataColumns.domain_id,
        BaseColumns.cell_id,
        BaseColumns.cycle_index,
    )

    hppc: tuple[ColumnSpec, ...] = (
        MetadataColumns.protocol,
        MetadataColumns.domain_id,
        BaseColumns.cell_id,
        BaseColumns.cycle_index,
    )

    rpt: tuple[ColumnSpec, ...] = (
        MetadataColumns.protocol,
        MetadataColumns.domain_id,
        BaseColumns.cell_id,
        BaseColumns.cycle_index,
    )

    eis: tuple[ColumnSpec, ...] = (
        MetadataColumns.protocol,
        MetadataColumns.domain_id,
        BaseColumns.cell_id,
        BaseColumns.cycle_index,
        MetadataColumns.soc_pct,
    )

    protocol_layouts: tuple[ProtocolMetadataLayout, ...] = (
        ProtocolMetadataLayout(BaseValues.cycling_protocol, Domains.time, cycling),
        ProtocolMetadataLayout(BaseValues.hppc_protocol, Domains.time, hppc),
        ProtocolMetadataLayout(BaseValues.rpt_protocol, Domains.time, rpt),
        ProtocolMetadataLayout(BaseValues.eis_protocol, Domains.freq, eis),
    )

    parquet_manifest: tuple[ColumnSpec, ...] = (
        MetadataColumns.raw_file_paths,
        MetadataColumns.parquet_segments,
        BaseColumns.row_count,
        MetadataColumns.protocol,
        MetadataColumns.domain_id,
        BaseColumns.cell_id,
        BaseColumns.cycle_index,
        MetadataColumns.soc_pct,
    )

    normalized_manifest: tuple[ColumnSpec, ...] = (
        BaseColumns.dataset_id,
        MetadataColumns.raw_file_paths,
        MetadataColumns.parquet_segments,
        MetadataColumns.normalized_segments,
        BaseColumns.row_count,
        MetadataColumns.domain_id,
        MetadataColumns.protocol,
        BaseColumns.axis_kind,
        BaseColumns.axis_col,
        MetadataColumns.resampling_method,
        MetadataColumns.resampling_params,
        MetadataColumns.time_convention,
    )

    parquet_footer: tuple[ColumnSpec, ...] = (
        BaseColumns.dataset_id,
        MetadataColumns.processing_stage,
        MetadataColumns.git_commit,
        MetadataColumns.git_dirty,
        MetadataColumns.manifest_path,
        MetadataColumns.protocols,
        MetadataColumns.domains,
        BaseColumns.row_count,
    )

    normalized_footer: tuple[ColumnSpec, ...] = (
        BaseColumns.dataset_id,
        MetadataColumns.processing_stage,
        MetadataColumns.git_commit,
        MetadataColumns.git_dirty,
        MetadataColumns.manifest_path,
        MetadataColumns.protocols,
        MetadataColumns.domains,
        BaseColumns.row_count,
        MetadataColumns.resampling_method,
        MetadataColumns.resampling_params,
        MetadataColumns.time_convention,
    )


def schema_from_layout(
    layout: tuple[ColumnSpec, ...] | MetadataLayoutSpec,
) -> dict[str, pl.DataType]:
    columns = layout.columns if isinstance(layout, MetadataLayoutSpec) else layout
    schema: dict[str, pl.DataType] = {}
    for column in columns:
        if column.dtype is None:
            raise ValueError(f"Column {column!r} has no dtype")
        schema[column] = column.dtype
    return schema


def validate_layout_values(
    values: dict[ColumnSpec, object],
    layout: tuple[ColumnSpec, ...] | MetadataLayoutSpec,
    *,
    context: str,
) -> None:
    required = layout.required if isinstance(layout, MetadataLayoutSpec) else layout
    missing = [column for column in required if column not in values]
    if missing:
        raise ValueError(f"{context} is missing required metadata columns: {missing}")


def validate_no_extra_layout_values(
    values: dict[ColumnSpec, object],
    layout: tuple[ColumnSpec, ...] | MetadataLayoutSpec,
    *,
    context: str,
) -> None:
    columns = layout.columns if isinstance(layout, MetadataLayoutSpec) else layout
    expected = set(columns)
    extra = [column for column in values if column not in expected]
    if extra:
        raise ValueError(f"{context} has metadata columns outside the layout: {extra}")
