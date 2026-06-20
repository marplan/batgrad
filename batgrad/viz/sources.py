from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetStageId, MappingSpec
from batgrad.data.processing.interactive import InteractiveProtocolSpec, run_load_interactive
from batgrad.data.processing.io import (
    SegmentSource,
    mapping_column_expr,
    mapping_column_exprs,
    mapping_column_sources,
    segment_values,
)
from batgrad.data.transforms.checks import (
    TimeCheckState,
    rebuild_time_axis_chunk,
    rebuild_time_axis_lazy,
)
from batgrad.viz.viewport import viewport_expr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from batgrad.data.processing.interactive import InteractiveStageRun
    from batgrad.viz.viewport import TraceSource


_MIN_OVERLAY_TIME_ROWS = 2


@dataclass(frozen=True)
class RunEntry:
    row: dict[str, object]
    source: SegmentSource
    protocol_spec: InteractiveProtocolSpec
    schema: dict[str, pl.DataType]


@dataclass(frozen=True)
class OverlayEntry:
    entry: RunEntry
    source: DatasetStageId


def iter_entries(run: InteractiveStageRun) -> list[RunEntry]:
    entries = []
    for row, source in run.iter_sources():
        protocol = row[str(BaseColumns.proto)]
        protocol_spec = run.protocol_spec(protocol)
        entries.append(
            RunEntry(
                row=row,
                source=source,
                protocol_spec=protocol_spec,
                schema=dict(source.scan().collect_schema()),
            )
        )
    return entries


def overlay_entries(
    run: InteractiveStageRun,
    overlay_sources: tuple[DatasetStageId, ...],
) -> list[OverlayEntry]:
    if not overlay_sources:
        return []
    if run.dataset_spec is None or run.input_store is None:
        raise ValueError("overlay_sources requires run.dataset_spec and run.input_store")
    entries: list[OverlayEntry] = []
    for source in overlay_sources:
        if source != DatasetStageId.ingested:
            raise NotImplementedError(f"Overlay source {source!r} is not supported yet")
        overlay_run = run_load_interactive(
            run.dataset_spec,
            run.input_store,
            source=source,
            protocols=run.protocol_order,
            group_values=run.group_values,
        )
        entries.extend(OverlayEntry(entry, source) for entry in coalesced_entries(overlay_run))
    return entries


def coalesced_entries(run: InteractiveStageRun) -> list[RunEntry]:
    groups: dict[tuple[str, tuple[tuple[str, object], ...]], list[dict[str, object]]] = {}
    specs: dict[str, InteractiveProtocolSpec] = {}
    for row in run.manifest().iter_rows(named=True):
        protocol = str(row[str(BaseColumns.proto)])
        protocol_spec = run.protocol_spec(protocol)
        specs[protocol] = protocol_spec
        key = (
            protocol,
            tuple(
                (str(column), row.get(str(column))) for column in protocol_group_by(protocol_spec)
            ),
        )
        groups.setdefault(key, []).append(row)

    entries = []
    for (protocol, _group_key), rows in groups.items():
        segments = tuple(
            segment for row in rows for segment in segment_values(row.get(str(run.segment_col)))
        )
        if not segments:
            continue
        merged = dict(rows[0])
        merged[str(run.segment_col)] = list(segments)
        merged[str(BaseColumns.row_n)] = sum(
            as_int_like(row[str(BaseColumns.row_n)]) for row in rows
        )
        source = SegmentSource.from_values(
            run.output_store,
            segments,
            row_count=as_int_like(merged[str(BaseColumns.row_n)]),
        )
        entries.append(
            RunEntry(
                row=merged,
                source=source,
                protocol_spec=specs[protocol],
                schema=dict(source.scan().collect_schema()),
            )
        )
    return entries


def as_int_like(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}")


def matching_overlay_entries(
    entries: list[RunEntry],
    overlays: list[OverlayEntry],
) -> list[OverlayEntry]:
    protocols = {str(entry.protocol_spec.protocol_id) for entry in entries}
    return [
        overlay for overlay in overlays if str(overlay.entry.protocol_spec.protocol_id) in protocols
    ]


def matching_overlays_for_entry(
    entry: RunEntry,
    overlays: list[OverlayEntry] | tuple[OverlayEntry, ...],
) -> list[OverlayEntry]:
    entry_key = entry_group_key(entry)
    return [
        overlay
        for overlay in overlays
        if str(overlay.entry.protocol_spec.protocol_id) == str(entry.protocol_spec.protocol_id)
        and entry_group_key(overlay.entry) == entry_key
    ]


def entry_group_key(entry: RunEntry) -> tuple[tuple[str, object], ...]:
    return tuple(
        (str(column), entry.row.get(str(column)))
        for column in protocol_group_by(entry.protocol_spec)
    )


def has_overlay_sources(
    overlay_entry: RunEntry,
    x_col: MappingSpec,
    source_y_col: MappingSpec,
) -> bool:
    schema = set(overlay_entry.schema)
    return bool(mapping_column_sources(x_col, schema)) and bool(
        mapping_column_sources(source_y_col, schema)
    )


def overlay_y_source(overlay_entry: RunEntry, source_y_col: MappingSpec) -> str | None:
    sources = mapping_column_sources(source_y_col, set(overlay_entry.schema))
    return sources[0] if sources else None


def overlay_lazy_frame(
    overlay_entry: RunEntry,
    protocol_spec: InteractiveProtocolSpec,
    x_col: MappingSpec,
    y_col: MappingSpec,
    source_y_col: MappingSpec,
) -> pl.LazyFrame:
    columns = _overlay_required_columns(x_col, source_y_col)
    data = overlay_entry.source.scan().select(
        mapping_column_expr(column, set(overlay_entry.schema)) for column in columns
    )
    data = _with_overlay_group_values_lazy(data, overlay_entry, protocol_spec)
    if x_col == BaseColumns.time:
        data = _rebuild_overlay_time_lazy(data, protocol_spec)
    if source_y_col != y_col:
        data = data.with_columns(pl.col(source_y_col).alias(y_col))
    return data.select(x_col, y_col)


def overlay_chunk_iter(
    overlay_entry: RunEntry,
    protocol_spec: InteractiveProtocolSpec,
    x_col: MappingSpec,
    y_col: MappingSpec,
    source_y_col: MappingSpec,
) -> Callable[
    [TraceSource, tuple[float, float] | None, tuple[float, float] | None, int, str],
    Iterator[pl.DataFrame],
]:
    def iter_chunks(
        source: TraceSource,
        x_range: tuple[float, float] | None,
        y_range: tuple[float, float] | None,
        max_batch_rows: int,
        row_id_col: str,
    ) -> Iterator[pl.DataFrame]:
        state = TimeCheckState()
        row_offset = 0
        for raw_chunk in overlay_entry.source.iter_chunks(max_batch_rows):
            chunk = _project_overlay_chunk(
                raw_chunk, overlay_entry, protocol_spec, x_col, y_col, source_y_col, state
            )
            expr = viewport_expr(x_col, y_col, x_range, y_range)
            if expr is not None:
                chunk = chunk.filter(expr)
            if chunk.height == 0:
                continue
            chunk = chunk.select(source.x_col, source.y_col)
            chunk = chunk.with_columns(
                pl.Series(row_id_col, range(row_offset, row_offset + chunk.height))
            )
            row_offset += chunk.height
            yield chunk

    return iter_chunks


def protocol_group_by(protocol_spec: InteractiveProtocolSpec) -> tuple[MappingSpec, ...]:
    group_by = getattr(protocol_spec, "group_by", None)
    if group_by is not None:
        return tuple(group_by)
    metadata = getattr(protocol_spec, "protocol_metadata", None) or protocol_spec.protocol.metadata
    return tuple(metadata.task_key)


def has_sources(entry: RunEntry, column: MappingSpec) -> bool:
    return bool(mapping_column_sources(column, set(entry.schema)))


def protocol_output_column(
    protocol_spec: InteractiveProtocolSpec,
    column: MappingSpec,
) -> MappingSpec:
    for output_column in protocol_spec.output_columns:
        if isinstance(output_column, MappingSpec) and str(output_column) == str(column):
            return output_column
    return column


def source_exprs(entry: RunEntry, columns: tuple[MappingSpec, ...]) -> tuple[pl.Expr, ...]:
    return mapping_column_exprs(columns, set(entry.schema), metadata=entry.row)


def entry_row_count(entry: RunEntry) -> int | None:
    value = entry.row.get(str(BaseColumns.row_n))
    return as_int_like(value) if value is not None else None


def _overlay_required_columns(
    x_col: MappingSpec,
    source_y_col: MappingSpec,
) -> tuple[MappingSpec, ...]:
    return tuple(dict.fromkeys((x_col, source_y_col)))


def _with_overlay_group_values_lazy(
    data: pl.LazyFrame,
    overlay_entry: RunEntry,
    protocol_spec: InteractiveProtocolSpec,
) -> pl.LazyFrame:
    exprs = _overlay_group_value_exprs(overlay_entry, protocol_spec)
    return data.with_columns(exprs) if exprs else data


def _with_overlay_group_values_frame(
    data: pl.DataFrame,
    overlay_entry: RunEntry,
    protocol_spec: InteractiveProtocolSpec,
) -> pl.DataFrame:
    exprs = _overlay_group_value_exprs(overlay_entry, protocol_spec)
    return data.with_columns(exprs) if exprs else data


def _overlay_group_value_exprs(
    overlay_entry: RunEntry,
    protocol_spec: InteractiveProtocolSpec,
) -> tuple[pl.Expr, ...]:
    return tuple(
        pl.lit(overlay_entry.row[str(column)]).cast(column.dtype).alias(column)
        for column in protocol_group_by(protocol_spec)
        if str(column) in overlay_entry.row
    )


def _rebuild_overlay_time_lazy(
    data: pl.LazyFrame,
    protocol_spec: InteractiveProtocolSpec,
) -> pl.LazyFrame:
    group_by = protocol_group_by(protocol_spec)
    data = data.with_columns(
        pl.col(BaseColumns.time)
        .cast(pl.Float64)
        .diff()
        .shift(-1)
        .over(list(group_by))
        .alias(BaseColumns.dt)
    )
    data = data.drop_nulls(subset=[BaseColumns.dt]).filter(pl.col(BaseColumns.dt) > 0.0)
    return rebuild_time_axis_lazy(data, BaseColumns.time, BaseColumns.dt, group_by)


def _project_overlay_chunk(
    raw_chunk: pl.DataFrame,
    overlay_entry: RunEntry,
    protocol_spec: InteractiveProtocolSpec,
    x_col: MappingSpec,
    y_col: MappingSpec,
    source_y_col: MappingSpec,
    state: TimeCheckState,
) -> pl.DataFrame:
    columns = _overlay_required_columns(x_col, source_y_col)
    available = set(raw_chunk.columns)
    chunk = raw_chunk.select(mapping_column_expr(column, available) for column in columns)
    chunk = _with_overlay_group_values_frame(chunk, overlay_entry, protocol_spec)
    if x_col == BaseColumns.time:
        chunk = _rebuild_overlay_time_chunk(chunk, state)
    if source_y_col != y_col:
        chunk = chunk.with_columns(pl.col(source_y_col).alias(y_col))
    return chunk.select(x_col, y_col)


def _rebuild_overlay_time_chunk(data: pl.DataFrame, state: TimeCheckState) -> pl.DataFrame:
    if state.pending_tail is not None:
        data = pl.concat((state.pending_tail, data), how="diagonal_relaxed")
    if data.height < _MIN_OVERLAY_TIME_ROWS:
        state.pending_tail = data
        return data.limit(0)
    with_dt = data.with_columns(
        pl.col(BaseColumns.time).cast(pl.Float64).diff().shift(-1).alias(BaseColumns.dt),
    )
    emit = with_dt.slice(0, with_dt.height - 1).filter(pl.col(BaseColumns.dt) > 0.0)
    state.pending_tail = data.slice(data.height - 1, 1)
    if emit.height == 0:
        return emit
    return rebuild_time_axis_chunk(emit, state, BaseColumns.time, BaseColumns.dt)
