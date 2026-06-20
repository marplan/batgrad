from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, overload

import polars as pl

from batgrad.contracts.mapping import BaseColumns, MappingSpec
from batgrad.data.processing.metadata import as_int

type ResolvedColumns = dict[MappingSpec, tuple[str, ...] | None]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from batgrad.storage.store import DataProcessingStore


@overload
def add_metadata_columns(
    data: pl.DataFrame,
    metadata: dict[MappingSpec, object],
) -> pl.DataFrame: ...


@overload
def add_metadata_columns(
    data: pl.LazyFrame,
    metadata: dict[MappingSpec, object],
) -> pl.LazyFrame: ...


@dataclass(frozen=True, slots=True)
class ParquetSegment:
    path: str
    row_start: int
    row_count: int

    @classmethod
    def from_value(cls, value: ParquetSegment | Mapping[Any, Any]) -> ParquetSegment:
        if isinstance(value, ParquetSegment):
            return value
        return cls(
            path=str(value[str(BaseColumns.path)]),
            row_start=as_int(value[str(BaseColumns.row0)]),
            row_count=as_int(value[str(BaseColumns.row_n)]),
        )

    def as_manifest_dict(self) -> dict[str, object]:
        return {
            str(BaseColumns.path): self.path,
            str(BaseColumns.row0): self.row_start,
            str(BaseColumns.row_n): self.row_count,
        }


type SegmentLike = ParquetSegment | Mapping[Any, Any]


@dataclass(frozen=True, slots=True)
class SegmentSource:
    store: DataProcessingStore
    segments: tuple[ParquetSegment, ...]
    row_count: int | None = None

    @classmethod
    def from_values(
        cls,
        store: DataProcessingStore,
        segments: tuple[SegmentLike, ...],
        *,
        row_count: int | None = None,
    ) -> SegmentSource:
        normalized = normalize_segments(segments)
        return cls(
            store=store,
            segments=normalized,
            row_count=segment_row_count(normalized) if row_count is None else row_count,
        )

    def scan(self) -> pl.LazyFrame:
        return scan_segment_frames(self.store, self.segments)

    def iter_chunks(
        self,
        chunk_rows: int,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[pl.DataFrame]:
        yield from iter_segment_frames(self.store, self.segments, chunk_rows, columns=columns)


def normalize_segments(segments: tuple[SegmentLike, ...]) -> tuple[ParquetSegment, ...]:
    return tuple(ParquetSegment.from_value(segment) for segment in segments)


def segment_values(value: object) -> tuple[SegmentLike, ...]:
    if not isinstance(value, list | tuple):
        return ()
    segments: list[SegmentLike] = []
    for segment in value:
        if isinstance(segment, ParquetSegment):
            segments.append(segment)
            continue
        if isinstance(segment, Mapping):
            segments.append(ParquetSegment.from_value(segment))
            continue
        raise TypeError(f"Expected parquet segment mapping, got {type(segment).__name__}")
    return tuple(segments)


def segment_row_count(segments: tuple[ParquetSegment, ...]) -> int:
    return sum(segment.row_count for segment in segments)


def segment_manifest_dicts(segments: tuple[ParquetSegment, ...]) -> tuple[dict[str, object], ...]:
    return tuple(segment.as_manifest_dict() for segment in segments)


def collect_frame(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        return cast("pl.DataFrame", data.collect())
    return data


def add_metadata_columns(
    data: pl.DataFrame | pl.LazyFrame,
    metadata: dict[MappingSpec, object],
) -> pl.DataFrame | pl.LazyFrame:
    return data.with_columns(
        pl.lit(value).cast(column.dtype).alias(column) for column, value in metadata.items()
    )


def select_and_cast_columns(
    data: pl.DataFrame | pl.LazyFrame,
    columns: tuple[MappingSpec, ...],
    *,
    resolved_columns: ResolvedColumns | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    available = set(frame_columns(data))
    return data.select(
        mapping_column_expr(column, available, resolved_columns=resolved_columns)
        for column in columns
    )


def mapping_column_exprs(
    columns: tuple[MappingSpec, ...],
    available: set[str] | Sequence[str],
    *,
    resolved_columns: ResolvedColumns | None = None,
    metadata: Mapping[str, object] | None = None,
    skip_existing: bool = True,
    null_for_missing: bool = False,
) -> tuple[pl.Expr, ...]:
    available_set = set(available)
    produced = set(available_set)
    exprs: list[pl.Expr] = []
    for column in dict.fromkeys(columns):
        if skip_existing and str(column) in produced:
            continue
        sources = (
            resolved_columns.get(column)
            if resolved_columns is not None and column in resolved_columns
            else mapping_column_sources(column, available_set)
        )
        if sources:
            exprs.append(
                mapping_column_expr(column, available_set, resolved_columns=resolved_columns)
            )
            produced.add(str(column))
            continue
        if metadata is not None and str(column) in metadata:
            exprs.append(pl.lit(metadata[str(column)]).cast(column.dtype).alias(column))
            produced.add(str(column))
            continue
        if null_for_missing:
            exprs.append(pl.lit(None).cast(column.dtype).alias(column))
            produced.add(str(column))
    return tuple(exprs)


def mapping_column_expr(
    column: MappingSpec,
    available: set[str] | Sequence[str],
    *,
    resolved_columns: ResolvedColumns | None = None,
) -> pl.Expr:
    available_set = set(available)
    sources = (
        resolved_columns.get(column)
        if resolved_columns is not None and column in resolved_columns
        else mapping_column_sources(column, available_set)
    )
    if not sources:
        return pl.lit(None).cast(column.dtype).alias(column)
    expr = (
        pl.col(sources[0])
        if len(sources) == 1
        else pl.coalesce(pl.col(source) for source in sources)
    )
    return expr.cast(column.dtype).alias(column)


def mapping_column_sources(
    column: MappingSpec, available: set[str] | Sequence[str]
) -> tuple[str, ...]:
    available_set = set(available)
    return tuple(alias for alias in column.alias if alias in available_set)


def resolve_mapping_columns_for_segments(
    store: DataProcessingStore,
    segments: tuple[SegmentLike, ...],
    requested: tuple[MappingSpec, ...],
    required: tuple[MappingSpec, ...],
    *,
    context: str,
    alias_count_chunk_rows: int = 500_000,
) -> tuple[tuple[str, ...], ResolvedColumns]:
    available = set(_scan_segments_schema(store, segments))
    del alias_count_chunk_rows
    input_columns: list[str] = []
    resolved_columns: ResolvedColumns = {}
    for column in requested:
        sources = mapping_column_sources(column, available)
        if len(sources) != 1 or (sources and sources[0] != str(column)):
            resolved_columns[column] = sources or None
        input_columns.extend(sources)
    missing = [column for column in required if not mapping_column_sources(column, available)]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")
    return tuple(dict.fromkeys(input_columns)), resolved_columns


def iter_segment_frames(
    store: DataProcessingStore,
    segments: tuple[SegmentLike, ...],
    chunk_rows: int,
    columns: tuple[str, ...] | None = None,
) -> Iterator[pl.DataFrame]:
    for segment in normalize_segments(segments):
        if segment.row_count <= 0:
            continue
        yield from store.iter_table_slices(
            segment.path,
            ((segment.row_start, segment.row_count),),
            chunk_rows,
            columns=columns,
        )


def scan_segment_frames(
    store: DataProcessingStore,
    segments: tuple[SegmentLike, ...],
) -> pl.LazyFrame:
    frames = [
        store.scan_table(segment.path).slice(
            segment.row_start,
            segment.row_count,
        )
        for segment in normalize_segments(segments)
    ]
    if not frames:
        raise ValueError("No table segments to scan")
    return frames[0] if len(frames) == 1 else pl.concat(frames)


def _scan_segments_schema(
    store: DataProcessingStore,
    segments: tuple[SegmentLike, ...],
) -> tuple[str, ...]:
    if not segments:
        return ()
    return tuple(scan_segment_frames(store, segments).collect_schema().names())


def consume_temp_table(
    store: DataProcessingStore,
    location: str | Path,
    *,
    chunk_rows: int,
    on_chunk: Callable[[pl.DataFrame], None],
    delete: bool = True,
) -> None:
    try:
        for chunk in store.iter_table_chunks(location, chunk_rows):
            on_chunk(chunk)
    finally:
        if delete:
            store.delete_file(location, missing_ok=True)


def iter_data_chunks(data: pl.DataFrame, chunk_rows: int) -> Iterator[pl.DataFrame]:
    if data.height <= chunk_rows:
        if data.height > 0:
            yield data
        return
    for offset in range(0, data.height, chunk_rows):
        chunk = data.slice(offset, chunk_rows)
        if chunk.height > 0:
            yield chunk


def coalesce_frames(
    frames: Iterator[pl.DataFrame],
    chunk_rows: int,
) -> Iterator[pl.DataFrame]:
    pending: list[pl.DataFrame] = []
    pending_rows = 0
    for frame in frames:
        if frame.height == 0:
            continue
        pending.append(frame)
        pending_rows += frame.height
        if pending_rows < chunk_rows:
            continue
        combined = pl.concat(pending, how="vertical")
        while combined.height >= chunk_rows:
            yield combined.slice(0, chunk_rows)
            combined = combined.slice(chunk_rows)
        pending = [combined] if combined.height > 0 else []
        pending_rows = combined.height
    if pending:
        yield from iter_data_chunks(pl.concat(pending, how="vertical"), chunk_rows)


def frame_columns(data: pl.DataFrame | pl.LazyFrame) -> tuple[str, ...]:
    if isinstance(data, pl.LazyFrame):
        return tuple(data.collect_schema().names())
    return tuple(data.columns)
