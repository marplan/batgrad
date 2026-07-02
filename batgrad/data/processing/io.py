from __future__ import annotations

from typing import TYPE_CHECKING, overload

import polars as pl

from batgrad.contracts.mapping import MappingSpec
from batgrad.storage.segments import scan_segment_frames

type ResolvedColumns = dict[MappingSpec, tuple[str, ...] | None]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from batgrad.contracts.segments import SegmentLike
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


def add_metadata_columns(
    data: pl.DataFrame | pl.LazyFrame,
    metadata: dict[MappingSpec, object],
) -> pl.DataFrame | pl.LazyFrame:
    """Add typed literal metadata columns to a dataframe or lazy frame."""
    return data.with_columns(
        pl.lit(value).cast(column.dtype).alias(column) for column, value in metadata.items()
    )


def select_and_cast_columns(
    data: pl.DataFrame | pl.LazyFrame,
    columns: tuple[MappingSpec, ...],
    *,
    resolved_columns: ResolvedColumns | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Select canonical mapping columns and cast them to declared dtypes."""
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
    """Build expressions that resolve mapping aliases, metadata, or nulls."""
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
    """Build one canonical output expression from available alias columns."""
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
    """Return available source columns matching a mapping spec's aliases."""
    available_set = set(available)
    return tuple(alias for alias in column.alias if alias in available_set)


def resolve_mapping_columns_for_segments(
    store: DataProcessingStore,
    segments: tuple[SegmentLike, ...],
    requested: tuple[MappingSpec, ...],
    required: tuple[MappingSpec, ...],
    *,
    context: str,
    one_of_col_groups: tuple[tuple[MappingSpec, ...], ...] = (),
) -> tuple[tuple[str, ...], ResolvedColumns]:
    """Resolve source columns needed to read requested mappings from segments.

    The result lists physical input columns to scan and a mapping from canonical
    specs to alias sources when the source differs from the canonical name.
    """
    available = set(_scan_segments_schema(store, segments))
    input_columns: list[str] = []
    resolved_columns: ResolvedColumns = {}
    for column in requested:
        sources = mapping_column_sources(column, available)
        if len(sources) != 1 or (sources and sources[0] != str(column)):
            resolved_columns[column] = sources or None
        input_columns.extend(sources)
    one_of_columns = {column for group in one_of_col_groups for column in group}
    missing = [
        column
        for column in required
        if column not in one_of_columns and not mapping_column_sources(column, available)
    ]
    if one_of_col_groups and not any(
        all(mapping_column_sources(column, available) for column in group)
        for group in one_of_col_groups
    ):
        expected = [tuple(str(column) for column in group) for group in one_of_col_groups]
        missing.append(MappingSpec(f"one of column groups {expected}", dtype=pl.String))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")
    return tuple(dict.fromkeys(input_columns)), resolved_columns


def _scan_segments_schema(
    store: DataProcessingStore,
    segments: tuple[SegmentLike, ...],
) -> tuple[str, ...]:
    if not segments:
        return ()
    return tuple(scan_segment_frames(store, segments).collect_schema().names())


def frame_columns(data: pl.DataFrame | pl.LazyFrame) -> tuple[str, ...]:
    if isinstance(data, pl.LazyFrame):
        return tuple(data.collect_schema().names())
    return tuple(data.columns)
