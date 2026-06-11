from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from batgrad.contracts.columns import ColumnSpec


def collect_frame(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        collected = data.collect()
        if not isinstance(collected, pl.DataFrame):
            raise TypeError(
                f"Expected LazyFrame.collect() to return DataFrame, got {type(collected).__name__}",
            )
        return collected
    return data


def add_metadata_columns(
    data: pl.DataFrame | pl.LazyFrame,
    metadata: dict[ColumnSpec, object],
) -> pl.DataFrame | pl.LazyFrame:
    exprs: list[pl.Expr] = []
    for column, value in metadata.items():
        dtype = column.dtype
        expr = pl.lit(value)
        if dtype is not None:
            expr = expr.cast(dtype, strict=False)
        exprs.append(expr.alias(column))

    if not exprs:
        return data
    return data.with_columns(exprs)


def select_and_cast_columns(
    data: pl.DataFrame | pl.LazyFrame,
    output_columns: tuple[ColumnSpec, ...],
    extra_source_columns: tuple[str, ...] = (),
    resolved_columns: dict[ColumnSpec, str | None] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    available_columns = frame_columns(data)
    exprs: list[pl.Expr] = []
    for column in output_columns:
        if resolved_columns is not None and column in resolved_columns:
            source_column = resolved_columns.get(column)
        else:
            source_column = column.matching_name(available_columns)
        expr = pl.col(source_column) if source_column is not None else pl.lit(None)
        if column.dtype is not None:
            expr = expr.cast(column.dtype, strict=False)
        exprs.append(expr.alias(column))
    exprs.extend(pl.col(column) for column in extra_source_columns)
    return data.select(exprs)


def frame_columns(data: pl.DataFrame | pl.LazyFrame) -> tuple[str, ...]:
    if isinstance(data, pl.LazyFrame):
        return tuple(data.collect_schema().names())
    return tuple(data.columns)


def iter_data_chunks(data: pl.DataFrame, chunk_rows: int) -> Iterator[pl.DataFrame]:
    if data.height <= chunk_rows:
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


def iter_parquet_chunks(path: str | Path, chunk_rows: int) -> Iterator[pl.DataFrame]:
    parquet_file = pq.ParquetFile(path)
    for arrow_batch in parquet_file.iter_batches(batch_size=chunk_rows):
        frame = pl.from_arrow(arrow_batch)
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"Expected batch conversion to return DataFrame, got {type(frame).__name__}",
            )
        if frame.height > 0:
            yield frame


def validate_required_metadata(
    metadata: dict[ColumnSpec, object],
    required_columns: tuple[ColumnSpec, ...],
    *,
    context: str,
) -> None:
    missing = [column for column in required_columns if column not in metadata]
    if missing:
        raise ValueError(f"{context} metadata is missing required columns: {missing}")
