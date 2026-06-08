from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from batgrad.contracts.columns import ColumnSpec


def collect_frame(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        collected = data.collect()
        if not isinstance(collected, pl.DataFrame):
            raise TypeError(
                "Expected LazyFrame.collect() to return DataFrame, "
                f"got {type(collected).__name__}",
            )
        return collected
    return data


def add_metadata_columns(
    data: pl.DataFrame,
    metadata: dict[ColumnSpec, object],
) -> pl.DataFrame:
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
    data: pl.DataFrame,
    output_columns: tuple[ColumnSpec, ...],
    extra_source_columns: tuple[str, ...] = (),
) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for column in output_columns:
        expr = pl.col(column) if column in data.columns else pl.lit(None)
        if column.dtype is not None:
            expr = expr.cast(column.dtype, strict=False)
        exprs.append(expr.alias(column))
    exprs.extend(pl.col(column) for column in extra_source_columns)
    return data.select(exprs)


def validate_required_metadata(
    metadata: dict[ColumnSpec, object],
    required_columns: tuple[ColumnSpec, ...],
    *,
    context: str,
) -> None:
    missing = [column for column in required_columns if column not in metadata]
    if missing:
        raise ValueError(f"{context} metadata is missing required columns: {missing}")
