from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.processing.io import SegmentSource, collect_frame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from batgrad.contracts.mapping import MappingSpec
    from batgrad.data.transforms.resampling import ResamplingSpec


_VIEWPORT_ROW_ID = "__batgrad_viewport_row_id"
_ANNOTATION_MASK = "__batgrad_annotation_mask"
_ANNOTATION_BOUNDARY = "__batgrad_annotation_boundary"


@dataclass(frozen=True)
class TraceSource:
    trace_idx: int
    lf: pl.LazyFrame
    x_col: MappingSpec
    y_col: MappingSpec
    resampling: ResamplingSpec
    customdata_cols: tuple[MappingSpec, ...] = ()
    segment_source: SegmentSource | None = None
    extra_exprs: tuple[pl.Expr, ...] = ()
    row_count: int | None = None
    chunk_iter: Callable[
        [TraceSource, tuple[float, float] | None, tuple[float, float] | None, int, str],
        Iterator[pl.DataFrame],
    ] | None = None


@dataclass(frozen=True)
class TraceSample:
    trace_idx: int
    x: list[object]
    y: list[object]
    customdata: list[list[object]] | None
    row_count: int
    shown_points: int
    budget: int

    @property
    def downsampled(self) -> bool:
        return self.row_count > self.shown_points


@dataclass(frozen=True)
class AnnotationSource:
    trace_idx: int
    parent_trace_idx: int
    lf: pl.LazyFrame
    x_col: MappingSpec
    y_col: MappingSpec
    annotation_columns: tuple[str, ...]
    annotation_reason: str
    segment_source: SegmentSource | None = None
    extra_exprs: tuple[pl.Expr, ...] = ()


def sample_trace_viewport(
    source: TraceSource,
    *,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    budget: int,
    max_batch_rows: int | None,
) -> TraceSample:
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if max_batch_rows is not None and max_batch_rows < 1:
        raise ValueError(f"max_batch_rows must be >= 1 or None, got {max_batch_rows}")

    source_lf = source.lf.with_columns(source.extra_exprs) if source.extra_exprs else source.lf
    lf = _filter_viewport(source_lf, source.x_col, source.y_col, x_range, y_range)
    columns = _sample_columns(source)
    lf = lf.select(columns)
    row_count = (
        source.row_count
        if source.row_count is not None and x_range is None and y_range is None
        else int(collect_frame(lf.select(pl.len().alias("__n")))["__n"].item())
    )
    if row_count == 0:
        return TraceSample(
            trace_idx=source.trace_idx,
            x=[],
            y=[],
            customdata=[] if source.customdata_cols else None,
            row_count=0,
            shown_points=0,
            budget=budget,
        )

    resampling = _resampling_with_budget(source.resampling, budget)
    if max_batch_rows is None or row_count <= max_batch_rows:
        sampled = resampling.apply_full(collect_frame(lf), apply_physics_compensation=False)
    else:
        if source.segment_source is None:
            raise ValueError("Bounded viewport sampling requires store-backed trace segments")
        chunk_iter = source.chunk_iter or _iter_bounded_chunks
        sampled_chunks = list(
            resampling.apply_bounded(
                lambda: chunk_iter(
                    source,
                    x_range,
                    y_range,
                    max_batch_rows,
                    _VIEWPORT_ROW_ID,
                ),
                row_count=row_count,
                max_batch_rows=max_batch_rows,
                row_id_col=_VIEWPORT_ROW_ID,
                apply_physics_compensation=False,
            )
        )
        sampled = (
            pl.concat(sampled_chunks, how="vertical")
            if sampled_chunks
            else collect_frame(lf.limit(0))
        )

    return TraceSample(
        trace_idx=source.trace_idx,
        x=sampled[source.x_col].to_list(),
        y=sampled[source.y_col].to_list(),
        customdata=_customdata_rows(sampled, source.customdata_cols),
        row_count=row_count,
        shown_points=sampled.height,
        budget=budget,
    )


def sample_annotation_viewport(
    source: AnnotationSource,
    *,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    max_batch_rows: int | None,
) -> TraceSample:
    if max_batch_rows is not None and max_batch_rows < 1:
        raise ValueError(f"max_batch_rows must be >= 1 or None, got {max_batch_rows}")
    if not source.annotation_columns:
        return _empty_annotation_sample(source.trace_idx)

    source_lf = source.lf.with_columns(source.extra_exprs) if source.extra_exprs else source.lf
    lf = _filter_viewport(source_lf, source.x_col, source.y_col, x_range, y_range).select(
        _annotation_sample_columns(source)
    )
    mask = _annotation_mask_expr(source.annotation_columns, source.annotation_reason)
    sampled = collect_frame(
        lf.with_columns(mask.alias(_ANNOTATION_MASK))
        .with_columns(
            (
                pl.col(_ANNOTATION_MASK)
                & (
                    ~pl.col(_ANNOTATION_MASK).shift(1).fill_null(value=False)
                    | ~pl.col(_ANNOTATION_MASK).shift(-1).fill_null(value=False)
                )
            ).alias(_ANNOTATION_BOUNDARY)
        )
        .filter(pl.col(_ANNOTATION_BOUNDARY) & pl.col(source.y_col).is_not_null())
        .select(source.x_col, source.y_col)
    )
    return TraceSample(
        trace_idx=source.trace_idx,
        x=sampled[source.x_col].to_list(),
        y=sampled[source.y_col].to_list(),
        customdata=None,
        row_count=sampled.height,
        shown_points=sampled.height,
        budget=max(1, sampled.height),
    )


def _filter_viewport(
    lf: pl.LazyFrame,
    x_col: MappingSpec,
    y_col: MappingSpec,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
) -> pl.LazyFrame:
    expr = viewport_expr(x_col, y_col, x_range, y_range)
    return lf if expr is None else lf.filter(expr)


def _filter_viewport_frame(
    frame: pl.DataFrame,
    x_col: MappingSpec,
    y_col: MappingSpec,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
) -> pl.DataFrame:
    expr = viewport_expr(x_col, y_col, x_range, y_range)
    return frame if expr is None else frame.filter(expr)


def viewport_expr(
    x_col: MappingSpec,
    y_col: MappingSpec,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
) -> pl.Expr | None:
    expr = None
    if x_range is not None:
        x_low, x_high = sorted(x_range)
        expr = (pl.col(x_col) >= x_low) & (pl.col(x_col) <= x_high)
    if y_range is not None:
        y_low, y_high = sorted(y_range)
        y_expr = (pl.col(y_col) >= y_low) & (pl.col(y_col) <= y_high)
        expr = y_expr if expr is None else expr & y_expr
    return expr


def _iter_bounded_chunks(
    source: TraceSource,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    max_batch_rows: int,
    row_id_col: str,
) -> Iterator[pl.DataFrame]:
    if source.segment_source is None:
        return
    columns = _segment_input_columns(source)
    row_offset = 0
    chunks = source.segment_source.iter_chunks(max_batch_rows, columns=columns)
    for raw_chunk in chunks:
        chunk = raw_chunk
        if source.extra_exprs:
            chunk = chunk.with_columns(source.extra_exprs)
        chunk = _filter_viewport_frame(chunk, source.x_col, source.y_col, x_range, y_range)
        if chunk.height == 0:
            continue
        chunk = chunk.select(_sample_columns(source))
        chunk = chunk.with_columns(
            pl.Series(row_id_col, range(row_offset, row_offset + chunk.height)),
        )
        row_offset += chunk.height
        yield chunk


def _sample_columns(source: TraceSource) -> list[MappingSpec]:
    columns = [source.x_col, source.y_col]
    columns.extend(column for column in source.customdata_cols if column not in columns)
    return list(dict.fromkeys(columns))


def _annotation_sample_columns(source: AnnotationSource) -> list[MappingSpec]:
    return list(dict.fromkeys((source.x_col, source.y_col, BaseColumns.anns)))


def _customdata_rows(
    sampled: pl.DataFrame,
    customdata_cols: tuple[MappingSpec, ...],
) -> list[list[object]] | None:
    cols = [column for column in customdata_cols if column in sampled.columns]
    if not cols:
        return None
    return [list(row) for row in sampled.select(cols).rows()]


def _segment_input_columns(source: TraceSource) -> tuple[str, ...] | None:
    if source.extra_exprs:
        return None
    return tuple(dict.fromkeys(str(column) for column in _sample_columns(source)))


def _empty_annotation_sample(trace_idx: int) -> TraceSample:
    return TraceSample(
        trace_idx=trace_idx,
        x=[],
        y=[],
        customdata=None,
        row_count=0,
        shown_points=0,
        budget=1,
    )


def _annotation_mask_expr(columns: tuple[str, ...], reason: str) -> pl.Expr:
    return (
        pl.col(BaseColumns.anns)
        .list.eval(
            pl.element().struct.field("column").is_in(columns)
            & (pl.element().struct.field("reason") == reason)
        )
        .list.any()
        .fill_null(value=False)
    )


def _resampling_with_budget(spec: ResamplingSpec, budget: int) -> ResamplingSpec:
    if not is_dataclass(spec):
        return spec
    field_names = {field.name for field in fields(spec)}
    if "points" not in field_names:
        return spec
    updates: dict[str, object] = {"points": budget}
    if "points_ratio" in field_names:
        updates["points_ratio"] = None
    return cast("ResamplingSpec", replace(cast("Any", spec), **updates))
