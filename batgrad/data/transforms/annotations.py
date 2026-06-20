from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from batgrad.contracts.mapping import BaseColumns

type AnnotationUpdate = tuple[pl.Expr, str, str]

ANNOTATION_ITEM_SEPARATOR = "\x1f"
MAX_FLAT_ANNOTATIONS = 16


def ensure_annotation_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(_annotation_base_exprs(data.columns))


def ensure_annotation_columns_lazy(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.with_columns(_annotation_base_exprs(data.collect_schema().names()))


def add_annotations(data: pl.DataFrame, updates: list[AnnotationUpdate]) -> pl.DataFrame:
    if not updates:
        return data
    any_violation = data.select(
        pl.any_horizontal(condition.fill_null(value=False) for condition, _, _ in updates)
        .any()
        .alias("__has_annotation"),
    ).item()
    return data.with_columns(_flat_annotation_exprs(updates)) if any_violation else data


def add_annotations_lazy(
    data: pl.LazyFrame,
    updates: list[AnnotationUpdate],
) -> pl.LazyFrame:
    return data.with_columns(_flat_annotation_exprs(updates)) if updates else data


def flat_annotations_to_structs(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    max_annotations: int = MAX_FLAT_ANNOTATIONS,
) -> pl.DataFrame | pl.LazyFrame:
    columns = data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.columns
    if BaseColumns.ann_cols not in columns or BaseColumns.ann_reasons not in columns:
        return data
    if isinstance(data, pl.DataFrame) and _flat_annotations_all_null(data):
        return data.with_columns(
            pl.lit(None, dtype=BaseColumns.anns.dtype).alias(BaseColumns.anns),
        ).drop(BaseColumns.ann_cols, BaseColumns.ann_reasons)
    if isinstance(data, pl.DataFrame) and _flat_annotations_single_item(data):
        return _flat_single_annotation_to_struct(data)
    annotation_columns = pl.col(BaseColumns.ann_cols).str.split(ANNOTATION_ITEM_SEPARATOR)
    annotation_reasons = pl.col(BaseColumns.ann_reasons).str.split(ANNOTATION_ITEM_SEPARATOR)
    structs = [
        pl.when(
            annotation_columns.list.get(idx, null_on_oob=True).is_not_null()
            & annotation_reasons.list.get(idx, null_on_oob=True).is_not_null(),
        )
        .then(
            pl.struct(
                annotation_columns.list.get(idx, null_on_oob=True).alias("column"),
                annotation_reasons.list.get(idx, null_on_oob=True).alias("reason"),
            ),
        )
        .otherwise(None)
        for idx in range(max_annotations)
    ]
    return data.with_columns(
        pl.when(pl.col(BaseColumns.ann_cols).is_not_null())
        .then(pl.concat_list(structs).list.drop_nulls())
        .otherwise(None)
        .cast(BaseColumns.anns.dtype)
        .alias(BaseColumns.anns),
    ).drop(BaseColumns.ann_cols, BaseColumns.ann_reasons)


def finalize_annotations(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    include_annotations: bool,
) -> pl.DataFrame | pl.LazyFrame:
    return flat_annotations_to_structs(data) if include_annotations else data


def _flat_annotations_all_null(data: pl.DataFrame) -> bool:
    if data.height == 0:
        return True
    return not bool(
        data.select(
            (
                pl.col(BaseColumns.ann_cols).is_not_null()
                | pl.col(BaseColumns.ann_reasons).is_not_null()
            )
            .any()
            .alias("__has_annotations"),
        )["__has_annotations"].item(),
    )


def _flat_annotations_single_item(data: pl.DataFrame) -> bool:
    if data.height == 0:
        return True
    return not bool(
        data.select(
            (
                pl.col(BaseColumns.ann_cols)
                .str.contains(ANNOTATION_ITEM_SEPARATOR, literal=True)
                .fill_null(value=False)
                | pl.col(BaseColumns.ann_reasons)
                .str.contains(ANNOTATION_ITEM_SEPARATOR, literal=True)
                .fill_null(value=False)
            )
            .any()
            .alias("__has_multi_annotations"),
        )["__has_multi_annotations"].item(),
    )


def _flat_single_annotation_to_struct(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(_single_annotation_series(data)).drop(
        BaseColumns.ann_cols, BaseColumns.ann_reasons
    )


def _single_annotation_series(data: pl.DataFrame) -> pl.Series:
    columns = data[str(BaseColumns.ann_cols)].to_arrow()
    reasons = data[str(BaseColumns.ann_reasons)].to_arrow()
    valid = pc.and_(pc.is_valid(columns), pc.is_valid(reasons))
    valid_values = np.asarray(valid).astype(np.int32, copy=False)
    offsets = np.empty(data.height + 1, dtype=np.int32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(valid_values)
    child = pa.StructArray.from_arrays(
        [pc.filter(columns, valid), pc.filter(reasons, valid)],
        names=["column", "reason"],
    )
    annotations = pa.ListArray.from_arrays(pa.array(offsets), child, mask=pc.invert(valid))
    return pl.Series(str(BaseColumns.anns), annotations)


def _annotation_base_exprs(columns: list[str]) -> list[pl.Expr]:
    exprs = []
    if BaseColumns.ann_cols not in columns:
        exprs.append(pl.lit(None, dtype=BaseColumns.ann_cols.dtype).alias(BaseColumns.ann_cols))
    if BaseColumns.ann_reasons not in columns:
        exprs.append(
            pl.lit(None, dtype=BaseColumns.ann_reasons.dtype).alias(BaseColumns.ann_reasons)
        )
    return exprs


def _flat_annotation_exprs(updates: list[AnnotationUpdate]) -> list[pl.Expr]:
    columns = [
        pl.when(condition.fill_null(value=False)).then(pl.lit(str(column))).otherwise(None)
        for condition, _reason, column in updates
    ]
    reasons = [
        pl.when(condition.fill_null(value=False)).then(pl.lit(str(reason))).otherwise(None)
        for condition, reason, _column in updates
    ]
    any_annotation = pl.any_horizontal(
        condition.fill_null(value=False) for condition, _reason, _column in updates
    )
    new_columns = pl.concat_str(columns, separator=ANNOTATION_ITEM_SEPARATOR, ignore_nulls=True)
    new_reasons = pl.concat_str(reasons, separator=ANNOTATION_ITEM_SEPARATOR, ignore_nulls=True)
    return [
        pl.when(any_annotation)
        .then(
            pl.concat_str(
                [pl.col(BaseColumns.ann_cols), new_columns],
                separator=ANNOTATION_ITEM_SEPARATOR,
                ignore_nulls=True,
            )
        )
        .otherwise(pl.col(BaseColumns.ann_cols))
        .alias(BaseColumns.ann_cols),
        pl.when(any_annotation)
        .then(
            pl.concat_str(
                [pl.col(BaseColumns.ann_reasons), new_reasons],
                separator=ANNOTATION_ITEM_SEPARATOR,
                ignore_nulls=True,
            )
        )
        .otherwise(pl.col(BaseColumns.ann_reasons))
        .alias(BaseColumns.ann_reasons),
    ]
