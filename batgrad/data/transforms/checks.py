from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import polars as pl

from batgrad.contracts.columns import BaseColumns, BatteryColumns, ColumnSpec, collect_column_specs
from batgrad.contracts.values import BaseValues

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import DatasetSpec


class CheckSpecBase:
    name: ClassVar[str]
    scope: ClassVar[TransformScope]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define class variable 'name'")
        if "scope" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define class variable 'scope'")


class TransformScope(StrEnum):
    BATCH = "batch"
    BOUNDED = "bounded"
    TWO_PASS = "two_pass"  # noqa: S105 - transform scope, not a credential.
    FULL_TASK = "full_task"


type FullTaskCheckHandler = Callable[
    [pl.LazyFrame, DatasetSpec, tuple[ColumnSpec, ...], CheckSpecBase],
    pl.LazyFrame,
]
type BoundedCheckHandler = Callable[
    [pl.DataFrame, DatasetSpec, CheckSpecBase, BoundedCheckState],
    pl.DataFrame,
]


@dataclass(frozen=True, slots=True)
class CheckHandler:
    full_task: FullTaskCheckHandler
    bounded: BoundedCheckHandler | None = None


@dataclass(slots=True)
class BoundedCheckState:
    time_pending_tail: pl.DataFrame | None = None
    cumulative_time: float = 0.0
    previous_axis_values: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class CheckFailure:
    check: str
    column: str
    reason: str
    count: int


CHECK_REGISTRY: dict[type[CheckSpecBase], CheckHandler] = {}
ANNOTATION_ITEM_SEPARATOR = "\x1f"
_MIN_BOUNDED_TIME_ROWS = 2


def register_check(spec_type: type[CheckSpecBase], handler: CheckHandler) -> None:
    if spec_type in CHECK_REGISTRY:
        raise ValueError(f"Check {spec_type.__name__} is already registered")
    if any(registered.name == spec_type.name for registered in CHECK_REGISTRY):
        raise ValueError(f"Check name {spec_type.name!r} is already registered")
    CHECK_REGISTRY[spec_type] = handler


@dataclass(frozen=True, slots=True)
class MissingCheckSpec(CheckSpecBase):
    name: ClassVar[str] = "missing"
    scope: ClassVar[TransformScope] = TransformScope.BATCH


@dataclass(frozen=True, slots=True)
class TimeCheckSpec(CheckSpecBase):
    time_col: ColumnSpec
    dt_col: ColumnSpec
    max_big_dt_count: int = 5
    big_dt_floor_s: float = 5.0
    max_diff_factor: float | None = 100.0

    name: ClassVar[str] = "time"
    scope: ClassVar[TransformScope] = TransformScope.FULL_TASK


@dataclass(frozen=True, slots=True)
class ColumnBoundsCheckSpec(CheckSpecBase):
    columns: tuple[ColumnSpec, ...] | None = None

    name: ClassVar[str] = "column_bounds"
    scope: ClassVar[TransformScope] = TransformScope.BATCH


@dataclass(frozen=True, slots=True)
class ImpedanceComponentsCheckSpec(CheckSpecBase):
    name: ClassVar[str] = "impedance_components"
    scope: ClassVar[TransformScope] = TransformScope.BATCH


@dataclass(frozen=True, slots=True)
class DomainAxisCheckSpec(CheckSpecBase):
    axis_col: ColumnSpec
    zero_replacement: float | None = None
    enforce_positive: bool = False

    name: ClassVar[str] = "domain_axis"
    scope: ClassVar[TransformScope] = TransformScope.FULL_TASK


def missing_check(
    data: pl.DataFrame,
    _dataset_spec: DatasetSpec,
    _check_spec: MissingCheckSpec,
) -> pl.DataFrame:
    data = ensure_annotation_column(data)
    numeric_columns = _numeric_check_columns(data.schema)
    if not numeric_columns:
        return data

    return add_annotations(
        data,
        [
            (pl.col(column).is_null() | pl.col(column).is_nan(), BaseValues.missing, column)
            for column in numeric_columns
        ],
    )


def missing_check_lazy(data: pl.LazyFrame) -> pl.LazyFrame:
    data = ensure_annotation_column_lazy(data)
    numeric_columns = _numeric_check_columns(data.collect_schema())
    if not numeric_columns:
        return data

    return add_annotations_lazy(
        data,
        [
            (pl.col(column).is_null() | pl.col(column).is_nan(), BaseValues.missing, column)
            for column in numeric_columns
        ],
    )


def run_check_full_task(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    group_by: tuple[ColumnSpec, ...],
    check_spec: CheckSpecBase,
) -> pl.LazyFrame:
    handler = CHECK_REGISTRY.get(type(check_spec))
    if handler is None:
        raise ValueError(f"No handler registered for check {type(check_spec).__name__}")
    return handler.full_task(data, dataset_spec, group_by, check_spec)


def apply_checks_full_task(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    group_by: tuple[ColumnSpec, ...],
    checks: tuple[CheckSpecBase, ...],
) -> pl.LazyFrame:
    for check in checks:
        data = run_check_full_task(data, dataset_spec, group_by, check)
    return data


def apply_checks_bounded_chunk(
    data: pl.DataFrame,
    dataset_spec: DatasetSpec,
    checks: tuple[CheckSpecBase, ...],
    state: BoundedCheckState,
) -> pl.DataFrame:
    for check in checks:
        handler = CHECK_REGISTRY.get(type(check))
        if handler is None:
            raise ValueError(f"No handler registered for check {type(check).__name__}")
        if handler.bounded is None:
            raise ValueError(
                f"Normalize check {type(check).__name__} does not support bounded execution",
            )
        data = handler.bounded(data, dataset_spec, check, state)
        if data.height == 0:
            return data
    return data


def collect_check_failures_full_task(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    group_by: tuple[ColumnSpec, ...],
    checks: tuple[CheckSpecBase, ...],
) -> tuple[CheckFailure, ...]:
    del group_by
    exprs, keys = _aggregate_check_exprs(dataset_spec, checks, data.collect_schema())
    if not exprs:
        return ()
    collected = data.select(exprs).collect()
    if not isinstance(collected, pl.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(collected).__name__}")
    return _aggregate_failures_from_row(collected.to_dicts()[0], keys)


def collect_check_failures_bounded_chunk(
    data: pl.DataFrame,
    dataset_spec: DatasetSpec,
    checks: tuple[CheckSpecBase, ...],
) -> tuple[CheckFailure, ...]:
    exprs, keys = _aggregate_check_exprs(dataset_spec, checks, data.schema)
    if not exprs:
        return ()
    return _aggregate_failures_from_row(data.select(exprs).row(0, named=True), keys)


def time_check_lazy(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    group_by: tuple[ColumnSpec, ...],
    check_spec: TimeCheckSpec,
) -> pl.LazyFrame:
    validate_max_big_dt_count(check_spec.max_big_dt_count)
    data = ensure_annotation_column_lazy(data)
    schema = data.collect_schema()
    time_col = check_spec.time_col
    dt_col = check_spec.dt_col
    date_time_col = getattr(dataset_spec.cols, "date_time", BatteryColumns.date_time)
    has_time = time_col in schema
    has_datetime = not has_time and date_time_col in schema
    if not has_time and not has_datetime:
        raise ValueError(
            f"No time column found. Expected {date_time_col!r} or {time_col!r} "
            f"but got columns: {sorted(schema.names())}",
        )

    group_columns = list(group_by)
    helper_cols: list[str] = []
    if has_datetime:
        data = data.with_columns(
            (
                (
                    pl.col(date_time_col) - pl.col(date_time_col).first().over(group_columns)
                ).dt.total_nanoseconds()
                / 1e9
            ).alias("_time_from_datetime"),
        )
        helper_cols.append("_time_from_datetime")
    if has_time:
        data = data.with_columns(
            (
                pl.col(time_col).cast(pl.Float64, strict=False)
                - pl.col(time_col).cast(pl.Float64, strict=False).first().over(group_columns)
            ).alias("_time_from_time"),
        )
        helper_cols.append("_time_from_time")

    if has_datetime:
        data = data.with_columns(
            pl.col("_time_from_datetime")
            .diff()
            .shift(-1)
            .over(group_columns)
            .cast(pl.Float64)
            .alias(dt_col),
        )
    else:
        data = data.with_columns(
            pl.col("_time_from_time")
            .diff()
            .shift(-1)
            .over(group_columns)
            .cast(pl.Float64)
            .alias(dt_col),
        )

    data = data.drop_nulls(subset=[dt_col]).filter(pl.col(dt_col) > 0.0)
    if check_spec.max_diff_factor is not None:
        max_dt_col = "_max_dt"
        big_dt_row_col = "_big_dt_row"
        excessive_big_dt_col = "_excessive_big_dt"
        data = data.with_columns(
            (
                pl.col(dt_col).quantile(0.5).over(group_columns) * float(check_spec.max_diff_factor)
            ).alias(max_dt_col),
        ).with_columns(
            pl.max_horizontal(
                pl.lit(check_spec.big_dt_floor_s),
                pl.coalesce([pl.col(max_dt_col), pl.lit(check_spec.big_dt_floor_s)]),
            ).alias(max_dt_col),
        )
        data = data.with_columns(
            (pl.col(dt_col) > pl.col(max_dt_col)).alias(big_dt_row_col),
        ).with_columns(
            (
                pl.col(big_dt_row_col).cast(pl.Int64).sum().over(group_columns)
                >= check_spec.max_big_dt_count
            ).alias(excessive_big_dt_col),
        )
        data = add_annotation_lazy(
            data,
            pl.col(excessive_big_dt_col) & pl.col(big_dt_row_col),
            BaseValues.big_dt,
            dt_col,
        )
        data = data.filter(
            pl.when(pl.col(excessive_big_dt_col))
            .then(pl.lit(value=True))
            .otherwise(~pl.col(big_dt_row_col)),
        )
        helper_cols.extend([max_dt_col, big_dt_row_col, excessive_big_dt_col])

    data = data.with_columns(
        (pl.col(dt_col).cum_sum().over(group_columns) - pl.col(dt_col)).alias(time_col),
    )
    duplicate_col = "_duplicate_time_count"
    data = data.with_columns(pl.len().over([*group_columns, time_col]).alias(duplicate_col))
    data = add_annotation_lazy(
        data,
        pl.col(duplicate_col) > 1,
        BaseValues.dup_time,
        time_col,
    )
    helper_cols.append(duplicate_col)
    return data.drop([column for column in helper_cols if column in data.collect_schema()])


def column_bounds_check_lazy(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    check_spec: ColumnBoundsCheckSpec,
) -> pl.LazyFrame:
    schema = data.collect_schema()
    updates: list[tuple[pl.Expr, str, str]] = []
    for column in _bounded_columns(dataset_spec, check_spec, schema.names()):
        valid = pl.col(column).is_not_null() & ~pl.col(column).is_nan()
        if column.col_min is not None:
            updates.append(
                (valid & (pl.col(column) < float(column.col_min)), BaseValues.col_min, column),
            )
        if column.col_max is not None:
            updates.append(
                (valid & (pl.col(column) > float(column.col_max)), BaseValues.col_max, column),
            )
    return add_annotations_lazy(data, updates)


def column_bounds_check(
    data: pl.DataFrame,
    dataset_spec: DatasetSpec,
    check_spec: ColumnBoundsCheckSpec,
) -> pl.DataFrame:
    updates: list[tuple[pl.Expr, str, str]] = []
    for column in _bounded_columns(dataset_spec, check_spec, data.columns):
        valid = pl.col(column).is_not_null() & ~pl.col(column).is_nan()
        if column.col_min is not None:
            updates.append(
                (valid & (pl.col(column) < float(column.col_min)), BaseValues.col_min, column),
            )
        if column.col_max is not None:
            updates.append(
                (valid & (pl.col(column) > float(column.col_max)), BaseValues.col_max, column),
            )
    return add_annotations(data, updates)


def impedance_components_check_lazy(data: pl.LazyFrame, dataset_spec: DatasetSpec) -> pl.LazyFrame:
    exprs = _impedance_component_exprs(dataset_spec, data.collect_schema().names())
    return data.with_columns(exprs) if exprs else data


def impedance_components_check(data: pl.DataFrame, dataset_spec: DatasetSpec) -> pl.DataFrame:
    exprs = _impedance_component_exprs(dataset_spec, data.columns)
    return data.with_columns(exprs) if exprs else data


def domain_axis_check_lazy(
    data: pl.LazyFrame,
    group_by: tuple[ColumnSpec, ...],
    check_spec: DomainAxisCheckSpec,
) -> pl.LazyFrame:
    axis_col = check_spec.axis_col
    if axis_col not in data.collect_schema():
        raise ValueError(f"Domain axis column {axis_col!r} is missing")
    data = ensure_annotation_column_lazy(data)
    if check_spec.zero_replacement is not None:
        data = data.with_columns(
            pl.when(pl.col(axis_col) == 0)
            .then(pl.lit(check_spec.zero_replacement))
            .otherwise(pl.col(axis_col))
            .alias(axis_col),
        )

    previous = pl.col(axis_col).diff().over(list(group_by))
    invalid = pl.col(axis_col).is_null() | pl.col(axis_col).is_nan() | (previous <= 0)
    if check_spec.enforce_positive:
        invalid |= pl.col(axis_col) <= 0
    return add_annotation_lazy(
        data,
        invalid.fill_null(value=False),
        BaseValues.domain_x_axis,
        axis_col,
    )


def domain_axis_check(
    data: pl.DataFrame,
    check_spec: DomainAxisCheckSpec,
    state: BoundedCheckState,
) -> pl.DataFrame:
    axis_col = check_spec.axis_col
    if axis_col not in data.columns:
        raise ValueError(f"Domain axis column {axis_col!r} is missing")
    data = ensure_annotation_column(data)
    if check_spec.zero_replacement is not None:
        data = data.with_columns(
            pl.when(pl.col(axis_col) == 0)
            .then(pl.lit(check_spec.zero_replacement))
            .otherwise(pl.col(axis_col))
            .alias(axis_col),
        )
    if data.height == 0:
        return data

    if state.previous_axis_values is None:
        state.previous_axis_values = {}
    previous_value = state.previous_axis_values.get(str(axis_col))
    diff = pl.col(axis_col).diff()
    if previous_value is not None:
        diff = (
            pl.when(pl.int_range(pl.len()) == 0)
            .then(
                pl.col(axis_col) - pl.lit(previous_value),
            )
            .otherwise(diff)
        )

    invalid = pl.col(axis_col).is_null() | pl.col(axis_col).is_nan() | (diff <= 0)
    if check_spec.enforce_positive:
        invalid |= pl.col(axis_col) <= 0
    data = add_annotation(data, invalid.fill_null(value=False), BaseValues.domain_x_axis, axis_col)
    last_value = data[axis_col].drop_nulls()
    if len(last_value) > 0:
        state.previous_axis_values[str(axis_col)] = float(last_value[-1])
    return data


def _bounded_columns(
    dataset_spec: DatasetSpec,
    check_spec: ColumnBoundsCheckSpec,
    available_columns: list[str],
) -> tuple[ColumnSpec, ...]:
    candidates = check_spec.columns
    if candidates is None:
        candidates = tuple(collect_column_specs(dataset_spec.cols).values())
    available = set(available_columns)
    return tuple(
        column
        for column in candidates
        if column in available and (column.col_min is not None or column.col_max is not None)
    )


def _impedance_component_exprs(
    dataset_spec: DatasetSpec,
    available_columns: list[str],
) -> list[pl.Expr]:
    cols = dataset_spec.cols
    z_real = getattr(cols, "z_real", BatteryColumns.z_real)
    z_imag = getattr(cols, "z_imag", BatteryColumns.z_imag)
    z_mag = getattr(cols, "z_mag", BatteryColumns.z_mag)
    z_phase = getattr(cols, "z_phase", BatteryColumns.z_phase)
    available = set(available_columns)
    has_rectangular = z_real in available and z_imag in available
    has_polar = z_mag in available and z_phase in available
    if not has_rectangular and not has_polar:
        raise ValueError(
            "EIS data requires either (z_real, z_imag) or (z_mag, z_phase); "
            f"available columns: {sorted(available)}",
        )

    exprs: list[pl.Expr] = []
    if has_polar:
        real_from_polar = pl.col(z_mag) * pl.col(z_phase).radians().cos()
        imag_from_polar = pl.col(z_mag) * pl.col(z_phase).radians().sin()
        exprs.extend(
            [
                pl.coalesce([pl.col(z_real), real_from_polar]).alias(z_real)
                if z_real in available
                else real_from_polar.alias(z_real),
                pl.coalesce([pl.col(z_imag), imag_from_polar]).alias(z_imag)
                if z_imag in available
                else imag_from_polar.alias(z_imag),
            ],
        )
    if has_rectangular:
        mag_from_rect = (pl.col(z_real).pow(2) + pl.col(z_imag).pow(2)).sqrt()
        phase_from_rect = pl.arctan2(pl.col(z_imag), pl.col(z_real)).degrees()
        exprs.extend(
            [
                pl.coalesce([pl.col(z_mag), mag_from_rect]).alias(z_mag)
                if z_mag in available
                else mag_from_rect.alias(z_mag),
                pl.coalesce([pl.col(z_phase), phase_from_rect]).alias(z_phase)
                if z_phase in available
                else phase_from_rect.alias(z_phase),
            ],
        )
    return exprs


def _missing_check_bounded_handler(
    data: pl.DataFrame,
    dataset_spec: DatasetSpec,
    check_spec: CheckSpecBase,
    _state: BoundedCheckState,
) -> pl.DataFrame:
    if not isinstance(check_spec, MissingCheckSpec):
        raise TypeError(f"Expected MissingCheckSpec, got {type(check_spec).__name__}")
    return missing_check(data, dataset_spec, check_spec)


def _missing_check_full_task_handler(
    data: pl.LazyFrame,
    _dataset_spec: DatasetSpec,
    _group_by: tuple[ColumnSpec, ...],
    check_spec: CheckSpecBase,
) -> pl.LazyFrame:
    if not isinstance(check_spec, MissingCheckSpec):
        raise TypeError(f"Expected MissingCheckSpec, got {type(check_spec).__name__}")
    return missing_check_lazy(data)


def _time_check_full_task_handler(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    group_by: tuple[ColumnSpec, ...],
    check_spec: CheckSpecBase,
) -> pl.LazyFrame:
    if not isinstance(check_spec, TimeCheckSpec):
        raise TypeError(f"Expected TimeCheckSpec, got {type(check_spec).__name__}")
    return time_check_lazy(data, dataset_spec, group_by, check_spec)


def _time_check_bounded_handler(
    data: pl.DataFrame,
    _dataset_spec: DatasetSpec,
    check_spec: CheckSpecBase,
    state: BoundedCheckState,
) -> pl.DataFrame:
    if not isinstance(check_spec, TimeCheckSpec):
        raise TypeError(f"Expected TimeCheckSpec, got {type(check_spec).__name__}")
    time_col = check_spec.time_col
    dt_col = check_spec.dt_col
    if time_col not in data.columns:
        raise ValueError(f"Bounded normalize requires time column {time_col!r}")

    if state.time_pending_tail is not None:
        data = pl.concat((state.time_pending_tail, data), how="diagonal_relaxed")
    if data.height < _MIN_BOUNDED_TIME_ROWS:
        state.time_pending_tail = data
        return data.limit(0)

    with_dt = data.with_columns(
        pl.col(time_col).cast(pl.Float64, strict=False).diff().shift(-1).alias(dt_col),
    )
    emit = with_dt.slice(0, with_dt.height - 1).filter(pl.col(dt_col) > 0.0)
    state.time_pending_tail = data.slice(data.height - 1, 1)
    if emit.height == 0:
        return emit

    dt_values = emit[dt_col].to_numpy().astype(np.float64)
    rebuilt_time = state.cumulative_time + np.cumsum(dt_values) - dt_values
    state.cumulative_time += float(dt_values.sum())
    emit = emit.with_columns(pl.Series(time_col, rebuilt_time))
    return add_annotation(
        emit,
        pl.col(time_col).is_duplicated(),
        BaseValues.dup_time,
        time_col,
    )


def _column_bounds_full_task_handler(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    _group_by: tuple[ColumnSpec, ...],
    check_spec: CheckSpecBase,
) -> pl.LazyFrame:
    if not isinstance(check_spec, ColumnBoundsCheckSpec):
        raise TypeError(f"Expected ColumnBoundsCheckSpec, got {type(check_spec).__name__}")
    return column_bounds_check_lazy(data, dataset_spec, check_spec)


def _column_bounds_bounded_handler(
    data: pl.DataFrame,
    dataset_spec: DatasetSpec,
    check_spec: CheckSpecBase,
    _state: BoundedCheckState,
) -> pl.DataFrame:
    if not isinstance(check_spec, ColumnBoundsCheckSpec):
        raise TypeError(f"Expected ColumnBoundsCheckSpec, got {type(check_spec).__name__}")
    return column_bounds_check(data, dataset_spec, check_spec)


def _impedance_components_full_task_handler(
    data: pl.LazyFrame,
    dataset_spec: DatasetSpec,
    _group_by: tuple[ColumnSpec, ...],
    check_spec: CheckSpecBase,
) -> pl.LazyFrame:
    if not isinstance(check_spec, ImpedanceComponentsCheckSpec):
        raise TypeError(f"Expected ImpedanceComponentsCheckSpec, got {type(check_spec).__name__}")
    result = impedance_components_check_lazy(data, dataset_spec)
    if not isinstance(result, pl.LazyFrame):
        raise TypeError(f"Expected LazyFrame, got {type(result).__name__}")
    return result


def _impedance_components_bounded_handler(
    data: pl.DataFrame,
    dataset_spec: DatasetSpec,
    check_spec: CheckSpecBase,
    _state: BoundedCheckState,
) -> pl.DataFrame:
    if not isinstance(check_spec, ImpedanceComponentsCheckSpec):
        raise TypeError(f"Expected ImpedanceComponentsCheckSpec, got {type(check_spec).__name__}")
    result = impedance_components_check(data, dataset_spec)
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(result).__name__}")
    return result


def _domain_axis_full_task_handler(
    data: pl.LazyFrame,
    _dataset_spec: DatasetSpec,
    group_by: tuple[ColumnSpec, ...],
    check_spec: CheckSpecBase,
) -> pl.LazyFrame:
    if not isinstance(check_spec, DomainAxisCheckSpec):
        raise TypeError(f"Expected DomainAxisCheckSpec, got {type(check_spec).__name__}")
    return domain_axis_check_lazy(data, group_by, check_spec)


def _domain_axis_bounded_handler(
    data: pl.DataFrame,
    _dataset_spec: DatasetSpec,
    check_spec: CheckSpecBase,
    state: BoundedCheckState,
) -> pl.DataFrame:
    if not isinstance(check_spec, DomainAxisCheckSpec):
        raise TypeError(f"Expected DomainAxisCheckSpec, got {type(check_spec).__name__}")
    return domain_axis_check(data, check_spec, state)


def _numeric_check_columns(schema: dict[str, pl.DataType]) -> list[str]:
    return [
        name
        for name, dtype in schema.items()
        if dtype.is_numeric() and name not in {BaseColumns.cycle_index, BaseColumns.row_count}
    ]


def _missing_condition(columns: list[str]) -> pl.Expr:
    return pl.any_horizontal(
        pl.col(column).is_null() | pl.col(column).is_nan() for column in columns
    )


def _aggregate_check_exprs(
    dataset_spec: DatasetSpec,
    checks: tuple[CheckSpecBase, ...],
    schema: dict[str, pl.DataType],
) -> tuple[list[pl.Expr], list[tuple[str, str, str, str]]]:
    exprs: list[pl.Expr] = []
    keys: list[tuple[str, str, str, str]] = []
    for check in checks:
        if isinstance(check, MissingCheckSpec):
            for column in _numeric_check_columns(schema):
                alias = f"__check_missing__{len(keys)}"
                exprs.append(
                    (pl.col(column).is_null() | pl.col(column).is_nan()).sum().alias(alias),
                )
                keys.append((alias, check.name, str(column), BaseValues.missing))
        elif isinstance(check, ColumnBoundsCheckSpec):
            for column in _bounded_columns(dataset_spec, check, list(schema)):
                valid = pl.col(column).is_not_null() & ~pl.col(column).is_nan()
                if column.col_min is not None:
                    alias = f"__check_col_min__{len(keys)}"
                    exprs.append(
                        (valid & (pl.col(column) < float(column.col_min))).sum().alias(alias),
                    )
                    keys.append((alias, check.name, str(column), BaseValues.col_min))
                if column.col_max is not None:
                    alias = f"__check_col_max__{len(keys)}"
                    exprs.append(
                        (valid & (pl.col(column) > float(column.col_max))).sum().alias(alias),
                    )
                    keys.append((alias, check.name, str(column), BaseValues.col_max))
    return exprs, keys


def _aggregate_failures_from_row(
    row: dict[str, object],
    keys: list[tuple[str, str, str, str]],
) -> tuple[CheckFailure, ...]:
    failures: list[CheckFailure] = []
    for alias, check, column, reason in keys:
        value = row.get(alias)
        count = int(value) if isinstance(value, int | float) else 0
        if count > 0:
            failures.append(CheckFailure(check=check, column=column, reason=reason, count=count))
    return tuple(failures)


def _empty_annotations() -> pl.Expr:
    return pl.lit([], dtype=_annotation_dtype())


def _annotation_dtype() -> pl.DataType:
    dtype = BaseColumns.annotations.dtype
    if dtype is None:
        raise ValueError(f"Column {BaseColumns.annotations!r} has no dtype")
    return dtype


def _annotation_base_expr(columns: list[str]) -> pl.Expr:
    if BaseColumns.annotations not in columns:
        return _empty_annotations().alias(BaseColumns.annotations)
    return pl.coalesce(
        [pl.col(BaseColumns.annotations).cast(_annotation_dtype()), _empty_annotations()],
    ).alias(BaseColumns.annotations)


def ensure_annotation_column(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(_annotation_base_expr(data.columns))


def ensure_annotation_column_lazy(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.with_columns(_annotation_base_expr(data.collect_schema().names()))


def _append_annotation_expr(condition: pl.Expr, reason: str, column: str) -> pl.Expr:
    annotation = pl.struct(
        pl.lit(str(column)).alias("column"),
        pl.lit(str(reason)).alias("reason"),
    )
    return (
        pl.when(condition)
        .then(pl.concat_list([pl.col(BaseColumns.annotations), annotation]))
        .otherwise(pl.col(BaseColumns.annotations))
        .alias(BaseColumns.annotations)
    )


def add_annotation(
    data: pl.DataFrame,
    condition: pl.Expr,
    reason: str,
    column: str,
) -> pl.DataFrame:
    return add_annotations(data, [(condition, reason, column)])


def add_annotations(
    data: pl.DataFrame,
    updates: list[tuple[pl.Expr, str, str]],
) -> pl.DataFrame:
    if not updates:
        return data
    any_violation = data.select(
        pl.any_horizontal(condition.fill_null(value=False) for condition, _, _ in updates)
        .any()
        .alias("__has_annotation"),
    ).item()
    if not any_violation:
        return data
    return data.with_columns(_flat_annotation_exprs(updates))


def add_annotation_lazy(
    data: pl.LazyFrame,
    condition: pl.Expr,
    reason: str,
    column: str,
) -> pl.LazyFrame:
    return add_annotations_lazy(data, [(condition, reason, column)])


def add_annotations_lazy(
    data: pl.LazyFrame,
    updates: list[tuple[pl.Expr, str, str]],
) -> pl.LazyFrame:
    if not updates:
        return data
    return data.with_columns(_flat_annotation_exprs(updates))


def _flat_annotation_exprs(updates: list[tuple[pl.Expr, str, str]]) -> list[pl.Expr]:
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
    return [
        pl.when(any_annotation)
        .then(pl.concat_str(columns, separator=ANNOTATION_ITEM_SEPARATOR, ignore_nulls=True))
        .otherwise(None)
        .alias(BaseColumns.annotation_columns),
        pl.when(any_annotation)
        .then(pl.concat_str(reasons, separator=ANNOTATION_ITEM_SEPARATOR, ignore_nulls=True))
        .otherwise(None)
        .alias(BaseColumns.annotation_reasons),
    ]


def validate_max_big_dt_count(max_big_dt_count: int) -> int:
    if max_big_dt_count < 1:
        raise ValueError(f"max_big_dt_count must be >= 1, got {max_big_dt_count}")
    return int(max_big_dt_count)


register_check(
    MissingCheckSpec,
    CheckHandler(
        full_task=_missing_check_full_task_handler,
        bounded=_missing_check_bounded_handler,
    ),
)
register_check(
    TimeCheckSpec,
    CheckHandler(
        full_task=_time_check_full_task_handler,
        bounded=_time_check_bounded_handler,
    ),
)
register_check(
    ColumnBoundsCheckSpec,
    CheckHandler(
        full_task=_column_bounds_full_task_handler,
        bounded=_column_bounds_bounded_handler,
    ),
)
register_check(
    ImpedanceComponentsCheckSpec,
    CheckHandler(
        full_task=_impedance_components_full_task_handler,
        bounded=_impedance_components_bounded_handler,
    ),
)
register_check(
    DomainAxisCheckSpec,
    CheckHandler(
        full_task=_domain_axis_full_task_handler,
        bounded=_domain_axis_bounded_handler,
    ),
)
