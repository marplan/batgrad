from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, overload

import numpy as np
import polars as pl

from batgrad.contracts.mapping import BaseColumns, MappingSpec
from batgrad.data.transforms.annotations import (
    ANNOTATION_ITEM_SEPARATOR,
    AnnotationUpdate,
    add_annotations,
    add_annotations_lazy,
    ensure_annotation_columns,
    ensure_annotation_columns_lazy,
)

type CheckViolation = tuple[str, str]


@dataclass
class TimeCheckState:
    pending_tail: pl.DataFrame | None = None
    cumulative_time: float = 0.0


@dataclass
class DomainAxisCheckState:
    previous_axis_value: float | None = None


type CheckState = TimeCheckState | DomainAxisCheckState | None


@dataclass(frozen=True)
class MissingCheckSpec:
    """Annotate null or NaN numeric values.

    If `columns` is omitted, all numeric columns present in the input frame are
    checked. The check produces internal annotation columns that are finalized by
    normalization when annotations are requested.

    Attributes:
        columns: Optional columns to check. Missing requested columns are ignored.
    """

    columns: tuple[MappingSpec, ...] | None = None

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return self.columns or ()

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return (BaseColumns.ann_cols, BaseColumns.ann_reasons)

    def apply_full(
        self, data: pl.LazyFrame, group_by: tuple[MappingSpec, ...], *, annotate: bool = True
    ) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
        del group_by
        return _apply_or_validate_updates_lazy(
            data,
            self._updates(self._columns(data.collect_schema())),
            annotate=annotate,
        )

    def init_state(self) -> CheckState:
        return None

    def apply_chunk(
        self, data: pl.DataFrame, state: CheckState, *, annotate: bool = True
    ) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
        del state
        return _apply_or_validate_updates(
            data,
            self._updates(self._columns(data.schema)),
            annotate=annotate,
        )

    def _columns(self, schema: dict[str, pl.DataType]) -> list[str]:
        if self.columns is not None:
            candidates = [str(column) for column in self.columns]
        else:
            candidates = list(schema)
        return [name for name in candidates if name in schema and schema[name].is_numeric()]

    def _updates(self, columns: list[str]) -> list[AnnotationUpdate]:
        return [
            (
                pl.col(column).is_null() | pl.col(column).is_nan(),
                BaseColumns.ann_reasons.values.missing,
                column,
            )
            for column in columns
        ]


@dataclass(frozen=True)
class TimeCheckSpec:
    """Validate and rebuild a monotonically increasing task time axis.

    The check computes `dt_col` from consecutive `time_col` differences, drops
    duplicate or non-positive intervals, optionally flags large intervals, and
    rebuilds time from cumulative `dt_col`. Full-task normalization applies the
    diff within `group_by`; because this uses a forward interval, the final row
    has no `dt_col` and is dropped. Bounded normalization carries a pending tail
    row across chunks and may emit no rows until at least two rows are available.

    Attributes:
        time_col: Source time column.
        dt_col: Output time-step column.
        max_dt_s: Maximum allowed interval in seconds. Set to `None` to skip the
            large time-step annotation.
    """

    time_col: MappingSpec
    dt_col: MappingSpec
    max_dt_s: float | None = 86_400.0

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return (self.time_col,)

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return (self.dt_col, BaseColumns.ann_cols, BaseColumns.ann_reasons)

    def apply_full(
        self, data: pl.LazyFrame, group_by: tuple[MappingSpec, ...], *, annotate: bool = True
    ) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
        if self.time_col not in data.collect_schema():
            raise ValueError(f"No time column found. Expected {self.time_col!r}")
        group_columns = list(group_by)
        violations: list[CheckViolation] = []
        dt_expr = pl.col(self.time_col).cast(pl.Float64).diff().shift(-1)
        if group_columns:
            dt_expr = dt_expr.over(group_columns)
        data = data.with_columns(dt_expr.cast(pl.Float64).alias(self.dt_col))
        helper_cols: list[str] = []
        duplicate_col = "__duplicate_time_count"
        helper_cols.append(duplicate_col)
        data = data.with_columns(
            pl.len().over([*group_columns, self.time_col]).alias(duplicate_col)
        )
        data, update_violations = _apply_or_validate_updates_lazy(
            data,
            [self._duplicate_time_update(pl.col(duplicate_col) > 1)],
            annotate=annotate,
        )
        violations.extend(update_violations)
        data = data.drop_nulls(subset=[self.dt_col]).filter(pl.col(self.dt_col) > 0.0)
        if self.max_dt_s is not None:
            self._validate_max_dt()
            big_dt_row_col = "__big_dt_row"
            helper_cols.append(big_dt_row_col)
            data = data.with_columns((pl.col(self.dt_col) > self.max_dt_s).alias(big_dt_row_col))
            data, update_violations = _apply_or_validate_updates_lazy(
                data,
                [self._big_dt_update(pl.col(big_dt_row_col))],
                annotate=annotate,
            )
            violations.extend(update_violations)
        data = rebuild_time_axis_lazy(data, self.time_col, self.dt_col, tuple(group_columns))
        return (
            data.drop([column for column in helper_cols if column in data.collect_schema()]),
            tuple(dict.fromkeys(violations)),
        )

    def init_state(self) -> CheckState:
        return TimeCheckState()

    def apply_chunk(
        self, data: pl.DataFrame, state: CheckState, *, annotate: bool = True
    ) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
        if not isinstance(state, TimeCheckState):
            raise TypeError("TimeCheckSpec requires chunk state")
        if self.time_col not in data.columns:
            raise ValueError(f"Bounded normalize requires time column {self.time_col!r}")
        if state.pending_tail is not None:
            data = pl.concat((state.pending_tail, data), how="diagonal_relaxed")
        if data.height < _MIN_BOUNDED_TIME_ROWS:
            state.pending_tail = data
            return data.limit(0), ()
        with_dt = data.with_columns(
            pl.col(self.time_col).cast(pl.Float64).diff().shift(-1).alias(self.dt_col),
        )
        violations: list[CheckViolation] = []
        with_dt, update_violations = _apply_or_validate_updates(
            with_dt,
            [self._duplicate_time_update(pl.col(self.time_col).is_duplicated())],
            annotate=annotate,
        )
        violations.extend(update_violations)
        emit = with_dt.slice(0, with_dt.height - 1).filter(pl.col(self.dt_col) > 0.0)
        state.pending_tail = data.slice(data.height - 1, 1)
        if emit.height == 0:
            return emit, tuple(dict.fromkeys(violations))
        emit = rebuild_time_axis_chunk(emit, state, self.time_col, self.dt_col)
        if self.max_dt_s is not None:
            self._validate_max_dt()
            emit, update_violations = _apply_or_validate_updates(
                emit,
                [self._big_dt_update(pl.col(self.dt_col) > self.max_dt_s)],
                annotate=annotate,
            )
            violations.extend(update_violations)
        return emit, tuple(dict.fromkeys(violations))

    def _validate_max_dt(self) -> None:
        if self.max_dt_s is not None and self.max_dt_s <= 0.0:
            raise ValueError(f"max_dt_s must be > 0, got {self.max_dt_s}")

    def _duplicate_time_update(self, condition: pl.Expr) -> AnnotationUpdate:
        return (condition, BaseColumns.ann_reasons.values.dup_time, self.time_col)

    def _big_dt_update(self, condition: pl.Expr) -> AnnotationUpdate:
        return (condition, BaseColumns.ann_reasons.values.big_dt, self.dt_col)


@dataclass(frozen=True)
class ColumnBoundsCheckSpec:
    """Annotate numeric values outside configured inclusive bounds.

    Bounds use `(lower, upper)` pairs. `None` disables one side. Values equal to
    a bound are accepted; the check annotates values below `lower` or above
    `upper` and never clips data.

    Attributes:
        bounds: Per-column `(lower, upper)` bounds.
    """

    bounds: dict[MappingSpec, tuple[float | None, float | None]]

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return tuple(self.bounds)

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return (BaseColumns.ann_cols, BaseColumns.ann_reasons)

    def apply_full(
        self, data: pl.LazyFrame, group_by: tuple[MappingSpec, ...], *, annotate: bool = True
    ) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
        del group_by
        return _apply_or_validate_updates_lazy(
            data,
            self._updates(data.collect_schema().names()),
            annotate=annotate,
        )

    def init_state(self) -> CheckState:
        return None

    def apply_chunk(
        self, data: pl.DataFrame, state: CheckState, *, annotate: bool = True
    ) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
        del state
        return _apply_or_validate_updates(
            data,
            self._updates(data.columns),
            annotate=annotate,
        )

    def _updates(self, columns: list[str]) -> list[AnnotationUpdate]:
        available = set(columns)
        updates: list[AnnotationUpdate] = []
        for column, (lower, upper) in self.bounds.items():
            if str(column) not in available:
                continue
            valid = pl.col(column).is_not_null() & ~pl.col(column).is_nan()
            if lower is not None:
                updates.append(
                    (
                        valid & (pl.col(column) < float(lower)),
                        BaseColumns.ann_reasons.values.col_min,
                        column,
                    )
                )
            if upper is not None:
                updates.append(
                    (
                        valid & (pl.col(column) > float(upper)),
                        BaseColumns.ann_reasons.values.col_max,
                        column,
                    )
                )
        return updates


@dataclass(frozen=True)
class ImpedanceComponentsCheckSpec:
    """Ensure EIS data has both rectangular and polar impedance components.

    Input must contain either `(z_real, z_imag)` or `(z_mag, z_phase)`. Missing
    counterparts are derived from the available representation. Existing columns
    are preserved by coalescing existing values before derived values.
    """

    z_real: MappingSpec = BaseColumns.z_real
    z_imag: MappingSpec = BaseColumns.z_imag
    z_mag: MappingSpec = BaseColumns.z_mag
    z_phase: MappingSpec = BaseColumns.z_phase

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return (self.z_real, self.z_imag, self.z_mag, self.z_phase)

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return (self.z_real, self.z_imag, self.z_mag, self.z_phase)

    def apply_full(
        self, data: pl.LazyFrame, group_by: tuple[MappingSpec, ...], *, annotate: bool = True
    ) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
        del group_by
        del annotate
        exprs = self._exprs(data.collect_schema().names())
        return (data.with_columns(exprs) if exprs else data), ()

    def init_state(self) -> CheckState:
        return None

    def apply_chunk(
        self, data: pl.DataFrame, state: CheckState, *, annotate: bool = True
    ) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
        del state
        del annotate
        exprs = self._exprs(data.columns)
        return (data.with_columns(exprs) if exprs else data), ()

    def _exprs(self, columns: list[str]) -> list[pl.Expr]:
        available = set(columns)
        has_rectangular = self.z_real in available and self.z_imag in available
        has_polar = self.z_mag in available and self.z_phase in available
        if not has_rectangular and not has_polar:
            raise ValueError(
                "EIS data requires either (z_real, z_imag) or (z_mag, z_phase); "
                f"available columns: {sorted(available)}",
            )
        exprs: list[pl.Expr] = []
        if has_polar:
            real_from_polar = pl.col(self.z_mag) * pl.col(self.z_phase).radians().cos()
            imag_from_polar = pl.col(self.z_mag) * pl.col(self.z_phase).radians().sin()
            exprs.extend(
                [
                    pl.coalesce([pl.col(self.z_real), real_from_polar]).alias(self.z_real)
                    if self.z_real in available
                    else real_from_polar.alias(self.z_real),
                    pl.coalesce([pl.col(self.z_imag), imag_from_polar]).alias(self.z_imag)
                    if self.z_imag in available
                    else imag_from_polar.alias(self.z_imag),
                ],
            )
        if has_rectangular:
            mag_from_rect = (pl.col(self.z_real).pow(2) + pl.col(self.z_imag).pow(2)).sqrt()
            phase_from_rect = pl.arctan2(pl.col(self.z_imag), pl.col(self.z_real)).degrees()
            exprs.extend(
                [
                    pl.coalesce([pl.col(self.z_mag), mag_from_rect]).alias(self.z_mag)
                    if self.z_mag in available
                    else mag_from_rect.alias(self.z_mag),
                    pl.coalesce([pl.col(self.z_phase), phase_from_rect]).alias(self.z_phase)
                    if self.z_phase in available
                    else phase_from_rect.alias(self.z_phase),
                ],
            )
        return exprs


@dataclass(frozen=True)
class DomainAxisCheckSpec:
    """Annotate invalid domain-axis values within a task or group.

    The axis is invalid when it is null, NaN, or not strictly increasing. Use
    `zero_replacement` before validation for sources that encode the first EIS
    frequency as zero. Set `enforce_positive` to also reject zero or negative
    values after replacement. Bounded normalization carries the previous axis
    value across chunks; the first row is not invalid only because it has no
    previous value.

    Attributes:
        axis_col: Domain column to validate.
        zero_replacement: Optional value used to replace zeros before checking.
        enforce_positive: Whether to reject non-positive axis values.
    """

    axis_col: MappingSpec
    zero_replacement: float | None = None
    enforce_positive: bool = False

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return (self.axis_col,)

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return (BaseColumns.ann_cols, BaseColumns.ann_reasons)

    def apply_full(
        self, data: pl.LazyFrame, group_by: tuple[MappingSpec, ...], *, annotate: bool = True
    ) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
        if self.axis_col not in data.collect_schema():
            raise ValueError(f"Domain axis column {self.axis_col!r} is missing")
        data = self._apply_zero_replacement(data)
        previous = pl.col(self.axis_col).diff()
        if group_by:
            previous = previous.over(list(group_by))
        return _apply_or_validate_updates_lazy(
            data,
            [self._invalid_axis_update(previous)],
            annotate=annotate,
        )

    def init_state(self) -> CheckState:
        return DomainAxisCheckState()

    def apply_chunk(
        self, data: pl.DataFrame, state: CheckState, *, annotate: bool = True
    ) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
        if not isinstance(state, DomainAxisCheckState):
            raise TypeError("DomainAxisCheckSpec requires chunk state")
        if self.axis_col not in data.columns:
            raise ValueError(f"Domain axis column {self.axis_col!r} is missing")
        data = self._apply_zero_replacement(data)
        if data.height == 0:
            return data, ()
        previous_value = state.previous_axis_value
        diff = pl.col(self.axis_col).diff()
        if previous_value is not None:
            diff = (
                pl.when(pl.int_range(pl.len()) == 0)
                .then(
                    pl.col(self.axis_col) - pl.lit(previous_value),
                )
                .otherwise(diff)
            )
        data, violations = _apply_or_validate_updates(
            data,
            [self._invalid_axis_update(diff)],
            annotate=annotate,
        )
        last_value = data[self.axis_col].drop_nulls()
        if len(last_value) > 0:
            state.previous_axis_value = float(last_value[-1])
        return data, violations

    @overload
    def _apply_zero_replacement(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _apply_zero_replacement(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def _apply_zero_replacement(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        if self.zero_replacement is None:
            return data
        expr = (
            pl.when(pl.col(self.axis_col) == 0)
            .then(pl.lit(self.zero_replacement))
            .otherwise(pl.col(self.axis_col))
            .alias(self.axis_col)
        )
        if isinstance(data, pl.LazyFrame):
            return data.with_columns(expr)
        return data.with_columns(expr)

    def _invalid_axis_update(self, previous: pl.Expr) -> AnnotationUpdate:
        invalid = pl.col(self.axis_col).is_null() | pl.col(self.axis_col).is_nan() | (previous <= 0)
        if self.enforce_positive:
            invalid |= pl.col(self.axis_col) <= 0
        return (
            invalid.fill_null(value=False),
            BaseColumns.ann_reasons.values.invalid_axis,
            self.axis_col,
        )


class CheckSpec(Protocol):
    @property
    def input_columns(self) -> tuple[MappingSpec, ...]: ...

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]: ...

    def apply_full(
        self, data: pl.LazyFrame, group_by: tuple[MappingSpec, ...], *, annotate: bool = True
    ) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]: ...

    def init_state(self) -> CheckState: ...

    def apply_chunk(
        self, data: pl.DataFrame, state: CheckState, *, annotate: bool = True
    ) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]: ...


_MIN_BOUNDED_TIME_ROWS = 2


def rebuild_time_axis_lazy(
    data: pl.LazyFrame,
    time_col: MappingSpec,
    dt_col: MappingSpec,
    group_by: tuple[MappingSpec, ...],
) -> pl.LazyFrame:
    cumulative = pl.col(dt_col).cum_sum()
    if group_by:
        cumulative = cumulative.over(list(group_by))
    return data.with_columns((cumulative - pl.col(dt_col)).alias(time_col))


def rebuild_time_axis_chunk(
    data: pl.DataFrame,
    state: TimeCheckState,
    time_col: MappingSpec,
    dt_col: MappingSpec,
) -> pl.DataFrame:
    dt_values = data[dt_col].to_numpy().astype(np.float64)
    rebuilt_time = state.cumulative_time + np.cumsum(dt_values) - dt_values
    state.cumulative_time += float(dt_values.sum())
    return data.with_columns(pl.Series(str(time_col), rebuilt_time))


def apply_checks_full_task(
    data: pl.LazyFrame,
    group_by: tuple[MappingSpec, ...],
    checks: tuple[CheckSpec, ...],
    *,
    annotate: bool = True,
) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
    """Apply checks to a full lazy normalization task.

    Args:
        data: Task frame.
        group_by: Columns that define independent time/domain groups.
        checks: Checks to run in order.
        annotate: When `True`, write annotation columns. When `False`, leave the
            frame unchanged and return detected violations.

    Returns:
        Updated frame and unique `(column, reason)` violations.
    """
    violations: list[CheckViolation] = []
    for check in checks:
        data, check_violations = check.apply_full(data, group_by, annotate=annotate)
        violations.extend(check_violations)
    return data, tuple(dict.fromkeys(violations))


def apply_checks_bounded_chunk(
    data: pl.DataFrame,
    checks: tuple[CheckSpec, ...],
    states: tuple[CheckState, ...],
    *,
    annotate: bool = True,
) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
    """Apply checks to one bounded normalization chunk.

    Args:
        data: Chunk frame.
        checks: Checks to run in order.
        states: Mutable per-check state created with `CheckSpec.init_state`.
        annotate: When `True`, write annotation columns. When `False`, return
            detected violations for the chunk.

    Returns:
        Updated chunk and unique `(column, reason)` violations.
    """
    violations: list[CheckViolation] = []
    for check, state in zip(checks, states, strict=True):
        data, check_violations = check.apply_chunk(data, state, annotate=annotate)
        violations.extend(check_violations)
        if data.height == 0:
            return data, tuple(dict.fromkeys(violations))
    return data, tuple(dict.fromkeys(violations))


def _apply_or_validate_updates(
    data: pl.DataFrame,
    updates: list[tuple[pl.Expr, str, str]],
    *,
    annotate: bool,
) -> tuple[pl.DataFrame, tuple[CheckViolation, ...]]:
    if not updates:
        return data, ()
    if annotate:
        return add_annotations(ensure_annotation_columns(data), updates), ()
    return data, _violations_from_row(data.select(_violation_exprs(updates)).row(0, named=True))


def _apply_or_validate_updates_lazy(
    data: pl.LazyFrame,
    updates: list[tuple[pl.Expr, str, str]],
    *,
    annotate: bool,
) -> tuple[pl.LazyFrame, tuple[CheckViolation, ...]]:
    if not updates:
        return data, ()
    if annotate:
        return add_annotations_lazy(ensure_annotation_columns_lazy(data), updates), ()
    frame = data.select(_violation_exprs(updates)).collect()
    row = frame.row(0, named=True)
    return data, _violations_from_row(row)


def _violation_exprs(updates: list[tuple[pl.Expr, str, str]]) -> list[pl.Expr]:
    return [
        condition.fill_null(value=False).any().alias(f"{column}{ANNOTATION_ITEM_SEPARATOR}{reason}")
        for condition, reason, column in updates
    ]


def _violations_from_row(row: dict[str, object]) -> tuple[CheckViolation, ...]:
    violations = []
    for key, value in row.items():
        if not value:
            continue
        column, reason = key.split(ANNOTATION_ITEM_SEPARATOR, maxsplit=1)
        violations.append((column, reason))
    return tuple(violations)
