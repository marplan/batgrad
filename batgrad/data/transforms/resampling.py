from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import numpy as np
import polars as pl
from tsdownsample import MinMaxLTTBDownsampler

from batgrad.contracts.columns import BatteryColumns

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batgrad.contracts.columns import ColumnSpec


DOWNSAMPLE_OVERSAMPLE_FACTOR = 4
TINY_DOWNSAMPLE_BUDGET = 2


class ResamplingSpecBase:
    method: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "method" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define class variable 'method'")


type ResamplingHandler = Callable[[pl.DataFrame, ResamplingSpecBase], pl.DataFrame]
RESAMPLING_REGISTRY: dict[type[ResamplingSpecBase], ResamplingHandler] = {}


def register_resampling[SpecT: ResamplingSpecBase](
    spec_type: type[SpecT],
) -> Callable[
    [Callable[[pl.DataFrame, SpecT], pl.DataFrame]],
    Callable[[pl.DataFrame, SpecT], pl.DataFrame],
]:
    def decorator(
        fn: Callable[[pl.DataFrame, SpecT], pl.DataFrame],
    ) -> Callable[[pl.DataFrame, SpecT], pl.DataFrame]:
        if spec_type in RESAMPLING_REGISTRY:
            raise ValueError(f"Resampling {spec_type.__name__} is already registered")
        if any(registered.method == spec_type.method for registered in RESAMPLING_REGISTRY):
            raise ValueError(f"Resampling method {spec_type.method!r} is already registered")

        RESAMPLING_REGISTRY[spec_type] = cast("ResamplingHandler", fn)
        return fn

    return decorator


@dataclass(frozen=True, slots=True)
class MinMaxLTTBResamplingSpec(ResamplingSpecBase):
    x_col: ColumnSpec
    y_col: ColumnSpec
    points: int | None = None
    points_ratio: float | None = None
    min_points: int = 3

    method: ClassVar[str] = "min_max_lttb"


@dataclass(frozen=True, slots=True)
class LinearResamplingSpec(ResamplingSpecBase):
    x_col: ColumnSpec
    points: int
    scale: Literal["linear", "log"] = "linear"
    value_range: tuple[float, float] | None = None

    method: ClassVar[str] = "linear"


@register_resampling(MinMaxLTTBResamplingSpec)
def min_max_lttb_resampling(
    data: pl.DataFrame,
    resampling_spec: MinMaxLTTBResamplingSpec,
) -> pl.DataFrame:
    budget = resolve_min_max_lttb_budget(resampling_spec, data.height)
    return downsample_min_max_lttb_frame(data, budget, resampling_spec)


@register_resampling(LinearResamplingSpec)
def linear_resampling(
    data: pl.DataFrame,
    resampling_spec: LinearResamplingSpec,
) -> pl.DataFrame:
    return resample_linear_frame(data, resampling_spec)


def run_resampling(data: pl.DataFrame, resampling_spec: ResamplingSpecBase) -> pl.DataFrame:
    handler = RESAMPLING_REGISTRY.get(type(resampling_spec))
    if handler is None:
        raise ValueError(f"No handler registered for resampling {type(resampling_spec).__name__}")
    return handler(data, resampling_spec)


def resampling_metadata_values(resampling_spec: ResamplingSpecBase | None) -> tuple[str, str]:
    if resampling_spec is None:
        return "none", "{}"
    if isinstance(resampling_spec, MinMaxLTTBResamplingSpec):
        params = {
            "x_col": str(resampling_spec.x_col),
            "y_col": str(resampling_spec.y_col),
            "points": resampling_spec.points,
            "points_ratio": resampling_spec.points_ratio,
            "min_points": resampling_spec.min_points,
        }
    elif isinstance(resampling_spec, LinearResamplingSpec):
        params = {
            "x_col": str(resampling_spec.x_col),
            "points": resampling_spec.points,
            "scale": resampling_spec.scale,
            "value_range": resampling_spec.value_range,
        }
    else:
        raise TypeError(f"Unsupported resampling spec {type(resampling_spec).__name__}")
    return resampling_spec.method, json.dumps(params, sort_keys=True, separators=(",", ":"))


def resolve_min_max_lttb_budget(
    spec: MinMaxLTTBResamplingSpec,
    row_count: int,
) -> int:
    if spec.min_points < 1:
        raise ValueError(f"min_points must be >= 1, got {spec.min_points}")
    if spec.points is not None:
        if spec.points < 1:
            raise ValueError(f"points must be >= 1, got {spec.points}")
        budget = spec.points
    elif spec.points_ratio is not None:
        if spec.points_ratio <= 0.0:
            raise ValueError(f"points_ratio must be > 0, got {spec.points_ratio}")
        budget = math.ceil(row_count * spec.points_ratio)
    else:
        raise ValueError("MinMaxLTTB resampling requires points or points_ratio")
    return min(row_count, max(spec.min_points, budget))


def downsample_min_max_lttb_frame(
    data: pl.DataFrame,
    budget: int,
    spec: MinMaxLTTBResamplingSpec,
) -> pl.DataFrame:
    if data.height <= budget:
        return data
    if spec.x_col not in data.columns or spec.y_col not in data.columns:
        raise ValueError(
            f"MinMaxLTTB requires columns {spec.x_col!r} and {spec.y_col!r}; "
            f"available columns: {sorted(data.columns)}",
        )
    if budget <= TINY_DOWNSAMPLE_BUDGET:
        return downsample_tiny_budget_frame(data, budget)

    x = data[spec.x_col].to_numpy().astype(np.float64)
    y = data[spec.y_col].to_numpy().astype(np.float64)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not finite_mask.all():
        valid_idx = np.where(finite_mask)[0]
        if len(valid_idx) <= budget:
            return data.filter(pl.Series(finite_mask))
        indices = MinMaxLTTBDownsampler().downsample(
            x[finite_mask],
            y[finite_mask],
            n_out=budget,
        )
        resolved = np.sort(valid_idx[indices]).tolist()
        return data[resolved]

    indices = MinMaxLTTBDownsampler().downsample(x, y, n_out=budget)
    return data[np.sort(indices).tolist()]


def downsample_tiny_budget_frame(data: pl.DataFrame, budget: int) -> pl.DataFrame:
    if budget <= 0:
        return data.limit(0)
    if budget == 1:
        return data[[0]]
    return data[[0, data.height - 1]]


def resample_linear_frame(data: pl.DataFrame, spec: LinearResamplingSpec) -> pl.DataFrame:
    if spec.points < 1:
        raise ValueError(f"points must be >= 1, got {spec.points}")
    if spec.scale not in {"linear", "log"}:
        raise ValueError(f"scale must be 'linear' or 'log', got {spec.scale!r}")
    if data.height == 0:
        return data
    if spec.x_col not in data.columns:
        raise ValueError(
            f"Linear resampling requires column {spec.x_col!r}; "
            f"available columns: {sorted(data.columns)}",
        )

    source = _linear_source_frame(data, spec.x_col)
    if source.height == 0:
        return data.limit(0)
    x_source = source[spec.x_col].to_numpy().astype(np.float64)
    x_target = _linear_target_grid(x_source, spec)
    if x_target.size == 0:
        return data.limit(0)

    output: dict[str, object] = {spec.x_col: x_target}
    for column, dtype in source.schema.items():
        if column == spec.x_col:
            continue
        if dtype.is_numeric():
            output[column] = _interp_numeric_with_extrapolation(
                x_target,
                x_source,
                source[column].to_numpy().astype(np.float64),
            )
        else:
            output[column] = _nearest_values(x_target, x_source, source[column].to_list())
    return pl.DataFrame(output).select(source.columns)


def _linear_source_frame(data: pl.DataFrame, x_col: ColumnSpec) -> pl.DataFrame:
    source = (
        data.with_columns(pl.col(x_col).cast(pl.Float64, strict=False).alias(x_col))
        .filter(pl.col(x_col).is_finite())
        .sort(x_col)
    )
    if source.height == 0:
        return source
    return source.group_by(x_col, maintain_order=True).last()


def _linear_target_grid(x_source: np.ndarray, spec: LinearResamplingSpec) -> np.ndarray:
    if spec.value_range is None:
        x_min = float(x_source[0])
        x_max = float(x_source[-1])
    else:
        x_min, x_max = spec.value_range
    if x_min > x_max:
        raise ValueError(f"value_range must satisfy min <= max, got ({x_min}, {x_max})")
    if spec.scale == "log":
        if x_min <= 0.0 or x_max <= 0.0:
            raise ValueError(
                f"log scale linear resampling requires positive range, got ({x_min}, {x_max})",
            )
        return np.logspace(np.log10(x_min), np.log10(x_max), spec.points)
    return np.linspace(x_min, x_max, spec.points)


def _interp_numeric_with_extrapolation(
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(x_source) & np.isfinite(y_source)
    if not valid.any():
        return np.full(x_target.size, np.nan, dtype=np.float64)
    x = x_source[valid]
    y = y_source[valid]
    if x.size == 1:
        return np.full(x_target.size, y[0], dtype=np.float64)

    out = np.interp(x_target, x, y)
    left = x_target < x[0]
    if left.any():
        left_span = x[1] - x[0] or 1.0
        left_slope = (y[1] - y[0]) / left_span
        out[left] = y[0] + left_slope * (x_target[left] - x[0])

    right = x_target > x[-1]
    if right.any():
        right_span = x[-1] - x[-2] or 1.0
        right_slope = (y[-1] - y[-2]) / right_span
        out[right] = y[-1] + right_slope * (x_target[right] - x[-1])
    return out


def _nearest_values(
    x_target: np.ndarray,
    x_source: np.ndarray,
    values: list[object],
) -> list[object]:
    positions = np.searchsorted(x_source, x_target, side="left")
    out: list[object] = []
    for target, pos in zip(x_target, positions, strict=True):
        if pos <= 0:
            out.append(values[0])
            continue
        if pos >= x_source.size:
            out.append(values[-1])
            continue
        left = pos - 1
        right = pos
        out.append(
            values[left] if target - x_source[left] <= x_source[right] - target else values[right],
        )
    return out


def select_min_max_lttb_row_ids(
    chunks: Iterable[pl.DataFrame],
    *,
    row_id_col: str,
    row_count: int,
    budget: int,
    spec: MinMaxLTTBResamplingSpec,
    chunk_rows: int,
) -> np.ndarray:
    chunk_count = math.ceil(row_count / chunk_rows)
    chunk_budget = max(
        spec.min_points,
        math.ceil((budget * DOWNSAMPLE_OVERSAMPLE_FACTOR) / chunk_count),
    )
    slim_columns = [row_id_col, spec.x_col, spec.y_col]
    sampled_chunks = [
        downsample_min_max_lttb_frame(chunk.select(slim_columns), chunk_budget, spec)
        for chunk in chunks
    ]
    if not sampled_chunks:
        return np.array([], dtype=np.int64)
    candidates = pl.concat(sampled_chunks, how="vertical")
    selected = downsample_min_max_lttb_frame(candidates, budget, spec).sort(row_id_col)
    return selected[row_id_col].to_numpy().astype(np.int64)


def compute_physics_arrays_from_chunks(
    chunks: Iterable[pl.DataFrame],
    selected_ids: np.ndarray,
    *,
    row_id_col: str,
    row_count: int,
    signal_col: ColumnSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_ends = np.empty(selected_ids.size, dtype=np.int64)
    if selected_ids.size > 1:
        target_ends[:-1] = selected_ids[1:]
    if selected_ids.size:
        target_ends[-1] = row_count

    dt_sums = np.zeros(selected_ids.size, dtype=np.float64)
    signal_dt_sums = np.zeros(selected_ids.size, dtype=np.float64)
    sample_idx = 0
    for chunk in chunks:
        chunk_row_ids = chunk[row_id_col].to_numpy().astype(np.int64)
        chunk_start = int(chunk_row_ids[0])
        chunk_end = int(chunk_row_ids[-1]) + 1
        dt = chunk[BatteryColumns.dt].to_numpy().astype(np.float64)
        signal = chunk[signal_col].to_numpy().astype(np.float64)
        while sample_idx < selected_ids.size:
            start = int(selected_ids[sample_idx])
            end = int(target_ends[sample_idx])
            if end <= chunk_start:
                sample_idx += 1
                continue
            if start >= chunk_end:
                break
            local_start = max(0, start - chunk_start)
            local_end = min(chunk.height, end - chunk_start)
            if local_end > local_start:
                dt_slice = dt[local_start:local_end]
                signal_slice = signal[local_start:local_end]
                dt_sums[sample_idx] += float(dt_slice.sum())
                signal_dt_sums[sample_idx] += float((signal_slice * dt_slice).sum())
            if end <= chunk_end:
                sample_idx += 1
                continue
            break

    averaged_signal = np.divide(
        signal_dt_sums,
        dt_sums,
        out=np.zeros(selected_ids.size, dtype=np.float64),
        where=dt_sums > 0.0,
    )
    rebuilt_time = np.cumsum(dt_sums) - dt_sums
    return dt_sums, rebuilt_time, averaged_signal


def resolve_downsampling_signal_col(columns: Iterable[str]) -> ColumnSpec | None:
    names = set(columns)
    if BatteryColumns.c_rate in names:
        return BatteryColumns.c_rate
    if BatteryColumns.current in names:
        return BatteryColumns.current
    return None


def apply_physics_preserving_downsampling(
    *,
    source_lf: pl.LazyFrame,
    sampled_lf: pl.LazyFrame,
    row_count: int,
    row_id_col: str,
    dt_col: ColumnSpec,
    time_col: ColumnSpec,
    signal_col: ColumnSpec,
) -> pl.LazyFrame:
    source_schema = set(source_lf.collect_schema().names())
    sampled_schema = set(sampled_lf.collect_schema().names())
    required_source = {row_id_col, dt_col, signal_col}
    missing_source = sorted(required_source - source_schema)
    if missing_source:
        raise ValueError(
            "Cannot apply physics-preserving downsampling: "
            f"source is missing columns {missing_source}",
        )
    if row_id_col not in sampled_schema:
        raise ValueError(
            "Cannot apply physics-preserving downsampling: "
            f"sampled data is missing row id column {row_id_col!r}",
        )

    prefix_lf = source_lf.select(
        pl.col(row_id_col).cast(pl.Int64).alias(row_id_col),
        pl.col(dt_col).cast(pl.Float64).cum_sum().alias("__physics_cum_dt"),
        (pl.col(signal_col).cast(pl.Float64) * pl.col(dt_col).cast(pl.Float64))
        .cum_sum()
        .alias("__physics_cum_signal_dt"),
    )
    sampled_windows_lf = (
        sampled_lf.sort(row_id_col)
        .with_columns(pl.col(row_id_col).cast(pl.Int64).alias(row_id_col))
        .with_columns(pl.col(row_id_col).shift(-1).alias("__physics_next_row_id"))
        .with_columns(
            pl.coalesce([pl.col("__physics_next_row_id"), pl.lit(row_count)])
            .cast(pl.Int64)
            .alias("__physics_end_row_id"),
            pl.when(pl.col("__physics_sample_pos") == 0)
            .then(pl.lit(-1))
            .otherwise(pl.col(row_id_col) - 1)
            .cast(pl.Int64)
            .alias("__physics_start_prev_row_id"),
        )
        .with_columns((pl.col("__physics_end_row_id") - 1).alias("__physics_end_prev_row_id"))
    )
    prefix_end_lf = prefix_lf.rename(
        {
            row_id_col: "__physics_end_prev_row_id",
            "__physics_cum_dt": "__physics_cum_dt_end",
            "__physics_cum_signal_dt": "__physics_cum_signal_dt_end",
        },
    )
    prefix_start_lf = prefix_lf.rename(
        {
            row_id_col: "__physics_start_prev_row_id",
            "__physics_cum_dt": "__physics_cum_dt_start",
            "__physics_cum_signal_dt": "__physics_cum_signal_dt_start",
        },
    )
    return (
        sampled_windows_lf.join(prefix_end_lf, on="__physics_end_prev_row_id", how="left")
        .join(prefix_start_lf, on="__physics_start_prev_row_id", how="left")
        .sort("__physics_sample_pos")
        .with_columns(
            (
                pl.coalesce([pl.col("__physics_cum_dt_end"), pl.lit(0.0)])
                - pl.coalesce([pl.col("__physics_cum_dt_start"), pl.lit(0.0)])
            ).alias("__physics_dt_sum"),
            (
                pl.coalesce([pl.col("__physics_cum_signal_dt_end"), pl.lit(0.0)])
                - pl.coalesce([pl.col("__physics_cum_signal_dt_start"), pl.lit(0.0)])
            ).alias("__physics_signal_dt_sum"),
        )
        .with_columns(
            pl.col("__physics_dt_sum").cast(pl.Float64).alias(dt_col),
            pl.when(pl.col("__physics_dt_sum") > 0.0)
            .then(pl.col("__physics_signal_dt_sum") / pl.col("__physics_dt_sum"))
            .otherwise(pl.col(signal_col).cast(pl.Float64))
            .alias(signal_col),
        )
        .with_columns((pl.col(dt_col).cum_sum() - pl.col(dt_col)).alias(time_col))
        .drop(
            [
                row_id_col,
                "__physics_sample_pos",
                "__physics_next_row_id",
                "__physics_end_row_id",
                "__physics_start_prev_row_id",
                "__physics_end_prev_row_id",
                "__physics_cum_dt_end",
                "__physics_cum_signal_dt_end",
                "__physics_cum_dt_start",
                "__physics_cum_signal_dt_start",
                "__physics_dt_sum",
                "__physics_signal_dt_sum",
            ],
        )
    )
