from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import polars as pl
from tsdownsample import MinMaxLTTBDownsampler

from batgrad.contracts.mapping import BaseColumns, MappingSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

DOWNSAMPLE_OVERSAMPLE_FACTOR = 4
TINY_DOWNSAMPLE_BUDGET = 2


@dataclass(frozen=True)
class MinMaxLTTBResamplingSpec:
    x_col: MappingSpec
    y_col: MappingSpec
    points: int | None = None
    points_ratio: float | None = None
    min_points: int = 3

    method: str = "min_max_lttb"

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return (self.x_col, self.y_col)

    def metadata_values(self) -> tuple[str, str]:
        params = {
            "x_col": str(self.x_col),
            "y_col": str(self.y_col),
            "points": self.points,
            "points_ratio": self.points_ratio,
            "min_points": self.min_points,
        }
        return self.method, json.dumps(params, sort_keys=True, separators=(",", ":"))

    def apply_full(
        self,
        data: pl.DataFrame,
        *,
        apply_physics_compensation: bool = True,
    ) -> pl.DataFrame:
        _ = apply_physics_compensation
        return self._downsample_frame(data, resolve_min_max_lttb_budget(self, data.height))

    def apply_bounded(
        self,
        chunks: Callable[[], Iterator[pl.DataFrame]],
        *,
        row_count: int,
        max_batch_rows: int,
        row_id_col: str,
        apply_physics_compensation: bool = True,
    ) -> Iterator[pl.DataFrame]:
        budget = resolve_min_max_lttb_budget(self, row_count)
        if row_count <= budget:
            for chunk in chunks():
                yield chunk.drop(row_id_col)
            return

        selected_ids = self._select_row_ids(
            chunks,
            row_id_col=row_id_col,
            row_count=row_count,
            budget=budget,
            max_batch_rows=max_batch_rows,
        )
        dt_sums = rebuilt_time = averaged_signal = None
        signal_col = self._resolve_physics_signal_col(chunks())
        if signal_col is not None and apply_physics_compensation:
            dt_sums, rebuilt_time, averaged_signal = self._compute_physics_arrays(
                chunks(),
                selected_ids,
                row_id_col=row_id_col,
                row_count=row_count,
                signal_col=signal_col,
            )
        yield from self._selected_rows(
            chunks(),
            selected_ids,
            row_id_col=row_id_col,
            dt_sums=dt_sums,
            rebuilt_time=rebuilt_time,
            averaged_signal=averaged_signal,
            signal_col=signal_col,
        )

    def _downsample_frame(self, data: pl.DataFrame, budget: int) -> pl.DataFrame:
        if data.height <= budget:
            return data
        if self.x_col not in data.columns or self.y_col not in data.columns:
            raise ValueError(
                f"MinMaxLTTB requires columns {self.x_col!r} and {self.y_col!r}; "
                f"available columns: {sorted(data.columns)}",
            )
        if budget <= TINY_DOWNSAMPLE_BUDGET:
            return downsample_tiny_budget_frame(data, budget)
        x = data[self.x_col].to_numpy().astype(np.float64)
        y = data[self.y_col].to_numpy().astype(np.float64)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if not finite_mask.all():
            valid_idx = np.where(finite_mask)[0]
            if len(valid_idx) <= budget:
                return data.filter(pl.Series(finite_mask))
            indices = MinMaxLTTBDownsampler().downsample(
                x[finite_mask], y[finite_mask], n_out=budget
            )
            return data[np.sort(valid_idx[indices]).tolist()]
        indices = MinMaxLTTBDownsampler().downsample(x, y, n_out=budget)
        return data[np.sort(indices).tolist()]

    def _select_row_ids(
        self,
        chunks: Callable[[], Iterator[pl.DataFrame]],
        *,
        row_id_col: str,
        row_count: int,
        budget: int,
        max_batch_rows: int,
    ) -> np.ndarray:
        chunk_count = math.ceil(row_count / max_batch_rows)
        chunk_budget = max(
            self.min_points,
            math.ceil((budget * DOWNSAMPLE_OVERSAMPLE_FACTOR) / chunk_count),
        )
        sample_columns = tuple(dict.fromkeys((row_id_col, self.x_col, self.y_col)))
        sampled_chunks = [
            self._downsample_frame(chunk.select(sample_columns), chunk_budget)
            for chunk in chunks()
        ]
        if not sampled_chunks:
            return np.array([], dtype=np.int64)
        candidates = pl.concat(sampled_chunks, how="vertical")
        selected = self._downsample_frame(candidates, budget).sort(row_id_col)
        return selected[row_id_col].to_numpy().astype(np.int64)

    @staticmethod
    def _selected_rows(
        chunks: Iterator[pl.DataFrame],
        selected_ids: np.ndarray,
        *,
        row_id_col: str,
        dt_sums: np.ndarray | None = None,
        rebuilt_time: np.ndarray | None = None,
        averaged_signal: np.ndarray | None = None,
        signal_col: MappingSpec | None = None,
    ) -> Iterator[pl.DataFrame]:
        for chunk in chunks:
            chunk_row_ids = chunk[row_id_col].to_numpy().astype(np.int64)
            start = int(np.searchsorted(selected_ids, int(chunk_row_ids[0]), side="left"))
            end = int(np.searchsorted(selected_ids, int(chunk_row_ids[-1]) + 1, side="left"))
            if end <= start:
                continue
            selected = chunk[(selected_ids[start:end] - int(chunk_row_ids[0])).tolist()]
            if (
                dt_sums is not None
                and rebuilt_time is not None
                and averaged_signal is not None
                and signal_col is not None
            ):
                selected = selected.with_columns(
                    pl.Series(BaseColumns.dt, dt_sums[start:end]),
                    pl.Series(BaseColumns.time, rebuilt_time[start:end]),
                    pl.Series(signal_col, averaged_signal[start:end]),
                )
            yield selected.drop(row_id_col)

    @staticmethod
    def _compute_physics_arrays(
        chunks: Iterator[pl.DataFrame],
        selected_ids: np.ndarray,
        *,
        row_id_col: str,
        row_count: int,
        signal_col: MappingSpec,
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
            dt = chunk[BaseColumns.dt].to_numpy().astype(np.float64)
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

    @staticmethod
    def _resolve_physics_signal_col(chunks: Iterator[pl.DataFrame]) -> MappingSpec | None:
        for chunk in chunks:
            names = set(chunk.columns)
            if BaseColumns.crate in names:
                return BaseColumns.crate
            if BaseColumns.curr in names:
                return BaseColumns.curr
            return None
        return None


@dataclass(frozen=True)
class LinearResamplingSpec:
    x_col: MappingSpec
    points: int
    scale: Literal["linear", "log"] = "linear"
    value_range: tuple[float, float] | None = None

    method: str = "linear"

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return (self.x_col,)

    def metadata_values(self) -> tuple[str, str]:
        params = {
            "x_col": str(self.x_col),
            "points": self.points,
            "scale": self.scale,
            "value_range": self.value_range,
        }
        return self.method, json.dumps(params, sort_keys=True, separators=(",", ":"))

    def apply_full(
        self,
        data: pl.DataFrame,
        *,
        apply_physics_compensation: bool = True,
    ) -> pl.DataFrame:
        _ = apply_physics_compensation
        return resample_linear_frame(data, self)

    def apply_bounded(
        self,
        chunks: Callable[[], Iterator[pl.DataFrame]],
        *,
        row_count: int,
        max_batch_rows: int,
        row_id_col: str,
        apply_physics_compensation: bool = True,
    ) -> Iterator[pl.DataFrame]:
        _ = chunks, row_count, max_batch_rows, row_id_col, apply_physics_compensation
        raise NotImplementedError("LinearResamplingSpec does not support bounded resampling")


class ResamplingSpec(Protocol):
    @property
    def input_columns(self) -> tuple[MappingSpec, ...]: ...

    def metadata_values(self) -> tuple[str, str]: ...

    def apply_full(
        self,
        data: pl.DataFrame,
        *,
        apply_physics_compensation: bool = True,
    ) -> pl.DataFrame: ...

    def apply_bounded(
        self,
        chunks: Callable[[], Iterator[pl.DataFrame]],
        *,
        row_count: int,
        max_batch_rows: int,
        row_id_col: str,
        apply_physics_compensation: bool = True,
    ) -> Iterator[pl.DataFrame]: ...


class MinMaxLTTBLikeSpec(ResamplingSpec, Protocol):
    x_col: MappingSpec
    y_col: MappingSpec
    points: int | None
    points_ratio: float | None
    min_points: int


def run_resampling(
    data: pl.DataFrame,
    spec: ResamplingSpec,
    *,
    apply_physics_compensation: bool = True,
) -> pl.DataFrame:
    return spec.apply_full(data, apply_physics_compensation=apply_physics_compensation)


def resampling_metadata_values(spec: ResamplingSpec | None) -> tuple[str, str]:
    if spec is None:
        return "none", "{}"
    return spec.metadata_values()


def resolve_min_max_lttb_budget(spec: MinMaxLTTBLikeSpec, row_count: int) -> int:
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


def _linear_source_frame(data: pl.DataFrame, x_col: MappingSpec) -> pl.DataFrame:
    source = (
        data.with_columns(pl.col(x_col).cast(pl.Float64).alias(x_col))
        .filter(pl.col(x_col).is_finite())
        .sort(x_col)
    )
    return source.group_by(x_col, maintain_order=True).last() if source.height else source


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
                f"log scale linear resampling requires positive range, got ({x_min}, {x_max})"
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
        out[left] = y[0] + ((y[1] - y[0]) / left_span) * (x_target[left] - x[0])
    right = x_target > x[-1]
    if right.any():
        right_span = x[-1] - x[-2] or 1.0
        out[right] = y[-1] + ((y[-1] - y[-2]) / right_span) * (x_target[right] - x[-1])
    return out


def _nearest_values(
    x_target: np.ndarray, x_source: np.ndarray, values: list[object]
) -> list[object]:
    positions = np.searchsorted(x_source, x_target, side="left")
    out = []
    for target, pos in zip(x_target, positions, strict=True):
        if pos <= 0:
            out.append(values[0])
        elif pos >= x_source.size:
            out.append(values[-1])
        else:
            left = pos - 1
            right = pos
            out.append(
                values[left]
                if target - x_source[left] <= x_source[right] - target
                else values[right]
            )
    return out
