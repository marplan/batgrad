from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anywidget
import polars as pl
import traitlets

from batgrad.data.processing.io import collect_frame
from batgrad.viz.viewport import (
    AnnotationSource,
    TraceSample,
    TraceSource,
    sample_annotation_viewport,
    sample_trace_viewport,
    viewport_expr,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import plotly.graph_objects as go

    from batgrad.contracts.mapping import MappingSpec
    from batgrad.data.processing.io import SegmentSource
    from batgrad.data.transforms.resampling import ResamplingSpec


_ESM = Path(__file__).with_suffix(".js")
_CSS = Path(__file__).with_suffix(".css")
_AXIS_RANGE_VALUES = 2


class PlotlyTraceResampler(anywidget.AnyWidget):
    _fig_json = traitlets.Dict(default_value={}).tag(sync=True)
    _update = traitlets.Dict(default_value={}).tag(sync=True)
    _evt = traitlets.Dict(default_value={}).tag(sync=True)
    selection = traitlets.Dict(default_value={}).tag(sync=True)
    _status = traitlets.Unicode(default_value="Ready").tag(sync=True)
    _height = traitlets.Int(default_value=600).tag(sync=True)

    _esm = _ESM
    _css = _CSS

    def __init__(
        self,
        fig: go.Figure,
        *,
        max_points_per_trace: int = 1_000,
        max_points_per_figure: int = 100_000,
        max_batch_rows: int | None = 500_000,
        height: int | None = None,
    ) -> None:
        super().__init__()
        if max_points_per_trace < 1:
            raise ValueError(f"max_points_per_trace must be >= 1, got {max_points_per_trace}")
        if max_points_per_figure < 1:
            raise ValueError(f"max_points_per_figure must be >= 1, got {max_points_per_figure}")
        if max_batch_rows is not None and max_batch_rows < 1:
            raise ValueError(f"max_batch_rows must be >= 1 or None, got {max_batch_rows}")

        self._fig = fig
        self._sources: dict[int, TraceSource] = {}
        self._inspection_sources: dict[int, TraceSource] = {}
        self._annotation_sources: dict[int, AnnotationSource] = {}
        self._trace_axes: dict[int, tuple[str, str]] = {}
        self._initial_samples: dict[int, TraceSample] = {}
        self._initial_annotation_samples: dict[int, TraceSample] = {}
        self._max_points_per_trace = int(max_points_per_trace)
        self._max_points_per_figure = int(max_points_per_figure)
        self._max_batch_rows = max_batch_rows
        layout_height = getattr(fig.layout, "height", None)
        self._height = int(height or layout_height or 600)
        self.observe(self._on_evt, names=["_evt"])

    def register_trace(
        self,
        trace_idx: int,
        lf: pl.LazyFrame,
        x_col: MappingSpec,
        y_col: MappingSpec,
        resampling: ResamplingSpec,
        *,
        customdata_cols: tuple[MappingSpec, ...] = (),
        segment_source: SegmentSource | None = None,
        extra_exprs: tuple[pl.Expr, ...] = (),
        row_count: int | None = None,
        chunk_iter: Callable[
            [TraceSource, tuple[float, float] | None, tuple[float, float] | None, int, str],
            Iterator[pl.DataFrame],
        ]
        | None = None,
        inspection_lf: pl.LazyFrame | None = None,
        inspection_segment_source: SegmentSource | None = None,
        inspection_extra_exprs: tuple[pl.Expr, ...] | None = None,
        inspection_row_count: int | None = None,
        inspection_chunk_iter: Callable[
            [TraceSource, tuple[float, float] | None, tuple[float, float] | None, int, str],
            Iterator[pl.DataFrame],
        ]
        | None = None,
    ) -> None:
        if trace_idx < 0 or trace_idx >= len(self._fig.data):
            raise IndexError(f"trace_idx {trace_idx} is outside figure trace range")
        self._sources[trace_idx] = TraceSource(
            trace_idx=trace_idx,
            lf=lf,
            x_col=x_col,
            y_col=y_col,
            resampling=resampling,
            customdata_cols=customdata_cols,
            segment_source=segment_source,
            extra_exprs=extra_exprs,
            row_count=row_count,
            chunk_iter=chunk_iter,
        )
        self._inspection_sources[trace_idx] = TraceSource(
            trace_idx=trace_idx,
            lf=lf if inspection_lf is None else inspection_lf,
            x_col=x_col,
            y_col=y_col,
            resampling=resampling,
            customdata_cols=customdata_cols,
            segment_source=(
                segment_source
                if inspection_segment_source is None
                else inspection_segment_source
            ),
            extra_exprs=(
                extra_exprs if inspection_extra_exprs is None else inspection_extra_exprs
            ),
            row_count=row_count if inspection_row_count is None else inspection_row_count,
            chunk_iter=chunk_iter if inspection_chunk_iter is None else inspection_chunk_iter,
        )
        self._trace_axes[trace_idx] = _trace_axis_keys(self._fig.data[trace_idx].to_plotly_json())

    def register_annotation_trace(
        self,
        trace_idx: int,
        parent_trace_idx: int,
        lf: pl.LazyFrame,
        x_col: MappingSpec,
        y_col: MappingSpec,
        *,
        annotation_columns: tuple[str, ...],
        annotation_reason: str,
        segment_source: SegmentSource | None = None,
        extra_exprs: tuple[pl.Expr, ...] = (),
    ) -> None:
        if trace_idx < 0 or trace_idx >= len(self._fig.data):
            raise IndexError(f"trace_idx {trace_idx} is outside figure trace range")
        if parent_trace_idx not in self._sources:
            raise ValueError(f"parent_trace_idx {parent_trace_idx} is not registered")
        self._annotation_sources[trace_idx] = AnnotationSource(
            trace_idx=trace_idx,
            parent_trace_idx=parent_trace_idx,
            lf=lf,
            x_col=x_col,
            y_col=y_col,
            annotation_columns=annotation_columns,
            annotation_reason=annotation_reason,
            segment_source=segment_source,
            extra_exprs=extra_exprs,
        )
        self._trace_axes[trace_idx] = _trace_axis_keys(self._fig.data[trace_idx].to_plotly_json())

    def show(self) -> PlotlyTraceResampler:
        fig_json = self._fig.to_plotly_json()
        samples = self._sample_traces(sorted(self._sources), axes={})
        annotation_samples = self._sample_annotations(sorted(self._annotation_sources), axes={})
        self._initial_samples = {sample.trace_idx: sample for sample in samples}
        self._initial_annotation_samples = {
            sample.trace_idx: sample for sample in annotation_samples
        }
        for sample in (*samples, *annotation_samples):
            trace = fig_json["data"][sample.trace_idx]
            trace["x"] = sample.x or [None]
            trace["y"] = sample.y or [None]
            if sample.customdata is not None:
                trace["customdata"] = sample.customdata
        self._fig_json = fig_json
        self._status = _status_text(samples)
        self._update = {"updates": [], "status": self._status, "_rid": 0}
        self.selection = {}
        return self

    def selected_data(  # noqa: C901, PLR0912
        self,
        *,
        selection: Mapping[str, object] | None = None,
        widget_index: int | None = None,
        offset: int = 0,
        limit: int = 100_000,
    ) -> tuple[pl.DataFrame, int]:
        del widget_index
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        selection_payload = self.selection if selection is None else selection
        selected_traces = selection_payload.get("traces")
        if not isinstance(selected_traces, list):
            return pl.DataFrame(), 0

        frames = []
        total_rows = 0
        remaining = limit
        for selected_trace in selected_traces:
            if remaining <= 0:
                break
            if not isinstance(selected_trace, Mapping):
                continue
            selected_trace_map = cast("Mapping[str, object]", selected_trace)
            try:
                trace_idx = int(cast("Any", selected_trace_map.get("trace_idx")))
            except (TypeError, ValueError):
                continue
            source = self._inspection_sources.get(trace_idx, self._sources.get(trace_idx))
            if source is None:
                continue

            lf = source.lf.with_row_index("data index")
            if source.extra_exprs:
                lf = lf.with_columns(source.extra_exprs)
            expr = viewport_expr(
                source.x_col,
                source.y_col,
                _numeric_range(selected_trace_map.get("x_range")),
                _numeric_range(selected_trace_map.get("y_range")),
            )
            if expr is not None:
                lf = lf.filter(expr)

            columns = list(
                dict.fromkeys((source.x_col, source.y_col, *source.customdata_cols))
            )
            lf = lf.select(
                pl.lit(_figure_title(self._fig)).alias("widget title"),
                pl.lit(_selected_trace_label(selected_trace_map, trace_idx)).alias("trace"),
                pl.lit(str(source.x_col)).alias("x column"),
                pl.lit(str(source.y_col)).alias("y column"),
                "data index",
                *columns,
            )
            trace_rows = int(collect_frame(lf.select(pl.len().alias("__n")))["__n"].item())
            trace_offset = max(0, offset - total_rows)
            if trace_offset < trace_rows:
                frame = collect_frame(lf.slice(trace_offset, remaining))
                if frame.height:
                    frames.append(frame)
                    remaining -= frame.height
            total_rows += trace_rows

        if not frames:
            return pl.DataFrame(), total_rows
        return pl.concat(frames, how="diagonal_relaxed"), total_rows

    def _on_evt(self, change: dict[str, Any]) -> None:
        evt = change.get("new")
        if not isinstance(evt, dict):
            return
        visible = _dict_payload(evt.get("visible"))
        indices = [idx for idx in sorted(self._sources) if bool(visible.get(str(idx), True))]
        annotation_indices = [
            idx
            for idx, source in sorted(self._annotation_sources.items())
            if bool(visible.get(str(source.parent_trace_idx), True))
        ]
        axes = _dict_payload(evt.get("axes"))
        samples = self._sample_traces(indices, axes=axes)
        annotation_samples = self._sample_annotations(annotation_indices, axes=axes)
        status = _status_text(samples)
        self._status = status
        self._update = {
            "updates": [_sample_payload(sample) for sample in (*samples, *annotation_samples)],
            "status": status,
            "_rid": int(evt.get("_rid") or 0),
        }

    def _sample_traces(
        self,
        trace_indices: list[int],
        *,
        axes: dict[str, object],
    ) -> list[TraceSample]:
        if not trace_indices:
            return []
        if not axes and all(idx in self._initial_samples for idx in trace_indices):
            return [self._initial_samples[idx] for idx in trace_indices]
        budget = max(
            1,
            min(self._max_points_per_trace, self._max_points_per_figure // len(trace_indices)),
        )
        samples: list[TraceSample] = []
        for idx in trace_indices:
            x_axis, y_axis = self._trace_axes[idx]
            samples.append(
                sample_trace_viewport(
                    self._sources[idx],
                    x_range=_axis_range(axes.get(x_axis)),
                    y_range=_axis_range(axes.get(y_axis)),
                    budget=budget,
                    max_batch_rows=self._max_batch_rows,
                )
            )
        return samples

    def _sample_annotations(
        self,
        trace_indices: list[int],
        *,
        axes: dict[str, object],
    ) -> list[TraceSample]:
        if not trace_indices:
            return []
        if not axes and all(idx in self._initial_annotation_samples for idx in trace_indices):
            return [self._initial_annotation_samples[idx] for idx in trace_indices]
        samples: list[TraceSample] = []
        for idx in trace_indices:
            x_axis, y_axis = self._trace_axes[idx]
            samples.append(
                sample_annotation_viewport(
                    self._annotation_sources[idx],
                    x_range=_axis_range(axes.get(x_axis)),
                    y_range=_axis_range(axes.get(y_axis)),
                    max_batch_rows=self._max_batch_rows,
                )
            )
        return samples


def _trace_axis_keys(trace: dict[str, object]) -> tuple[str, str]:
    return _layout_axis_key(str(trace.get("xaxis", "x")), "x"), _layout_axis_key(
        str(trace.get("yaxis", "y")), "y"
    )


def _dict_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _layout_axis_key(axis_ref: str, prefix: str) -> str:
    if axis_ref == prefix:
        return f"{prefix}axis"
    return f"{prefix}axis{axis_ref.removeprefix(prefix)}"


def _axis_range(axis: object) -> tuple[float, float] | None:
    if not isinstance(axis, Mapping):
        return None
    axis_map: Mapping[Any, Any] = axis
    if axis_map.get("autorange") is True:
        return None
    values = axis_map.get("range")
    if not isinstance(values, list | tuple) or len(values) < _AXIS_RANGE_VALUES:
        return None
    try:
        low, high = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    if axis_map.get("type") == "log":
        return math.pow(10.0, low), math.pow(10.0, high)
    return low, high


def _numeric_range(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) < _AXIS_RANGE_VALUES:
        return None
    values = cast("tuple[Any, ...] | list[Any]", value)
    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None


def _figure_title(fig: go.Figure) -> str:
    title = getattr(getattr(fig, "layout", None), "title", None)
    text = getattr(title, "text", None)
    return str(text) if text else ""


def _selected_trace_label(selected_trace: Mapping[str, object], trace_idx: int) -> str:
    name = str(selected_trace.get("name") or trace_idx)
    legendgroup = str(selected_trace.get("legendgroup") or "")
    if name == "ingested" and legendgroup:
        return f"ingested | {legendgroup}"
    return name


def _sample_payload(sample: TraceSample) -> dict[str, object]:
    payload: dict[str, object] = {
        "trace_idx": sample.trace_idx,
        "x": sample.x,
        "y": sample.y,
    }
    if sample.customdata is not None:
        payload["customdata"] = sample.customdata
    return payload


def _status_text(samples: list[TraceSample]) -> str:
    if not samples:
        return "No visible traces"
    total = sum(sample.shown_points for sample in samples)
    if total == 0:
        return "No points in view"
    if any(sample.downsampled for sample in samples):
        budget = min(sample.budget for sample in samples)
        return f"Downsampled: {budget:,} pts/trace"
    return "Raw"
