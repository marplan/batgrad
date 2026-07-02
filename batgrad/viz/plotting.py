from __future__ import annotations

from typing import TYPE_CHECKING, cast

import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec
from batgrad.viz.widgets.plotly_trace_resampler import PlotlyTraceResampler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import polars as pl

    from batgrad.contracts.mapping import MappingSpec
    from batgrad.data.transforms.resampling import ResamplingSpec
    from batgrad.storage.segments import SegmentSource
    from batgrad.viz.viewport import TraceSource

COLORWAY = tuple(qualitative.Plotly)
DEFAULT_TRACE_MODE = "lines+markers"
DEFAULT_MARKER_SIZE = 6
DEFAULT_MAX_POINTS_PER_TRACE = 1_000
DEFAULT_MAX_POINTS_PER_FIGURE = 100_000
DEFAULT_TIMESERIES_ROW_HEIGHT = 240
DEFAULT_TIMESERIES_MIN_HEIGHT = 350
DEFAULT_EIS_HEIGHT = 650
EIS_NEG_IMAG = "__batgrad_neg_z_imag"
EIS_COLUMNS = (
    BaseColumns.freq,
    BaseColumns.z_real,
    BaseColumns.z_imag,
    BaseColumns.z_mag,
    BaseColumns.z_phase,
)


def colors_by_label(labels: tuple[str, ...]) -> dict[str, str]:
    unique = tuple(dict.fromkeys(labels))
    return {label: COLORWAY[idx % len(COLORWAY)] for idx, label in enumerate(unique)}


def consume_showlegend(label: str, shown_labels: set[str]) -> bool:
    showlegend = label not in shown_labels
    shown_labels.add(label)
    return showlegend


def timeseries_height(rows: int) -> int:
    return max(DEFAULT_TIMESERIES_MIN_HEIGHT, DEFAULT_TIMESERIES_ROW_HEIGHT * rows)


def timeseries_subplot_kwargs(rows: int) -> dict[str, object]:
    return {
        "rows": rows,
        "cols": 1,
        "shared_xaxes": True,
        "vertical_spacing": 0.08,
    }


def eis_subplot_kwargs() -> dict[str, object]:
    return {
        "rows": 2,
        "cols": 2,
        "specs": [[{"rowspan": 2}, {}], [None, {}]],
        "column_widths": [0.5, 0.5],
        "row_heights": [0.5, 0.5],
        "horizontal_spacing": 0.12,
        "vertical_spacing": 0.2,
    }


def make_timeseries_figure(
    y_cols: tuple[str, ...],
    axis_col: str,
    title: str | dict[str, object],
) -> tuple[go.Figure, int]:
    height = timeseries_height(len(y_cols))
    fig = make_subplots(**timeseries_subplot_kwargs(len(y_cols)))
    fig.update_layout(
        height=height,
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        title=title,
    )
    fig.update_xaxes(title_text=str(axis_col), row=len(y_cols), col=1)
    for row_idx, y_col in enumerate(y_cols, start=1):
        fig.update_yaxes(title_text=str(y_col), row=row_idx, col=1)
    return fig, height


def make_eis_figure(title: str | dict[str, object]) -> tuple[go.Figure, int]:
    fig = make_subplots(**eis_subplot_kwargs())
    fig.update_layout(
        height=DEFAULT_EIS_HEIGHT,
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        title=title,
    )
    fig.update_xaxes(title_text=str(BaseColumns.z_real), row=1, col=1)
    fig.update_yaxes(title_text=f"-{BaseColumns.z_imag}", row=1, col=1)
    fig.update_xaxes(title_text=str(BaseColumns.freq), type="log", row=1, col=2)
    fig.update_yaxes(title_text=str(BaseColumns.z_mag), row=1, col=2)
    fig.update_xaxes(title_text=str(BaseColumns.freq), type="log", row=2, col=2)
    fig.update_yaxes(title_text=str(BaseColumns.z_phase), row=2, col=2)
    return fig, DEFAULT_EIS_HEIGHT


def make_trace_resampler(
    fig: go.Figure,
    height: int,
    *,
    max_points_per_trace: int = DEFAULT_MAX_POINTS_PER_TRACE,
    max_points_per_figure: int = DEFAULT_MAX_POINTS_PER_FIGURE,
    max_batch_rows: int | None = 500_000,
) -> PlotlyTraceResampler:
    return PlotlyTraceResampler(
        fig,
        max_points_per_trace=max_points_per_trace,
        max_points_per_figure=max_points_per_figure,
        max_batch_rows=max_batch_rows,
        height=height,
    )


def add_plotly_trace(fig: go.Figure, trace: go.Scattergl, row: int, col: int) -> int:
    trace_idx = len(fig.data)
    fig.add_trace(trace, row=row, col=col)
    return trace_idx


def add_registered_xy_trace(
    fig: go.Figure,
    widget: PlotlyTraceResampler,
    lf: pl.LazyFrame,
    *,
    x_col: str | MappingSpec,
    y_col: str | MappingSpec,
    row: int,
    col: int,
    label: str,
    color: str,
    showlegend: bool,
    hovertemplate: str,
    resampling: ResamplingSpec | None = None,
    legendgroup: str | None = None,
    customdata_cols: tuple[str | MappingSpec, ...] = (),
    segment_source: SegmentSource | None = None,
    extra_exprs: tuple[pl.Expr, ...] = (),
    row_count: int | None = None,
    chunk_iter: Callable[
        [TraceSource, tuple[float, float] | None, tuple[float, float] | None, int, str],
        Iterator[pl.DataFrame],
    ]
    | None = None,
    marker: dict[str, object] | None = None,
    mode: str = DEFAULT_TRACE_MODE,
) -> int:
    trace_idx = add_plotly_trace(
        fig,
        go.Scattergl(
            name=label,
            legendgroup=legendgroup or label,
            showlegend=showlegend,
            line={"color": color},
            marker=marker or {"color": color, "size": DEFAULT_MARKER_SIZE},
            mode=mode,
            hovertemplate=hovertemplate,
        ),
        row,
        col,
    )
    widget.register_trace(
        trace_idx,
        lf,
        x_col,
        y_col,
        resampling
        or MinMaxLTTBResamplingSpec(
            x_col=cast("MappingSpec", x_col),
            y_col=cast("MappingSpec", y_col),
            points=DEFAULT_MAX_POINTS_PER_TRACE,
        ),
        customdata_cols=customdata_cols,
        segment_source=segment_source,
        extra_exprs=extra_exprs,
        row_count=row_count,
        chunk_iter=chunk_iter,
    )
    return trace_idx


def axis_hovertemplate(
    title: str,
    x_col: str | MappingSpec,
    y_col: str | MappingSpec,
    customdata_cols: tuple[str | MappingSpec, ...] = (),
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    custom_labels: tuple[str, ...] = (),
) -> str:
    rows = [
        f"<b>{title}</b>",
        f"{x_label or x_col}: %{{x}}",
        f"{y_label or y_col}: %{{y}}",
    ]
    for idx, column in enumerate(customdata_cols):
        label = custom_labels[idx] if idx < len(custom_labels) else str(column)
        rows.append(f"{label}: %{{customdata[{idx}]}}")
    return "<br>".join(rows) + "<extra></extra>"


def annotation_hovertemplate(title: str) -> str:
    return f"<b>{title}</b><extra></extra>"
