from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, MappingSpec
from batgrad.data.processing.io import collect_frame, mapping_column_sources
from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec
from batgrad.viz.layouts import eis_subplot_kwargs, timeseries_subplot_kwargs
from batgrad.viz.sources import (
    OverlayEntry as _OverlayEntry,
    RunEntry as _RunEntry,
    entry_row_count as _entry_row_count,
    has_overlay_sources as _has_overlay_sources,
    has_sources as _has_sources,
    iter_entries as _iter_entries,
    matching_overlay_entries as _matching_overlay_entries,
    matching_overlays_for_entry as _matching_overlays_for_entry,
    overlay_chunk_iter as _overlay_chunk_iter,
    overlay_entries as _overlay_entries,
    overlay_lazy_frame as _overlay_lazy_frame,
    overlay_y_source as _overlay_y_source,
    protocol_group_by as _protocol_group_by,
    protocol_output_column as _protocol_output_column,
    source_exprs as _source_exprs,
)
from batgrad.viz.widgets.plotly_trace_resampler import PlotlyTraceResampler

if TYPE_CHECKING:
    from batgrad.data.processing.interactive import InteractiveProtocolSpec, InteractiveStageRun
    from batgrad.data.transforms.resampling import ResamplingSpec


_EIS_NEG_IMAG = "__batgrad_neg_z_imag"
_EIS_COLS = {
    BaseColumns.freq,
    BaseColumns.z_real,
    BaseColumns.z_imag,
    BaseColumns.z_mag,
    BaseColumns.z_phase,
}
_COLORWAY = tuple(qualitative.Plotly)


def make_widgets(
    run: InteractiveStageRun,
    cols: list[MappingSpec] | tuple[MappingSpec, ...],
    *,
    overlay_sources: tuple[DatasetStageId, ...] = (),
    max_points_per_trace: int = 1_000,
    max_points_per_figure: int = 100_000,
    max_batch_rows: int | None = 500_000,
) -> tuple[PlotlyTraceResampler, ...]:
    entries = list(_iter_entries(run))
    overlay_entries = _overlay_entries(run, overlay_sources)
    widgets = []
    for protocol_entries in _entries_by_protocol(entries, run.protocol_order):
        protocol_overlays = _matching_overlay_entries(protocol_entries, overlay_entries)
        if _is_eis(protocol_entries[0].protocol_spec):
            if not _include_eis(cols):
                continue
            widget = make_eis_widget(
                protocol_entries,
                max_points_per_trace=max_points_per_trace,
                max_points_per_figure=max_points_per_figure,
                max_batch_rows=max_batch_rows,
            )
        else:
            widget = make_timeseries_widget(
                protocol_entries,
                overlay_entries=tuple(protocol_overlays),
                cols=cols,
                max_points_per_trace=max_points_per_trace,
                max_points_per_figure=max_points_per_figure,
                max_batch_rows=max_batch_rows,
            )
        if widget is not None:
            widgets.append(widget)

    return tuple(widgets)


def make_timeseries_widget(
    entries: list[_RunEntry],
    *,
    overlay_entries: tuple[_OverlayEntry, ...] = (),
    cols: list[MappingSpec] | tuple[MappingSpec, ...],
    max_points_per_trace: int,
    max_points_per_figure: int,
    max_batch_rows: int | None,
) -> PlotlyTraceResampler | None:
    y_cols = _timeseries_y_cols(entries, cols)
    if not y_cols:
        return None

    fig = make_subplots(**timeseries_subplot_kwargs(len(y_cols)))
    fig.update_layout(
        height=max(350, 240 * len(y_cols)),
        hovermode="closest",
        title=_protocol_title_layout(entries),
    )
    fig.update_xaxes(
        title_text=str(entries[0].protocol_spec.protocol.axis_col), row=len(y_cols), col=1
    )

    widget = PlotlyTraceResampler(
        fig,
        max_points_per_trace=max_points_per_trace,
        max_points_per_figure=max_points_per_figure,
        max_batch_rows=max_batch_rows,
        height=max(350, 240 * len(y_cols)),
    )

    colors = _entry_colors(entries)
    shown_labels: set[str] = set()
    shown_overlay_labels: set[str] = set()
    for entry in entries:
        axis_col = entry.protocol_spec.protocol.axis_col
        if axis_col not in entry.schema:
            continue
        label = _entry_label(entry)
        color = colors[label]
        annotation_messages = _annotation_messages(entry)
        display_label = _entry_display_label(entry, label, y_cols, annotation_messages)
        for row_idx, y_col in enumerate(y_cols, start=1):
            if not _has_sources(entry, y_col):
                continue
            output_y_col = _protocol_output_column(entry.protocol_spec, y_col)
            trace_idx = _add_registered_trace(
                fig,
                widget,
                entry,
                x_col=axis_col,
                y_col=y_col,
                row=row_idx,
                col=1,
                label=display_label,
                legendgroup=label,
                color=color,
                showlegend=_consume_showlegend(label, shown_labels),
                hovertemplate=_axis_hovertemplate(display_label, axis_col, y_col, ()),
                customdata_cols=(),
                resampling=MinMaxLTTBResamplingSpec(
                    x_col=axis_col, y_col=y_col, points=max_points_per_trace
                ),
                marker={"color": color, "size": 6},
            )
            for annotation_reason, annotation_columns in _annotation_overlays_for_y_col(
                annotation_messages, output_y_col
            ):
                annotation_label = _annotation_label("", y_col, annotation_reason).removeprefix(
                    " | "
                )
                _add_annotation_trace(
                    fig,
                    widget,
                    entry,
                    parent_trace_idx=trace_idx,
                    x_col=axis_col,
                    y_col=y_col,
                    row=row_idx,
                    col=1,
                    label=annotation_label,
                    legendgroup=label,
                    annotation_columns=annotation_columns,
                    annotation_reason=annotation_reason,
                    showlegend=False,
                    extra_exprs=_source_exprs(entry, (axis_col, y_col)),
                )
            for overlay in _matching_overlays_for_entry(entry, overlay_entries):
                overlay_source = _overlay_y_source(overlay.entry, output_y_col)
                if overlay_source is None:
                    continue
                hover_y_label = _raw_overlay_y_label(overlay_source, y_col)
                _add_ingested_overlay_trace(
                    fig,
                    widget,
                    overlay.entry,
                    normalized_entry=entry,
                    x_col=axis_col,
                    y_col=y_col,
                    source_y_col=output_y_col,
                    hover_y_label=hover_y_label,
                    row=row_idx,
                    col=1,
                    label="raw",
                    legendgroup=label,
                    showlegend=_consume_showlegend("raw", shown_overlay_labels),
                    max_points_per_trace=max_points_per_trace,
                )
            fig.update_yaxes(title_text=str(y_col), row=row_idx, col=1)
    return widget if fig.data else None


def _add_registered_trace(
    fig: go.Figure,
    widget: PlotlyTraceResampler,
    entry: _RunEntry,
    *,
    x_col: MappingSpec,
    y_col: MappingSpec,
    row: int,
    col: int,
    label: str,
    color: str,
    showlegend: bool,
    hovertemplate: str,
    customdata_cols: tuple[MappingSpec, ...],
    resampling: ResamplingSpec,
    legendgroup: str | None = None,
    marker: dict[str, object] | None = None,
    extra_exprs: tuple[pl.Expr, ...] = (),
) -> int:
    source_exprs = _source_exprs(entry, (x_col, y_col, *customdata_cols))
    trace_idx = _add_plotly_trace(
        fig,
        go.Scattergl(
            name=label,
            legendgroup=legendgroup or label,
            showlegend=showlegend,
            line={"color": color},
            marker=marker or {"color": color},
            mode="lines+markers",
            hovertemplate=hovertemplate,
        ),
        row,
        col,
    )
    widget.register_trace(
        trace_idx,
        entry.source.scan(),
        x_col,
        y_col,
        resampling,
        customdata_cols=customdata_cols,
        segment_source=entry.source,
        extra_exprs=(*source_exprs, *extra_exprs),
        row_count=_entry_row_count(entry),
    )
    return trace_idx


def _add_annotation_trace(
    fig: go.Figure,
    widget: PlotlyTraceResampler,
    entry: _RunEntry,
    *,
    parent_trace_idx: int,
    x_col: MappingSpec,
    y_col: MappingSpec,
    row: int,
    col: int,
    label: str,
    legendgroup: str,
    annotation_columns: tuple[str, ...],
    annotation_reason: str,
    showlegend: bool,
    extra_exprs: tuple[pl.Expr, ...],
) -> None:
    trace_idx = _add_plotly_trace(
        fig,
        go.Scattergl(
            name=label,
            legendgroup=legendgroup,
            showlegend=showlegend,
            marker={"color": "red", "size": 8},
            mode="markers",
            hovertemplate=_annotation_hovertemplate(label),
        ),
        row,
        col,
    )
    widget.register_annotation_trace(
        trace_idx,
        parent_trace_idx,
        entry.source.scan(),
        x_col,
        y_col,
        annotation_columns=annotation_columns,
        annotation_reason=annotation_reason,
        segment_source=entry.source,
        extra_exprs=extra_exprs,
    )


def _add_ingested_overlay_trace(
    fig: go.Figure,
    widget: PlotlyTraceResampler,
    overlay_entry: _RunEntry,
    *,
    normalized_entry: _RunEntry,
    x_col: MappingSpec,
    y_col: MappingSpec,
    source_y_col: MappingSpec,
    hover_y_label: str | None,
    row: int,
    col: int,
    label: str,
    legendgroup: str,
    showlegend: bool,
    max_points_per_trace: int,
) -> None:
    if not _has_overlay_sources(overlay_entry, x_col, source_y_col):
        return
    lf = _overlay_lazy_frame(
        overlay_entry, normalized_entry.protocol_spec, x_col, y_col, source_y_col
    )
    trace_idx = _add_plotly_trace(
        fig,
        go.Scattergl(
            name=label,
            legendgroup=legendgroup,
            showlegend=showlegend,
            marker={"color": "silver", "size": 6, "symbol": "circle-open"},
            mode="markers",
            hovertemplate=_axis_hovertemplate(
                label,
                x_col,
                y_col,
                (),
                y_label=hover_y_label,
            ),
        ),
        row,
        col,
    )
    widget.register_trace(
        trace_idx,
        lf,
        x_col,
        y_col,
        MinMaxLTTBResamplingSpec(x_col=x_col, y_col=y_col, points=max_points_per_trace),
        segment_source=overlay_entry.source,
        row_count=_entry_row_count(overlay_entry),
        chunk_iter=_overlay_chunk_iter(
            overlay_entry, normalized_entry.protocol_spec, x_col, y_col, source_y_col
        ),
    )


def _add_plotly_trace(fig: go.Figure, trace: go.Scattergl, row: int, col: int) -> int:
    trace_idx = len(fig.data)
    fig.add_trace(trace, row=row, col=col)
    return trace_idx


def _consume_showlegend(label: str, shown_labels: set[str]) -> bool:
    showlegend = label not in shown_labels
    shown_labels.add(label)
    return showlegend


def _protocol_title_layout(entries: list[_RunEntry]) -> dict[str, object]:
    return {"text": _protocol_title(entries), "x": 0.5, "xanchor": "center"}


def make_eis_widget(
    entries: list[_RunEntry],
    *,
    max_points_per_trace: int,
    max_points_per_figure: int,
    max_batch_rows: int | None,
) -> PlotlyTraceResampler | None:
    fig = make_subplots(**eis_subplot_kwargs())
    fig.update_layout(
        height=650,
        hovermode="closest",
        title=_protocol_title_layout(entries),
    )
    fig.update_xaxes(title_text=str(BaseColumns.z_real), row=1, col=1)
    fig.update_yaxes(title_text=f"-{BaseColumns.z_imag}", row=1, col=1)
    fig.update_xaxes(title_text=str(BaseColumns.freq), type="log", row=1, col=2)
    fig.update_yaxes(title_text=str(BaseColumns.z_mag), row=1, col=2)
    fig.update_xaxes(title_text=str(BaseColumns.freq), type="log", row=2, col=2)
    fig.update_yaxes(title_text=str(BaseColumns.z_phase), row=2, col=2)

    widget = PlotlyTraceResampler(
        fig,
        max_points_per_trace=max_points_per_trace,
        max_points_per_figure=max_points_per_figure,
        max_batch_rows=max_batch_rows,
        height=650,
    )

    colors = _entry_colors(entries)
    shown_labels: set[str] = set()
    for entry in entries:
        if not _has_eis_columns(entry.schema):
            continue
        label = _entry_label(entry)
        color = colors[label]
        group_cols = _available_group_cols(entry)
        neg_imag_expr = (-pl.col(BaseColumns.z_imag)).alias(_EIS_NEG_IMAG)
        neg_imag_col = MappingSpec(_EIS_NEG_IMAG, dtype=BaseColumns.z_imag.dtype)
        _add_registered_trace(
            fig,
            widget,
            entry,
            x_col=BaseColumns.z_real,
            y_col=neg_imag_col,
            row=1,
            col=1,
            label=label,
            color=color,
            showlegend=_consume_showlegend(label, shown_labels),
            hovertemplate=_axis_hovertemplate(
                label,
                BaseColumns.z_real,
                neg_imag_col,
                (BaseColumns.freq, *group_cols),
                y_label=f"-{BaseColumns.z_imag}",
                custom_labels=("Frequency [Hz]",),
            ),
            customdata_cols=(BaseColumns.freq, *group_cols),
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.z_real, y_col=neg_imag_col, points=max_points_per_trace
            ),
            extra_exprs=(neg_imag_expr,),
        )
        _add_registered_trace(
            fig,
            widget,
            entry,
            x_col=BaseColumns.freq,
            y_col=BaseColumns.z_mag,
            row=1,
            col=2,
            label=label,
            color=color,
            showlegend=False,
            hovertemplate=_axis_hovertemplate(
                label, BaseColumns.freq, BaseColumns.z_mag, group_cols
            ),
            customdata_cols=group_cols,
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.freq, y_col=BaseColumns.z_mag, points=max_points_per_trace
            ),
        )
        _add_registered_trace(
            fig,
            widget,
            entry,
            x_col=BaseColumns.freq,
            y_col=BaseColumns.z_phase,
            row=2,
            col=2,
            label=label,
            color=color,
            showlegend=False,
            hovertemplate=_axis_hovertemplate(
                label, BaseColumns.freq, BaseColumns.z_phase, group_cols
            ),
            customdata_cols=group_cols,
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.freq, y_col=BaseColumns.z_phase, points=max_points_per_trace
            ),
        )
    return widget if fig.data else None


def _raw_overlay_y_label(source_y_name: str, y_col: MappingSpec) -> str | None:
    if source_y_name == str(y_col):
        return None
    return f"{y_col} | alias: {source_y_name}"


def _entries_by_protocol(
    entries: list[_RunEntry],
    protocol_order: tuple[str, ...],
) -> list[list[_RunEntry]]:
    groups: dict[str, list[_RunEntry]] = {}
    for entry in entries:
        protocol = str(entry.protocol_spec.protocol_id)
        groups.setdefault(protocol, []).append(entry)
    order = {protocol: idx for idx, protocol in enumerate(protocol_order)}
    return [
        group
        for _protocol, group in sorted(
            groups.items(), key=lambda item: order.get(item[0], len(order))
        )
    ]


def _timeseries_y_cols(
    entries: list[_RunEntry],
    cols: list[MappingSpec] | tuple[MappingSpec, ...],
) -> list[MappingSpec]:
    requested = [col for col in cols if col not in _EIS_COLS]
    return [col for col in requested if any(_has_sources(entry, col) for entry in entries)]


def _include_eis(cols: list[MappingSpec] | tuple[MappingSpec, ...]) -> bool:
    return any(col in _EIS_COLS for col in cols)


def _has_eis_columns(schema: dict[str, pl.DataType]) -> bool:
    return all(mapping_column_sources(column, set(schema)) for column in _EIS_COLS)


def _is_eis(protocol_spec: InteractiveProtocolSpec) -> bool:
    return protocol_spec.protocol.protocol_id == DatasetProtocolId.eis


def _protocol_title(entries: list[_RunEntry]) -> str:
    return str(entries[0].protocol_spec.protocol_id)


def _entry_colors(entries: list[_RunEntry]) -> dict[str, str]:
    labels = list(dict.fromkeys(_entry_label(entry) for entry in entries))
    return {label: _COLORWAY[idx % len(_COLORWAY)] for idx, label in enumerate(labels)}


def _available_group_cols(entry: _RunEntry) -> tuple[MappingSpec, ...]:
    return _protocol_group_by(entry.protocol_spec)


def _annotation_messages(entry: _RunEntry) -> tuple[tuple[str, str], ...]:
    if str(BaseColumns.anns) not in entry.schema:
        return ()
    try:
        frame = collect_frame(
            entry.source.scan()
            .select(BaseColumns.anns)
            .explode(BaseColumns.anns)
            .drop_nulls(BaseColumns.anns)
            .unnest(BaseColumns.anns)
            .select("column", "reason")
            .drop_nulls()
            .unique()
            .sort("column", "reason")
        )
    except pl.exceptions.PolarsError:
        return ()
    return tuple((str(row["column"]), str(row["reason"])) for row in frame.rows(named=True))


def _annotation_label(label: str, column: MappingSpec, reason: str) -> str:
    return f"{label} | {reason}: {column}"


def _entry_display_label(
    entry: _RunEntry,
    label: str,
    y_cols: list[MappingSpec],
    messages: tuple[tuple[str, str], ...],
) -> str:
    suffixes = []
    for y_col in y_cols:
        output_y_col = _protocol_output_column(entry.protocol_spec, y_col)
        for reason, _annotation_columns in _annotation_overlays_for_y_col(messages, output_y_col):
            suffixes.append(f"{reason}: {y_col}")
    unique_suffixes = tuple(dict.fromkeys(suffixes))
    if not unique_suffixes:
        return label
    return " | ".join((label, *unique_suffixes))


def _annotation_overlays_for_y_col(
    messages: tuple[tuple[str, str], ...],
    y_col: MappingSpec,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    aliases = set(y_col.alias)
    by_reason: dict[str, list[str]] = {}
    for annotation_column, reason in messages:
        if annotation_column not in aliases:
            continue
        by_reason.setdefault(reason, []).append(annotation_column)
    return tuple(
        (reason, tuple(dict.fromkeys(columns))) for reason, columns in sorted(by_reason.items())
    )


def _axis_hovertemplate(
    title: str,
    x_col: MappingSpec,
    y_col: MappingSpec,
    customdata_cols: tuple[MappingSpec, ...],
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


def _annotation_hovertemplate(title: str) -> str:
    return f"<b>{title}</b><extra></extra>"


def _entry_label(entry: _RunEntry) -> str:
    parts = []
    for column in _protocol_group_by(entry.protocol_spec):
        value = entry.row.get(str(column))
        if value is not None:
            parts.append(f"{column}={value}")
    return " | ".join(parts) or "selected"
