from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import polars as pl
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.protocols import BatteryProtocols
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN
from batgrad.ml.data.config import LoaderConfig, ScalingRule, WindowConfig
from batgrad.ml.data.index import sort_index_frame
from batgrad.ml.data.materialization import materialize_batch_plan
from batgrad.ml.data.planning import iter_batch_plans
from batgrad.ml.data.scaling import inverse_scale_tensor, scale_data
from batgrad.ml.train_utils import scaling_rules
from batgrad.storage.segments import SegmentSource, collect_segment_window_frames
from batgrad.viz.plotting import (
    COLORWAY,
    EIS_COLUMNS,
    EIS_NEG_IMAG,
    add_registered_xy_trace,
    axis_hovertemplate,
    colors_by_label,
    consume_showlegend,
    make_eis_figure,
    make_timeseries_figure,
    make_trace_resampler,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from batgrad.contracts.segments import ParquetSegment
    from batgrad.ml.config import ExperimentConfig
    from batgrad.ml.data.batch import Batch
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.data.planning import WindowRef
    from batgrad.storage.store import DatasetStoreReader
    from batgrad.viz.widgets.plotly_trace_resampler import PlotlyTraceResampler


@dataclass(frozen=True, slots=True)
class MlBatchPreviewSubmission:
    submit_id: int
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    batch_size: int
    seq_len: int
    batch_group_index: int
    sample_index: int = 0
    consecutive_step: int = 0
    strategy: str = "shuffled_protocol_groups"
    stateful_n_windows: int = 1
    active_protocol: str = str(DatasetProtocolId.cycling)
    scaling: tuple[ScalingRule, ...] = ()


@dataclass(frozen=True, slots=True)
class MlBatchPreview:
    widget: PlotlyTraceResampler
    metadata: pl.DataFrame
    batch_index: int
    total_batches: int
    overlay_trace_indices: tuple[int, ...]
    current_stream_key: tuple[object, ...]
    store: DatasetStoreReader
    index: MlDatasetIndex
    submission: MlBatchPreviewSubmission


@dataclass(frozen=True, slots=True)
class _PreviewSelection:
    store: DatasetStoreReader
    index: MlDatasetIndex
    submission: MlBatchPreviewSubmission
    ref: WindowRef
    batch: Batch
    sample_index: int
    batch_index: int
    total_batches: int


_TRACE_LABELS = ("stream", "input batch", "target batch")
PLOT_SEQUENCE_RANK = 2


@dataclass(frozen=True, slots=True)
class RolloutPlotSeries:
    inputs: torch.Tensor
    context_prediction: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    match: dict[str, object]
    anchor: int


@dataclass(frozen=True, slots=True)
class _BatchPlotSpec:
    x: str
    y: str
    row: int
    col: int
    input: bool
    target: bool


def build_rollout_figure(
    config: ExperimentConfig,
    series: list[RolloutPlotSeries],
    context_len: int,
    run_name: str | None,
) -> object:
    import plotly.graph_objects as go  # noqa: PLC0415 - only needed for plot payloads.
    from plotly.subplots import make_subplots  # noqa: PLC0415

    figure = make_subplots(
        rows=len(config.data.target_columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )
    scaling = scaling_rules(config)
    for channel_idx, channel_name in enumerate(config.data.target_columns, start=1):
        value_idx = channel_idx - 1
        for series_idx, item in enumerate(series):
            input_cpu = inverse_scale_tensor(
                item.inputs.detach().cpu(), config.data.input_columns, scaling
            )
            target_cpu = inverse_scale_tensor(
                item.target.detach().cpu(), config.data.target_columns, scaling
            )
            context_prediction_cpu = inverse_scale_tensor(
                item.context_prediction.detach().cpu(), config.data.target_columns, scaling
            )
            prediction_cpu = inverse_scale_tensor(
                item.prediction.detach().cpu(), config.data.target_columns, scaling
            )
            if target_cpu.ndim != PLOT_SEQUENCE_RANK or prediction_cpu.ndim != PLOT_SEQUENCE_RANK:
                raise ValueError(
                    "rollout plot expects prediction/target tensors shaped (T, C), got "
                    f"prediction={tuple(prediction_cpu.shape)} target={tuple(target_cpu.shape)}"
                )
            total_steps = int(target_cpu.shape[0])
            pred_steps = int(prediction_cpu.shape[0])
            x_values = _rollout_time_axis(config, input_cpu)
            pred_full = torch.full(
                (total_steps, int(prediction_cpu.shape[1])),
                float("nan"),
                dtype=prediction_cpu.dtype,
            )
            pred_full[:context_len, :] = context_prediction_cpu[:context_len, :]
            pred_full[context_len : context_len + pred_steps, :] = prediction_cpu
            color = COLORWAY[series_idx % len(COLORWAY)]
            cell_id = item.match.get("cell id", "unknown-cell")
            cycle = item.match.get("cycle index", "unknown-cycle")
            group_id = f"{cell_id}:{cycle}:{item.anchor}"
            label = f"{cell_id} cycle={cycle} anchor={item.anchor}"
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=target_cpu[:, value_idx].tolist(),
                    mode="lines",
                    line={"color": color},
                    name=f"{label} target",
                    legendgroup=group_id,
                    showlegend=channel_idx == 1,
                ),
                row=channel_idx,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=pred_full[:, value_idx].tolist(),
                    mode="lines",
                    line={"color": color, "dash": "dash"},
                    name=f"{label} prediction",
                    legendgroup=group_id,
                    showlegend=channel_idx == 1,
                ),
                row=channel_idx,
                col=1,
            )
            context_x = x_values[min(context_len, len(x_values) - 1)]
            y_values = torch.cat((target_cpu[:, value_idx], pred_full[:, value_idx]))
            finite = y_values[torch.isfinite(y_values)]
            if int(finite.numel()) > 0:
                y_min = float(finite.min().item())
                y_max = float(finite.max().item())
                if y_min == y_max:
                    y_min -= 1.0
                    y_max += 1.0
                figure.add_trace(
                    go.Scatter(
                        x=[context_x, context_x],
                        y=[y_min, y_max],
                        mode="lines",
                        line={"color": color, "dash": "dot"},
                        name=f"{label} context",
                        legendgroup=group_id,
                        showlegend=False,
                    ),
                    row=channel_idx,
                    col=1,
                )
        figure.update_yaxes(title_text=channel_name, row=channel_idx, col=1)
    first_match = series[0].match
    dataset_id = first_match.get("dataset id", "unknown-dataset")
    protocol = first_match.get("protocol", config.data.protocols[0])
    display_name = run_name or config.run.name or "local-run"
    figure.update_layout(
        title=(
            f"rollout | run={display_name} | protocol={protocol} | dataset={dataset_id} | "
            f"context={context_len}"
        ),
        xaxis_title="Time [s]",
        legend={"groupclick": "togglegroup"},
        height=max(360, 260 * len(config.data.target_columns)),
    )
    return figure


def _rollout_time_axis(config: ExperimentConfig, inputs: torch.Tensor) -> list[float]:
    try:
        dt_idx = config.data.input_columns.index("Time diff [s]")
    except ValueError:
        return [float(idx) for idx in range(int(inputs.shape[0]))]
    dt = inputs[:, dt_idx].to(dtype=torch.float32).clamp_min(0.0)
    elapsed = torch.cumsum(dt, dim=0) - dt[0]
    return [float(value) for value in elapsed.tolist()]


def build_ml_batch_preview(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    submission: MlBatchPreviewSubmission,
) -> MlBatchPreview:
    return _render_ml_batch_preview(_prepare_preview_selection(store, index, submission))


def _prepare_preview_selection(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    submission: MlBatchPreviewSubmission,
) -> _PreviewSelection:
    config = _loader_config(submission)
    sorted_index = _sorted_preview_index(index)
    protocol = DatasetProtocolId(submission.active_protocol)
    requested_batch_index = _preview_raw_plan_index(submission)
    selected_plan = None
    selected_index = 0
    total_batches = 0
    for plan_idx, plan in enumerate(iter_batch_plans(sorted_index, protocol, config)):
        total_batches = plan_idx + 1
        if plan_idx <= requested_batch_index:
            selected_plan = plan
            selected_index = plan_idx
    if selected_plan is None:
        raise ValueError("No batch windows are available for this preview selection")

    selected_sample_index = min(max(0, int(submission.sample_index)), len(selected_plan.refs) - 1)
    return _PreviewSelection(
        store=store,
        index=sorted_index,
        submission=submission,
        ref=selected_plan.refs[selected_sample_index],
        batch=materialize_batch_plan(
            store,
            selected_plan,
            submission.input_columns,
            submission.target_columns,
            submission.scaling,
            config,
            selected_index,
        ),
        sample_index=selected_sample_index,
        batch_index=selected_index,
        total_batches=total_batches,
    )


def _render_ml_batch_preview(data: _PreviewSelection) -> MlBatchPreview:
    store = data.store
    index = data.index
    submission = data.submission
    ref = data.ref
    batch = data.batch
    batch_index = data.batch_index
    total_batches = data.total_batches
    sample_index = data.sample_index
    if ref.protocol == DatasetProtocolId.eis:
        return _build_eis_batch_preview(
            store, index, submission, ref, batch, batch_index, total_batches, sample_index
        )
    title = _ref_title(index, ref)
    axis_col = _axis_column(ref.protocol)
    y_columns = _plot_columns(axis_col, submission.input_columns, submission.target_columns)
    if not y_columns:
        raise ValueError("Select at least one non-axis input or target column to plot")

    stream_lf = _scaled_lazy_frame(
        _stream_lazy_frame(store, ref.segments, (axis_col, *y_columns)), submission.scaling
    )
    stream = _scaled_stream_window(
        store,
        ref.segments,
        ref.offset,
        _preview_rows(submission) + 1,
        (axis_col, *y_columns),
        submission.scaling,
    )
    fig, height = make_timeseries_figure(y_columns, axis_col, title)
    widget = make_trace_resampler(fig, height, max_batch_rows=None)

    overlay_trace_indices = []
    shown_roles: set[str] = set()
    colors = colors_by_label(_TRACE_LABELS)
    for row_idx, column in enumerate(y_columns, start=1):
        _add_ml_trace(
            fig,
            widget,
            stream_lf,
            axis_col=axis_col,
            y_col=column,
            row=row_idx,
            col=1,
            label="stream",
            color=colors["stream"],
            showlegend=consume_showlegend("stream", shown_roles),
        )
    overlay_trace_indices.extend(
        _add_batch_overlay_traces(
            fig,
            widget,
            stream,
            0,
            _preview_rows(submission),
            _timeseries_plot_specs(axis_col, y_columns, submission),
            colors,
            shown_roles,
        )
    )

    return MlBatchPreview(
        widget=widget,
        metadata=_metadata_frame(batch, batch_index, total_batches, sample_index, submission),
        batch_index=batch_index,
        total_batches=total_batches,
        overlay_trace_indices=tuple(overlay_trace_indices),
        current_stream_key=_stream_key(ref),
        store=store,
        index=index,
        submission=submission,
    )


def update_ml_batch_preview(
    preview: MlBatchPreview,
    batch_group_index: int,
    sample_index: int | None = None,
    consecutive_step: int | None = None,
) -> MlBatchPreview:
    submission = replace(
        preview.submission,
        batch_group_index=int(batch_group_index),
        sample_index=preview.submission.sample_index if sample_index is None else int(sample_index),
        consecutive_step=preview.submission.consecutive_step
        if consecutive_step is None
        else int(consecutive_step),
    )
    config = _loader_config(submission)
    selected = _prepare_preview_selection(preview.store, preview.index, submission)
    batch_index = selected.batch_index
    sample_index = selected.sample_index
    ref = selected.ref
    if _stream_key(ref) != preview.current_stream_key:
        return build_ml_batch_preview(
            preview.store,
            preview.index,
            replace(submission, batch_group_index=submission.batch_group_index),
        )
    batch = selected.batch
    if ref.protocol == DatasetProtocolId.eis:
        return _update_eis_batch_preview(
            preview,
            ref,
            batch,
            batch_index,
            selected.total_batches,
            sample_index,
            submission,
            config,
        )
    axis_col = _axis_column(ref.protocol)
    y_columns = _plot_columns(axis_col, submission.input_columns, submission.target_columns)
    rows = _preview_rows(submission)
    stream = _scaled_stream_window(
        preview.store,
        ref.segments,
        ref.offset,
        rows + 1,
        (axis_col, *y_columns),
        submission.scaling,
    )
    overlay_frames = _batch_overlay_frames(
        stream,
        0,
        rows,
        _timeseries_plot_specs(axis_col, y_columns, submission),
    )
    preview.widget.update_registered_traces(
        tuple(
            (trace_idx, frame.lazy(), frame.height)
            for trace_idx, frame in zip(preview.overlay_trace_indices, overlay_frames, strict=True)
        )
    )
    return MlBatchPreview(
        widget=preview.widget,
        metadata=_metadata_frame(
            batch, batch_index, selected.total_batches, sample_index, submission
        ),
        batch_index=batch_index,
        total_batches=selected.total_batches,
        overlay_trace_indices=preview.overlay_trace_indices,
        current_stream_key=preview.current_stream_key,
        store=preview.store,
        index=preview.index,
        submission=submission,
    )


def count_ml_batch_preview_groups(
    index: MlDatasetIndex,
    *,
    strategy: str,
    active_protocol: str,
    batch_size: int,
    seq_len: int,
    stateful_n_windows: int,
) -> int:
    if index.frame.is_empty() or BaseColumns.proto not in index.frame.columns:
        return 0
    if ml_batch_preview_unavailable_message(
        strategy=strategy,
        active_protocol=active_protocol,
    ):
        return 0
    submission = MlBatchPreviewSubmission(
        submit_id=0,
        input_columns=("__unused__",),
        target_columns=("__unused__",),
        batch_size=batch_size,
        seq_len=seq_len,
        batch_group_index=0,
        strategy=strategy,
        stateful_n_windows=stateful_n_windows,
        active_protocol=active_protocol,
    )
    config = _loader_config(submission)
    batch_count = _count_preview_batches(
        _sorted_preview_index(index), DatasetProtocolId(submission.active_protocol), config
    )
    if batch_count == 0:
        return 0
    return (batch_count + stateful_n_windows - 1) // stateful_n_windows


def ml_batch_preview_unavailable_message(*, strategy: str, active_protocol: str) -> str | None:
    try:
        protocol = DatasetProtocolId(active_protocol)
    except ValueError:
        return None
    if protocol == DatasetProtocolId.eis and strategy != "sequential":
        return (
            "EIS batch preview is not supported with shuffled protocol groups yet. "
            "Use Sequential debug or select cycling/HPPC/RPT."
        )
    return None


def _count_preview_batches(
    index: MlDatasetIndex,
    active_protocol: DatasetProtocolId | object,
    config: LoaderConfig,
) -> int:
    return sum(1 for _plan in iter_batch_plans(index, active_protocol, config))


def _loader_config(submission: MlBatchPreviewSubmission) -> LoaderConfig:
    return LoaderConfig(
        strategy="sequential"
        if submission.strategy == "sequential"
        else "shuffled_protocol_groups",
        active_protocol=DatasetProtocolId(submission.active_protocol),
        stateful_n_windows=int(submission.stateful_n_windows),
        default_window=WindowConfig(
            batch_size=submission.batch_size,
            seq_len=submission.seq_len,
            drop_incomplete=False,
        ),
    )


def _sorted_preview_index(
    index: MlDatasetIndex,
) -> MlDatasetIndex:
    return type(index)(sort_index_frame(index.frame))


def _build_eis_batch_preview(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    submission: MlBatchPreviewSubmission,
    ref: WindowRef,
    batch: Batch,
    batch_index: int,
    total_batches: int,
    sample_index: int,
) -> MlBatchPreview:
    _validate_eis_preview_columns((*submission.input_columns, *submission.target_columns))
    stream_lf = _eis_stream_lazy_frame(store, ref.segments, submission.scaling)
    stream = _eis_stream_window_frame(
        store, ref.segments, ref.offset, _preview_rows(submission) + 1, submission.scaling
    )
    title = _ref_title(index, ref)
    fig, height = make_eis_figure(title)
    widget = make_trace_resampler(fig, height, max_batch_rows=None)

    overlay_trace_indices = []
    shown_roles: set[str] = set()
    colors = colors_by_label(_TRACE_LABELS)
    for spec in _eis_plot_specs(submission):
        _add_ml_trace(
            fig,
            widget,
            stream_lf,
            axis_col=spec.x,
            y_col=spec.y,
            row=spec.row,
            col=spec.col,
            label="stream",
            color=colors["stream"],
            showlegend=consume_showlegend("stream", shown_roles),
        )
    overlay_trace_indices.extend(
        _add_batch_overlay_traces(
            fig,
            widget,
            stream,
            0,
            _preview_rows(submission),
            _eis_plot_specs(submission),
            colors,
            shown_roles,
        )
    )

    return MlBatchPreview(
        widget=widget,
        metadata=_metadata_frame(batch, batch_index, total_batches, sample_index, submission),
        batch_index=batch_index,
        total_batches=total_batches,
        overlay_trace_indices=tuple(overlay_trace_indices),
        current_stream_key=_stream_key(ref),
        store=store,
        index=index,
        submission=submission,
    )


def _update_eis_batch_preview(
    preview: MlBatchPreview,
    ref: WindowRef,
    batch: Batch,
    batch_index: int,
    total_batches: int,
    sample_index: int,
    submission: MlBatchPreviewSubmission,
    config: LoaderConfig,
) -> MlBatchPreview:
    rows = config.default_window.batch_size * config.default_window.seq_len
    stream = _eis_stream_window_frame(
        preview.store, ref.segments, ref.offset, rows + 1, submission.scaling
    )
    overlay_frames = _batch_overlay_frames(
        stream,
        0,
        rows,
        _eis_plot_specs(preview.submission),
    )
    preview.widget.update_registered_traces(
        tuple(
            (trace_idx, frame.lazy(), frame.height)
            for trace_idx, frame in zip(preview.overlay_trace_indices, overlay_frames, strict=True)
        )
    )
    return MlBatchPreview(
        widget=preview.widget,
        metadata=_metadata_frame(batch, batch_index, total_batches, sample_index, submission),
        batch_index=batch_index,
        total_batches=total_batches,
        overlay_trace_indices=preview.overlay_trace_indices,
        current_stream_key=preview.current_stream_key,
        store=preview.store,
        index=preview.index,
        submission=submission,
    )


def _eis_stream_window_frame(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    offset: int,
    rows: int,
    scaling: tuple[ScalingRule, ...] = (),
) -> pl.DataFrame:
    stream = _stream_window_frame(store, segments, offset, rows, EIS_COLUMNS)
    stream = _scaled_frame(stream, scaling)
    if stream.is_empty():
        return stream.with_columns(pl.lit(None).cast(pl.Float64).alias(EIS_NEG_IMAG))
    return stream.with_columns((-pl.col(BaseColumns.z_imag)).alias(EIS_NEG_IMAG))


def _scaled_frame(data: pl.DataFrame, scaling: tuple[ScalingRule, ...]) -> pl.DataFrame:
    return scale_data(data, scaling) if scaling else data


def _scaled_lazy_frame(data: pl.LazyFrame, scaling: tuple[ScalingRule, ...]) -> pl.LazyFrame:
    return scale_data(data, scaling) if scaling else data


def _scaled_stream_window(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    offset: int,
    rows: int,
    columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
) -> pl.DataFrame:
    data = _stream_window_frame(store, segments, offset, rows, columns)
    return _scaled_frame(data, scaling)


def _eis_stream_lazy_frame(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    scaling: tuple[ScalingRule, ...] = (),
) -> pl.LazyFrame:
    stream = _scaled_lazy_frame(_stream_lazy_frame(store, segments, EIS_COLUMNS), scaling)
    return stream.with_columns((-pl.col(BaseColumns.z_imag)).alias(EIS_NEG_IMAG))


def _validate_eis_preview_columns(columns: tuple[str, ...]) -> None:
    available = set(columns)
    required = {BaseColumns.z_real, BaseColumns.z_imag, BaseColumns.freq}
    missing = sorted(str(column) for column in required if column not in available)
    if missing:
        raise ValueError(
            "EIS batch preview requires EIS impedance columns in the selected inputs/targets: "
            f"missing={missing}"
        )


def _eis_plot_specs(submission: MlBatchPreviewSubmission) -> tuple[_BatchPlotSpec, ...]:
    input_columns = set(submission.input_columns)
    target_columns = set(submission.target_columns)
    return (
        _BatchPlotSpec(
            x=BaseColumns.z_real,
            y=EIS_NEG_IMAG,
            row=1,
            col=1,
            input=BaseColumns.z_real in input_columns or BaseColumns.z_imag in input_columns,
            target=BaseColumns.z_real in target_columns or BaseColumns.z_imag in target_columns,
        ),
        _BatchPlotSpec(
            x=BaseColumns.freq,
            y=BaseColumns.z_mag,
            row=1,
            col=2,
            input=BaseColumns.z_mag in input_columns,
            target=BaseColumns.z_mag in target_columns,
        ),
        _BatchPlotSpec(
            x=BaseColumns.freq,
            y=BaseColumns.z_phase,
            row=2,
            col=2,
            input=BaseColumns.z_phase in input_columns,
            target=BaseColumns.z_phase in target_columns,
        ),
    )


def _timeseries_plot_specs(
    axis_col: str,
    y_columns: tuple[str, ...],
    submission: MlBatchPreviewSubmission,
) -> tuple[_BatchPlotSpec, ...]:
    return tuple(
        _BatchPlotSpec(
            x=axis_col,
            y=column,
            row=row_idx,
            col=1,
            input=column in submission.input_columns,
            target=column in submission.target_columns,
        )
        for row_idx, column in enumerate(y_columns, start=1)
    )


def _add_batch_overlay_traces(
    fig: go.Figure,
    widget: PlotlyTraceResampler,
    stream: pl.DataFrame,
    offset: int,
    rows: int,
    specs: tuple[_BatchPlotSpec, ...],
    colors: dict[str, str],
    shown_roles: set[str],
) -> list[int]:
    trace_indices = []
    for label, spec, frame in _iter_batch_overlay_frames(stream, offset, rows, specs):
        trace_indices.append(
            _add_ml_trace(
                fig,
                widget,
                frame.lazy(),
                axis_col=spec.x,
                y_col=spec.y,
                row=spec.row,
                col=spec.col,
                label=label,
                color=colors[label],
                showlegend=consume_showlegend(label, shown_roles),
            )
        )
    return trace_indices


def _batch_overlay_frames(
    stream: pl.DataFrame,
    offset: int,
    rows: int,
    specs: tuple[_BatchPlotSpec, ...],
) -> list[pl.DataFrame]:
    return [
        frame for _label, _spec, frame in _iter_batch_overlay_frames(stream, offset, rows, specs)
    ]


def _iter_batch_overlay_frames(
    stream: pl.DataFrame,
    offset: int,
    rows: int,
    specs: tuple[_BatchPlotSpec, ...],
) -> list[tuple[str, _BatchPlotSpec, pl.DataFrame]]:
    frames = []
    for spec in specs:
        selected = stream.select(spec.x, spec.y)
        if spec.input:
            frames.append(("input batch", spec, _overlay_frame(selected, offset, rows)))
        if spec.target:
            frames.append(("target batch", spec, _overlay_frame(selected, offset + 1, rows)))
    return frames


def _preview_rows(submission: MlBatchPreviewSubmission) -> int:
    if submission.strategy != "sequential":
        return submission.seq_len
    return submission.batch_size * submission.seq_len


def _stream_window_frame(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    offset: int,
    rows: int,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    return collect_segment_window_frames(
        store,
        segments,
        offset=offset,
        rows=rows,
        columns=columns,
    )


def _stream_lazy_frame(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    columns: tuple[str, ...],
) -> pl.LazyFrame:
    if not segments:
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.Float64)).lazy()
    return SegmentSource.from_values(store, segments).scan().select(columns)


def _overlay_frame(stream: pl.DataFrame, offset: int, rows: int) -> pl.DataFrame:
    return stream.slice(max(0, offset), rows)


def _add_ml_trace(
    fig: go.Figure,
    widget: PlotlyTraceResampler,
    lf: pl.LazyFrame,
    *,
    axis_col: str,
    y_col: str,
    row: int,
    col: int,
    label: str,
    color: str,
    showlegend: bool,
) -> int:
    return add_registered_xy_trace(
        fig,
        widget,
        lf,
        x_col=axis_col,
        y_col=y_col,
        row=row,
        col=col,
        label=label,
        legendgroup=label,
        color=color,
        showlegend=showlegend,
        hovertemplate=axis_hovertemplate(label, axis_col, y_col),
    )


def _axis_column(protocol: DatasetProtocolId) -> str:
    specs = {
        DatasetProtocolId.cycling: BatteryProtocols.cyc,
        DatasetProtocolId.hppc: BatteryProtocols.hppc,
        DatasetProtocolId.rpt: BatteryProtocols.rpt,
        DatasetProtocolId.eis: BatteryProtocols.eis,
    }
    return specs[protocol].axis_col


def _plot_columns(
    axis_col: str,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column for column in dict.fromkeys((*input_columns, *target_columns)) if column != axis_col
    )


def _metadata_frame(
    batch: Batch,
    plan_index: int,
    total_batches: int,
    sample_index: int,
    submission: MlBatchPreviewSubmission,
) -> pl.DataFrame:
    state = batch.state
    sample_index = _clamp_sample_index(sample_index, len(state.window_offsets))
    consecutive_index = _clamp_consecutive_index(
        submission.consecutive_step,
        submission.stateful_n_windows,
    )
    rows = [
        {"scope": "preview", "field": "preview_group", "value": submission.batch_group_index},
        {"scope": "preview", "field": "batch_index", "value": sample_index},
        {"scope": "preview", "field": "consecutive_index", "value": consecutive_index},
        {"scope": "preview", "field": "plan_index", "value": plan_index},
        {"scope": "preview", "field": "total_batches", "value": total_batches},
        {"scope": "selected_sample", "field": "sample_index", "value": sample_index},
        {
            "scope": "selected_sample",
            "field": "manifest_path",
            "value": state.manifest_paths[sample_index],
        },
        {
            "scope": "selected_sample",
            "field": "manifest_row_id",
            "value": state.manifest_row_ids[sample_index],
        },
        {
            "scope": "selected_sample",
            "field": "group_key",
            "value": state.group_keys[sample_index],
        },
        {
            "scope": "selected_sample",
            "field": "alignment_key",
            "value": state.alignment_keys[sample_index],
        },
        {
            "scope": "selected_sample",
            "field": "window_offset",
            "value": state.window_offsets[sample_index],
        },
        {"scope": "tensor", "field": "inputs_shape", "value": tuple(batch.inputs.shape)},
        {"scope": "tensor", "field": "targets_shape", "value": tuple(batch.targets.shape)},
        {"scope": "tensor", "field": "mask_true", "value": int(batch.mask.sum().item())},
    ]
    rows.extend(
        {"scope": "batch.state", "field": field, "value": value}
        for field, value in asdict(batch.state).items()
    )
    return pl.DataFrame(
        {
            "scope": [str(row["scope"]) for row in rows],
            "field": [str(row["field"]) for row in rows],
            "value": [str(row["value"]) for row in rows],
        }
    )


def _preview_raw_plan_index(submission: MlBatchPreviewSubmission) -> int:
    consecutive_index = _clamp_consecutive_index(
        submission.consecutive_step,
        submission.stateful_n_windows,
    )
    return (
        int(submission.batch_group_index) * int(submission.stateful_n_windows) + consecutive_index
    )


def _clamp_sample_index(sample_index: int, sample_count: int) -> int:
    if sample_count <= 0:
        return 0
    return min(max(0, int(sample_index)), sample_count - 1)


def _clamp_consecutive_index(consecutive_index: int, consecutive_count: int) -> int:
    if consecutive_count <= 0:
        return 0
    return min(max(0, int(consecutive_index)), consecutive_count - 1)


def _ref_title(index: MlDatasetIndex, ref: WindowRef) -> str:
    row = _ref_row(index, ref)
    columns = (
        BaseColumns.set_id,
        BaseColumns.cell_id,
        BaseColumns.cidx,
        BaseColumns.proto,
        BaseColumns.split,
    )
    values = [
        f"{column}={row[column]}" for column in columns if column in row and row[column] is not None
    ]
    return " | ".join(values) if values else "ML batch preview"


def _ref_row(index: MlDatasetIndex, ref: WindowRef) -> dict[str, object]:
    rows = index.frame.filter(
        (pl.col(BaseColumns.manifest) == ref.manifest_path)
        & (pl.col(MANIFEST_ROW_ID_COLUMN) == ref.manifest_row_id)
    )
    if rows.height:
        return rows.row(0, named=True)
    return {
        BaseColumns.proto: ref.protocol,
        BaseColumns.split: ref.split,
        BaseColumns.manifest: ref.manifest_path,
        MANIFEST_ROW_ID_COLUMN: ref.manifest_row_id,
    }


def _stream_key(ref: WindowRef) -> tuple[object, ...]:
    return (ref.protocol, ref.manifest_path, ref.manifest_row_id, ref.group_key)
