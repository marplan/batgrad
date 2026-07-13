from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import polars as pl
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.protocols import BatteryProtocols
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN
from batgrad.ml.data.preview import (
    MlBatchPreviewData,
    MlBatchPreviewSpec,
    prepare_ml_batch_preview,
)
from batgrad.ml.data.scaling import inverse_scale_tensor, scale_data
from batgrad.ml.experiment import scaling_rules
from batgrad.ml.metrics import loss_metric_payload
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
    from batgrad.ml.data.config import ScalingRule
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.data.planning import WindowRef
    from batgrad.ml.inference import InferencePrediction, InferenceResult
    from batgrad.ml.validation import RolloutExample
    from batgrad.storage.store import DatasetStoreReader
    from batgrad.viz.widgets.plotly_trace_resampler import PlotlyTraceResampler


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
    spec: MlBatchPreviewSpec


_TRACE_LABELS = ("stream", "input batch", "target batch")
PLOT_SEQUENCE_RANK = 2


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
    series: list[RolloutExample] | tuple[RolloutExample, ...],
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
            x_values = _rollout_target_time_axis(config, input_cpu)
            pred_full = torch.full(
                (total_steps, int(prediction_cpu.shape[1])),
                float("nan"),
                dtype=prediction_cpu.dtype,
            )
            pred_full[:context_len, :] = context_prediction_cpu[:context_len, :]
            # Target index k represents the source row after input index k. The
            # first rollout prediction therefore aligns with the final context
            # logit at target index context_len - 1.
            rollout_start = item.target_start
            pred_full[rollout_start : rollout_start + pred_steps, :] = prediction_cpu
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
            context_x = x_values[min(rollout_start, len(x_values) - 1)]
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


def _rollout_target_time_axis(config: ExperimentConfig, inputs: torch.Tensor) -> list[float]:
    """Return timestamps aligned with next-row targets rather than input rows."""
    try:
        dt_idx = config.data.input_columns.index("Time diff [s]")
    except ValueError:
        return [float(idx + 1) for idx in range(int(inputs.shape[0]))]
    if int(inputs.shape[0]) == 0:
        return []
    dt = inputs[:, dt_idx].to(dtype=torch.float32).clamp_min(0.0)
    elapsed = torch.cumsum(dt, dim=0) - dt[0]
    target_elapsed = torch.cat((elapsed[1:], (elapsed[-1:] + dt[-1:])))
    return [float(value) for value in target_elapsed.tolist()]


def build_inference_widget(result: InferenceResult, batch_index: int) -> PlotlyTraceResampler:
    config = result.config
    columns = tuple(dict.fromkeys((*config.data.input_columns, *config.data.target_columns)))
    axis_col = _axis_column(_inference_protocol(result, batch_index))
    scaling = scaling_rules(config)
    inputs = inverse_scale_tensor(result.inputs, config.data.input_columns, scaling)
    targets = inverse_scale_tensor(result.targets, config.data.target_columns, scaling)
    predictions = tuple(
        replace(
            series,
            predictions=inverse_scale_tensor(
                series.predictions,
                config.data.target_columns,
                scaling,
            ),
        )
        for series in result.predictions
    )
    label = inference_group_label(result.group_keys[batch_index])
    fig, height = make_timeseries_figure(columns, axis_col, label)
    widget = make_trace_resampler(fig, height, max_batch_rows=None)
    frame = _inference_plot_frame(
        config,
        inputs,
        targets,
        predictions,
        batch_index,
        columns,
        axis_col,
    )
    lf = frame.lazy()
    trace_labels = (
        "ground truth",
        *(_inference_prediction_label(series) for series in predictions),
    )
    colors = colors_by_label(trace_labels)
    shown_roles: set[str] = set()
    for row_idx, column in enumerate(columns, start=1):
        target_idx = _column_index(config.data.target_columns, column)
        add_registered_xy_trace(
            fig,
            widget,
            lf,
            x_col=axis_col,
            y_col=_inference_base_col(column),
            row=row_idx,
            col=1,
            label="ground truth",
            color=colors["ground truth"],
            showlegend=consume_showlegend("ground truth", shown_roles),
            hovertemplate=axis_hovertemplate("ground truth", axis_col, _inference_base_col(column)),
            row_count=frame.height,
        )
        if target_idx is None:
            continue
        for series in predictions:
            y_col = _inference_prediction_col(column, series)
            series_label = _inference_prediction_label(series)
            add_registered_xy_trace(
                fig,
                widget,
                lf,
                x_col=axis_col,
                y_col=y_col,
                row=row_idx,
                col=1,
                label=series_label,
                color=colors[series_label],
                showlegend=consume_showlegend(series_label, shown_roles),
                hovertemplate=axis_hovertemplate(series_label, axis_col, y_col),
                row_count=frame.height,
            )
    return widget


def inference_metrics_frame(result: InferenceResult) -> pl.DataFrame:
    rows = []
    for series in result.predictions:
        row: dict[str, object] = {
            "checkpoint": series.checkpoint_alias,
            "checkpoint_path": series.checkpoint_path,
            "strategy": "classic" if series.suffix_steps == 0 else "masked_suffix",
            "suffix_steps": series.suffix_steps,
        }
        if series.metrics is not None:
            row.update(
                loss_metric_payload(
                    "loss_ce",
                    "rmse",
                    result.config.data.target_columns,
                    series.metrics,
                )
            )
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _inference_plot_frame(
    config: ExperimentConfig,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: tuple[InferencePrediction, ...],
    batch_index: int,
    columns: tuple[str, ...],
    axis_col: str,
) -> pl.DataFrame:
    x_values = _rollout_target_time_axis(config, inputs[batch_index])
    data: dict[str, list[float | None]] = {axis_col: [float(value) for value in x_values]}
    for column in columns:
        target_idx = _column_index(config.data.target_columns, column)
        input_idx = _column_index(config.data.input_columns, column)
        if target_idx is not None:
            base = targets[batch_index, :, target_idx].tolist()
            for series in predictions:
                prediction = [None] * len(x_values)
                values = series.predictions[batch_index, :, target_idx].tolist()
                prediction[series.target_start : series.target_start + len(values)] = values
                data[_inference_prediction_col(column, series)] = prediction
        elif input_idx is not None:
            base = inputs[batch_index, :, input_idx].tolist()
        else:
            base = [None] * len(x_values)
        data[_inference_base_col(column)] = [
            None if value is None else float(value) for value in base
        ]
    return pl.DataFrame(data)


def _column_index(columns: tuple[str, ...], column: str) -> int | None:
    try:
        return columns.index(column)
    except ValueError:
        return None


def _inference_base_col(column: str) -> str:
    return f"base::{column}"


def _inference_prediction_col(column: str, series: InferencePrediction) -> str:
    return f"prediction::{series.checkpoint_alias}::{series.suffix_steps}::{column}"


def _inference_prediction_label(series: InferencePrediction) -> str:
    strategy = "classic" if series.suffix_steps == 0 else f"suffix steps {series.suffix_steps}"
    return f"prediction | {strategy} | {series.checkpoint_alias}"


def inference_group_label(group_key: tuple[object, ...]) -> str:
    if not group_key:
        return "inference"
    protocol = group_key[-1]
    identifiers = tuple(str(item) for item in group_key[1:-1]) or tuple(
        str(item) for item in group_key
    )
    return ":".join((*identifiers, str(protocol)))


def _inference_protocol(result: InferenceResult, batch_index: int) -> DatasetProtocolId:
    group_key = result.group_keys[batch_index]
    return (
        DatasetProtocolId(group_key[-1])
        if group_key
        else DatasetProtocolId(result.config.data.protocols[0])
    )


def build_ml_batch_preview(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    spec: MlBatchPreviewSpec,
) -> MlBatchPreview:
    return _render_ml_batch_preview(prepare_ml_batch_preview(store, index, spec))


def _render_ml_batch_preview(data: MlBatchPreviewData) -> MlBatchPreview:
    store = data.store
    index = data.index
    submission = data.spec
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
        submission.preview_rows + 1,
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
            submission.preview_rows,
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
        spec=submission,
    )


def update_ml_batch_preview(
    preview: MlBatchPreview,
    batch_group_index: int,
    sample_index: int | None = None,
    consecutive_step: int | None = None,
) -> MlBatchPreview:
    submission = replace(
        preview.spec,
        batch_group_index=int(batch_group_index),
        sample_index=preview.spec.sample_index if sample_index is None else int(sample_index),
        consecutive_step=preview.spec.consecutive_step
        if consecutive_step is None
        else int(consecutive_step),
    )
    selected = prepare_ml_batch_preview(preview.store, preview.index, submission)
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
        )
    axis_col = _axis_column(ref.protocol)
    y_columns = _plot_columns(axis_col, submission.input_columns, submission.target_columns)
    rows = submission.preview_rows
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
        spec=submission,
    )


def _build_eis_batch_preview(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    submission: MlBatchPreviewSpec,
    ref: WindowRef,
    batch: Batch,
    batch_index: int,
    total_batches: int,
    sample_index: int,
) -> MlBatchPreview:
    _validate_eis_preview_columns((*submission.input_columns, *submission.target_columns))
    stream_lf = _eis_stream_lazy_frame(store, ref.segments, submission.scaling)
    stream = _eis_stream_window_frame(
        store, ref.segments, ref.offset, submission.preview_rows + 1, submission.scaling
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
            submission.preview_rows,
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
        spec=submission,
    )


def _update_eis_batch_preview(
    preview: MlBatchPreview,
    ref: WindowRef,
    batch: Batch,
    batch_index: int,
    total_batches: int,
    sample_index: int,
    submission: MlBatchPreviewSpec,
) -> MlBatchPreview:
    rows = submission.preview_rows
    stream = _eis_stream_window_frame(
        preview.store, ref.segments, ref.offset, rows + 1, submission.scaling
    )
    overlay_frames = _batch_overlay_frames(
        stream,
        0,
        rows,
        _eis_plot_specs(preview.spec),
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
        spec=submission,
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


def _eis_plot_specs(submission: MlBatchPreviewSpec) -> tuple[_BatchPlotSpec, ...]:
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
    submission: MlBatchPreviewSpec,
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
    submission: MlBatchPreviewSpec,
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
