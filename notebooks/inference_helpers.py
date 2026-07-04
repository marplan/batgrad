# ruff: noqa: INP001

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import polars as pl
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.protocols import BatteryProtocols
from batgrad.ml.config import ExperimentConfig, _coerce_dataclass
from batgrad.ml.data.config import LoaderConfig, WindowConfig
from batgrad.ml.data.index import MlDatasetIndex, sort_index_frame
from batgrad.ml.data.materialization import materialize_batch_plan
from batgrad.ml.data.planning import BatchPlan, WindowRef, build_stream_plans
from batgrad.ml.data.scaling import inverse_scale_tensor
from batgrad.ml.nn import SequenceMixer
from batgrad.ml.rollout import (
    masked_suffix_rollout_predictions,
    one_step_rollout_predictions,
)
from batgrad.ml.train_utils import feedback_pairs, scaling_rules
from batgrad.notebook_helpers import wrap_anywidget_blocks
from batgrad.viz.plotting import (
    add_registered_xy_trace,
    axis_hovertemplate,
    colors_by_label,
    consume_showlegend,
    make_timeseries_figure,
    make_trace_resampler,
)


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: str
    label: str


@dataclass(frozen=True, slots=True)
class InferenceSubmission:
    submit_id: int
    checkpoints: tuple[SelectedCheckpoint, ...]
    device: str
    masked_suffix_steps: tuple[int, ...]
    rollout_steps: int


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    submission: InferenceSubmission
    selected_index_frame: pl.DataFrame
    store: object


@dataclass(frozen=True, slots=True)
class CheckpointModel:
    config: ExperimentConfig
    model: torch.nn.Module
    step: int | None


@dataclass(frozen=True, slots=True)
class SelectedCheckpoint:
    alias: str
    path: str


@dataclass(frozen=True, slots=True)
class PredictionSeries:
    checkpoint_label: str
    checkpoint_path: str
    suffix_steps: int
    label: str
    predictions: torch.Tensor


@dataclass(frozen=True, slots=True)
class BatchInferenceResult:
    config: ExperimentConfig
    inputs: torch.Tensor
    targets: torch.Tensor
    predictions: tuple[PredictionSeries, ...]
    context_len: int
    rollout_len: int
    group_labels: tuple[str, ...]
    warning: str | None


def available_devices() -> tuple[str, ...]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{idx}" for idx in range(torch.cuda.device_count()))
    return tuple(devices)


def discover_checkpoints(root: str | Path = ".") -> tuple[CheckpointInfo, ...]:
    root_path = Path(root)
    paths = sorted(root_path.glob("**/checkpoints/*.pt"))
    return tuple(
        CheckpointInfo(path=str(path), label=str(path.relative_to(root_path))) for path in paths
    )


def checkpoint_frame(checkpoints: tuple[CheckpointInfo, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "alias": [f"ckpt {idx}" for idx in range(1, len(checkpoints) + 1)],
            "checkpoint": [item.label for item in checkpoints],
            "checkpoint_path": [item.path for item in checkpoints],
        }
    )


def make_checkpoint_table(frame: pl.DataFrame) -> object:
    import marimo as mo  # noqa: PLC0415

    return mo.ui.table(
        frame,
        selection="multi",
        hidden_columns=["checkpoint_path"],
    )


def selected_checkpoints_from_table(
    checkpoint_table_value: object,
    frame: pl.DataFrame,
) -> tuple[SelectedCheckpoint, ...]:
    selected_aliases = {
        str(row["alias"])
        for row in _table_rows(checkpoint_table_value)
        if row.get("alias")
    }
    if not selected_aliases:
        return ()
    rows = frame.filter(pl.col("alias").is_in(selected_aliases)).iter_rows(named=True)
    return tuple(
        SelectedCheckpoint(alias=str(row["alias"]), path=str(row["checkpoint_path"]))
        for row in rows
        if row.get("checkpoint_path")
    )


def _table_rows(value: object) -> tuple[Mapping[str, object], ...]:
    rows: object
    if isinstance(value, pl.DataFrame):
        rows = value.iter_rows(named=True)
    elif isinstance(value, Mapping):
        selection = value.get("selection")
        if isinstance(selection, pl.DataFrame):
            rows = selection.iter_rows(named=True)
        elif isinstance(selection, Iterable) and not isinstance(selection, (str, bytes)):
            rows = selection
        elif all(key in value for key in ("alias", "checkpoint_path")):
            rows = (value,)
        else:
            rows = ()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        rows = value
    else:
        rows = ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def load_checkpoint_config(path: str | Path) -> tuple[ExperimentConfig | None, str | None]:
    if not path:
        return None, None
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
            return None, f"Checkpoint is missing a config dict: {path}"
        config = _coerce_dataclass(ExperimentConfig, checkpoint["config"], "checkpoint.config")
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return None, str(exc)
    return config, None


def make_inference_submission(
    *,
    submit_id: int,
    checkpoints: tuple[SelectedCheckpoint, ...],
    device: str,
    masked_suffix_steps: str,
    rollout_steps: int,
) -> InferenceSubmission | None:
    checkpoints = tuple(checkpoint for checkpoint in checkpoints if checkpoint.path)
    if not checkpoints:
        raise ValueError("Select one or more checkpoints before running inference")
    parsed_suffix_steps = parse_masked_suffix_steps(masked_suffix_steps)
    if rollout_steps <= 0:
        raise ValueError("Rollout steps must be > 0")
    return InferenceSubmission(
        submit_id=submit_id,
        checkpoints=checkpoints,
        device=device,
        masked_suffix_steps=parsed_suffix_steps,
        rollout_steps=rollout_steps,
    )


def parse_masked_suffix_steps(value: str) -> tuple[int, ...]:
    steps = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise ValueError("Masked suffix steps must be >= 0")
        steps.append(step)
    if not steps:
        raise ValueError("Enter at least one masked suffix step")
    return tuple(dict.fromkeys(steps))


def checkpoint_summary(config: ExperimentConfig | None, error: str | None) -> object:
    import marimo as mo  # noqa: PLC0415

    if error is not None:
        return mo.callout(error, kind="danger")
    if config is None:
        return mo.md("Select a checkpoint to inspect model/data settings.")
    rows = {
        "seq_len": config.loader.seq_len,
        "protocols": ", ".join(config.data.protocols),
        "input_columns": ", ".join(config.data.input_columns),
        "target_columns": ", ".join(config.data.target_columns),
        "feedback_columns": ", ".join(config.data.feedback_columns) or "<none>",
    }
    return mo.ui.table(
        pl.DataFrame(
            {
                "field": list(rows.keys()),
                "value": [str(value) for value in rows.values()],
            }
        )
    )


def make_inference_request(
    *,
    store: object,
    selected_index_frame: pl.DataFrame,
    submission: InferenceSubmission | None,
) -> InferenceRequest | None:
    if submission is None:
        return None
    if store is None:
        raise ValueError("Select a valid store root before running inference")
    if selected_index_frame.is_empty():
        raise ValueError("Select one or more ML index rows before running inference")
    return InferenceRequest(
        submission=submission,
        selected_index_frame=selected_index_frame.clone(),
        store=store,
    )


def build_batch_inference(
    request: InferenceRequest | None,
) -> tuple[str | None, BatchInferenceResult | None]:
    if request is None:
        return None, None
    try:
        submission = request.submission
        device = _resolve_device(submission.device)
        checkpoints = tuple(
            _load_checkpoint_model(checkpoint.path, device)
            for checkpoint in submission.checkpoints
        )
        result = _run_selected_batch_inference(
            store=request.store,
            selected_index_frame=request.selected_index_frame,
            selected_checkpoints=submission.checkpoints,
            checkpoints=checkpoints,
            device=device,
            masked_suffix_steps=submission.masked_suffix_steps,
            requested_rollout_steps=submission.rollout_steps,
        )
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        NotImplementedError,
    ) as exc:
        return str(exc), None
    return None, result


def inference_view(
    *,
    submission_error: str | None,
    inference_error: str | None,
    result_view: object | None,
) -> object:
    import marimo as mo  # noqa: PLC0415

    if submission_error is not None:
        return mo.callout(submission_error, kind="danger")
    if inference_error is not None:
        return mo.callout(inference_error, kind="danger")
    if result_view is None:
        return mo.md("Select checkpoint and ML index rows, then click Run inference.")
    return result_view


def render_batch_result(
    result: BatchInferenceResult | None,
    batch_index: int,
) -> object | None:
    import marimo as mo  # noqa: PLC0415

    if result is None:
        return None
    items = []
    if result.warning is not None:
        items.append(mo.callout(result.warning, kind="warn"))
    idx = min(max(0, int(batch_index)), int(result.inputs.shape[0]) - 1)
    items.append(_build_inference_widget(result, idx))
    return mo.vstack(items)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but torch.cuda.is_available() is false")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device is not available: {value}")
    return device


def _load_checkpoint_model(path: str, device: torch.device) -> CheckpointModel:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint must contain a 'model' state dict: {path}")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError(f"Checkpoint must contain a 'config' dict: {path}")
    config = _coerce_dataclass(ExperimentConfig, raw_config, "checkpoint.config")
    model = SequenceMixer(
        config.model,
        input_dim=len(config.data.input_columns),
        output_dim=len(config.data.target_columns),
        device=device,
    ).to(device)
    model.load_state_dict(cast("dict[str, torch.Tensor]", checkpoint["model"]))
    model.eval()
    step = checkpoint.get("step")
    return CheckpointModel(config=config, model=model, step=step if isinstance(step, int) else None)


@torch.no_grad()
def _run_selected_batch_inference(
    *,
    store: object,
    selected_index_frame: pl.DataFrame,
    selected_checkpoints: tuple[SelectedCheckpoint, ...],
    checkpoints: tuple[CheckpointModel, ...],
    device: torch.device,
    masked_suffix_steps: tuple[int, ...],
    requested_rollout_steps: int,
) -> BatchInferenceResult:
    if not checkpoints:
        raise ValueError("Select at least one checkpoint before running inference")
    _validate_compatible_checkpoints(checkpoints)
    checkpoint = checkpoints[0]
    config = checkpoint.config
    context_len = int(config.loader.seq_len)
    protocol = DatasetProtocolId(config.data.protocols[0])
    index = MlDatasetIndex(sort_index_frame(selected_index_frame))
    loader = LoaderConfig(
        split=BaseColumns.split.values.train,
        default_window=WindowConfig(
            batch_size=max(1, selected_index_frame.height),
            seq_len=context_len,
        ),
        seed=config.run.seed,
        strategy="sequential",
        active_protocol=protocol,
        stateful_n_windows=1,
        drop_incomplete_batches=False,
        data_access="windowed",
        num_workers=0,
        multiprocessing_context=None,
        device=str(device),
    )
    streams = build_stream_plans(index, protocol, loader)
    if not streams:
        raise ValueError(f"No selected ML index rows match checkpoint protocol {protocol}")
    if len(streams) != selected_index_frame.height:
        raise ValueError(
            f"Selected rows must all match checkpoint protocol {protocol}; "
            f"got {len(streams)} matching streams for {selected_index_frame.height} rows"
        )
    max_rollout = min(max(0, int(stream.row_count) - context_len - 1) for stream in streams)
    if max_rollout <= 0:
        raise ValueError(
            "Selected rows are too short for checkpoint context length "
            f"seq_len={context_len}. Shortest selected row_count="
            f"{min(stream.row_count for stream in streams)}"
        )
    effective_rollout = min(int(requested_rollout_steps), max_rollout)
    warning = None
    if effective_rollout < requested_rollout_steps:
        warning = (
            f"Requested {requested_rollout_steps:,} rollout steps, clipped to "
            f"{effective_rollout:,} by the shortest selected file."
        )
    inference_loader = replace(
        loader,
        default_window=WindowConfig(
            batch_size=len(streams),
            seq_len=context_len + effective_rollout,
            drop_incomplete=False,
        ),
    )
    plan = BatchPlan(
        active_protocol=protocol,
        refs=tuple(WindowRef(stream, 0) for stream in streams),
    )
    batch = materialize_batch_plan(
        store,
        plan,
        config.data.input_columns,
        config.data.target_columns,
        scaling_rules(config),
        inference_loader,
        batch_idx=0,
    )
    inputs = batch.active.inputs.to(device=device)
    targets = batch.active.targets.to(device=device)
    mask = batch.active.mask.to(device=device)
    predictions = tuple(
        PredictionSeries(
            checkpoint_label=selected_checkpoint.alias,
            checkpoint_path=selected_checkpoint.path,
            suffix_steps=suffix_steps,
            label=_prediction_label(suffix_steps, selected_checkpoint.alias),
            predictions=_batch_predictions(
                _inference_config(item.config, suffix_steps),
                item.model,
                inputs,
                targets,
                mask,
                context_len,
                effective_rollout,
                suffix_steps,
                device,
            ).detach().cpu(),
        )
        for selected_checkpoint, item in zip(selected_checkpoints, checkpoints, strict=True)
        for suffix_steps in masked_suffix_steps
    )
    return BatchInferenceResult(
        config=config,
        inputs=inputs.detach().cpu(),
        targets=targets.detach().cpu(),
        predictions=predictions,
        context_len=context_len,
        rollout_len=effective_rollout,
        group_labels=tuple(
            _group_label(idx, key) for idx, key in enumerate(batch.active.state.group_keys)
        ),
        warning=warning,
    )


def _validate_compatible_checkpoints(checkpoints: tuple[CheckpointModel, ...]) -> None:
    reference = checkpoints[0].config
    for idx, checkpoint in enumerate(checkpoints[1:], start=2):
        config = checkpoint.config
        differences = []
        if tuple(config.data.protocols) != tuple(reference.data.protocols):
            differences.append("protocols")
        if tuple(config.data.input_columns) != tuple(reference.data.input_columns):
            differences.append("input_columns")
        if tuple(config.data.target_columns) != tuple(reference.data.target_columns):
            differences.append("target_columns")
        if int(config.loader.seq_len) != int(reference.loader.seq_len):
            differences.append("seq_len")
        if scaling_rules(config) != scaling_rules(reference):
            differences.append("scaling")
        if differences:
            raise ValueError(
                "Selected checkpoints must use the same protocols, input columns, "
                "target columns, seq_len, and scaling. "
                f"ckpt {idx} differs: {', '.join(differences)}"
            )


def _prediction_label(suffix_steps: int, checkpoint_label: str) -> str:
    if suffix_steps == 0:
        return f"prediction | classic | {checkpoint_label}"
    return f"prediction | suffix steps {suffix_steps} | {checkpoint_label}"


def _inference_config(config: ExperimentConfig, masked_suffix_steps: int) -> ExperimentConfig:
    masked_suffix = replace(
        config.validation.masked_suffix,
        enabled=masked_suffix_steps > 0,
        suffix_steps=masked_suffix_steps if masked_suffix_steps > 0 else None,
    )
    return replace(config, validation=replace(config.validation, masked_suffix=masked_suffix))


def _batch_predictions(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    rollout_len: int,
    masked_suffix_steps: int,
    device: torch.device,
) -> torch.Tensor:
    pairs = list(feedback_pairs(config))
    results = []
    for sample_idx in range(int(inputs.shape[0])):
        sample_inputs = inputs[sample_idx : sample_idx + 1]
        sample_targets = targets[sample_idx : sample_idx + 1]
        sample_mask = mask[sample_idx : sample_idx + 1]
        if masked_suffix_steps > 0:
            result = masked_suffix_rollout_predictions(
                config,
                model,
                sample_inputs,
                sample_targets,
                sample_mask,
                context_len,
                rollout_len,
                pairs,
                device,
            )
        else:
            result = one_step_rollout_predictions(
                config,
                model,
                sample_inputs,
                sample_targets,
                sample_mask,
                context_len,
                rollout_len,
                pairs,
                device,
            )
        results.append(result.prediction)
    return torch.stack(results, dim=0)


def _build_inference_widget(result: BatchInferenceResult, batch_index: int) -> object:
    config = result.config
    columns = tuple(dict.fromkeys((*config.data.input_columns, *config.data.target_columns)))
    axis_col = _axis_column(DatasetProtocolId(config.data.protocols[0]))
    scaling = scaling_rules(config)
    input_cpu = inverse_scale_tensor(result.inputs, config.data.input_columns, scaling)
    target_cpu = inverse_scale_tensor(result.targets, config.data.target_columns, scaling)
    prediction_series = tuple(
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
    label = result.group_labels[batch_index]
    fig, height = make_timeseries_figure(columns, axis_col, label)
    widget = make_trace_resampler(fig, height, max_batch_rows=None)
    frame = _batch_plot_frame(
        config,
        input_cpu,
        target_cpu,
        prediction_series,
        result.context_len,
        result.rollout_len,
        batch_index,
        columns,
        axis_col,
    )
    lf = frame.lazy()
    row_count = frame.height
    trace_labels = ("ground truth", *(series.label for series in prediction_series))
    colors = colors_by_label(trace_labels)
    shown_roles: set[str] = set()
    for row_idx, column in enumerate(columns, start=1):
        target_idx = _column_index(config.data.target_columns, column)
        add_registered_xy_trace(
            fig,
            widget,
            lf,
            x_col=axis_col,
            y_col=_base_col(column),
            row=row_idx,
            col=1,
            label="ground truth",
            color=colors["ground truth"],
            showlegend=consume_showlegend("ground truth", shown_roles),
            hovertemplate=axis_hovertemplate("ground truth", axis_col, _base_col(column)),
            row_count=row_count,
        )
        if target_idx is not None:
            for series in prediction_series:
                y_col = _pred_col(column, series.checkpoint_label, series.suffix_steps)
                add_registered_xy_trace(
                    fig,
                    widget,
                    lf,
                    x_col=axis_col,
                    y_col=y_col,
                    row=row_idx,
                    col=1,
                    label=series.label,
                    color=colors[series.label],
                    showlegend=consume_showlegend(series.label, shown_roles),
                    hovertemplate=axis_hovertemplate(series.label, axis_col, y_col),
                    row_count=row_count,
                )
    return wrap_anywidget_blocks((widget,))[0]


def _batch_plot_frame(
    config: ExperimentConfig,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: tuple[PredictionSeries, ...],
    context_len: int,
    rollout_len: int,
    batch_index: int,
    columns: tuple[str, ...],
    axis_col: str,
) -> pl.DataFrame:
    x_values = _time_axis(config, inputs[batch_index : batch_index + 1])
    data: dict[str, list[float | None]] = {axis_col: x_values}
    for column in columns:
        target_idx = _column_index(config.data.target_columns, column)
        input_idx = _column_index(config.data.input_columns, column)
        if target_idx is not None:
            base = targets[batch_index, :, target_idx].tolist()
            for series in predictions:
                pred = [None] * len(x_values)
                values = series.predictions[batch_index, :rollout_len, target_idx].tolist()
                pred[context_len : context_len + len(values)] = values
                data[_pred_col(column, series.checkpoint_label, series.suffix_steps)] = pred
        elif input_idx is not None:
            base = inputs[batch_index, :, input_idx].tolist()
        else:
            base = [None] * len(x_values)
        data[_base_col(column)] = [None if value is None else float(value) for value in base]
    return pl.DataFrame(data)


def _time_axis(config: ExperimentConfig, inputs: torch.Tensor) -> list[float]:
    dt_idx = _column_index(config.data.input_columns, BaseColumns.dt)
    if dt_idx is None:
        return [float(idx) for idx in range(int(inputs.shape[1]))]
    dt = inputs[0, :, dt_idx].to(dtype=torch.float32).clamp_min(0.0)
    elapsed = torch.cumsum(dt, dim=0) - dt[0]
    return [float(value) for value in elapsed.tolist()]


def _column_index(columns: tuple[str, ...], column: str) -> int | None:
    try:
        return columns.index(column)
    except ValueError:
        return None


def _base_col(column: str) -> str:
    return f"base::{column}"


def _pred_col(column: str, checkpoint_label: str, suffix_steps: int) -> str:
    return f"prediction::{checkpoint_label}::{suffix_steps}::{column}"


def _group_label(batch_index: int, group_key: tuple[object, ...]) -> str:
    del batch_index
    if not group_key:
        return "inference"
    protocol = group_key[-1]
    identifiers = tuple(str(item) for item in group_key[1:-1]) or tuple(
        str(item) for item in group_key
    )
    return ":".join((*identifiers, str(protocol)))


def _axis_column(protocol: DatasetProtocolId) -> str:
    specs = {
        DatasetProtocolId.cycling: BatteryProtocols.cyc,
        DatasetProtocolId.hppc: BatteryProtocols.hppc,
        DatasetProtocolId.rpt: BatteryProtocols.rpt,
        DatasetProtocolId.eis: BatteryProtocols.eis,
    }
    return specs[protocol].axis_col
