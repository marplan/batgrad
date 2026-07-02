from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.logging import configure_logger, get_logger
from batgrad.ml.config import ExperimentConfig, config_to_dict, load_experiment_config
from batgrad.ml.data.config import (
    LoaderConfig as DataLoaderConfig,
    ScalingRule,
    ValidationConfig as DataValidationConfig,
    WindowConfig,
)
from batgrad.ml.data.loader import MlDataIterable, create_dataloader
from batgrad.ml.data.materialization import materialize_window_ref
from batgrad.ml.data.planning import WindowRef, build_stream_plans
from batgrad.ml.data.scaling import inverse_scale_tensor, scale_data
from batgrad.ml.loggers import build_logger
from batgrad.ml.nn import (
    SequenceMixer,
    categorical_ce_loss,
    categorical_ce_loss_components,
    decode_categorical_logits,
    encode_categorical_values,
)
from batgrad.storage.local import LocalDataProcessingStore

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batgrad.ml.data.batch import Batch
    from batgrad.ml.loggers import RunLogger
    from batgrad.ml.nn import MambaCarryState

logger = get_logger(__name__)
PLOT_SEQUENCE_RANK = 2


def train_from_config(path: str | Path) -> Path | None:
    config = load_experiment_config(path)
    run_dir = _prepare_run_dir(config)

    torch.manual_seed(config.seed)
    device = torch.device(config.run.device)
    store = LocalDataProcessingStore(config.data.store_root)
    train_loader, val_loader, train_dataset = _create_loaders(config, store)
    max_steps = config.train.max_steps or _max_steps_for_epochs(train_dataset, config.train.epochs)
    model: torch.nn.Module = SequenceMixer(
        config.model,
        input_dim=len(config.data.input_columns),
        output_dim=len(config.data.target_columns),
        device=device,
    ).to(device)
    if config.run.compile_model:
        model = cast("torch.nn.Module", torch.compile(model))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = _build_scheduler(config, optimizer, max_steps)
    logger = build_logger(
        config.logging, run_dir, cast("dict[str, object]", config_to_dict(config))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.run.use_amp and device.type == "cuda")
    step = 0
    epoch_idx = 0
    try:
        while step < max_steps:
            steps_per_epoch = train_dataset.steps_per_epoch(epoch_idx)
            if steps_per_epoch <= 0:
                raise ValueError("training split produced no batches")
            log_every = _cadence_steps(
                config.train.log_every_steps, config.train.log_per_epoch, steps_per_epoch
            )
            validate_every = _cadence_steps(
                config.train.validate_every_steps,
                config.train.validate_per_epoch,
                steps_per_epoch,
                allow_disabled=True,
            )
            train_dataset.set_epoch(epoch_idx)
            for epoch_step, batch in enumerate(train_loader, start=1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(
                    config,
                    model,
                    batch.active.inputs,
                    batch.active.targets,
                    batch.active.mask,
                    device,
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                if step % log_every == 0 or step == 1:
                    epoch, epoch_pct = _epoch_progress(epoch_idx, epoch_step, steps_per_epoch)
                    logger.log_metrics(
                        step,
                        {
                            "train/loss": float(loss.detach().cpu()),
                            "lr": float(scheduler.get_last_lr()[0]),
                        },
                        epoch=epoch,
                        epoch_pct=epoch_pct,
                    )
                if validate_every and step % validate_every == 0:
                    epoch, epoch_pct = _epoch_progress(epoch_idx, epoch_step, steps_per_epoch)
                    _validate(
                        config,
                        model,
                        val_loader,
                        train_dataset,
                        store,
                        device,
                        logger,
                        step,
                        epoch,
                        epoch_pct,
                    )
                if step >= max_steps:
                    break
            epoch_idx += 1
        _save_checkpoint(config, model, run_dir)
    finally:
        logger.finish()
    return run_dir


def _create_loaders(
    config: ExperimentConfig,
    store: LocalDataProcessingStore,
) -> tuple[Iterable[Batch], Iterable[Batch], MlDataIterable]:
    train_loader = create_dataloader(
        store=store,
        manifest_paths=config.data.manifest_paths,
        input_columns=config.data.input_columns,
        target_columns=config.data.target_columns,
        protocols=config.data.protocols,
        active_protocol=config.data.protocols[0],
        validation=_data_validation_config(config),
        scaling=_scaling_rules(config),
        config=_loader_config(
            config, BaseColumns.split.values.train, expand_roll_forward=True
        ),
    )
    val_loader = create_dataloader(
        store=store,
        manifest_paths=config.data.manifest_paths,
        input_columns=config.data.input_columns,
        target_columns=config.data.target_columns,
        protocols=config.data.protocols,
        active_protocol=config.data.protocols[0],
        validation=_data_validation_config(config),
        scaling=_scaling_rules(config),
        config=_loader_config(
            config, BaseColumns.split.values.val, expand_roll_forward=False
        ),
    )
    return (train_loader, val_loader, _dataset(train_loader))


def _prepare_run_dir(config: ExperimentConfig) -> Path | None:
    run_dir = _make_run_dir(config)
    if run_dir is None:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_to_dict(config), file, indent=2)
    return run_dir


def _save_checkpoint(
    config: ExperimentConfig, model: torch.nn.Module, run_dir: Path | None
) -> None:
    if not config.checkpoint.enabled or run_dir is None:
        return
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config_to_dict(config)},
        checkpoint_dir / "final.pt",
    )


def _cadence_steps(
    every_steps: int | None,
    per_epoch: int,
    steps_per_epoch: int,
    *,
    allow_disabled: bool = False,
) -> int:
    if every_steps is not None:
        return every_steps
    if allow_disabled and per_epoch == 0:
        return 0
    return max(1, steps_per_epoch // per_epoch)


def _max_steps_for_epochs(dataset: MlDataIterable, epochs: float) -> int:
    full_epochs = math.floor(epochs)
    partial = epochs - full_epochs
    total = sum(dataset.steps_per_epoch(epoch_idx) for epoch_idx in range(full_epochs))
    if partial > 0.0:
        total += math.ceil(dataset.steps_per_epoch(full_epochs) * partial)
    return max(1, total)


def _epoch_progress(epoch_idx: int, epoch_step: int, steps_per_epoch: int) -> tuple[int, int]:
    pct = math.ceil(epoch_step / steps_per_epoch * 100)
    return epoch_idx + 1, min(100, max(1, pct))


def _batch_loss(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    mask = mask.to(device=device)
    if not config.train.masked_suffix.enabled:
        return _teacher_forced_loss(config, model, inputs, targets, mask, device)

    return _masked_suffix_loss(config, model, inputs, targets, mask, device)


def _teacher_forced_loss(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(
        device_type=device.type, enabled=config.run.use_amp and device.type == "cuda"
    ):
        logits = model(_encode_inputs(config, inputs, device), mask=mask)
        return categorical_ce_loss(
            cast("torch.Tensor", logits),
            targets,
            mask,
            config.model.output_sigma,
            _target_ranges(config, device),
        )


def _masked_suffix_loss(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    suffix = config.train.masked_suffix
    context_len = int(inputs.shape[1]) - suffix.roll_forward_steps
    if context_len <= suffix.suffix_steps:
        raise ValueError("masked suffix roll_forward_steps leaves too little context")
    feedback = inputs.clone()
    target_indices = tuple(config.data.target_columns.index(name) for name in suffix.channels)
    input_indices = tuple(config.data.input_columns.index(name) for name in suffix.channels)
    windows = _masked_suffix_windows(suffix.suffix_steps, suffix.roll_forward_steps, context_len)
    total_loss = torch.zeros((), dtype=targets.dtype, device=device)
    total_count = torch.zeros((), dtype=targets.dtype, device=device)
    feedback_bin_overrides = torch.zeros(
        (*feedback.shape, config.model.num_bins), dtype=feedback.dtype, device=feedback.device
    )
    feedback_bin_override_mask = torch.zeros_like(feedback_bin_overrides, dtype=torch.bool)
    states: dict[str, MambaCarryState] | None = None
    for start, current_suffix_steps, next_shift_steps in windows:
        window_inputs = feedback[:, start : start + context_len, :].clone()
        window_targets = targets[:, start : start + context_len, :]
        window_mask = mask[:, start : start + context_len]
        suffix_slice = slice(context_len - current_suffix_steps, context_len)
        masked_input_mask = torch.zeros_like(window_inputs, dtype=torch.bool)
        for input_idx in input_indices:
            window_inputs[:, suffix_slice, input_idx] = suffix.fill_value
            masked_input_mask[:, suffix_slice, input_idx] = True
        loss_mask = _masked_suffix_loss_mask(
            window_mask,
            suffix_slice,
            target_indices,
            len(config.data.target_columns),
            loss_on_masked_only=suffix.loss_on_masked_only,
        )
        with torch.autocast(
            device_type=device.type, enabled=config.run.use_amp and device.type == "cuda"
        ):
            encoded_window_inputs = _encode_inputs(
                config, window_inputs, device, masked_input_mask=masked_input_mask
            )
            encoded_window_inputs = _apply_binned_feedback_overrides(
                encoded_window_inputs,
                feedback_bin_overrides,
                feedback_bin_override_mask,
                slice(start, start + context_len),
            )
            result = model(
                encoded_window_inputs,
                mask=window_mask,
                states=states,
                return_states=suffix.carry_mamba_state,
            )
            if suffix.carry_mamba_state:
                logits, _final_states = cast(
                    "tuple[torch.Tensor, dict[str, MambaCarryState]]", result
                )
            else:
                logits = cast("torch.Tensor", result)
            loss_sum, loss_count = categorical_ce_loss_components(
                logits,
                window_targets,
                loss_mask,
                config.model.output_sigma,
                _target_ranges(config, device),
            )
            total_loss = total_loss + loss_sum
            total_count = total_count + loss_count
        write_slice = slice(start + context_len - current_suffix_steps, start + context_len)
        _write_masked_suffix_feedback(
            config,
            feedback,
            feedback_bin_overrides,
            feedback_bin_override_mask,
            logits,
            suffix_slice,
            write_slice,
            target_indices,
            input_indices,
            device,
        )
        if suffix.carry_mamba_state and next_shift_steps > 0:
            with torch.autocast(
                device_type=device.type, enabled=config.run.use_amp and device.type == "cuda"
            ):
                states = _prefix_mamba_states(
                    model,
                    encoded_window_inputs,
                    window_mask,
                    states,
                    next_shift_steps,
                )
            if suffix.detach_between_windows:
                states = {key: value.detach() for key, value in states.items()}
        else:
            states = None
    if bool((total_count <= 0).item()):
        return torch.zeros((), dtype=targets.dtype, device=device)
    return total_loss / total_count


def _masked_suffix_windows(
    suffix_steps: int, roll_forward_steps: int, context_len: int
) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = [
        (0, min(suffix_steps, context_len), min(suffix_steps, roll_forward_steps))
    ]
    remaining = roll_forward_steps
    start = 0
    while remaining > 0:
        shift = min(suffix_steps, remaining)
        start += shift
        remaining -= shift
        next_shift_steps = min(suffix_steps, remaining)
        windows.append((start, min(suffix_steps, context_len), next_shift_steps))
    return windows


def _prefix_mamba_states(
    model: torch.nn.Module,
    encoded_inputs: torch.Tensor,
    mask: torch.Tensor,
    states: dict[str, MambaCarryState] | None,
    prefix_steps: int,
) -> dict[str, MambaCarryState]:
    if prefix_steps <= 0:
        raise ValueError(f"prefix_steps must be > 0, got {prefix_steps}")
    _logits, next_states = cast(
        "tuple[torch.Tensor, dict[str, MambaCarryState]]",
        model(
            encoded_inputs[:, :prefix_steps, :, :],
            mask=mask[:, :prefix_steps],
            states=states,
            return_states=True,
        ),
    )
    return next_states


def _masked_suffix_loss_mask(
    window_mask: torch.Tensor,
    suffix_slice: slice,
    target_indices: tuple[int, ...],
    target_count: int,
    *,
    loss_on_masked_only: bool,
) -> torch.Tensor:
    if not loss_on_masked_only:
        return window_mask
    loss_mask = torch.zeros(
        (*window_mask.shape, target_count), dtype=torch.bool, device=window_mask.device
    )
    for target_idx in target_indices:
        loss_mask[:, suffix_slice, target_idx] = window_mask[:, suffix_slice]
    return loss_mask


def _apply_binned_feedback_overrides(
    encoded_inputs: torch.Tensor,
    feedback_bin_overrides: torch.Tensor,
    feedback_bin_override_mask: torch.Tensor,
    override_slice: slice,
) -> torch.Tensor:
    return torch.where(
        feedback_bin_override_mask[:, override_slice, :, :],
        feedback_bin_overrides[:, override_slice, :, :],
        encoded_inputs,
    )


def _write_masked_suffix_feedback(
    config: ExperimentConfig,
    feedback: torch.Tensor,
    feedback_bin_overrides: torch.Tensor,
    feedback_bin_override_mask: torch.Tensor,
    logits: torch.Tensor,
    suffix_slice: slice,
    write_slice: slice,
    target_indices: tuple[int, ...],
    input_indices: tuple[int, ...],
    device: torch.device,
) -> None:
    if write_slice.stop is None or write_slice.stop > int(feedback.shape[1]):
        return
    if config.model.feedback_mode == "probabilities":
        pred_bins = torch.softmax(logits.detach().float(), dim=-1).to(dtype=feedback.dtype)
        for selected_idx, input_idx in zip(target_indices, input_indices, strict=True):
            feedback_bin_overrides[:, write_slice, input_idx, :] = pred_bins[
                :, suffix_slice, selected_idx, :
            ]
            feedback_bin_override_mask[:, write_slice, input_idx, :] = True
        return
    pred = decode_categorical_logits(logits, _target_ranges(config, device)).detach()
    for selected_idx, input_idx in zip(target_indices, input_indices, strict=True):
        feedback[:, write_slice, input_idx] = pred[:, suffix_slice, selected_idx]


def _validation_masked_suffix_enabled(config: ExperimentConfig) -> bool:
    enabled = config.validation.masked_suffix.enabled
    return config.train.masked_suffix.enabled if enabled is None else enabled


def _validation_suffix_steps(config: ExperimentConfig) -> int:
    return config.validation.masked_suffix.suffix_steps or config.train.masked_suffix.suffix_steps


def _validation_carry_mamba_state(config: ExperimentConfig) -> bool:
    carry = config.validation.masked_suffix.carry_mamba_state
    return config.train.masked_suffix.carry_mamba_state if carry is None else carry


def _validation_loss_config(config: ExperimentConfig) -> ExperimentConfig:
    suffix = replace(
        config.train.masked_suffix,
        enabled=_validation_masked_suffix_enabled(config),
        suffix_steps=_validation_suffix_steps(config),
        carry_mamba_state=_validation_carry_mamba_state(config),
        roll_forward_steps=0,
    )
    return replace(config, train=replace(config.train, masked_suffix=suffix))


@dataclass(frozen=True, slots=True)
class RolloutResult:
    prediction: torch.Tensor
    loss_sum: torch.Tensor
    loss_count: torch.Tensor

    @property
    def loss(self) -> torch.Tensor | None:
        if bool((self.loss_count <= 0).item()):
            return None
        return self.loss_sum / self.loss_count


@dataclass(frozen=True, slots=True)
class RolloutPlotSeries:
    inputs: torch.Tensor
    context_prediction: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    match: dict[str, object]
    anchor: int


@torch.no_grad()
def _validate(
    config: ExperimentConfig,
    model: torch.nn.Module,
    val_loader: Iterable[Batch],
    train_dataset: MlDataIterable,
    store: LocalDataProcessingStore,
    device: torch.device,
    logger: RunLogger,
    step: int,
    epoch: int | None = None,
    epoch_pct: int | None = None,
) -> None:
    model.eval()
    if config.validation.max_tf_batches > 0:
        val_loss_config = _validation_loss_config(config)
        loss_name = (
            "val/tf/suffix_loss"
            if val_loss_config.train.masked_suffix.enabled
            else "val/tf/full_loss"
        )
        losses: list[float] = []
        for idx, batch in enumerate(val_loader):
            loss = _batch_loss(
                val_loss_config,
                model,
                batch.active.inputs,
                batch.active.targets,
                batch.active.mask,
                device,
            )
            losses.append(float(loss.detach().cpu()))
            if idx + 1 >= config.validation.max_tf_batches:
                break
        if losses:
            logger.log_metrics(
                step,
                {loss_name: sum(losses) / len(losses)},
                epoch=epoch,
                epoch_pct=epoch_pct,
            )
    if config.validation.rollout_steps > 0:
        _run_rollouts(config, model, train_dataset, store, device, logger, step)


@torch.no_grad()
def _run_rollouts(
    config: ExperimentConfig,
    model: torch.nn.Module,
    dataset: MlDataIterable,
    store: LocalDataProcessingStore,
    device: torch.device,
    logger: RunLogger,
    step: int,
) -> None:
    protocol = DatasetProtocolId(config.data.protocols[0])
    val_index = dataset.full_index.filter_split(BaseColumns.split.values.val)
    context_len = config.loader.seq_len
    stored_rollout_len = config.validation.rollout_steps
    total_rollout_len = stored_rollout_len + (
        config.validation.rollout_extension.steps
        if config.validation.rollout_extension.enabled
        else 0
    )
    window_config = replace(
        _loader_config(config, BaseColumns.split.values.val),
        default_window=WindowConfig(batch_size=1, seq_len=context_len + stored_rollout_len),
    )
    stream_plans = build_stream_plans(val_index, protocol, window_config)
    scaling = _scaling_rules(config)
    rollout_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    rollout_loss_count = torch.zeros((), dtype=torch.float32, device=device)
    plot_series: list[RolloutPlotSeries] = []
    for group in config.validation.split.groups:
        if not group.rollout_start_offsets:
            continue
        matches = [
            stream
            for stream in stream_plans
            if _stream_matches(stream.group_key, config.validation.split.group_by, group.match)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"rollout selector must match exactly one stream, got {len(matches)}: {group.match}"
            )
        stream = matches[0]
        for anchor in group.rollout_start_offsets:
            offset = max(0, anchor - context_len + 1)
            batch = materialize_window_ref(
                store,
                WindowRef(stream, offset),
                config.data.input_columns,
                config.data.target_columns,
                scaling,
                window_config,
                batch_idx=0,
            )
            inputs = batch.active.inputs.to(device=device)
            targets = batch.active.targets.to(device=device)
            rollout_mask = batch.active.mask.to(device=device)
            if config.validation.rollout_extension.enabled:
                inputs = _append_rollout_extension(config, inputs)
                targets = torch.cat(
                    (
                        targets,
                        torch.full(
                            (
                                targets.shape[0],
                                config.validation.rollout_extension.steps,
                                targets.shape[2],
                            ),
                            float("nan"),
                            dtype=targets.dtype,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
                rollout_mask = torch.cat(
                    (
                        rollout_mask,
                        torch.zeros(
                            (rollout_mask.shape[0], config.validation.rollout_extension.steps),
                            dtype=torch.bool,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
            feedback_pairs = [
                (config.data.target_columns.index(name), config.data.input_columns.index(name))
                for name in config.data.feedback_columns
                if name in config.data.target_columns and name in config.data.input_columns
            ]
            if _validation_masked_suffix_enabled(config):
                rollout_result = _masked_suffix_rollout_predictions(
                    config,
                    model,
                    inputs,
                    targets,
                    rollout_mask,
                    context_len,
                    total_rollout_len,
                    feedback_pairs,
                    device,
                )
            else:
                rollout_result = _one_step_rollout_predictions(
                    config,
                    model,
                    inputs,
                    targets,
                    rollout_mask,
                    context_len,
                    total_rollout_len,
                    feedback_pairs,
                    device,
                )
            rollout_loss_sum = rollout_loss_sum + rollout_result.loss_sum
            rollout_loss_count = rollout_loss_count + rollout_result.loss_count
            pred_tensor = rollout_result.prediction
            if int(pred_tensor.shape[0]) > 0 and config.validation.log_rollout_plots:
                context_prediction = _context_predictions(
                    config, model, inputs, context_len, device
                )
                input_tensor = inputs[
                    :, : context_len + pred_tensor.shape[0], :
                ].cpu()[0]
                target_tensor = targets[
                    :, : context_len + pred_tensor.shape[0], :
                ].cpu()[0]
                plot_series.append(
                    RolloutPlotSeries(
                        inputs=input_tensor,
                        context_prediction=context_prediction,
                        prediction=pred_tensor,
                        target=target_tensor,
                        match=group.match,
                        anchor=anchor,
                    )
                )
    if bool((rollout_loss_count > 0).item()):
        logger.log_metrics(
            step,
            {"val/rollout/loss": float((rollout_loss_sum / rollout_loss_count).detach().cpu())},
        )
    if plot_series:
        logger.log_payload(
            step,
            "val/rollout/plot",
            _build_rollout_figure(config, plot_series, context_len, logger.run_name()),
        )


def _build_rollout_figure(
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
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    scaling = _scaling_rules(config)
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
            color = colors[series_idx % len(colors)]
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


def _context_predictions(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    context_len: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = _encode_inputs(config, inputs[:, :context_len, :].clone(), device)
    logits = cast(
        "torch.Tensor",
        model(encoded, mask=torch.ones(encoded.shape[:2], dtype=torch.bool, device=device)),
    )
    return decode_categorical_logits(logits, _target_ranges(config, device)).cpu()[0]


def _one_step_rollout_predictions(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    rollout_len: int,
    feedback_pairs: list[tuple[int, int]],
    device: torch.device,
) -> RolloutResult:
    current = _encode_inputs(config, inputs[:, :context_len, :].clone(), device)
    future_inputs = inputs[:, context_len:, :]
    predictions: list[torch.Tensor] = []
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_count = torch.zeros((), dtype=torch.float32, device=device)
    states: dict[str, MambaCarryState] | None = None
    carry_mamba_state = _validation_carry_mamba_state(config)
    for future_idx in range(min(rollout_len, int(future_inputs.shape[1]))):
        window_mask = torch.ones(current.shape[:2], dtype=torch.bool, device=device)
        result = model(
            current,
            mask=window_mask,
            states=states,
            return_states=carry_mamba_state,
        )
        if carry_mamba_state:
            logits, _final_states = cast(
                "tuple[torch.Tensor, dict[str, MambaCarryState]]", result
            )
        else:
            logits = cast("torch.Tensor", result)
        pred = decode_categorical_logits(logits[:, -1:, :, :], _target_ranges(config, device))
        predictions.append(pred.cpu())
        target_slice = slice(context_len + future_idx, context_len + future_idx + 1)
        loss_sum, loss_count = categorical_ce_loss_components(
            logits[:, -1:, :, :],
            targets[:, target_slice, :],
            mask[:, target_slice],
            config.model.output_sigma,
            _target_ranges(config, device),
        )
        total_loss = total_loss + loss_sum
        total_count = total_count + loss_count
        next_scalar = future_inputs[:, future_idx : future_idx + 1, :].clone()
        next_input = _rollout_next_input_bins(
            config,
            next_scalar,
            logits[:, -1:, :, :],
            pred,
            feedback_pairs,
            device,
        )
        if carry_mamba_state:
            states = _prefix_mamba_states(
                model, current[:, :1, :, :], window_mask[:, :1], states, 1
            )
        current = torch.cat((current[:, 1:, :, :], next_input), dim=1)
    if not predictions:
        prediction = torch.empty((0, len(config.data.target_columns)), dtype=inputs.dtype)
    else:
        prediction = torch.cat(predictions, dim=1).cpu()[0]
    return RolloutResult(prediction=prediction, loss_sum=total_loss, loss_count=total_count)


def _masked_suffix_rollout_predictions(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    rollout_len: int,
    feedback_pairs: list[tuple[int, int]],
    device: torch.device,
) -> RolloutResult:
    suffix_steps = _validation_suffix_steps(config)
    feedback = inputs.clone()
    rollout_len = min(rollout_len, max(0, int(feedback.shape[1]) - context_len))
    predictions: list[torch.Tensor] = []
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_count = torch.zeros((), dtype=torch.float32, device=device)
    feedback_bin_overrides = torch.zeros(
        (*feedback.shape, config.model.num_bins), dtype=feedback.dtype, device=feedback.device
    )
    feedback_bin_override_mask = torch.zeros_like(feedback_bin_overrides, dtype=torch.bool)
    target_indices = tuple(target_idx for target_idx, _input_idx in feedback_pairs)
    input_indices = tuple(input_idx for _target_idx, input_idx in feedback_pairs)
    states: dict[str, MambaCarryState] | None = None
    carry_mamba_state = _validation_carry_mamba_state(config)
    completed = 0
    while completed < rollout_len:
        current_suffix_steps = min(suffix_steps, rollout_len - completed)
        window_end = context_len + completed + current_suffix_steps
        start = window_end - context_len
        window_inputs = feedback[:, start:window_end, :].clone()
        if int(window_inputs.shape[1]) < context_len:
            break
        suffix_slice = slice(context_len - current_suffix_steps, context_len)
        masked_input_mask = torch.zeros_like(window_inputs, dtype=torch.bool)
        for _target_idx, input_idx in feedback_pairs:
            window_inputs[:, suffix_slice, input_idx] = config.train.masked_suffix.fill_value
            masked_input_mask[:, suffix_slice, input_idx] = True
        encoded_window_inputs = _encode_inputs(
            config, window_inputs, device, masked_input_mask=masked_input_mask
        )
        encoded_window_inputs = _apply_binned_feedback_overrides(
            encoded_window_inputs,
            feedback_bin_overrides,
            feedback_bin_override_mask,
            slice(start, window_end),
        )
        window_mask = torch.ones(encoded_window_inputs.shape[:2], dtype=torch.bool, device=device)
        result = model(
            encoded_window_inputs,
            mask=window_mask,
            states=states,
            return_states=carry_mamba_state,
        )
        if carry_mamba_state:
            logits, _final_states = cast(
                "tuple[torch.Tensor, dict[str, MambaCarryState]]", result
            )
        else:
            logits = cast("torch.Tensor", result)
        pred = decode_categorical_logits(
            logits[:, suffix_slice, :, :], _target_ranges(config, device)
        )
        predictions.append(pred.cpu())
        write_slice = slice(window_end - current_suffix_steps, window_end)
        loss_mask = torch.zeros(
            (*mask[:, write_slice].shape, len(config.data.target_columns)),
            dtype=torch.bool,
            device=device,
        )
        for target_idx in target_indices:
            loss_mask[:, :, target_idx] = mask[:, write_slice]
        loss_sum, loss_count = categorical_ce_loss_components(
            logits[:, suffix_slice, :, :],
            targets[:, write_slice, :],
            loss_mask,
            config.model.output_sigma,
            _target_ranges(config, device),
        )
        total_loss = total_loss + loss_sum
        total_count = total_count + loss_count
        _write_masked_suffix_feedback(
            config,
            feedback,
            feedback_bin_overrides,
            feedback_bin_override_mask,
            logits,
            suffix_slice,
            write_slice,
            target_indices,
            input_indices,
            device,
        )
        completed += current_suffix_steps
        if carry_mamba_state and completed < rollout_len:
            states = _prefix_mamba_states(
                model,
                encoded_window_inputs,
                window_mask,
                states,
                current_suffix_steps,
            )
    if not predictions:
        prediction = torch.empty((0, len(config.data.target_columns)), dtype=inputs.dtype)
    else:
        prediction = torch.cat(predictions, dim=1).cpu()[0]
    return RolloutResult(prediction=prediction, loss_sum=total_loss, loss_count=total_count)


def _stream_matches(
    group_key: tuple[object, ...], group_by: tuple[str, ...], match: dict[str, object]
) -> bool:
    key_map = dict(zip(group_by, group_key, strict=True))
    return all(key_map.get(key) == value for key, value in match.items())


def _data_validation_config(config: ExperimentConfig) -> DataValidationConfig:
    provided = tuple(group.match for group in config.validation.split.groups)
    split = config.validation.split
    if split.strategy == "provide":
        return DataValidationConfig.provide(provided, group_by=split.group_by)
    if split.strategy == "merge":
        return DataValidationConfig.merge(
            provided, fraction=split.fraction, group_by=split.group_by
        )
    return DataValidationConfig.sample(fraction=split.fraction, group_by=split.group_by)


def _scaling_rules(config: ExperimentConfig) -> tuple[ScalingRule, ...]:
    return tuple(
        ScalingRule(
            column=rule.column,
            input_min=rule.input_min,
            input_max=rule.input_max,
            output_min=rule.output_min,
            output_max=rule.output_max,
            clip=rule.clip,
            transform=rule.transform,
        )
        for rule in config.data.scaling
    )


def _target_ranges(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    scaling_by_column = {rule.column: rule for rule in config.data.scaling}
    return torch.tensor(
        [
            [scaling_by_column[column].output_min, scaling_by_column[column].output_max]
            for column in config.data.target_columns
        ],
        dtype=torch.float32,
        device=device,
    )


def _input_ranges(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    scaling_by_column = {rule.column: rule for rule in config.data.scaling}
    return torch.tensor(
        [
            [scaling_by_column[column].output_min, scaling_by_column[column].output_max]
            for column in config.data.input_columns
        ],
        dtype=torch.float32,
        device=device,
    )


def _encode_inputs(
    config: ExperimentConfig,
    inputs: torch.Tensor,
    device: torch.device,
    *,
    masked_input_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    encoded = encode_categorical_values(
        inputs,
        config.model.num_bins,
        config.model.input_sigma,
        _input_ranges(config, device),
    )
    if masked_input_mask is None or not bool(masked_input_mask.any().item()):
        return encoded
    return torch.where(masked_input_mask.unsqueeze(-1), torch.zeros_like(encoded), encoded)


def _rollout_next_input_bins(
    config: ExperimentConfig,
    next_scalar: torch.Tensor,
    logits: torch.Tensor,
    pred_scalar: torch.Tensor,
    feedback_pairs: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    if config.model.feedback_mode == "decoded_scalar":
        for target_idx, input_idx in feedback_pairs:
            next_scalar[:, :, input_idx] = pred_scalar[:, :, target_idx]
        return _encode_inputs(config, next_scalar, device)

    next_bins = _encode_inputs(config, next_scalar, device)
    pred_bins = torch.softmax(logits.float(), dim=-1).to(dtype=next_bins.dtype)
    for target_idx, input_idx in feedback_pairs:
        next_bins[:, :, input_idx, :] = pred_bins[:, :, target_idx, :]
    return next_bins


def _append_rollout_extension(config: ExperimentConfig, inputs: torch.Tensor) -> torch.Tensor:
    extension = config.validation.rollout_extension
    if not extension.enabled or extension.steps <= 0:
        return inputs
    suffix = inputs[:, -1:, :].clone().repeat(1, extension.steps, 1)
    scaling = _scaling_rules(config)
    for column, physical_value in extension.input_values.items():
        column_idx = config.data.input_columns.index(column)
        value = torch.tensor([[[float(physical_value)]]], dtype=inputs.dtype, device=inputs.device)
        rule = tuple(rule for rule in scaling if rule.name == column)
        if not rule:
            raise ValueError(f"Missing scaling rule for rollout extension column: {column}")
        suffix[:, :, column_idx] = scale_data(value, rule).reshape(())
    return torch.cat((inputs, suffix), dim=1)


def _loader_config(
    config: ExperimentConfig, split: str, *, expand_roll_forward: bool = False
) -> DataLoaderConfig:
    seq_len = config.loader.seq_len
    if expand_roll_forward and config.train.masked_suffix.enabled:
        seq_len += config.train.masked_suffix.roll_forward_steps
    return DataLoaderConfig(
        split=split,
        default_window=WindowConfig(
            batch_size=config.loader.batch_size, seq_len=seq_len
        ),
        strategy=config.loader.strategy,
        active_protocol=DatasetProtocolId(config.data.protocols[0]),
        stateful_n_windows=config.loader.stateful_n_windows,
        data_access=config.loader.data_access,
        num_workers=config.loader.num_workers,
        prefetch_to_device=config.loader.prefetch_to_device,
        device=config.run.device,
        multiprocessing_context=None if config.loader.num_workers == 0 else "spawn",
    )


def _dataset(loader: object) -> MlDataIterable:
    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, MlDataIterable):
        raise TypeError("expected loader created by batgrad.ml.data.create_dataloader")
    return dataset


def _build_scheduler(
    config: ExperimentConfig, optimizer: torch.optim.Optimizer, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.scheduler.kind == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    warmup_steps = max(1, int(max_steps * config.scheduler.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(max(1, max_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.scheduler.min_lr_ratio + (1.0 - config.scheduler.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _make_run_dir(config: ExperimentConfig) -> Path | None:
    if config.run.output_dir is None:
        return None
    base = Path(config.run.output_dir)
    name = config.run.name or datetime.now(tz=UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = base / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return run_dir


def main() -> None:
    configure_logger()
    parser = argparse.ArgumentParser(description="Run a compact batgrad ML training job")
    parser.add_argument("--config", required=True)
    run_dir = train_from_config(parser.parse_args().config)
    if run_dir is not None:
        logger.info("run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
