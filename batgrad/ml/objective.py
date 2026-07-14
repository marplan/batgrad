from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from batgrad.ml.distributed import globally_normalized_backward_scale
from batgrad.ml.experiment import encode_inputs, feedback_pairs, model_autocast, target_ranges
from batgrad.ml.loss import loss_metrics_from_logits, target_valid_mask
from batgrad.ml.masked_suffix import (
    attention_mask_or_none,
    create_feedback_buffer,
    masked_suffix_loss_mask,
    masked_suffix_window_slices,
    masked_suffix_windows,
    masked_window_forward,
    refresh_mamba_states_from_feedback,
)
from batgrad.ml.metrics import LossMetrics
from batgrad.ml.nn import decode_categorical_logits

if TYPE_CHECKING:
    from collections.abc import Callable

    from batgrad.ml.config import ExperimentConfig, MaskedSuffixConfig
    from batgrad.ml.masked_suffix import MaskedWindowOutput
    from batgrad.ml.nn import MambaCarryState


@dataclass(frozen=True, slots=True)
class ObjectiveWindowTrace:
    """Exact detached outputs and loss mask for one objective model call."""

    start: int
    logits: torch.Tensor
    predictions: torch.Tensor
    loss_mask: torch.Tensor
    prediction_slice: slice
    target_slice: slice


@dataclass(frozen=True, slots=True)
class ObjectiveTrace:
    """Detached objective inputs plus a merged display trajectory and exact windows."""

    inputs: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor
    mask: torch.Tensor
    mask_boundaries: tuple[int, ...]
    context_len: int
    roll_forward_steps: int
    windows: tuple[ObjectiveWindowTrace, ...] = ()


def batch_loss_with_metrics(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    *,
    suffix: MaskedSuffixConfig,
    include_rmse: bool = False,
    mask_all_valid: bool | None = None,
    initial_mamba_states: dict[str, MambaCarryState] | None = None,
    return_mamba_states: bool = False,
    trace_callback: Callable[[ObjectiveTrace], None] | None = None,
) -> LossMetrics:
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    mask = mask.to(device=device)
    if not suffix.enabled:
        return teacher_forced_loss_with_metrics(
            config,
            model,
            inputs,
            targets,
            mask,
            device,
            include_rmse=include_rmse,
            mask_all_valid=mask_all_valid,
            trace_callback=trace_callback,
        )
    return masked_suffix_loss_with_metrics(
        config,
        model,
        inputs,
        targets,
        mask,
        device,
        suffix=suffix,
        include_rmse=include_rmse,
        mask_all_valid=mask_all_valid,
        initial_mamba_states=initial_mamba_states,
        return_mamba_states=return_mamba_states,
        trace_callback=trace_callback,
    )


def backward_batch_loss_with_metrics(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    suffix: MaskedSuffixConfig,
    collect_metrics: bool,
    mask_all_valid: bool | None = None,
    initial_mamba_states: dict[str, MambaCarryState] | None = None,
    return_mamba_states: bool = False,
    trace_callback: Callable[[ObjectiveTrace], None] | None = None,
) -> LossMetrics:
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    mask = mask.to(device=device)
    local_count = batch_loss_count(config, targets, mask, suffix=suffix)
    backward_scale = globally_normalized_backward_scale(local_count)
    if suffix.enabled and suffix.detach_between_windows and suffix.roll_forward_steps > 0:
        return masked_suffix_loss_with_metrics(
            config,
            model,
            inputs,
            targets,
            mask,
            device,
            suffix=suffix,
            backward_scaler=scaler,
            backward_scale=backward_scale,
            collect_metrics=collect_metrics,
            mask_all_valid=mask_all_valid,
            initial_mamba_states=initial_mamba_states,
            return_mamba_states=return_mamba_states,
            trace_callback=trace_callback,
        )
    metrics = batch_loss_with_metrics(
        config,
        model,
        inputs,
        targets,
        mask,
        device,
        suffix=suffix,
        mask_all_valid=mask_all_valid,
        initial_mamba_states=initial_mamba_states,
        return_mamba_states=return_mamba_states,
        trace_callback=trace_callback,
    )
    scaler.scale(metrics.loss * local_count * backward_scale).backward()
    if collect_metrics:
        return metrics
    return LossMetrics(loss=metrics.loss, mamba_states=metrics.mamba_states)


def teacher_forced_loss_with_metrics(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    *,
    include_rmse: bool = False,
    mask_all_valid: bool | None = None,
    trace_callback: Callable[[ObjectiveTrace], None] | None = None,
) -> LossMetrics:
    with model_autocast(config, device):
        logits = cast(
            "torch.Tensor",
            model(
                encode_inputs(config, inputs, device),
                mask=attention_mask_or_none(mask, all_valid=mask_all_valid),
            ),
        )
        metrics = loss_metrics_from_logits(
            logits,
            targets,
            mask,
            sigma=config.model.output_sigma,
            target_ranges=target_ranges(config, device),
            include_rmse=include_rmse,
        )
        if trace_callback is not None:
            with torch.no_grad():
                predictions = decode_categorical_logits(
                    logits.detach(), target_ranges(config, device)
                )
            trace_callback(
                ObjectiveTrace(
                    inputs.detach().cpu(),
                    targets.detach().cpu(),
                    predictions.cpu(),
                    mask.detach().cpu(),
                    (),
                    int(inputs.shape[1]),
                    0,
                    (
                        ObjectiveWindowTrace(
                            0,
                            logits.detach().cpu(),
                            predictions.cpu(),
                            mask.detach().cpu(),
                            slice(0, int(logits.shape[1])),
                            slice(0, int(logits.shape[1])),
                        ),
                    ),
                )
            )
        return metrics


def masked_suffix_loss_with_metrics(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    *,
    suffix: MaskedSuffixConfig,
    backward_scaler: torch.amp.GradScaler | None = None,
    backward_scale: torch.Tensor | None = None,
    collect_metrics: bool = True,
    include_rmse: bool = False,
    mask_all_valid: bool | None = None,
    initial_mamba_states: dict[str, MambaCarryState] | None = None,
    return_mamba_states: bool = False,
    trace_callback: Callable[[ObjectiveTrace], None] | None = None,
) -> LossMetrics:
    context_len = int(inputs.shape[1]) - suffix.roll_forward_steps
    if context_len <= suffix.suffix_steps:
        raise ValueError("masked suffix roll_forward_steps leaves too little context")
    feedback = create_feedback_buffer(inputs, config.model.num_bins)
    pairs = feedback_pairs(config, suffix.channels)
    target_indices = tuple(target_idx for target_idx, _input_idx in pairs)
    input_indices = tuple(input_idx for _target_idx, input_idx in pairs)
    windows = masked_suffix_windows(suffix.suffix_steps, suffix.roll_forward_steps, context_len)
    total_loss = torch.zeros((), dtype=targets.dtype, device=device)
    total_count = torch.zeros((), dtype=targets.dtype, device=device)
    feature_loss_sum = torch.zeros(
        (len(config.data.target_columns),), dtype=torch.float32, device=device
    )
    feature_loss_count = torch.zeros_like(feature_loss_sum)
    feature_squared_error_sum = torch.zeros_like(feature_loss_sum)
    feature_squared_error_count = torch.zeros_like(feature_loss_sum)
    states = initial_mamba_states
    final_window_start_states: dict[str, MambaCarryState] | None = None
    final_window_start = 0
    ranges = target_ranges(config, device)
    trace_predictions = (
        torch.full_like(targets, float("nan"), device=device)
        if trace_callback is not None
        else None
    )
    trace_boundaries: list[int] = []
    trace_windows: list[ObjectiveWindowTrace] = []
    for start, current_suffix_steps, next_shift_steps in windows:
        final_window_start = start
        final_window_start_states = states
        output = masked_window_forward(
            config,
            model,
            feedback,
            mask,
            start=start,
            context_len=context_len,
            suffix_steps=current_suffix_steps,
            next_shift_steps=next_shift_steps,
            input_indices=input_indices,
            target_indices=target_indices,
            states=states,
            suffix=suffix,
            device=device,
            use_attention_mask=True,
            mask_all_valid=mask_all_valid,
        )
        states = output.next_states
        window_targets = targets[:, start : start + context_len, :]
        window_mask = mask[:, start : start + context_len]
        loss_mask = masked_suffix_loss_mask(
            window_mask,
            output.prediction_slice,
            target_indices,
            len(config.data.target_columns),
            loss_on_masked_only=suffix.loss_on_masked_only,
        )
        if trace_predictions is not None:
            trace_windows.append(
                _capture_masked_window_trace(
                    trace_predictions,
                    output,
                    start,
                    context_len,
                    ranges,
                    loss_mask,
                )
            )
            trace_boundaries.append(start + context_len - current_suffix_steps)
        current = loss_metrics_from_logits(
            output.logits,
            window_targets,
            loss_mask,
            sigma=config.model.output_sigma,
            target_ranges=ranges,
            include_rmse=include_rmse,
        )
        if current.feature_loss_sum is None or current.feature_loss_count is None:
            raise RuntimeError("loss calculation did not return feature components")
        total_loss, total_count = _accumulate_window_loss(
            total_loss,
            total_count,
            current.feature_loss_sum.sum(),
            current.feature_loss_count.sum(),
            backward_scaler,
            backward_scale,
        )
        if collect_metrics:
            feature_loss_sum += current.feature_loss_sum.detach()
            feature_loss_count += current.feature_loss_count.detach()
            if current.feature_squared_error_sum is not None:
                feature_squared_error_sum += current.feature_squared_error_sum.detach()
            if current.feature_squared_error_count is not None:
                feature_squared_error_count += current.feature_squared_error_count.detach()
    final_states = None
    if return_mamba_states:
        final_states = refresh_mamba_states_from_feedback(
            config,
            model,
            feedback,
            mask,
            final_window_start,
            context_len,
            final_window_start_states,
            device,
            mask_all_valid=mask_all_valid,
        )
    metrics = LossMetrics(
        loss=total_loss * 0.0 if bool((total_count <= 0).item()) else total_loss / total_count,
        feature_loss_sum=feature_loss_sum if collect_metrics else None,
        feature_loss_count=feature_loss_count if collect_metrics else None,
        feature_squared_error_sum=feature_squared_error_sum
        if collect_metrics and include_rmse
        else None,
        feature_squared_error_count=feature_squared_error_count
        if collect_metrics and include_rmse
        else None,
        mamba_states=final_states,
    )
    if trace_callback is not None and trace_predictions is not None:
        trace_callback(
            ObjectiveTrace(
                inputs.detach().cpu(),
                targets.detach().cpu(),
                trace_predictions.detach().cpu(),
                mask.detach().cpu(),
                tuple(trace_boundaries),
                context_len,
                suffix.roll_forward_steps,
                tuple(trace_windows),
            )
        )
    return metrics


def _capture_masked_window_trace(
    trace_predictions: torch.Tensor,
    output: MaskedWindowOutput,
    start: int,
    context_len: int,
    ranges: torch.Tensor,
    loss_mask: torch.Tensor,
) -> ObjectiveWindowTrace:
    with torch.no_grad():
        decoded = decode_categorical_logits(output.logits.detach(), ranges)
    window_predictions = trace_predictions[:, start : start + context_len, :]
    trace_predictions[:, start : start + context_len, :] = torch.where(
        torch.isnan(window_predictions),
        decoded,
        window_predictions,
    )
    trace_predictions[:, output.target_slice, :] = decoded[:, output.prediction_slice, :]
    return ObjectiveWindowTrace(
        start,
        output.logits.detach().cpu(),
        decoded.detach().cpu(),
        loss_mask.detach().cpu(),
        output.prediction_slice,
        output.target_slice,
    )


def _accumulate_window_loss(
    total_loss: torch.Tensor,
    total_count: torch.Tensor,
    loss_sum: torch.Tensor,
    loss_count: torch.Tensor,
    backward_scaler: torch.amp.GradScaler | None,
    backward_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backward_scaler is None:
        return total_loss + loss_sum, total_count + loss_count
    if backward_scale is None:
        raise ValueError("backward_scale is required for per-window backward")
    backward_scaler.scale(loss_sum * backward_scale).backward()
    return total_loss + loss_sum.detach(), total_count + loss_count.detach()


def batch_loss_count(
    config: ExperimentConfig,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    suffix: MaskedSuffixConfig,
) -> torch.Tensor:
    if not suffix.enabled:
        return target_valid_mask(mask, targets, targets.device).to(dtype=torch.float32).sum()
    context_len = int(targets.shape[1]) - suffix.roll_forward_steps
    if context_len <= suffix.suffix_steps:
        raise ValueError("masked suffix roll_forward_steps leaves too little context")
    target_indices = tuple(
        target_idx for target_idx, _input_idx in feedback_pairs(config, suffix.channels)
    )
    windows = masked_suffix_windows(suffix.suffix_steps, suffix.roll_forward_steps, context_len)
    total = torch.zeros((), dtype=torch.float32, device=mask.device)
    for start, current_suffix_steps, _next_shift_steps in windows:
        window_targets = targets[:, start : start + context_len, :]
        window_mask = mask[:, start : start + context_len]
        _input_suffix, prediction_slice, _write, _target = masked_suffix_window_slices(
            start, current_suffix_steps, context_len
        )
        loss_mask = masked_suffix_loss_mask(
            window_mask,
            prediction_slice,
            target_indices,
            len(config.data.target_columns),
            loss_on_masked_only=suffix.loss_on_masked_only,
        )
        total += (
            target_valid_mask(loss_mask, window_targets, targets.device)
            .to(dtype=torch.float32)
            .sum()
        )
    return total
