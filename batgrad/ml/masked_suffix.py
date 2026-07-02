from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from batgrad.ml.nn import (
    categorical_ce_loss,
    categorical_ce_loss_components,
    decode_categorical_logits,
)
from batgrad.ml.train_utils import encode_inputs, target_ranges

if TYPE_CHECKING:
    from batgrad.ml.config import ExperimentConfig
    from batgrad.ml.nn import MambaCarryState


def batch_loss(
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
        return teacher_forced_loss(config, model, inputs, targets, mask, device)

    return masked_suffix_loss(config, model, inputs, targets, mask, device)


def backward_batch_loss(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> torch.Tensor:
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    mask = mask.to(device=device)
    suffix = config.train.masked_suffix
    if suffix.enabled and suffix.detach_between_windows and suffix.roll_forward_steps > 0:
        return masked_suffix_loss(
            config,
            model,
            inputs,
            targets,
            mask,
            device,
            backward_scaler=scaler,
        )
    loss = batch_loss(config, model, inputs, targets, mask, device)
    scaler.scale(loss).backward()
    return loss


def teacher_forced_loss(
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
        logits = model(encode_inputs(config, inputs, device), mask=mask)
        return categorical_ce_loss(
            cast("torch.Tensor", logits),
            targets,
            mask,
            config.model.output_sigma,
            target_ranges(config, device),
        )


def masked_suffix_loss(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    backward_scaler: torch.amp.GradScaler | None = None,
) -> torch.Tensor:
    suffix = config.train.masked_suffix
    context_len = int(inputs.shape[1]) - suffix.roll_forward_steps
    if context_len <= suffix.suffix_steps:
        raise ValueError("masked suffix roll_forward_steps leaves too little context")
    feedback = inputs.clone()
    target_indices = tuple(config.data.target_columns.index(name) for name in suffix.channels)
    input_indices = tuple(config.data.input_columns.index(name) for name in suffix.channels)
    windows = masked_suffix_windows(suffix.suffix_steps, suffix.roll_forward_steps, context_len)
    backward_total_count = None
    if backward_scaler is not None:
        backward_total_count = masked_suffix_total_loss_count(
            mask,
            windows,
            context_len,
            target_indices,
            len(config.data.target_columns),
            loss_on_masked_only=suffix.loss_on_masked_only,
        )
        if bool((backward_total_count <= 0).item()):
            return torch.zeros((), dtype=targets.dtype, device=device)
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
        loss_mask = masked_suffix_loss_mask(
            window_mask,
            suffix_slice,
            target_indices,
            len(config.data.target_columns),
            loss_on_masked_only=suffix.loss_on_masked_only,
        )
        with torch.autocast(
            device_type=device.type, enabled=config.run.use_amp and device.type == "cuda"
        ):
            encoded_window_inputs = encode_inputs(
                config, window_inputs, device, masked_input_mask=masked_input_mask
            )
            encoded_window_inputs = apply_binned_feedback_overrides(
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
                target_ranges(config, device),
            )
            total_loss, total_count = accumulate_masked_suffix_window_loss(
                total_loss,
                total_count,
                loss_sum,
                loss_count,
                backward_scaler,
                backward_total_count,
            )
        write_slice = slice(start + context_len - current_suffix_steps, start + context_len)
        write_masked_suffix_feedback(
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
                states = prefix_mamba_states(
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


def accumulate_masked_suffix_window_loss(
    total_loss: torch.Tensor,
    total_count: torch.Tensor,
    loss_sum: torch.Tensor,
    loss_count: torch.Tensor,
    backward_scaler: torch.amp.GradScaler | None,
    backward_total_count: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backward_scaler is None:
        return total_loss + loss_sum, total_count + loss_count
    if backward_total_count is None:
        raise ValueError("backward_total_count is required for per-window backward")
    backward_scaler.scale(loss_sum / backward_total_count).backward()
    return total_loss + loss_sum.detach(), total_count + loss_count.detach()


def masked_suffix_total_loss_count(
    mask: torch.Tensor,
    windows: list[tuple[int, int, int]],
    context_len: int,
    target_indices: tuple[int, ...],
    target_count: int,
    *,
    loss_on_masked_only: bool,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=mask.device)
    for start, current_suffix_steps, _next_shift_steps in windows:
        window_mask = mask[:, start : start + context_len]
        suffix_slice = slice(context_len - current_suffix_steps, context_len)
        loss_mask = masked_suffix_loss_mask(
            window_mask,
            suffix_slice,
            target_indices,
            target_count,
            loss_on_masked_only=loss_on_masked_only,
        )
        total = total + loss_mask.to(dtype=torch.float32).sum()
    return total


def masked_suffix_windows(
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


def prefix_mamba_states(
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


def masked_suffix_loss_mask(
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


def apply_binned_feedback_overrides(
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


def write_masked_suffix_feedback(
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
    pred = decode_categorical_logits(logits, target_ranges(config, device)).detach()
    for selected_idx, input_idx in zip(target_indices, input_indices, strict=True):
        feedback[:, write_slice, input_idx] = pred[:, suffix_slice, selected_idx]
