from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from batgrad.ml.experiment import encode_inputs, feedback_value, model_autocast

if TYPE_CHECKING:
    from batgrad.ml.config import ExperimentConfig, MaskedSuffixConfig
    from batgrad.ml.nn import MambaCarryState


@dataclass(slots=True)
class FeedbackBuffer:
    values: torch.Tensor
    bin_overrides: torch.Tensor
    bin_override_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class MaskedWindowOutput:
    logits: torch.Tensor
    prediction_slice: slice
    target_slice: slice
    next_states: dict[str, MambaCarryState] | None


def create_feedback_buffer(inputs: torch.Tensor, num_bins: int) -> FeedbackBuffer:
    overrides = torch.zeros((*inputs.shape, num_bins), dtype=inputs.dtype, device=inputs.device)
    return FeedbackBuffer(
        values=inputs.clone(),
        bin_overrides=overrides,
        bin_override_mask=torch.zeros_like(overrides, dtype=torch.bool),
    )


def masked_window_forward(
    config: ExperimentConfig,
    model: torch.nn.Module,
    feedback: FeedbackBuffer,
    mask: torch.Tensor,
    *,
    start: int,
    context_len: int,
    suffix_steps: int,
    next_shift_steps: int,
    input_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    states: dict[str, MambaCarryState] | None,
    suffix: MaskedSuffixConfig,
    device: torch.device,
    use_attention_mask: bool,
    mask_all_valid: bool | None = None,
) -> MaskedWindowOutput:
    window_slice = slice(start, start + context_len)
    window_inputs = feedback.values[:, window_slice, :].clone()
    if int(window_inputs.shape[1]) < context_len:
        raise ValueError("masked suffix window is shorter than context length")
    window_mask = mask[:, window_slice]
    input_suffix, prediction_slice, write_slice, target_slice = masked_suffix_window_slices(
        start, suffix_steps, context_len
    )
    masked_input_mask = torch.zeros_like(window_inputs, dtype=torch.bool)
    for input_idx in input_indices:
        masked_input_mask[:, input_suffix, input_idx] = True
    with model_autocast(config, device):
        encoded = encode_inputs(
            config,
            window_inputs,
            device,
            masked_input_mask=masked_input_mask,
        )
        encoded = apply_binned_feedback_overrides(encoded, feedback, window_slice)
        result = model(
            encoded,
            mask=attention_mask_or_none(window_mask, all_valid=mask_all_valid)
            if use_attention_mask
            else None,
            states=states,
            return_states=suffix.carry_mamba_state,
        )
        if suffix.carry_mamba_state:
            logits, _ignored_states = cast(
                "tuple[torch.Tensor, dict[str, MambaCarryState]]", result
            )
        else:
            logits = cast("torch.Tensor", result)
        write_masked_suffix_feedback(
            config,
            feedback,
            logits,
            prediction_slice,
            write_slice,
            target_indices,
            input_indices,
            device,
        )
        next_states = None
        if suffix.carry_mamba_state and next_shift_steps > 0:
            next_states = prefix_mamba_states(
                model,
                encoded,
                window_mask if use_attention_mask else None,
                states,
                next_shift_steps,
                mask_all_valid=mask_all_valid,
            )
            if suffix.detach_between_windows:
                next_states = {key: value.detach() for key, value in next_states.items()}
    return MaskedWindowOutput(logits, prediction_slice, target_slice, next_states)


def refresh_mamba_states_from_feedback(
    config: ExperimentConfig,
    model: torch.nn.Module,
    feedback: FeedbackBuffer,
    mask: torch.Tensor,
    start: int,
    context_len: int,
    states: dict[str, MambaCarryState] | None,
    device: torch.device,
    *,
    mask_all_valid: bool | None = None,
) -> dict[str, MambaCarryState]:
    window_slice = slice(start, start + context_len)
    window_mask = mask[:, window_slice]
    with torch.no_grad(), model_autocast(config, device):
        encoded = encode_inputs(config, feedback.values[:, window_slice, :], device)
        encoded = apply_binned_feedback_overrides(encoded, feedback, window_slice)
        _logits, next_states = cast(
            "tuple[torch.Tensor, dict[str, MambaCarryState]]",
            model(
                encoded,
                mask=attention_mask_or_none(window_mask, all_valid=mask_all_valid),
                states=states,
                return_states=True,
            ),
        )
    return next_states


def masked_suffix_windows(
    suffix_steps: int, roll_forward_steps: int, context_len: int
) -> list[tuple[int, int, int]]:
    windows = [(0, min(suffix_steps, context_len), min(suffix_steps, roll_forward_steps))]
    remaining = roll_forward_steps
    start = 0
    while remaining > 0:
        shift = min(suffix_steps, remaining)
        start += shift
        remaining -= shift
        windows.append((start, min(shift, context_len), min(suffix_steps, remaining)))
    return windows


def masked_suffix_window_slices(
    start: int,
    suffix_steps: int,
    context_len: int,
) -> tuple[slice, slice, slice, slice]:
    input_suffix = slice(context_len - suffix_steps, context_len)
    prediction = slice(input_suffix.start - 1, context_len - 1)
    write = slice(start + input_suffix.start, start + input_suffix.stop)
    target = slice(write.start - 1, write.stop - 1)
    return input_suffix, prediction, write, target


def prefix_mamba_states(
    model: torch.nn.Module,
    encoded_inputs: torch.Tensor,
    mask: torch.Tensor | None,
    states: dict[str, MambaCarryState] | None,
    prefix_steps: int,
    *,
    mask_all_valid: bool | None = None,
) -> dict[str, MambaCarryState]:
    if prefix_steps <= 0:
        raise ValueError(f"prefix_steps must be > 0, got {prefix_steps}")
    _logits, next_states = cast(
        "tuple[torch.Tensor, dict[str, MambaCarryState]]",
        model(
            encoded_inputs[:, :prefix_steps, :, :],
            mask=attention_mask_or_none(
                None if mask is None else mask[:, :prefix_steps], all_valid=mask_all_valid
            ),
            states=states,
            return_states=True,
        ),
    )
    return next_states


def attention_mask_or_none(
    mask: torch.Tensor | None, *, all_valid: bool | None = None
) -> torch.Tensor | None:
    if mask is None:
        return None
    if all_valid is None:
        all_valid = bool(mask.all().item())
    return None if all_valid else mask


def masked_suffix_loss_mask(
    window_mask: torch.Tensor,
    prediction_slice: slice,
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
        loss_mask[:, prediction_slice, target_idx] = window_mask[:, prediction_slice]
    return loss_mask


def apply_binned_feedback_overrides(
    encoded_inputs: torch.Tensor,
    feedback: FeedbackBuffer,
    override_slice: slice,
) -> torch.Tensor:
    return torch.where(
        feedback.bin_override_mask[:, override_slice, :, :],
        feedback.bin_overrides[:, override_slice, :, :],
        encoded_inputs,
    )


def write_masked_suffix_feedback(
    config: ExperimentConfig,
    feedback: FeedbackBuffer,
    logits: torch.Tensor,
    prediction_slice: slice,
    write_slice: slice,
    target_indices: tuple[int, ...],
    input_indices: tuple[int, ...],
    device: torch.device,
) -> None:
    if write_slice.stop is None or write_slice.stop > int(feedback.values.shape[1]):
        return
    prediction = feedback_value(config, logits, device, feedback.values.dtype)
    if prediction.binned:
        for selected_idx, input_idx in zip(target_indices, input_indices, strict=True):
            feedback.bin_overrides[:, write_slice, input_idx, :] = prediction.data[
                :, prediction_slice, selected_idx, :
            ]
            feedback.bin_override_mask[:, write_slice, input_idx, :] = True
        return
    for selected_idx, input_idx in zip(target_indices, input_indices, strict=True):
        feedback.values[:, write_slice, input_idx] = prediction.data[
            :, prediction_slice, selected_idx
        ]
