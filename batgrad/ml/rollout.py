from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from batgrad.ml.experiment import (
    FeedbackValue,
    encode_inputs,
    feedback_pairs,
    feedback_value,
    model_autocast,
    target_ranges,
)
from batgrad.ml.loss import loss_metrics_from_logits
from batgrad.ml.masked_suffix import (
    create_feedback_buffer,
    masked_suffix_loss_mask,
    masked_suffix_windows,
    masked_window_forward,
    prefix_mamba_states,
)
from batgrad.ml.metrics import LossMetrics, add_loss_metrics
from batgrad.ml.nn import decode_categorical_logits

if TYPE_CHECKING:
    from batgrad.ml.config import ExperimentConfig, MaskedSuffixConfig
    from batgrad.ml.nn import MambaCarryState

ROLLOUT_INPUT_RANK = 3


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Predictions and optional metrics from one batched rollout.

    Attributes:
        prediction: Decoded output-scaled values shaped
            `(B, executed_steps, C_out)`.
        metrics: Count-weighted metrics when targets and a mask were supplied.
        target_start: Target-sequence offset aligned with prediction step zero;
            equal to `context_len - 1` under the next-row target contract.
    """

    prediction: torch.Tensor
    metrics: LossMetrics | None
    target_start: int


@torch.no_grad()
def rollout_batch(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    context_len: int,
    rollout_steps: int,
    suffix: MaskedSuffixConfig,
    device: torch.device,
    targets: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> RolloutResult:
    """Run classic or masked-suffix autoregressive rollout for a batch.

    Known future non-feedback controls must already be present in `inputs`.
    Feedback columns are replaced recursively with detached model predictions.
    Supplying both `targets` and `mask` enables scored evaluation; omitting
    both performs deployment-style prediction with identical execution semantics.

    Args:
        config: Experiment defining columns, scaling, encoding, and feedback.
        model: Compatible model in evaluation mode.
        inputs: Scaled scalar inputs shaped `(B, T, C_in)` containing context
            and enough future control rows for the requested horizon.
        context_len: Number of initially observed input rows.
        rollout_steps: Maximum future rows to predict.
        suffix: Disabled for one-step rollout or enabled for chunked suffix calls.
        device: Model execution device.
        targets: Optional scaled next-row targets shaped `(B, T, C_out)`.
        mask: Optional valid-target mask supplied together with `targets`.

    Returns:
        Decoded output-scaled predictions, optional metrics, and target alignment.

    Raises:
        ValueError: If shapes, context, horizon, suffix width, or optional scoring
            inputs are invalid.

    Examples:
        Run deployment-style one-step feedback without metrics:

        ```python
        from batgrad.ml.config import MaskedSuffixConfig
        from batgrad.ml.rollout import rollout_batch

        result = rollout_batch(
            config,
            model,
            inputs,
            context_len=512,
            rollout_steps=64,
            suffix=MaskedSuffixConfig(enabled=False),
            device=device,
        )
        assert result.prediction.shape[1] == 64
        ```

    Note:
        Attention recomputes over each visible context. Mamba state is advanced by
        exactly the rows leaving the next window and must not be shared across
        unrelated streams.
    """
    if inputs.ndim != ROLLOUT_INPUT_RANK:
        raise ValueError(f"rollout inputs must be shaped (B,T,C), got {tuple(inputs.shape)}")
    if context_len <= 0 or context_len > int(inputs.shape[1]):
        raise ValueError(
            "rollout context_len must be > 0 and no greater than the input length, "
            f"got context_len={context_len} input_steps={int(inputs.shape[1])}"
        )
    if rollout_steps < 0:
        raise ValueError(f"rollout_steps must be >= 0, got {rollout_steps}")
    if suffix.enabled and suffix.suffix_steps >= context_len:
        raise ValueError(
            "enabled rollout suffix_steps must be smaller than context_len, "
            f"got suffix_steps={suffix.suffix_steps} context_len={context_len}"
        )
    if (targets is None) != (mask is None):
        raise ValueError("rollout targets and mask must either both be provided or both omitted")
    inputs = inputs.to(device=device)
    if targets is not None:
        targets = targets.to(device=device)
    if mask is not None:
        mask = mask.to(device=device)
    if suffix.enabled:
        return _masked_suffix_rollout(
            config,
            model,
            inputs,
            context_len=context_len,
            rollout_steps=rollout_steps,
            suffix=suffix,
            device=device,
            targets=targets,
            mask=mask,
        )
    return _one_step_rollout(
        config,
        model,
        inputs,
        context_len=context_len,
        rollout_steps=rollout_steps,
        suffix=suffix,
        device=device,
        targets=targets,
        mask=mask,
    )


@torch.no_grad()
def predict_context(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    context_len: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_inputs(config, inputs[:, :context_len, :].to(device=device), device)
    with model_autocast(config, device):
        logits = cast("torch.Tensor", model(encoded, mask=None))
    return decode_categorical_logits(logits, target_ranges(config, device))


def _one_step_rollout(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    context_len: int,
    rollout_steps: int,
    suffix: MaskedSuffixConfig,
    device: torch.device,
    targets: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> RolloutResult:
    current = encode_inputs(config, inputs[:, :context_len, :].clone(), device)
    future_inputs = inputs[:, context_len:, :]
    pairs = feedback_pairs(config)
    ranges = target_ranges(config, device)
    predictions: list[torch.Tensor] = []
    metrics: LossMetrics | None = None
    states: dict[str, MambaCarryState] | None = None
    for future_idx in range(min(rollout_steps, int(future_inputs.shape[1]))):
        with model_autocast(config, device):
            result = model(
                current,
                mask=None,
                states=states,
                return_states=suffix.carry_mamba_state,
            )
        logits = (
            cast("tuple[torch.Tensor, dict[str, MambaCarryState]]", result)[0]
            if suffix.carry_mamba_state
            else cast("torch.Tensor", result)
        )
        selected_logits = logits[:, -1:, :, :]
        feedback = feedback_value(config, selected_logits, device, current.dtype)
        prediction = (
            decode_categorical_logits(selected_logits, ranges) if feedback.binned else feedback.data
        )
        predictions.append(prediction)
        if targets is not None and mask is not None:
            target_slice = slice(context_len + future_idx - 1, context_len + future_idx)
            metrics = add_loss_metrics(
                metrics,
                loss_metrics_from_logits(
                    selected_logits,
                    targets[:, target_slice, :],
                    mask[:, target_slice],
                    sigma=config.model.output_sigma,
                    target_ranges=ranges,
                    include_rmse=True,
                ),
            )
        next_scalar = future_inputs[:, future_idx : future_idx + 1, :].clone()
        next_input = rollout_next_input_bins(
            config,
            next_scalar,
            feedback,
            pairs,
            device,
        )
        if suffix.carry_mamba_state:
            with model_autocast(config, device):
                states = prefix_mamba_states(model, current[:, :1, :, :], None, states, 1)
        current = torch.cat((current[:, 1:, :, :], next_input), dim=1)
    return RolloutResult(
        prediction=_stack_predictions(inputs, predictions, config, device),
        metrics=metrics,
        target_start=context_len - 1,
    )


def _masked_suffix_rollout(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    context_len: int,
    rollout_steps: int,
    suffix: MaskedSuffixConfig,
    device: torch.device,
    targets: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> RolloutResult:
    feedback = create_feedback_buffer(inputs, config.model.num_bins)
    rollout_steps = min(rollout_steps, max(0, int(inputs.shape[1]) - context_len))
    pairs = feedback_pairs(config)
    target_indices = tuple(target_idx for target_idx, _input_idx in pairs)
    input_indices = tuple(input_idx for _target_idx, input_idx in pairs)
    ranges = target_ranges(config, device)
    predictions: list[torch.Tensor] = []
    metrics: LossMetrics | None = None
    states: dict[str, MambaCarryState] | None = None
    execution_mask = (
        mask if mask is not None else torch.ones(inputs.shape[:2], dtype=torch.bool, device=device)
    )
    windows = masked_suffix_windows(suffix.suffix_steps, rollout_steps, context_len)[1:]
    for start, current_suffix_steps, next_shift_steps in windows:
        if suffix.carry_mamba_state and states is None and start > 0:
            states = seed_mamba_states(config, model, feedback.values[:, :start, :], device)
        output = masked_window_forward(
            config,
            model,
            feedback,
            execution_mask,
            start=start,
            context_len=context_len,
            suffix_steps=current_suffix_steps,
            next_shift_steps=next_shift_steps,
            input_indices=input_indices,
            target_indices=target_indices,
            states=states,
            suffix=suffix,
            device=device,
            use_attention_mask=False,
        )
        states = output.next_states
        selected_logits = output.logits[:, output.prediction_slice, :, :]
        predictions.append(decode_categorical_logits(selected_logits, ranges))
        if targets is not None and mask is not None:
            window_mask = mask[:, start : start + context_len]
            loss_mask = masked_suffix_loss_mask(
                window_mask,
                output.prediction_slice,
                target_indices,
                len(config.data.target_columns),
                loss_on_masked_only=suffix.loss_on_masked_only,
            )[:, output.prediction_slice]
            metrics = add_loss_metrics(
                metrics,
                loss_metrics_from_logits(
                    selected_logits,
                    targets[:, output.target_slice, :],
                    loss_mask,
                    sigma=config.model.output_sigma,
                    target_ranges=ranges,
                    include_rmse=True,
                ),
            )
    return RolloutResult(
        prediction=_stack_predictions(inputs, predictions, config, device),
        metrics=metrics,
        target_start=context_len - 1,
    )


def _stack_predictions(
    inputs: torch.Tensor,
    predictions: list[torch.Tensor],
    config: ExperimentConfig,
    device: torch.device,
) -> torch.Tensor:
    if predictions:
        return torch.cat(predictions, dim=1)
    return torch.empty(
        (int(inputs.shape[0]), 0, len(config.data.target_columns)),
        dtype=inputs.dtype,
        device=device,
    )


def seed_mamba_states(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, MambaCarryState]:
    if int(inputs.shape[1]) <= 0:
        raise ValueError("Mamba state seed inputs must contain at least one timestep")
    encoded = encode_inputs(config, inputs, device)
    with model_autocast(config, device):
        _logits, states = cast(
            "tuple[torch.Tensor, dict[str, MambaCarryState]]",
            model(encoded, mask=None, return_states=True),
        )
    return states


def rollout_next_input_bins(
    config: ExperimentConfig,
    next_scalar: torch.Tensor,
    feedback: FeedbackValue,
    pairs: tuple[tuple[int, int], ...],
    device: torch.device,
) -> torch.Tensor:
    if not feedback.binned:
        for target_idx, input_idx in pairs:
            next_scalar[:, :, input_idx] = feedback.data[:, :, target_idx]
        return encode_inputs(config, next_scalar, device)
    next_bins = encode_inputs(config, next_scalar, device)
    for target_idx, input_idx in pairs:
        next_bins[:, :, input_idx, :] = feedback.data[:, :, target_idx, :]
    return next_bins
