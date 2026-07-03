from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.data.config import WindowConfig
from batgrad.ml.data.materialization import materialize_window_ref
from batgrad.ml.data.planning import WindowRef, build_stream_plans
from batgrad.ml.masked_suffix import (
    apply_binned_feedback_overrides,
    prefix_mamba_states,
    write_masked_suffix_feedback,
)
from batgrad.ml.metrics import aggregate_rmse, feature_metric_payload, feature_rmse_payload
from batgrad.ml.nn import (
    categorical_ce_loss_components,
    categorical_ce_loss_per_feature_components,
    categorical_rmse_per_feature_components,
    decode_categorical_logits,
)
from batgrad.ml.train_utils import (
    append_rollout_extension,
    encode_inputs,
    feedback_pairs,
    loader_config,
    scaling_rules,
    target_ranges,
    validation_carry_mamba_state,
    validation_masked_suffix_enabled,
    validation_suffix_steps,
)
from batgrad.viz.ml import RolloutPlotSeries, build_rollout_figure

if TYPE_CHECKING:
    from batgrad.ml.config import ExperimentConfig
    from batgrad.ml.data.loader import MlDataIterable
    from batgrad.ml.loggers import RunLogger
    from batgrad.ml.nn import MambaCarryState
    from batgrad.storage.local import LocalDataProcessingStore


@dataclass(frozen=True, slots=True)
class RolloutResult:
    prediction: torch.Tensor
    loss_sum: torch.Tensor
    loss_count: torch.Tensor
    feature_loss_sum: torch.Tensor
    feature_loss_count: torch.Tensor
    feature_squared_error_sum: torch.Tensor
    feature_squared_error_count: torch.Tensor

    @property
    def loss(self) -> torch.Tensor | None:
        if bool((self.loss_count <= 0).item()):
            return None
        return self.loss_sum / self.loss_count


@torch.no_grad()
def run_rollouts(  # noqa: C901, PLR0915
    config: ExperimentConfig,
    model: torch.nn.Module,
    dataset: MlDataIterable,
    store: LocalDataProcessingStore,
    device: torch.device,
    logger: RunLogger,
    step: int,
) -> dict[str, float]:
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
        loader_config(config, BaseColumns.split.values.val),
        default_window=WindowConfig(batch_size=1, seq_len=context_len + stored_rollout_len),
    )
    stream_plans = build_stream_plans(val_index, protocol, window_config)
    scaling = scaling_rules(config)
    rollout_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    rollout_loss_count = torch.zeros((), dtype=torch.float32, device=device)
    feature_loss_sum = torch.zeros(
        (len(config.data.target_columns),), dtype=torch.float32, device=device
    )
    feature_loss_count = torch.zeros_like(feature_loss_sum)
    feature_squared_error_sum = torch.zeros_like(feature_loss_sum)
    feature_squared_error_count = torch.zeros_like(feature_loss_sum)
    plot_series: list[RolloutPlotSeries] = []
    for group in config.validation.split.groups:
        if not group.rollout_start_offsets:
            continue
        matches = [
            stream
            for stream in stream_plans
            if stream_matches(stream.group_key, config.validation.split.group_by, group.match)
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
                inputs = append_rollout_extension(config, inputs)
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
            pairs = list(feedback_pairs(config))
            if validation_masked_suffix_enabled(config):
                rollout_result = masked_suffix_rollout_predictions(
                    config,
                    model,
                    inputs,
                    targets,
                    rollout_mask,
                    context_len,
                    total_rollout_len,
                    pairs,
                    device,
                )
            else:
                rollout_result = one_step_rollout_predictions(
                    config,
                    model,
                    inputs,
                    targets,
                    rollout_mask,
                    context_len,
                    total_rollout_len,
                    pairs,
                    device,
                )
            rollout_loss_sum = rollout_loss_sum + rollout_result.loss_sum
            rollout_loss_count = rollout_loss_count + rollout_result.loss_count
            feature_loss_sum = feature_loss_sum + rollout_result.feature_loss_sum
            feature_loss_count = feature_loss_count + rollout_result.feature_loss_count
            feature_squared_error_sum = (
                feature_squared_error_sum + rollout_result.feature_squared_error_sum
            )
            feature_squared_error_count = (
                feature_squared_error_count + rollout_result.feature_squared_error_count
            )
            pred_tensor = rollout_result.prediction
            if int(pred_tensor.shape[0]) > 0 and config.validation.log_rollout_plots:
                context_prediction = context_predictions(config, model, inputs, context_len, device)
                input_tensor = inputs[:, : context_len + pred_tensor.shape[0], :].cpu()[0]
                target_tensor = targets[:, : context_len + pred_tensor.shape[0], :].cpu()[0]
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
        metrics = {
            "val/rollout/loss_ce": float((rollout_loss_sum / rollout_loss_count).detach().cpu())
        }
        rmse = aggregate_rmse(feature_squared_error_sum, feature_squared_error_count)
        if rmse is not None:
            metrics["val/rollout/rmse"] = rmse
        metrics.update(
            feature_metric_payload(
                "val/rollout/loss_ce",
                config.data.target_columns,
                feature_loss_sum,
                feature_loss_count,
            )
        )
        metrics.update(
            feature_rmse_payload(
                "val/rollout/rmse",
                config.data.target_columns,
                feature_squared_error_sum,
                feature_squared_error_count,
            )
        )
        logger.log_metrics(
            step,
            metrics,
        )
    else:
        metrics = {}
    if plot_series:
        logger.log_payload(
            step,
            "val/rollout/plot",
            build_rollout_figure(config, plot_series, context_len, logger.run_name()),
        )
    return metrics


def context_predictions(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    context_len: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_inputs(config, inputs[:, :context_len, :].clone(), device)
    logits = cast(
        "torch.Tensor",
        model(encoded, mask=torch.ones(encoded.shape[:2], dtype=torch.bool, device=device)),
    )
    return decode_categorical_logits(logits, target_ranges(config, device)).cpu()[0]


def one_step_rollout_predictions(
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    rollout_len: int,
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> RolloutResult:
    current = encode_inputs(config, inputs[:, :context_len, :].clone(), device)
    future_inputs = inputs[:, context_len:, :]
    predictions: list[torch.Tensor] = []
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_count = torch.zeros((), dtype=torch.float32, device=device)
    feature_loss_sum = torch.zeros(
        (len(config.data.target_columns),), dtype=torch.float32, device=device
    )
    feature_loss_count = torch.zeros_like(feature_loss_sum)
    feature_squared_error_sum = torch.zeros_like(feature_loss_sum)
    feature_squared_error_count = torch.zeros_like(feature_loss_sum)
    states: dict[str, MambaCarryState] | None = None
    carry_mamba_state = validation_carry_mamba_state(config)
    for future_idx in range(min(rollout_len, int(future_inputs.shape[1]))):
        window_mask = torch.ones(current.shape[:2], dtype=torch.bool, device=device)
        result = model(
            current,
            mask=window_mask,
            states=states,
            return_states=carry_mamba_state,
        )
        if carry_mamba_state:
            logits, _final_states = cast("tuple[torch.Tensor, dict[str, MambaCarryState]]", result)
        else:
            logits = cast("torch.Tensor", result)
        pred = decode_categorical_logits(logits[:, -1:, :, :], target_ranges(config, device))
        predictions.append(pred.cpu())
        target_slice = slice(context_len + future_idx, context_len + future_idx + 1)
        loss_sum, loss_count = categorical_ce_loss_components(
            logits[:, -1:, :, :],
            targets[:, target_slice, :],
            mask[:, target_slice],
            config.model.output_sigma,
            target_ranges(config, device),
        )
        total_loss = total_loss + loss_sum
        total_count = total_count + loss_count
        current_feature_sum, current_feature_count = categorical_ce_loss_per_feature_components(
            logits[:, -1:, :, :],
            targets[:, target_slice, :],
            mask[:, target_slice],
            config.model.output_sigma,
            target_ranges(config, device),
        )
        current_squared_sum, current_squared_count = categorical_rmse_per_feature_components(
            logits[:, -1:, :, :],
            targets[:, target_slice, :],
            mask[:, target_slice],
            target_ranges(config, device),
        )
        feature_loss_sum = feature_loss_sum + current_feature_sum
        feature_loss_count = feature_loss_count + current_feature_count
        feature_squared_error_sum = feature_squared_error_sum + current_squared_sum
        feature_squared_error_count = feature_squared_error_count + current_squared_count
        next_scalar = future_inputs[:, future_idx : future_idx + 1, :].clone()
        next_input = rollout_next_input_bins(
            config,
            next_scalar,
            logits[:, -1:, :, :],
            pred,
            pairs,
            device,
        )
        if carry_mamba_state:
            states = prefix_mamba_states(
                model, current[:, :1, :, :], window_mask[:, :1], states, 1
            )
        current = torch.cat((current[:, 1:, :, :], next_input), dim=1)
    if not predictions:
        prediction = torch.empty((0, len(config.data.target_columns)), dtype=inputs.dtype)
    else:
        prediction = torch.cat(predictions, dim=1).cpu()[0]
    return RolloutResult(
        prediction=prediction,
        loss_sum=total_loss,
        loss_count=total_count,
        feature_loss_sum=feature_loss_sum,
        feature_loss_count=feature_loss_count,
        feature_squared_error_sum=feature_squared_error_sum,
        feature_squared_error_count=feature_squared_error_count,
    )


def masked_suffix_rollout_predictions(  # noqa: PLR0915
    config: ExperimentConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    rollout_len: int,
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> RolloutResult:
    suffix_steps = validation_suffix_steps(config)
    feedback = inputs.clone()
    rollout_len = min(rollout_len, max(0, int(feedback.shape[1]) - context_len))
    predictions: list[torch.Tensor] = []
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_count = torch.zeros((), dtype=torch.float32, device=device)
    feature_loss_sum = torch.zeros(
        (len(config.data.target_columns),), dtype=torch.float32, device=device
    )
    feature_loss_count = torch.zeros_like(feature_loss_sum)
    feature_squared_error_sum = torch.zeros_like(feature_loss_sum)
    feature_squared_error_count = torch.zeros_like(feature_loss_sum)
    feedback_bin_overrides = torch.zeros(
        (*feedback.shape, config.model.num_bins), dtype=feedback.dtype, device=feedback.device
    )
    feedback_bin_override_mask = torch.zeros_like(feedback_bin_overrides, dtype=torch.bool)
    target_indices = tuple(target_idx for target_idx, _input_idx in pairs)
    input_indices = tuple(input_idx for _target_idx, input_idx in pairs)
    states: dict[str, MambaCarryState] | None = None
    carry_mamba_state = validation_carry_mamba_state(config)
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
        for _target_idx, input_idx in pairs:
            window_inputs[:, suffix_slice, input_idx] = config.train.masked_suffix.fill_value
            masked_input_mask[:, suffix_slice, input_idx] = True
        encoded_window_inputs = encode_inputs(
            config, window_inputs, device, masked_input_mask=masked_input_mask
        )
        encoded_window_inputs = apply_binned_feedback_overrides(
            encoded_window_inputs,
            feedback_bin_overrides,
            feedback_bin_override_mask,
            slice(start, window_end),
        )
        window_mask = torch.ones(encoded_window_inputs.shape[:2], dtype=torch.bool, device=device)
        if carry_mamba_state and states is None and start > 0:
            states = seed_mamba_states(config, model, feedback[:, :start, :], device)
        result = model(
            encoded_window_inputs,
            mask=window_mask,
            states=states,
            return_states=carry_mamba_state,
        )
        if carry_mamba_state:
            logits, _final_states = cast("tuple[torch.Tensor, dict[str, MambaCarryState]]", result)
        else:
            logits = cast("torch.Tensor", result)
        pred = decode_categorical_logits(
            logits[:, suffix_slice, :, :], target_ranges(config, device)
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
            target_ranges(config, device),
        )
        total_loss = total_loss + loss_sum
        total_count = total_count + loss_count
        current_feature_sum, current_feature_count = categorical_ce_loss_per_feature_components(
            logits[:, suffix_slice, :, :],
            targets[:, write_slice, :],
            loss_mask,
            config.model.output_sigma,
            target_ranges(config, device),
        )
        current_squared_sum, current_squared_count = categorical_rmse_per_feature_components(
            logits[:, suffix_slice, :, :],
            targets[:, write_slice, :],
            loss_mask,
            target_ranges(config, device),
        )
        feature_loss_sum = feature_loss_sum + current_feature_sum
        feature_loss_count = feature_loss_count + current_feature_count
        feature_squared_error_sum = feature_squared_error_sum + current_squared_sum
        feature_squared_error_count = feature_squared_error_count + current_squared_count
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
        completed += current_suffix_steps
        if carry_mamba_state and completed < rollout_len:
            states = prefix_mamba_states(
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
    return RolloutResult(
        prediction=prediction,
        loss_sum=total_loss,
        loss_count=total_count,
        feature_loss_sum=feature_loss_sum,
        feature_loss_count=feature_loss_count,
        feature_squared_error_sum=feature_squared_error_sum,
        feature_squared_error_count=feature_squared_error_count,
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
    mask = torch.ones(encoded.shape[:2], dtype=torch.bool, device=device)
    _logits, states = cast(
        "tuple[torch.Tensor, dict[str, MambaCarryState]]",
        model(encoded, mask=mask, return_states=True),
    )
    return states


def stream_matches(
    group_key: tuple[object, ...], group_by: tuple[str, ...], match: dict[str, object]
) -> bool:
    key_map = dict(zip(group_by, group_key, strict=True))
    return all(key_map.get(key) == value for key, value in match.items())


def rollout_next_input_bins(
    config: ExperimentConfig,
    next_scalar: torch.Tensor,
    logits: torch.Tensor,
    pred_scalar: torch.Tensor,
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    if config.model.feedback_mode == "decoded_scalar":
        for target_idx, input_idx in pairs:
            next_scalar[:, :, input_idx] = pred_scalar[:, :, target_idx]
        return encode_inputs(config, next_scalar, device)

    next_bins = encode_inputs(config, next_scalar, device)
    pred_bins = torch.softmax(logits.float(), dim=-1).to(dtype=next_bins.dtype)
    for target_idx, input_idx in pairs:
        next_bins[:, :, input_idx, :] = pred_bins[:, :, target_idx, :]
    return next_bins
