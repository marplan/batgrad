from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.data.config import (
    LoaderConfig as DataLoaderConfig,
    ScalingRule,
    ValidationConfig as DataValidationConfig,
    WindowConfig,
)
from batgrad.ml.nn import decode_categorical_logits, encode_categorical_values

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from batgrad.ml.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class FeedbackValue:
    data: torch.Tensor
    binned: bool


def amp_enabled(config: ExperimentConfig, device: torch.device) -> bool:
    return config.run.use_amp and device.type == "cuda"


def model_autocast(config: ExperimentConfig, device: torch.device) -> AbstractContextManager[None]:
    return torch.autocast(
        device_type=device.type,
        enabled=amp_enabled(config, device),
    )


def scaling_rules(config: ExperimentConfig) -> tuple[ScalingRule, ...]:
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


def target_ranges(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    scaling_by_column = {rule.column: rule for rule in config.data.scaling}
    return torch.tensor(
        [
            [scaling_by_column[column].output_min, scaling_by_column[column].output_max]
            for column in config.data.target_columns
        ],
        dtype=torch.float32,
        device=device,
    )


def input_ranges(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    scaling_by_column = {rule.column: rule for rule in config.data.scaling}
    return torch.tensor(
        [
            [scaling_by_column[column].output_min, scaling_by_column[column].output_max]
            for column in config.data.input_columns
        ],
        dtype=torch.float32,
        device=device,
    )


def encode_inputs(
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
        input_ranges(config, device),
    )
    if masked_input_mask is None or not bool(masked_input_mask.any().item()):
        return encoded
    return torch.where(masked_input_mask.unsqueeze(-1), torch.zeros_like(encoded), encoded)


def feedback_pairs(
    config: ExperimentConfig,
    channels: tuple[str, ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    selected = config.data.feedback_columns if channels is None else channels
    return tuple(
        (config.data.target_columns.index(name), config.data.input_columns.index(name))
        for name in selected
    )


def feedback_value(
    config: ExperimentConfig,
    logits: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> FeedbackValue:
    if config.model.feedback_mode == "probabilities":
        probabilities = torch.softmax(logits.detach().float(), dim=-1).to(dtype=dtype)
        return FeedbackValue(probabilities, binned=True)
    decoded = decode_categorical_logits(logits, target_ranges(config, device)).detach()
    return FeedbackValue(decoded, binned=False)


def data_validation_config(config: ExperimentConfig) -> DataValidationConfig:
    provided = tuple(group.match for group in config.validation.split.groups)
    split = config.validation.split
    if split.strategy == "provide":
        return DataValidationConfig.provide(provided, group_by=split.group_by)
    if split.strategy == "merge":
        return DataValidationConfig.merge(
            provided,
            fraction=split.fraction,
            seed=config.run.seed,
            group_by=split.group_by,
        )
    return DataValidationConfig.sample(
        fraction=split.fraction,
        seed=config.run.seed,
        group_by=split.group_by,
    )


def loader_config(
    config: ExperimentConfig, split: str, *, expand_roll_forward: bool = False
) -> DataLoaderConfig:
    seq_len = config.loader.seq_len
    if expand_roll_forward and config.train.masked_suffix.enabled:
        seq_len += config.train.masked_suffix.roll_forward_steps
    return DataLoaderConfig(
        split=split,
        default_window=WindowConfig(batch_size=config.loader.batch_size, seq_len=seq_len),
        seed=config.run.seed,
        strategy=config.loader.strategy,
        protocol_order=tuple(DatasetProtocolId(protocol) for protocol in config.data.protocols),
        stateful_n_windows=config.loader.stateful_n_windows,
        cross_protocol_state_carry=config.loader.cross_protocol_state_carry,
        data_access=config.loader.data_access,
        num_workers=config.loader.num_workers,
        prefetch_to_device=config.loader.prefetch_to_device,
        device=config.run.device,
        multiprocessing_context=None if config.loader.num_workers == 0 else "spawn",
    )


def train_loader_config(config: ExperimentConfig) -> DataLoaderConfig:
    return loader_config(config, BaseColumns.split.values.train, expand_roll_forward=True)


def val_loader_config(config: ExperimentConfig) -> DataLoaderConfig:
    return loader_config(config, BaseColumns.split.values.val)
