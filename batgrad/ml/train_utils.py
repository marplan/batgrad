from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.data.config import (
    LoaderConfig as DataLoaderConfig,
    ScalingRule,
    ValidationConfig as DataValidationConfig,
    WindowConfig,
)
from batgrad.ml.data.scaling import scale_data
from batgrad.ml.nn import encode_categorical_values

if TYPE_CHECKING:
    from batgrad.ml.config import ExperimentConfig


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


def feedback_pairs(config: ExperimentConfig) -> tuple[tuple[int, int], ...]:
    return tuple(
        (config.data.target_columns.index(name), config.data.input_columns.index(name))
        for name in config.data.feedback_columns
        if name in config.data.target_columns and name in config.data.input_columns
    )


def validation_masked_suffix_enabled(config: ExperimentConfig) -> bool:
    enabled = config.validation.masked_suffix.enabled
    return config.train.masked_suffix.enabled if enabled is None else enabled


def validation_suffix_steps(config: ExperimentConfig) -> int:
    return config.validation.masked_suffix.suffix_steps or config.train.masked_suffix.suffix_steps


def validation_carry_mamba_state(config: ExperimentConfig) -> bool:
    carry = config.validation.masked_suffix.carry_mamba_state
    return config.train.masked_suffix.carry_mamba_state if carry is None else carry


def validation_loss_config(config: ExperimentConfig) -> ExperimentConfig:
    suffix = replace(
        config.train.masked_suffix,
        enabled=validation_masked_suffix_enabled(config),
        suffix_steps=validation_suffix_steps(config),
        carry_mamba_state=validation_carry_mamba_state(config),
        roll_forward_steps=0,
    )
    return replace(config, train=replace(config.train, masked_suffix=suffix))


def append_rollout_extension(config: ExperimentConfig, inputs: torch.Tensor) -> torch.Tensor:
    extension = config.validation.rollout_extension
    if not extension.enabled or extension.steps <= 0:
        return inputs
    suffix = inputs[:, -1:, :].clone().repeat(1, extension.steps, 1)
    scaling = scaling_rules(config)
    for column, physical_value in extension.input_values.items():
        column_idx = config.data.input_columns.index(column)
        value = torch.tensor([[[float(physical_value)]]], dtype=inputs.dtype, device=inputs.device)
        rule = tuple(rule for rule in scaling if rule.name == column)
        if not rule:
            raise ValueError(f"Missing scaling rule for rollout extension column: {column}")
        suffix[:, :, column_idx] = scale_data(value, rule).reshape(())
    return torch.cat((inputs, suffix), dim=1)


def data_validation_config(config: ExperimentConfig) -> DataValidationConfig:
    provided = tuple(group.match for group in config.validation.split.groups)
    split = config.validation.split
    if split.strategy == "provide":
        return DataValidationConfig.provide(provided, group_by=split.group_by)
    if split.strategy == "merge":
        return DataValidationConfig.merge(
            provided, fraction=split.fraction, group_by=split.group_by
        )
    return DataValidationConfig.sample(fraction=split.fraction, group_by=split.group_by)


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
        active_protocol=DatasetProtocolId(config.data.protocols[0]),
        stateful_n_windows=config.loader.stateful_n_windows,
        data_access=config.loader.data_access,
        num_workers=config.loader.num_workers,
        prefetch_to_device=config.loader.prefetch_to_device,
        device=config.run.device,
        multiprocessing_context=None if config.loader.num_workers == 0 else "spawn",
    )


def train_loader_config(config: ExperimentConfig) -> DataLoaderConfig:
    return loader_config(config, BaseColumns.split.values.train, expand_roll_forward=True)


def val_loader_config(config: ExperimentConfig) -> DataLoaderConfig:
    return loader_config(config, BaseColumns.split.values.val, expand_roll_forward=False)
