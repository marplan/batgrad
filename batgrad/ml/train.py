from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
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
from batgrad.ml.data.loader import MlDataIterable, create_dataloader, dataloader_for_split
from batgrad.ml.data.materialization import materialize_window_ref
from batgrad.ml.data.planning import WindowRef, build_stream_plans
from batgrad.ml.data.scaling import scale_data
from batgrad.ml.loggers import build_logger
from batgrad.ml.nn import (
    SequenceMixer,
    categorical_ce_loss,
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
        config=_loader_config(config, BaseColumns.split.values.train),
    )
    return (
        train_loader,
        dataloader_for_split(train_loader, BaseColumns.split.values.val),
        _dataset(train_loader),
    )


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
    losses: list[torch.Tensor] = []
    feedback_bin_overrides = torch.zeros(
        (*feedback.shape, config.model.num_bins), dtype=feedback.dtype, device=feedback.device
    )
    feedback_bin_override_mask = torch.zeros_like(feedback_bin_overrides, dtype=torch.bool)
    states = None
    for start, current_suffix_steps in windows:
        window_inputs = feedback[:, start : start + context_len, :].clone()
        window_targets = targets[:, start : start + context_len, :]
        window_mask = mask[:, start : start + context_len]
        suffix_slice = slice(context_len - current_suffix_steps, context_len)
        masked_input_mask = torch.zeros_like(window_inputs, dtype=torch.bool)
        for input_idx in input_indices:
            window_inputs[:, suffix_slice, input_idx] = suffix.fill_value
            masked_input_mask[:, suffix_slice, input_idx] = True
        loss_mask = _masked_suffix_loss_mask(
            window_mask, suffix_slice, loss_on_masked_only=suffix.loss_on_masked_only
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
                logits, states = cast("tuple[torch.Tensor, dict[str, MambaCarryState]]", result)
            else:
                logits = cast("torch.Tensor", result)
            losses.append(
                categorical_ce_loss(
                    logits,
                    window_targets,
                    loss_mask,
                    config.model.output_sigma,
                    _target_ranges(config, device),
                )
            )
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
        if states is not None and suffix.detach_between_windows:
            states = {key: value.detach() for key, value in states.items()}
    return torch.stack(losses).mean()


def _masked_suffix_windows(
    suffix_steps: int, roll_forward_steps: int, context_len: int
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = [(0, min(suffix_steps, context_len))]
    remaining = roll_forward_steps
    start = 0
    while remaining > 0:
        shift = min(suffix_steps, remaining)
        start += shift
        remaining -= shift
        windows.append((start, min(suffix_steps, context_len)))
    return windows


def _masked_suffix_loss_mask(
    window_mask: torch.Tensor, suffix_slice: slice, *, loss_on_masked_only: bool
) -> torch.Tensor:
    if not loss_on_masked_only:
        return window_mask
    loss_mask = torch.zeros_like(window_mask, dtype=torch.bool)
    loss_mask[:, suffix_slice] = window_mask[:, suffix_slice]
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
        losses: list[float] = []
        for idx, batch in enumerate(val_loader):
            loss = _batch_loss(
                config, model, batch.active.inputs, batch.active.targets, batch.active.mask, device
            )
            losses.append(float(loss.detach().cpu()))
            if idx + 1 >= config.validation.max_tf_batches:
                break
        if losses:
            logger.log_metrics(
                step,
                {"val_tf/loss": sum(losses) / len(losses)},
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
            if config.validation.rollout_extension.enabled:
                inputs = _append_rollout_extension(config, inputs)
                targets = torch.cat(
                    (
                        targets,
                        targets[:, -1:, :].repeat(1, config.validation.rollout_extension.steps, 1),
                    ),
                    dim=1,
                )
            current = _encode_inputs(config, inputs[:, :context_len, :].clone(), device)
            future_inputs = inputs[:, context_len:, :]
            predictions: list[torch.Tensor] = []
            feedback_pairs = [
                (config.data.target_columns.index(name), config.data.input_columns.index(name))
                for name in config.data.feedback_columns
                if name in config.data.target_columns and name in config.data.input_columns
            ]
            for future_idx in range(min(total_rollout_len, int(future_inputs.shape[1]))):
                logits = cast(
                    "torch.Tensor",
                    model(
                        current, mask=torch.ones(current.shape[:2], dtype=torch.bool, device=device)
                    ),
                )
                pred = decode_categorical_logits(
                    logits[:, -1:, :, :], _target_ranges(config, device)
                )
                predictions.append(pred.cpu())
                next_scalar = future_inputs[:, future_idx : future_idx + 1, :].clone()
                next_input = _rollout_next_input_bins(
                    config,
                    next_scalar,
                    logits[:, -1:, :, :],
                    pred,
                    feedback_pairs,
                    device,
                )
                current = torch.cat((current[:, 1:, :, :], next_input), dim=1)
            if predictions and config.validation.log_rollout_plots:
                pred_tensor = torch.cat(predictions, dim=1)[0]
                target_tensor = targets[
                    :, context_len : context_len + pred_tensor.shape[0], :
                ].cpu()[0]
                logger.log_payload(
                    step,
                    "validation/rollout",
                    {
                        "match": group.match,
                        "anchor": anchor,
                        "target_columns": list(config.data.target_columns),
                        "prediction": pred_tensor.tolist(),
                        "target": target_tensor.tolist(),
                    },
                )


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


def _loader_config(config: ExperimentConfig, split: str) -> DataLoaderConfig:
    return DataLoaderConfig(
        split=split,
        default_window=WindowConfig(
            batch_size=config.loader.batch_size, seq_len=config.loader.seq_len
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
