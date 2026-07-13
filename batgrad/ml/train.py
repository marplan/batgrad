from __future__ import annotations

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch.nn.parallel import DistributedDataParallel

from batgrad.logging import get_logger
from batgrad.ml.checkpoint import (
    checkpoint_dir,
    load_model_weights,
    save_training_checkpoint,
    save_validation_checkpoints,
)
from batgrad.ml.config import (
    ExperimentConfig,
    config_to_dict,
    load_experiment_config,
    resolve_store_root,
)
from batgrad.ml.data.loader import (
    MlDataIterable,
    create_dataloader_from_index,
    create_index,
)
from batgrad.ml.distributed import (
    DistributedContext,
    all_reduce_loss_metrics,
    barrier,
    cleanup_distributed,
    init_distributed,
    unwrap_model,
)
from batgrad.ml.experiment import (
    amp_enabled,
    data_validation_config,
    scaling_rules,
    train_loader_config,
    val_loader_config,
)
from batgrad.ml.loggers import NoOpRunLogger, build_logger
from batgrad.ml.masked_suffix import masked_suffix_windows
from batgrad.ml.metrics import grad_norm_metrics, loss_metric_payload
from batgrad.ml.nn import build_model
from batgrad.ml.objective import backward_batch_loss_with_metrics
from batgrad.ml.validation import validate
from batgrad.storage.local import LocalDataProcessingStore
from batgrad.viz.ml import build_rollout_figure

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from batgrad.ml.data.batch import Batch
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.nn import MambaCarryState
    from batgrad.ml.validation import ValidationResult

logger = get_logger(__name__)


def train_from_config(path: str | Path) -> Path | None:  # noqa: C901, PLR0912, PLR0915
    config = load_experiment_config(path)
    dist_ctx = init_distributed(config.run.device)
    _validate_distributed_state_carry(config, dist_ctx)
    run_dir = _setup_with_distributed_cleanup(
        lambda: _prepare_run_dir(config) if dist_ctx.is_main else None
    )
    _setup_with_distributed_cleanup(barrier)
    torch.manual_seed(config.run.seed)
    device = dist_ctx.device
    store = _setup_with_distributed_cleanup(
        lambda: LocalDataProcessingStore(resolve_store_root(config.data.store_root))
    )
    logger.info("Creating data loaders")
    train_loader, val_loader, train_dataset, index = _setup_with_distributed_cleanup(
        lambda: _create_loaders(config, store)
    )
    logger.info("Data loaders ready")
    max_steps = config.train.max_steps or _max_steps_for_epochs(
        train_dataset, config.train.epochs, dist_ctx
    )
    logger.info("Creating model on %s", device)
    model: torch.nn.Module = _setup_with_distributed_cleanup(lambda: build_model(config, device))
    logger.info("Model ready on %s", device)
    _setup_with_distributed_cleanup(lambda: _init_model_from_checkpoint(config, model, device))
    if config.run.compile_model:
        logger.info("Wrapping model with torch.compile")
        model = _setup_with_distributed_cleanup(
            lambda: cast("torch.nn.Module", torch.compile(model))
        )
        logger.info("Model compile wrapper ready")
    if dist_ctx.enabled:
        logger.info("Wrapping model with DistributedDataParallel")
        model = _setup_with_distributed_cleanup(
            lambda: DistributedDataParallel(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
            )
        )
        logger.info("DistributedDataParallel wrapper ready")
    optimizer = _setup_with_distributed_cleanup(
        lambda: torch.optim.AdamW(
            model.parameters(),
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    )
    scheduler = _setup_with_distributed_cleanup(
        lambda: _build_scheduler(config, optimizer, max_steps)
    )
    logger.info("Initializing run logger")
    run_logger = _setup_with_distributed_cleanup(
        lambda: (
            build_logger(config.logging, run_dir, _logged_config(config, model, device))
            if dist_ctx.is_main
            else NoOpRunLogger()
        )
    )
    logger.info("Run logger ready")
    checkpoints = _setup_with_distributed_cleanup(
        lambda: checkpoint_dir(run_dir, run_logger.run_id())
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled(config, device))
    step = 0
    epoch_idx = 0
    last_epoch_idx = 0
    last_epoch_step = 0
    clip_trigger_count = 0
    clip_observed_count = 0
    log_token_count = 0
    log_time = time.perf_counter()
    best_checkpoints: dict[str, float] = {}
    first_train_step = True
    carried_mamba_states: dict[str, MambaCarryState] | None = None
    carried_stateful_group_idx: int | None = None
    carried_stateful_step_idx: int | None = None
    try:
        logger.info("Starting training loop")
        while step < max_steps:
            steps_per_epoch = _local_steps_per_epoch(train_dataset, epoch_idx, dist_ctx)
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
                last_epoch_idx = epoch_idx
                last_epoch_step = epoch_step
                next_step = step + 1
                should_log = next_step % log_every == 0 or next_step == 1
                log_token_count += _model_compute_token_count(config, batch)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if first_train_step:
                    logger.info("Running first training step")
                carried_mamba_states = _initial_mamba_states_for_batch(
                    batch,
                    carried_mamba_states,
                    carried_stateful_group_idx,
                    carried_stateful_step_idx,
                )
                return_mamba_states = _should_return_mamba_states(config, batch)
                loss_metrics = backward_batch_loss_with_metrics(
                    config,
                    model,
                    batch.inputs,
                    batch.targets,
                    batch.mask,
                    device,
                    scaler,
                    suffix=config.train.masked_suffix,
                    collect_metrics=should_log,
                    mask_all_valid=batch.all_valid,
                    initial_mamba_states=carried_mamba_states,
                    return_mamba_states=return_mamba_states,
                )
                carried_mamba_states = _detach_mamba_states(loss_metrics.mamba_states)
                carried_stateful_group_idx = batch.state.stateful_group_idx
                carried_stateful_step_idx = batch.state.stateful_step_idx
                log_loss_metrics = (
                    all_reduce_loss_metrics(loss_metrics) if should_log else loss_metrics
                )
                loss = log_loss_metrics.loss
                scaler.unscale_(optimizer)
                grad_metrics = (
                    grad_norm_metrics(unwrap_model(model), config, log_loss_metrics)
                    if should_log
                    else {}
                )
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.grad_clip_norm
                )
                clip_observed_count += 1
                if float(total_grad_norm.detach().cpu()) > config.train.grad_clip_norm:
                    clip_trigger_count += 1
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step = next_step
                if first_train_step:
                    logger.info("First training step complete")
                    first_train_step = False
                if should_log:
                    now = time.perf_counter()
                    elapsed = max(now - log_time, 1e-9)
                    tokens_per_sec = (log_token_count / elapsed) * dist_ctx.world_size
                    log_time = now
                    log_token_count = 0
                    epoch, epoch_pct = _epoch_progress(epoch_idx, epoch_step, steps_per_epoch)
                    run_logger.log_metrics(
                        step,
                        {
                            "train/loss_ce": float(loss.detach().cpu()),
                            "train/lr": float(scheduler.get_last_lr()[0]),
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/epoch": epoch,
                            "train/epoch_pct": epoch_pct,
                            "train/grad_norm/model": float(total_grad_norm.detach().cpu()),
                            "train/grad_clip/trigger_fraction": clip_trigger_count
                            / clip_observed_count,
                            **grad_metrics,
                        },
                        epoch=epoch,
                        epoch_pct=epoch_pct,
                    )
                if validate_every and step % validate_every == 0:
                    epoch, epoch_pct = _epoch_progress(epoch_idx, epoch_step, steps_per_epoch)
                    validation_result = validate(
                        config,
                        model,
                        val_loader,
                        index,
                        store,
                        device,
                        dist_ctx,
                    )
                    validation_metrics = _validation_metric_payload(config, validation_result)
                    if validation_metrics:
                        run_logger.log_metrics(
                            step,
                            validation_metrics,
                            epoch=epoch,
                            epoch_pct=epoch_pct,
                        )
                    if validation_result.rollout_examples:
                        run_logger.log_payload(
                            step,
                            "val/rollout/plot",
                            build_rollout_figure(
                                config,
                                validation_result.rollout_examples,
                                config.loader.seq_len,
                                run_logger.run_name(),
                            ),
                        )
                    save_validation_checkpoints(
                        config,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        checkpoints,
                        step=step,
                        epoch_idx=epoch_idx,
                        epoch_step=epoch_step,
                        metrics=validation_metrics,
                        best=best_checkpoints,
                    )
                if step >= max_steps:
                    break
            epoch_idx += 1
        if config.checkpoint.save_final:
            save_training_checkpoint(
                config,
                model,
                optimizer,
                scheduler,
                scaler,
                None if checkpoints is None else checkpoints / "final.pt",
                step=step,
                epoch_idx=last_epoch_idx,
                epoch_step=last_epoch_step,
            )
    finally:
        run_logger.finish()
        cleanup_distributed()
    return run_dir


def _validate_distributed_state_carry(
    config: ExperimentConfig, dist_ctx: DistributedContext
) -> None:
    if (
        dist_ctx.world_size <= 1
        or not config.train.masked_suffix.enabled
        or not config.train.masked_suffix.carry_mamba_state
        or config.loader.stateful_n_windows == 1
    ):
        return
    cleanup_distributed()
    raise ValueError(
        "DDP does not support cross-batch Mamba state carry. Set "
        "loader.stateful_n_windows=1, disable train.masked_suffix.carry_mamba_state, "
        "or run single-process training."
    )


def _validation_metric_payload(
    config: ExperimentConfig,
    result: ValidationResult,
) -> dict[str, float]:
    payload: dict[str, float] = {}
    if result.teacher_forced_metrics is not None:
        payload.update(
            loss_metric_payload(
                "val/tf/loss_ce",
                "val/tf/rmse",
                config.data.target_columns,
                result.teacher_forced_metrics,
            )
        )
    if result.rollout_metrics is not None:
        payload.update(
            loss_metric_payload(
                "val/rollout/loss_ce",
                "val/rollout/rmse",
                config.data.target_columns,
                result.rollout_metrics,
            )
        )
    return payload


def _setup_with_distributed_cleanup[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except BaseException:
        cleanup_distributed()
        raise


def _create_loaders(
    config: ExperimentConfig,
    store: LocalDataProcessingStore,
) -> tuple[Iterable[Batch], Iterable[Batch], MlDataIterable, MlDatasetIndex]:
    index = create_index(
        store=store,
        manifest_paths=config.data.manifest_paths,
        protocols=config.data.protocols,
        protocol_mode=config.data.protocol_mode,
        validation=data_validation_config(config),
    )
    train_loader = create_dataloader_from_index(
        store=store,
        index=index,
        input_columns=config.data.input_columns,
        target_columns=config.data.target_columns,
        protocols=config.data.protocols,
        scaling=scaling_rules(config),
        config=train_loader_config(config),
    )
    val_loader = create_dataloader_from_index(
        store=store,
        index=index,
        input_columns=config.data.input_columns,
        target_columns=config.data.target_columns,
        protocols=config.data.protocols,
        scaling=scaling_rules(config),
        config=val_loader_config(config),
    )
    return (train_loader, val_loader, _dataset(train_loader), index)


def _logged_config(
    config: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, object]:
    payload = cast("dict[str, object]", config_to_dict(config))
    run_payload = cast("dict[str, object]", payload["run"])
    model_payload = cast("dict[str, object]", payload["model"])
    run_payload["device"] = _resolved_device_name(device)
    model_payload.update(_model_parameter_counts(unwrap_model(model)))
    return payload


def _model_compute_token_count(config: ExperimentConfig, batch: Batch) -> int:
    batch_size = int(batch.inputs.shape[0])
    seq_len = int(batch.inputs.shape[1])
    suffix = config.train.masked_suffix
    if not suffix.enabled or suffix.roll_forward_steps <= 0:
        return batch_size * seq_len
    context_len = seq_len - suffix.roll_forward_steps
    if context_len <= 0:
        raise ValueError("masked suffix roll_forward_steps leaves no model context")
    windows = masked_suffix_windows(suffix.suffix_steps, suffix.roll_forward_steps, context_len)
    return batch_size * len(windows) * context_len


def _initial_mamba_states_for_batch(
    batch: Batch,
    carried_states: dict[str, MambaCarryState] | None,
    carried_group_idx: int | None,
    carried_step_idx: int | None,
) -> dict[str, MambaCarryState] | None:
    group_idx = batch.state.stateful_group_idx
    step_idx = batch.state.stateful_step_idx
    if group_idx is None or step_idx is None or step_idx == 0:
        return None
    if carried_states is None or carried_group_idx != group_idx:
        return None
    if carried_step_idx is None or carried_step_idx + 1 != step_idx:
        return None
    return carried_states


def _should_return_mamba_states(config: ExperimentConfig, batch: Batch) -> bool:
    suffix = config.train.masked_suffix
    if not suffix.enabled or not suffix.carry_mamba_state:
        return False
    step_idx = batch.state.stateful_step_idx
    steps = batch.state.stateful_steps
    return step_idx is not None and steps is not None and step_idx < steps - 1


def _detach_mamba_states(
    states: dict[str, MambaCarryState] | None,
) -> dict[str, MambaCarryState] | None:
    if states is None:
        return None
    return {key: value.detach() for key, value in states.items()}


def _resolved_device_name(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{index}"


def _model_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_non_trainable": total - trainable,
    }


def _prepare_run_dir(config: ExperimentConfig) -> Path | None:
    run_dir = _make_run_dir(config)
    if run_dir is None:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_to_dict(config), file, indent=2)
    return run_dir


def _init_model_from_checkpoint(
    config: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    if config.run.init_from is None:
        logger.info("Initializing model weights from checkpoint: none")
        return
    checkpoint_path = Path(config.run.init_from)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"run.init_from checkpoint does not exist: {checkpoint_path}")
    logger.info("Initializing model weights from checkpoint: %s", checkpoint_path)
    load_model_weights(model, checkpoint_path, device, config)


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


def _max_steps_for_epochs(
    dataset: MlDataIterable, epochs: float, dist_ctx: DistributedContext
) -> int:
    full_epochs = math.floor(epochs)
    partial = epochs - full_epochs
    total = sum(
        _local_steps_per_epoch(dataset, epoch_idx, dist_ctx) for epoch_idx in range(full_epochs)
    )
    if partial > 0.0:
        total += math.ceil(_local_steps_per_epoch(dataset, full_epochs, dist_ctx) * partial)
    return max(1, total)


def _local_steps_per_epoch(
    dataset: MlDataIterable, epoch_idx: int, dist_ctx: DistributedContext
) -> int:
    global_steps = dataset.steps_per_epoch(epoch_idx)
    if not dist_ctx.enabled:
        return global_steps
    return global_steps // dist_ctx.world_size


def _epoch_progress(epoch_idx: int, epoch_step: int, steps_per_epoch: int) -> tuple[int, int]:
    pct = math.ceil(epoch_step / steps_per_epoch * 100)
    return epoch_idx + 1, min(100, max(1, pct))


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
