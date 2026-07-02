from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from batgrad.logging import configure_logger, get_logger
from batgrad.ml.config import (
    ExperimentConfig,
    config_to_dict,
    load_experiment_config,
)
from batgrad.ml.data.loader import MlDataIterable, create_dataloader
from batgrad.ml.loggers import build_logger
from batgrad.ml.masked_suffix import batch_loss
from batgrad.ml.nn import SequenceMixer
from batgrad.ml.rollout import run_rollouts
from batgrad.ml.train_utils import (
    data_validation_config,
    scaling_rules,
    train_loader_config,
    val_loader_config,
    validation_loss_config,
)
from batgrad.storage.local import LocalDataProcessingStore

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batgrad.ml.data.batch import Batch
    from batgrad.ml.loggers import RunLogger

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
                loss = batch_loss(
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
        validation=data_validation_config(config),
        scaling=scaling_rules(config),
        config=train_loader_config(config),
    )
    val_loader = create_dataloader(
        store=store,
        manifest_paths=config.data.manifest_paths,
        input_columns=config.data.input_columns,
        target_columns=config.data.target_columns,
        protocols=config.data.protocols,
        active_protocol=config.data.protocols[0],
        validation=data_validation_config(config),
        scaling=scaling_rules(config),
        config=val_loader_config(config),
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
        val_loss_config = validation_loss_config(config)
        loss_name = (
            "val/tf/suffix_loss"
            if val_loss_config.train.masked_suffix.enabled
            else "val/tf/full_loss"
        )
        losses: list[float] = []
        for idx, batch in enumerate(val_loader):
            loss = batch_loss(
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
        run_rollouts(config, model, train_dataset, store, device, logger, step)


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
