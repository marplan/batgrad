from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from batgrad.logging import get_logger
from batgrad.ml.config import ExperimentConfig, config_to_dict, parse_experiment_config
from batgrad.ml.distributed import unwrap_model
from batgrad.ml.nn import build_model

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Model and experiment metadata restored for evaluation.

    Attributes:
        config: Validated configuration embedded in the checkpoint.
        model: Reconstructed model in evaluation mode on the requested device.
        step: Saved training step, if present and valid.
    """

    config: ExperimentConfig
    model: torch.nn.Module
    step: int | None


def load_checkpoint_payload(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    trusted: bool = False,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=not trusted)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain an object payload: {path}")
    return cast("dict[str, object]", checkpoint)


def checkpoint_config(
    payload: Mapping[str, object], path: str | Path = "checkpoint"
) -> ExperimentConfig:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError(f"Checkpoint must contain a 'config' dict: {path}")
    return parse_experiment_config(raw_config)


def read_checkpoint_config(path: str | Path) -> ExperimentConfig:
    """Read and validate only the experiment configuration from a checkpoint.

    Args:
        path: PyTorch checkpoint path.

    Returns:
        The embedded experiment configuration.

    Note:
        PyTorch still deserializes the checkpoint payload; this helper merely
        avoids constructing the model.
    """
    return checkpoint_config(load_checkpoint_payload(path), path)


def load_checkpoint(path: str | Path, device: torch.device) -> LoadedCheckpoint:
    """Reconstruct a trained model for evaluation.

    Args:
        path: PyTorch checkpoint path.
        device: Device used for deserialization and model construction.

    Returns:
        The validated configuration, evaluation-mode model, and optional step.

    Raises:
        TypeError: If required configuration or model payloads are absent.
        RuntimeError: If model weights are incompatible with the configuration.

    Note:
        Checkpoints are loaded through PyTorch's restricted weights-only unpickler.
    """
    payload = load_checkpoint_payload(path, device)
    state_dict = payload.get("model")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint must contain a 'model' state dict: {path}")
    config = checkpoint_config(payload, path)
    model = build_model(config, device)
    model.load_state_dict(cast("dict[str, torch.Tensor]", state_dict))
    model.eval()
    step = payload.get("step")
    return LoadedCheckpoint(
        config=config,
        model=model,
        step=step if isinstance(step, int) else None,
    )


def load_model_weights(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    config: ExperimentConfig,
) -> None:
    payload = load_checkpoint_payload(path, device)
    state_dict = payload.get("model")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint must contain a 'model' state dict: {path}")
    checkpoint_config_ = checkpoint_config(payload, path)
    differences = _initialization_config_differences(config, checkpoint_config_)
    if differences:
        raise ValueError(
            "run.init_from checkpoint is incompatible with the current experiment: "
            + ", ".join(differences)
        )
    model.load_state_dict(cast("dict[str, torch.Tensor]", state_dict))


def _initialization_config_differences(
    current: ExperimentConfig,
    checkpoint: ExperimentConfig,
) -> tuple[str, ...]:
    differences = []
    if current.model != checkpoint.model:
        differences.append("model")
    if current.data.input_columns != checkpoint.data.input_columns:
        differences.append("input_columns")
    if current.data.target_columns != checkpoint.data.target_columns:
        differences.append("target_columns")
    if current.data.scaling != checkpoint.data.scaling:
        differences.append("scaling")
    return tuple(differences)


def training_checkpoint_payload(
    config: ExperimentConfig,
    model_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    step: int,
    epoch_idx: int,
    epoch_step: int,
) -> dict[str, object]:
    return {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config_to_dict(config),
        "step": step,
        "train_cursor": {
            "epoch_idx": epoch_idx,
            "epoch_step": epoch_step,
            "global_step": step,
        },
    }


def export_checkpoint(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Export a compact checkpoint for inference and weight initialization.

    The exported payload omits optimizer, scheduler, scaler, and cursor state. It
    retains the validated experiment configuration needed to reconstruct the
    model and check data compatibility.

    Args:
        source: Full or compact batgrad checkpoint.
        destination: Output `.pt` path.

    Returns:
        The created checkpoint path.
    """
    payload = load_checkpoint_payload(source, trusted=True)
    state_dict = payload.get("model")
    if not isinstance(state_dict, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise TypeError(f"Checkpoint must contain a tensor 'model' state dict: {source}")
    config = checkpoint_config(payload, source)
    step = payload.get("step")
    exported: dict[str, object] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": state_dict,
        "config": config_to_dict(config),
        "step": step if isinstance(step, int) else None,
    }
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    return output


def checkpoint_dir(run_dir: Path | None, run_id: str | None) -> Path | None:
    if run_dir is None:
        return None
    name = run_id or run_dir.name
    return run_dir / "checkpoints" / _path_component(name)


def save_validation_checkpoints(
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    directory: Path | None,
    *,
    step: int,
    epoch_idx: int,
    epoch_step: int,
    metrics: dict[str, float],
    best: dict[str, float],
) -> None:
    if directory is None:
        return
    if config.checkpoint.save_latest:
        save_training_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            scaler,
            directory / "latest.pt",
            step=step,
            epoch_idx=epoch_idx,
            epoch_step=epoch_step,
        )
    if not config.checkpoint.save_best:
        return
    for monitor in config.checkpoint.monitors:
        value = metrics.get(monitor)
        if value is None:
            logger.warning("Checkpoint monitor %s was not logged at step=%d", monitor, step)
            continue
        previous = best.get(monitor)
        if previous is not None and value >= previous:
            continue
        best[monitor] = value
        save_training_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            scaler,
            directory / f"best_{_metric_filename(monitor)}.pt",
            step=step,
            epoch_idx=epoch_idx,
            epoch_step=epoch_step,
        )


def save_training_checkpoint(
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    path: Path | None,
    *,
    step: int,
    epoch_idx: int,
    epoch_step: int,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        training_checkpoint_payload(
            config,
            unwrap_model(model).state_dict(),
            optimizer,
            scheduler,
            scaler,
            step=step,
            epoch_idx=epoch_idx,
            epoch_step=epoch_step,
        ),
        path,
    )


def _path_component(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-")
    return normalized or "run"


def _metric_filename(metric_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", metric_name).strip("_").lower()
    return normalized or "metric"
