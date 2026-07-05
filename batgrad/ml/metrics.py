from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from batgrad.ml.config import ExperimentConfig
    from batgrad.ml.nn import MambaCarryState


@dataclass(frozen=True, slots=True)
class LossMetrics:
    loss: torch.Tensor
    feature_loss_sum: torch.Tensor | None = None
    feature_loss_count: torch.Tensor | None = None
    feature_squared_error_sum: torch.Tensor | None = None
    feature_squared_error_count: torch.Tensor | None = None
    mamba_states: dict[str, MambaCarryState] | None = None


def grad_norm_metrics(
    model: torch.nn.Module,
    config: ExperimentConfig,
    loss_metrics: LossMetrics,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    group_squares: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group_name(name, config)
        square = parameter.grad.detach().float().square().sum()
        group_squares[group] = (
            square if group not in group_squares else group_squares[group] + square
        )
    for group, square in group_squares.items():
        metrics[f"train/grad_norm/{group}"] = float(torch.sqrt(square).cpu())
    metrics.update(
        feature_metric_payload(
            "train/loss_ce",
            config.data.target_columns,
            loss_metrics.feature_loss_sum,
            loss_metrics.feature_loss_count,
        )
    )
    return metrics


def parameter_group_name(name: str, config: ExperimentConfig) -> str:
    parts = name.split(".")
    if parts[0] == "feature_proj":
        return "input"
    if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
        idx = int(parts[1])
        if idx < len(config.model.layers):
            return f"layers/{idx:02d}_{config.model.layers[idx].kind}"
    if parts[0] == "head_layers" and len(parts) > 1 and parts[1].isdigit():
        idx = int(parts[1])
        if idx < len(config.model.head_layers):
            return f"head_layers/{idx:02d}_{config.model.head_layers[idx].kind}"
    if parts[0] in {"final_norm", "output"}:
        return "output"
    return "other"


def accumulate_loss_metrics(
    metrics: LossMetrics,
    feature_loss_sum: torch.Tensor,
    feature_loss_count: torch.Tensor,
    feature_squared_error_sum: torch.Tensor,
    feature_squared_error_count: torch.Tensor,
) -> None:
    if metrics.feature_loss_sum is not None and metrics.feature_loss_count is not None:
        feature_loss_sum += metrics.feature_loss_sum
        feature_loss_count += metrics.feature_loss_count
    if (
        metrics.feature_squared_error_sum is not None
        and metrics.feature_squared_error_count is not None
    ):
        feature_squared_error_sum += metrics.feature_squared_error_sum
        feature_squared_error_count += metrics.feature_squared_error_count


def feature_metric_payload(
    prefix: str,
    feature_names: tuple[str, ...],
    loss_sum: torch.Tensor | None,
    loss_count: torch.Tensor | None,
) -> dict[str, float]:
    if loss_sum is None or loss_count is None:
        return {}
    values = loss_sum / loss_count.clamp_min(1)
    return {
        f"{prefix}/{name}": float(value.detach().cpu())
        for name, value, count in zip(feature_names, values, loss_count, strict=True)
        if bool((count > 0).item())
    }


def feature_rmse_payload(
    prefix: str,
    feature_names: tuple[str, ...],
    squared_error_sum: torch.Tensor | None,
    squared_error_count: torch.Tensor | None,
) -> dict[str, float]:
    if squared_error_sum is None or squared_error_count is None:
        return {}
    values = torch.sqrt(squared_error_sum / squared_error_count.clamp_min(1))
    return {
        f"{prefix}/{name}": float(value.detach().cpu())
        for name, value, count in zip(feature_names, values, squared_error_count, strict=True)
        if bool((count > 0).item())
    }


def aggregate_rmse(
    squared_error_sum: torch.Tensor | None,
    squared_error_count: torch.Tensor | None,
) -> float | None:
    if squared_error_sum is None or squared_error_count is None:
        return None
    count = squared_error_count.sum()
    if bool((count <= 0).item()):
        return None
    return float(torch.sqrt(squared_error_sum.sum() / count).detach().cpu())
