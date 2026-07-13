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


def loss_metrics_to_cpu(metrics: LossMetrics | None) -> LossMetrics | None:
    if metrics is None:
        return None
    return LossMetrics(
        loss=metrics.loss.detach().cpu(),
        feature_loss_sum=_detach_cpu(metrics.feature_loss_sum),
        feature_loss_count=_detach_cpu(metrics.feature_loss_count),
        feature_squared_error_sum=_detach_cpu(metrics.feature_squared_error_sum),
        feature_squared_error_count=_detach_cpu(metrics.feature_squared_error_count),
    )


def _detach_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu()


def add_loss_metrics(total: LossMetrics | None, current: LossMetrics) -> LossMetrics:
    if total is None:
        return current
    feature_loss_sum = _add_optional(total.feature_loss_sum, current.feature_loss_sum)
    feature_loss_count = _add_optional(total.feature_loss_count, current.feature_loss_count)
    squared_sum = _add_optional(total.feature_squared_error_sum, current.feature_squared_error_sum)
    squared_count = _add_optional(
        total.feature_squared_error_count, current.feature_squared_error_count
    )
    if feature_loss_sum is None or feature_loss_count is None:
        raise ValueError("loss metrics require feature sums and counts for accumulation")
    count = feature_loss_count.sum()
    loss = (
        torch.zeros((), dtype=feature_loss_sum.dtype, device=feature_loss_sum.device)
        if bool((count <= 0).item())
        else feature_loss_sum.sum() / count
    )
    return LossMetrics(
        loss=loss,
        feature_loss_sum=feature_loss_sum,
        feature_loss_count=feature_loss_count,
        feature_squared_error_sum=squared_sum,
        feature_squared_error_count=squared_count,
        mamba_states=current.mamba_states,
    )


def _add_optional(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def loss_metric_payload(
    loss_prefix: str,
    rmse_prefix: str,
    feature_names: tuple[str, ...],
    metrics: LossMetrics,
) -> dict[str, float]:
    payload = {loss_prefix: float(metrics.loss.detach().cpu())}
    payload.update(
        feature_metric_payload(
            loss_prefix,
            feature_names,
            metrics.feature_loss_sum,
            metrics.feature_loss_count,
        )
    )
    rmse = aggregate_rmse(
        metrics.feature_squared_error_sum,
        metrics.feature_squared_error_count,
    )
    if rmse is not None:
        payload[rmse_prefix] = rmse
    payload.update(
        feature_rmse_payload(
            rmse_prefix,
            feature_names,
            metrics.feature_squared_error_sum,
            metrics.feature_squared_error_count,
        )
    )
    return payload


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
