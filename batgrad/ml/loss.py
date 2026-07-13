from __future__ import annotations

import torch

from batgrad.ml.metrics import LossMetrics
from batgrad.ml.nn import (
    categorical_target_distribution,
    decode_categorical_logits,
)


def categorical_ce_loss_per_feature_components(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    sigma: float,
    target_ranges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.to(device=logits.device)
    valid = target_valid_mask(mask, target, logits.device)
    target = torch.where(valid, target, torch.zeros_like(target))
    target_dist = categorical_target_distribution(
        target, int(logits.shape[-1]), sigma, target_ranges
    )
    loss = -(target_dist * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    dims = tuple(range(loss.ndim - 1))
    return (loss * valid).sum(dim=dims), valid.sum(dim=dims).to(dtype=loss.dtype)


def categorical_rmse_per_feature_components(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    target_ranges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.to(device=logits.device)
    valid = target_valid_mask(mask, target, logits.device)
    pred = decode_categorical_logits(logits, target_ranges).float()
    target = torch.where(valid, target, pred.to(dtype=target.dtype))
    squared = torch.square(pred - target.float())
    dims = tuple(range(squared.ndim - 1))
    return (squared * valid).sum(dim=dims), valid.sum(dim=dims).to(dtype=squared.dtype)


def target_valid_mask(
    mask: torch.Tensor | None, target: torch.Tensor, device: torch.device
) -> torch.Tensor:
    finite = torch.isfinite(target.to(device=device))
    if mask is None:
        return finite
    valid = mask.to(dtype=torch.bool, device=device)
    if valid.shape == target.shape[:-1]:
        return valid.unsqueeze(-1).expand(target.shape) & finite
    if valid.shape == target.shape:
        return valid & finite
    raise ValueError(
        "target mask must have shape target.shape[:-1] or target.shape, "
        f"got mask={tuple(valid.shape)} target={tuple(target.shape)}"
    )


def loss_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    sigma: float,
    target_ranges: torch.Tensor,
    include_rmse: bool = False,
) -> LossMetrics:
    feature_sum, feature_count = categorical_ce_loss_per_feature_components(
        logits,
        targets,
        mask,
        sigma,
        target_ranges,
    )
    total_count = feature_count.sum()
    loss_sum = feature_sum.sum()
    loss = loss_sum * 0.0 if bool((total_count <= 0).item()) else loss_sum / total_count
    squared_sum = squared_count = None
    if include_rmse:
        squared_sum, squared_count = categorical_rmse_per_feature_components(
            logits,
            targets,
            mask,
            target_ranges,
        )
    return LossMetrics(
        loss=loss,
        feature_loss_sum=feature_sum,
        feature_loss_count=feature_count,
        feature_squared_error_sum=squared_sum,
        feature_squared_error_count=squared_count,
    )
