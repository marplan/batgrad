from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from batgrad.ml.metrics import LossMetrics


@dataclass(frozen=True, slots=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def is_distributed_requested() -> bool:
    return all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(configured_device: str) -> DistributedContext:
    if not is_distributed_requested():
        device = torch.device(configured_device)
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    requested = torch.device(configured_device)
    if requested.type != "cuda":
        raise ValueError("DDP training requires run.device='cuda' and a torchrun CUDA launch")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requested but CUDA is not available")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if is_distributed_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def globally_normalized_backward_scale(local_count: torch.Tensor) -> torch.Tensor:
    total_count = all_reduce_sum(local_count.detach().clone())
    if bool((total_count <= 0).item()):
        return torch.zeros((), dtype=local_count.dtype, device=local_count.device)
    world_size = dist.get_world_size() if is_distributed_initialized() else 1
    return local_count.new_tensor(float(world_size)) / total_count


def all_reduce_loss_metrics(metrics: LossMetrics) -> LossMetrics:
    feature_loss_sum = _required_reduced(metrics.feature_loss_sum, "feature_loss_sum")
    feature_loss_count = _required_reduced(metrics.feature_loss_count, "feature_loss_count")
    squared_sum = _optional_reduced(metrics.feature_squared_error_sum)
    squared_count = _optional_reduced(metrics.feature_squared_error_count)
    count = feature_loss_count.sum()
    loss = (
        feature_loss_sum.sum() / count
        if bool((count > 0).item())
        else torch.zeros((), dtype=feature_loss_sum.dtype, device=feature_loss_sum.device)
    )
    return LossMetrics(
        loss=loss,
        feature_loss_sum=feature_loss_sum,
        feature_loss_count=feature_loss_count,
        feature_squared_error_sum=squared_sum,
        feature_squared_error_count=squared_count,
    )


def _required_reduced(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"loss metrics are missing {name}")
    return all_reduce_sum(value.detach().clone())


def _optional_reduced(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else all_reduce_sum(value.detach().clone())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = getattr(model, "module", None)
    if isinstance(module, torch.nn.Module):
        return unwrap_model(module)
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, torch.nn.Module):
        return unwrap_model(original)
    return model
