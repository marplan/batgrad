from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


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


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    all_reduce_sum(tensor)
    if is_distributed_initialized():
        tensor /= dist.get_world_size()
    return tensor


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = getattr(model, "module", None)
    if isinstance(module, torch.nn.Module):
        return unwrap_model(module)
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, torch.nn.Module):
        return unwrap_model(original)
    return model
