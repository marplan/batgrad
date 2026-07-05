from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

    import torch

    from batgrad.contracts.mapping import DatasetProtocolId


@dataclass(frozen=True, slots=True)
class BatchSegmentRef:
    """Traceable source range inside one normalized shard segment."""

    path: str
    row_start: int
    row_count: int
    window_row_start: int
    window_row_count: int


@dataclass(frozen=True, slots=True)
class BatchState:
    """Traceability metadata for a scheduled tensor batch."""

    split: str
    batch_idx: int
    protocols: tuple[DatasetProtocolId, ...]
    manifest_paths: tuple[str, ...]
    manifest_row_ids: tuple[int, ...]
    group_keys: tuple[tuple[object, ...], ...]
    alignment_keys: tuple[tuple[object, ...], ...]
    segments: tuple[BatchSegmentRef, ...]
    window_offsets: tuple[int, ...]
    stateful_group_idx: int | None = None
    stateful_step_idx: int | None = None
    stateful_steps: int | None = None


@dataclass(frozen=True, slots=True)
class Batch:
    """Protocol-agnostic time-series batch yielded by ML data loaders."""

    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    all_valid: bool
    state: BatchState

    def is_protocol(self, protocol: object) -> bool:
        value = str(protocol)
        return any(
            value == str(candidate) or value == candidate.name for candidate in self.state.protocols
        )

    def pin_memory(self) -> Self:
        return type(self)(
            inputs=self.inputs.pin_memory(),
            targets=self.targets.pin_memory(),
            mask=self.mask.pin_memory(),
            all_valid=self.all_valid,
            state=self.state,
        )

    def to(self, device: str | torch.device | int | None, *, non_blocking: bool = False) -> Self:
        return type(self)(
            inputs=self.inputs.to(device=device, non_blocking=non_blocking),
            targets=self.targets.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            all_valid=self.all_valid,
            state=self.state,
        )
