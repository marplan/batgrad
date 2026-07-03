from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batgrad.contracts.mapping import DatasetProtocolId

if TYPE_CHECKING:
    from typing import Self

    import torch


@dataclass(frozen=True, slots=True)
class BatchSegmentRef:
    """Traceable source range inside one normalized shard segment."""

    path: str
    row_start: int
    row_count: int
    window_row_start: int
    window_row_count: int


@dataclass(frozen=True, slots=True)
class ProtocolBatchState:
    """Traceability metadata for one protocol tensor payload.

    A protocol batch is materialized from one or more normalized segment ranges.
    The state contains enough information to reconstruct the source rows from the
    store without relying on caller-local state.
    """

    split: str
    batch_idx: int
    protocol: DatasetProtocolId
    manifest_paths: tuple[str, ...]
    manifest_row_ids: tuple[int, ...]
    group_keys: tuple[tuple[object, ...], ...]
    alignment_keys: tuple[tuple[object, ...], ...]
    segments: tuple[BatchSegmentRef, ...]
    window_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProtocolBatch:
    """Tensor payload for one protocol inside a `Batch`.

    `inputs`, `targets`, and `mask` have shapes `(B, T, C_in)`, `(B, T,
    C_target)`, and `(B, T)`. `all_valid` records whether `mask` is entirely
    true so attention can skip padding masks without losing loss-mask semantics.
    Future schedules can place multiple protocol payloads in the same parent
    `Batch`, for example cycling plus EIS for the same cell/cycle alignment key.

    Examples:
        Access the active cycling tensors::

            cycling = batch.protocols[DatasetProtocolId.cycling]
            assert cycling.inputs.shape[:2] == cycling.mask.shape
    """

    protocol: DatasetProtocolId
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    all_valid: bool
    state: ProtocolBatchState

    def pin_memory(self) -> Self:
        return type(self)(
            protocol=self.protocol,
            inputs=self.inputs.pin_memory(),
            targets=self.targets.pin_memory(),
            mask=self.mask.pin_memory(),
            all_valid=self.all_valid,
            state=self.state,
        )

    def to(self, device: str | torch.device | int | None, *, non_blocking: bool = False) -> Self:
        return type(self)(
            protocol=self.protocol,
            inputs=self.inputs.to(device=device, non_blocking=non_blocking),
            targets=self.targets.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            all_valid=self.all_valid,
            state=self.state,
        )


@dataclass(frozen=True, slots=True)
class BatchState:
    """Shared metadata for a scheduled batch.

    `active_protocol` identifies which protocol should drive the current model
    update. `protocols` on the parent `Batch` can still contain additional
    protocol payloads aligned to the active one. The initial sequential strategy
    yields one protocol payload, but later strategies can yield, for example,
    cycling as active and EIS as an aligned context payload.
    """

    split: str
    batch_idx: int
    active_protocol: DatasetProtocolId
    protocol_order: tuple[DatasetProtocolId, ...]


@dataclass(frozen=True, slots=True)
class Batch:
    """Protocol-aware batch yielded by ML data loaders.

    A `Batch` is a small container of protocol-specific tensor payloads. Today a
    sequential loader yields one active protocol at a time::

        batch.active_protocol == DatasetProtocolId.cycling
        batch.active.inputs

    Later, aligned schedules can return multiple entries without changing the
    outer type::

        batch.protocols[DatasetProtocolId.cycling]
        batch.protocols[DatasetProtocolId.eis]

    `active_protocol` tells training which head/objective should be updated.
    """

    state: BatchState
    protocols: dict[DatasetProtocolId, ProtocolBatch]

    @property
    def active_protocol(self) -> DatasetProtocolId:
        return self.state.active_protocol

    @property
    def active(self) -> ProtocolBatch:
        return self.protocols[self.active_protocol]

    def is_protocol(self, protocol: object) -> bool:
        value = str(protocol)
        return any(
            self.active_protocol == candidate
            for candidate in DatasetProtocolId
            if value == str(candidate) or value == candidate.name
        )

    def pin_memory(self) -> Self:
        return type(self)(
            state=self.state,
            protocols={protocol: batch.pin_memory() for protocol, batch in self.protocols.items()},
        )

    def to(self, device: str | torch.device | int | None, *, non_blocking: bool = False) -> Self:
        return type(self)(
            state=self.state,
            protocols={
                protocol: batch.to(device, non_blocking=non_blocking)
                for protocol, batch in self.protocols.items()
            },
        )
