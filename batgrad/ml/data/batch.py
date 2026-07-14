from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

    import torch

    from batgrad.contracts.mapping import DatasetProtocolId


@dataclass(frozen=True, slots=True)
class BatchSegmentRef:
    """Traceable source range inside one normalized shard segment.

    Attributes:
        path: Normalized parquet segment path.
        row_start: First source row read from the segment.
        row_count: Number of source rows read.
        window_row_start: First row within the logical materialized window.
        window_row_count: Number of window rows supplied by this segment.
    """

    path: str
    row_start: int
    row_count: int
    window_row_start: int
    window_row_count: int


@dataclass(frozen=True, slots=True)
class BatchState:
    """Traceability and recurrent-sequence metadata for a tensor batch.

    Tuple-valued stream fields are lane-aligned. Stateful identifiers are set
    only when consecutive batches deliberately form one recurrent sequence.

    Attributes:
        split: Source index split.
        batch_idx: Batch number within the loader iteration.
        protocols: Protocol associated with each lane.
        manifest_paths: Source manifest for each lane.
        manifest_row_ids: Source manifest row identifier for each lane.
        group_keys: Protocol-specific stream keys.
        alignment_keys: Physical-context keys used for cross-protocol alignment.
        segments: Normalized parquet ranges contributing to the batch.
        window_offsets: Source-row start offset for each lane.
        stateful_group_idx: Optional recurrent group identifier.
        stateful_step_idx: Zero-based position in the recurrent group.
        stateful_steps: Total number of steps in the recurrent group.
    """

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
    """Protocol-agnostic time-series batch yielded by ML data loaders.

    Targets are shifted one source row ahead of inputs. Incomplete source rows and
    null input values are represented by `-2.0`; null targets remain `NaN` so loss
    calculation can exclude them. `mask` identifies target positions backed by
    real source rows, while target finiteness provides feature-level validity.

    Attributes:
        inputs: Model-space tensor shaped `(B, T, C_in)`.
        targets: Model-space tensor shaped `(B, T, C_out)` for the following
            source rows.
        mask: Valid-target mask shaped `(B, T)` or `(B, T, C_out)`.
        all_valid: Whether every target row is valid, allowing mask optimizations.
        state: Source and stateful-sequence metadata; remains on the CPU.

    Examples:
        Inspect a batch without assuming a specific protocol:

        ```python
        batch = next(iter(loader))
        assert batch.inputs.shape[:2] == batch.targets.shape[:2]
        print(batch.state.group_keys)
        ```

    """

    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    all_valid: bool
    state: BatchState

    def is_protocol(self, protocol: object) -> bool:
        """Return whether any batch lane uses the requested protocol.

        Args:
            protocol: Protocol enum, name, or serialized value.
        """
        value = str(protocol)
        return any(
            value == str(candidate) or value == candidate.name for candidate in self.state.protocols
        )

    def pin_memory(self) -> Self:
        """Return a batch with tensor fields copied to pinned CPU memory."""
        return type(self)(
            inputs=self.inputs.pin_memory(),
            targets=self.targets.pin_memory(),
            mask=self.mask.pin_memory(),
            all_valid=self.all_valid,
            state=self.state,
        )

    def to(self, device: str | torch.device | int | None, *, non_blocking: bool = False) -> Self:
        """Return a batch whose tensor fields are moved to `device`.

        Traceability metadata is reused without modification.

        Args:
            device: PyTorch tensor destination.
            non_blocking: Attempt asynchronous transfer when supported.

        Returns:
            A new batch with transferred tensor fields.
        """
        return type(self)(
            inputs=self.inputs.to(device=device, non_blocking=non_blocking),
            targets=self.targets.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            all_valid=self.all_valid,
            state=self.state,
        )
