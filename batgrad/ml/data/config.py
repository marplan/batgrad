from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, MappingSpec

PADDING_VALUE = -2.0
type BatchStrategy = Literal["sequential", "shuffled_protocol_groups"]
type CrossProtocolStateCarry = Literal["chain"]
type MultiprocessingContext = Literal["fork", "spawn", "forkserver"]
type DataAccessMode = Literal["windowed", "full_in_mem"]
type ScalingTransform = Literal["linear", "log1p"]
GROUP_KEY_CELL_CYCLE_PROTOCOL = (
    BaseColumns.set_id,
    BaseColumns.cell_id,
    BaseColumns.cidx,
    BaseColumns.proto,
)
ALIGNMENT_KEY_CELL_CYCLE = (
    BaseColumns.set_id,
    BaseColumns.cell_id,
    BaseColumns.cidx,
)


def column_name(column: str | MappingSpec) -> str:
    return column


def coerce_protocol(protocol: object) -> DatasetProtocolId:
    if isinstance(protocol, DatasetProtocolId):
        return protocol
    value = str(protocol)
    for candidate in DatasetProtocolId:
        if value == str(candidate) or value == candidate.name:
            return candidate
    raise ValueError(f"Unknown dataset protocol: {value!r}")


@dataclass(frozen=True)
class ValidationConfig:
    """Group-aware split policy used while constructing an ML index.

    Attributes:
        strategy: `"sample"` hashes groups deterministically, `"provide"`
            selects explicit groups, and `"merge"` combines both.
        fraction: Fraction of groups assigned to validation by sampling.
        seed: Hash seed used by sampling.
        group_by: Manifest columns that define an indivisible group.
        provided: Explicit partial or complete group selectors.

    Examples:
        Hold out one cell explicitly:

        ```python
        from batgrad.contracts.mapping import BaseColumns
        from batgrad.ml.data.config import ValidationConfig

        validation = ValidationConfig.provide(
            ({BaseColumns.set_id: "pozzato-2022", BaseColumns.cell_id: "Cell1"},)
        )
        ```
    """

    strategy: Literal["sample", "provide", "merge"] = "sample"
    fraction: float = 0.2
    seed: int = 69
    group_by: tuple[str | MappingSpec, ...] = (
        BaseColumns.set_id,
        BaseColumns.cell_id,
        BaseColumns.cidx,
    )
    provided: tuple[dict[str | MappingSpec, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.strategy not in {"sample", "provide", "merge"}:
            raise ValueError(f"Unknown validation strategy: {self.strategy!r}")
        if not (0.0 <= self.fraction < 1.0):
            raise ValueError(f"fraction must be in [0, 1), got {self.fraction}")
        if not self.group_by:
            raise ValueError("validation group_by must not be empty")
        group_names = tuple(column_name(column) for column in self.group_by)
        if len(set(group_names)) != len(group_names):
            raise ValueError(f"validation group_by contains duplicates: {group_names}")
        if self.strategy == "provide" and not self.provided:
            raise ValueError("provide validation strategy requires provided groups")

    @classmethod
    def sample(
        cls,
        fraction: float = 0.2,
        seed: int = 69,
        group_by: tuple[str | MappingSpec, ...] = (
            BaseColumns.set_id,
            BaseColumns.cell_id,
            BaseColumns.cidx,
        ),
    ) -> ValidationConfig:
        """Create a deterministic sampled-group split.

        Args:
            fraction: Fraction of groups assigned to validation.
            seed: Hash seed.
            group_by: Columns defining one group.

        Returns:
            A sampled-group validation policy.
        """
        return cls(strategy="sample", fraction=fraction, seed=seed, group_by=group_by)

    @classmethod
    def provide(
        cls,
        provided: tuple[dict[str | MappingSpec, object], ...],
        group_by: tuple[str | MappingSpec, ...] = (
            BaseColumns.set_id,
            BaseColumns.cell_id,
            BaseColumns.cidx,
        ),
    ) -> ValidationConfig:
        """Create a split containing only explicitly selected groups.

        Args:
            provided: Partial or complete group selectors.
            group_by: Columns defining one group.

        Returns:
            An explicit-group validation policy.
        """
        return cls(strategy="provide", fraction=0.0, group_by=group_by, provided=provided)

    @classmethod
    def merge(
        cls,
        provided: tuple[dict[str | MappingSpec, object], ...],
        fraction: float = 0.2,
        seed: int = 69,
        group_by: tuple[str | MappingSpec, ...] = (
            BaseColumns.set_id,
            BaseColumns.cell_id,
            BaseColumns.cidx,
        ),
    ) -> ValidationConfig:
        """Create a split combining explicit and sampled groups.

        Args:
            provided: Partial or complete group selectors.
            fraction: Additional fraction selected by deterministic sampling.
            seed: Hash seed.
            group_by: Columns defining one group.

        Returns:
            A merged explicit-and-sampled validation policy.
        """
        return cls(
            strategy="merge",
            fraction=fraction,
            seed=seed,
            group_by=group_by,
            provided=provided,
        )


@dataclass(frozen=True)
class ScalingRule:
    """Runtime scaling rule for one tensor or frame column.

    Attributes:
        column: Canonical column name or mapping specification.
        input_min: Lower physical-unit bound.
        input_max: Upper physical-unit bound.
        output_min: Lower model-space bound.
        output_max: Upper model-space bound.
        clip: Clip forward-scaled values to output bounds.
        transform: Linear scaling or `log1p` followed by linear scaling.

    Note:
        Tensor functions associate rules with channels by tuple order. Frame
        functions associate rules by column name.
    """

    column: str | MappingSpec
    input_min: float
    input_max: float
    output_min: float = -1.0
    output_max: float = 1.0
    clip: bool = False
    transform: ScalingTransform = "linear"

    def __post_init__(self) -> None:
        if self.transform not in {"linear", "log1p"}:
            raise ValueError(f"Unknown scaling transform: {self.transform!r}")
        if not all(
            math.isfinite(value)
            for value in (self.input_min, self.input_max, self.output_min, self.output_max)
        ):
            raise ValueError("scaling bounds must be finite")
        if self.input_min >= self.input_max:
            raise ValueError(
                f"input_min must be < input_max, got {self.input_min} >= {self.input_max}"
            )
        if self.transform == "log1p" and self.input_min < 0.0:
            raise ValueError(f"log1p scaling requires input_min >= 0, got {self.input_min}")
        if self.output_min >= self.output_max:
            raise ValueError(
                f"output_min must be < output_max, got {self.output_min} >= {self.output_max}"
            )

    @property
    def name(self) -> str:
        return column_name(self.column)


@dataclass(frozen=True)
class WindowConfig:
    """Windowing settings for one protocol stream.

    `WindowConfig` controls how a normalized manifest row is split into model
    windows before tensor materialization. It is protocol-specific because time
    series protocols such as cycling/HPPC and frequency-domain protocols such as
    EIS usually need different sequence lengths.

    Attributes:
        batch_size: Number of contiguous sequence lanes carved from each window.
        seq_len: Number of input and shifted-target positions per lane.
        drop_incomplete: Drop a final source window that cannot fill every lane.
        step_rows: Source-row distance between windows. The default advances by
            `batch_size * seq_len`.

    Examples:
        Configure cycling as long ordered time windows and EIS as one compact
        frequency window:

        ```python
        windows = {
            DatasetProtocolId.cycling: WindowConfig(batch_size=8, seq_len=1024),
            DatasetProtocolId.eis: WindowConfig(batch_size=16, seq_len=48),
        }
        ```

    """

    batch_size: int = 32
    seq_len: int = 128
    drop_incomplete: bool = True
    step_rows: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.step_rows is not None and self.step_rows <= 0:
            raise ValueError(f"step_rows must be > 0 when set, got {self.step_rows}")

    @property
    def step(self) -> int:
        """Source-row distance between consecutive windows."""
        return self.step_rows or self.batch_size * self.seq_len

    @property
    def window_rows(self) -> int:
        """Source rows needed for inputs and one-row-ahead targets."""
        return self.batch_size * self.seq_len + 1


def _validate_loader_runtime(config: LoaderConfig) -> None:
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {config.num_workers}")
    if config.prefetch_factor <= 0:
        raise ValueError(f"prefetch_factor must be > 0, got {config.prefetch_factor}")
    if config.multiprocessing_context not in {None, "fork", "spawn", "forkserver"}:
        raise ValueError(
            "multiprocessing_context must be one of None, 'fork', 'spawn', or 'forkserver', "
            f"got {config.multiprocessing_context!r}"
        )
    if not str(config.device).strip():
        raise ValueError("device must not be empty")


def _validate_loader_strategy(config: LoaderConfig) -> None:
    if config.strategy not in {"sequential", "shuffled_protocol_groups"}:
        raise ValueError(f"Unknown batch strategy: {config.strategy!r}")
    if config.stateful_n_windows == 0 or config.stateful_n_windows < -1:
        raise ValueError(
            "stateful_n_windows must be -1 for whole-stream mode or a positive integer, "
            f"got {config.stateful_n_windows}"
        )
    if config.data_access not in {"windowed", "full_in_mem"}:
        raise ValueError(
            f"data_access must be 'windowed' or 'full_in_mem', got {config.data_access!r}"
        )
    if config.data_access == "full_in_mem" and config.num_workers != 0:
        raise ValueError("data_access='full_in_mem' requires num_workers=0")
    if config.cross_protocol_state_carry not in {None, "chain"}:
        raise ValueError(
            "cross_protocol_state_carry must be None or 'chain', "
            f"got {config.cross_protocol_state_carry!r}"
        )


@dataclass(frozen=True)
class LoaderConfig:
    """Runtime options for protocol-aware ML loading.

    The canonical ML index stores facts about available manifest rows. Loader
    configuration describes how those rows are planned into windows and yielded.
    `group_key` identifies one protocol-specific stream/window source, while
    `alignment_key` identifies the shared physical context used by future
    multi-protocol schedules.

    Usually `group_key` includes protocol and `alignment_key` does not:

    ```text
    group_key = (dataset id, cell id, cycle index, protocol)
    alignment_key = (dataset id, cell id, cycle index)
    ```

    With this setup cycling/HPPC/EIS remain separate streams but can later be
    bundled together for the same cell/cycle.

    `data_access` controls the speed/RAM trade-off:

    - `"windowed"` reads only the windows required for each batch from parquet.
      This has the lowest RAM footprint and is easiest to reason about, but it is
      IO-bound and can be orders of magnitude slower than memory-backed loading.
    - `"full_in_mem"` preloads the full selected split, active protocol, and
      requested columns into CPU `float32` tensors. RAM scales approximately as
      `rows * selected_columns * 4 bytes`, plus runtime and temporary conversion
      overhead. A synthetic normalized smoke with batch=64, seq=1024, three
       inputs and one target measured roughly 0.1-0.2M tokens/s for `windowed` and
       30M+ tokens/s for `full_in_mem` after the cache is built.

    Attributes:
        split: Index split yielded by the loader.
        default_window: Window shape used unless a protocol override exists.
        window_by_protocol: Protocol-specific window overrides.
        seed: Reproducible planning and epoch-phase seed.
        strategy: Sequential windows or shuffled protocol-group streams.
        protocol_order: Explicit protocol traversal order.
        stateful_n_windows: Consecutive windows per stateful group, or `-1` for
            whole-stream mode.
        cross_protocol_state_carry: `"chain"` groups aligned protocol streams
            for recurrent state carry.
        drop_incomplete_batches: Drop plans with fewer than the requested lanes.
        drop_incomplete_distributed: Equalize plan counts across distributed ranks.
        data_access: Parquet-window or full-memory access.
        group_key: Columns identifying a protocol-specific stream.
        alignment_key: Columns identifying streams that may be aligned.
        num_workers: PyTorch worker process count.
        prefetch_factor: Batches prefetched by each worker.
        persistent_workers: Retain workers between iterations.
        pin_memory: Pin CPU tensors before device transfer.
        multiprocessing_context: Worker start method.
        device: Destination device used by optional prefetch.
        prefetch_to_device: Asynchronously move batches to CUDA.
        non_blocking: Use non-blocking tensor transfers where possible.

    Note:
        Whole-stream shuffled batches truncate all lanes to the shortest lane.
        Finite stateful groups discard a final group shorter than
        `stateful_n_windows`. Shuffled protocol groups do not support EIS;
        sequential mode traverses requested protocols in order. `full_in_mem`
        requires `num_workers=0`; device prefetch requires CUDA.
    """

    split: str = BaseColumns.split.values.train
    default_window: WindowConfig = field(default_factory=WindowConfig)
    window_by_protocol: dict[DatasetProtocolId, WindowConfig] = field(default_factory=dict)
    seed: int = 69
    strategy: BatchStrategy = "shuffled_protocol_groups"
    protocol_order: tuple[DatasetProtocolId, ...] = ()
    stateful_n_windows: int = 1
    cross_protocol_state_carry: CrossProtocolStateCarry | None = None
    drop_incomplete_batches: bool = True
    drop_incomplete_distributed: bool = True
    data_access: DataAccessMode = "windowed"
    group_key: tuple[str | MappingSpec, ...] = GROUP_KEY_CELL_CYCLE_PROTOCOL
    alignment_key: tuple[str | MappingSpec, ...] = ALIGNMENT_KEY_CELL_CYCLE
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = False
    multiprocessing_context: MultiprocessingContext | None = "spawn"
    device: str = "cpu"
    prefetch_to_device: bool = False
    non_blocking: bool = True

    def __post_init__(self) -> None:
        splits = {BaseColumns.split.values.train, BaseColumns.split.values.val}
        if self.split not in splits:
            raise ValueError(f"split must be one of {sorted(splits)}, got {self.split!r}")
        if not self.group_key:
            raise ValueError("group_key must not be empty")
        if not self.alignment_key:
            raise ValueError("alignment_key must not be empty")
        _validate_loader_runtime(self)
        _validate_loader_strategy(self)

    def window_for(self, protocol: DatasetProtocolId) -> WindowConfig:
        """Return window settings for a protocol.

        Args:
            protocol: Protocol whose override should be resolved.

        Returns:
            The protocol override, or `default_window` when no override exists.
        """
        return self.window_by_protocol.get(protocol, self.default_window)
