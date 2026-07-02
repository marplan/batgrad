from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, MappingSpec

PADDING_VALUE = -2.0
type BatchStrategy = Literal["sequential", "shuffled_protocol_groups"]
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
        return cls(
            strategy="merge",
            fraction=fraction,
            seed=seed,
            group_by=group_by,
            provided=provided,
        )


@dataclass(frozen=True)
class ScalingRule:
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

    Examples:
        Configure cycling as long ordered time windows and EIS as one compact
        frequency window::

            windows = {
                DatasetProtocolId.cycling: WindowConfig(batch_size=8, seq_len=1024),
                DatasetProtocolId.eis: WindowConfig(batch_size=16, seq_len=48),
            }

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
        return self.step_rows or self.batch_size * self.seq_len

    @property
    def window_rows(self) -> int:
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


@dataclass(frozen=True)
class LoaderConfig:
    """Runtime options for protocol-aware ML loading.

    The canonical ML index stores facts about available manifest rows. Loader
    configuration describes how those rows are planned into windows and yielded.
    `group_key` identifies one protocol-specific stream/window source, while
    `alignment_key` identifies the shared physical context used by future
    multi-protocol schedules.

    Usually `group_key` includes protocol and `alignment_key` does not::

        group_key = (dataset id, cell id, cycle index, protocol)
        alignment_key = (dataset id, cell id, cycle index)

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
    """

    split: str = BaseColumns.split.values.train
    default_window: WindowConfig = field(default_factory=WindowConfig)
    window_by_protocol: dict[DatasetProtocolId, WindowConfig] = field(default_factory=dict)
    seed: int = 69
    strategy: BatchStrategy = "shuffled_protocol_groups"
    active_protocol: DatasetProtocolId | None = None
    stateful_n_windows: int = 1
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
        return self.window_by_protocol.get(protocol, self.default_window)
