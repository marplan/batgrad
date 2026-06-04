from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from batgrad.data.processing.config import ProcessingStage

if TYPE_CHECKING:
    from batgrad.contracts.columns import ColumnSpec
    from batgrad.data.datasets.registry import DatasetIds
    from batgrad.data.locations import DatasetLocation


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str | None = None
    year: int | None = None
    author: str | None = None
    parent_dataset_id: str | None = None
    misc: dict[str, str] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: DatasetIds
    location: DatasetLocation
    cols: type
    info: DatasetInfo = field(default_factory=DatasetInfo)
    raw: RawIngestSpec | None = None
    normalize: NormalizeSpec | None = None

    @property
    def default_stages(self) -> tuple[ProcessingStage, ...]:
        stages: list[ProcessingStage] = []
        if self.raw is not None:
            stages.append(ProcessingStage.RAW_TO_PARQUET)
        if self.normalize is not None:
            stages.append(ProcessingStage.NORMALIZE)
        return tuple(stages)


@dataclass(frozen=True, slots=True)
class RawIngestSpec:
    input_source: Literal["raw"] = "raw"
    output_source: Literal["parquet"] = "parquet"
    file_suffixes: tuple[str, ...] = field(default_factory=tuple)
    excluded_files: frozenset[str] = field(default_factory=frozenset)
    row_group_size: int = 256_000
    max_shard_size_bytes: int = 500 * 1024 * 1024
    shard_size_tolerance_ratio: float = 0.03


CheckName = Literal["missing", "time", "battery_signal_corr"]


@dataclass(frozen=True, slots=True)
class TokenNormalizeSpec:
    columns: tuple[ColumnSpec, ...]
    checks: tuple[CheckName, ...] = field(default_factory=tuple)
    resampling_profile_id: str = "none"
    scaling_profile_id: str = "none"


@dataclass(frozen=True, slots=True)
class NormalizeSpec:
    spec_id: str
    input_source: Literal["parquet"] = "parquet"
    output_source: Literal["normalized"] = "normalized"
    time_convention: str = "start_of_interval"
    token_specs: dict[str, TokenNormalizeSpec] = field(default_factory=dict)
    row_group_size: int = 256_000
    max_shard_size_bytes: int = 500 * 1024 * 1024
    shard_size_tolerance_ratio: float = 0.03
