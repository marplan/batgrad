from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ProcessingStage(StrEnum):
    RAW_TO_PARQUET = "raw_to_parquet"
    NORMALIZE = "normalize"


class FailureMode(StrEnum):
    CONTINUE = "continue"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class RawStageConfig:
    n_jobs: int = 1
    polars_max_threads: int | None = None
    chunk_rows: int = 1_000_000
    failure_mode: FailureMode = FailureMode.CONTINUE


@dataclass(frozen=True, slots=True)
class NormalizeStageConfig:
    n_jobs: int = 1
    polars_max_threads: int | None = None
    chunk_rows: int = 200_000
    failure_mode: FailureMode = FailureMode.CONTINUE
    apply_scaling: bool = True
    apply_resampling: bool = True
    resampling_profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingRunConfig:
    stages: tuple[ProcessingStage, ...] | None = None
    raw: RawStageConfig = field(default_factory=RawStageConfig)
    normalize: NormalizeStageConfig = field(default_factory=NormalizeStageConfig)
