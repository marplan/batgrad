from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ProcessingStage(StrEnum):
    TO_PARQUET = "raw_to_parquet"
    NORMALIZE = "normalize"


class FailureMode(StrEnum):
    CONTINUE = "continue"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class ProcessingStageSpec:
    stage: ProcessingStage
    dataset_spec_attr: str
    run_config_attr: str
    dataset_method: str


PROCESSING_STAGE_SPECS: dict[ProcessingStage, ProcessingStageSpec] = {
    ProcessingStage.TO_PARQUET: ProcessingStageSpec(
        stage=ProcessingStage.TO_PARQUET,
        dataset_spec_attr="raw",
        run_config_attr="raw",
        dataset_method="raw_to_parquet",
    ),
    ProcessingStage.NORMALIZE: ProcessingStageSpec(
        stage=ProcessingStage.NORMALIZE,
        dataset_spec_attr="normalize",
        run_config_attr="normalize",
        dataset_method="normalize",
    ),
}


@dataclass(frozen=True, slots=True)
class RawStageConfig:
    n_jobs: int = 1
    polars_max_threads: int | None = None
    chunk_rows: int = 256_000
    failure_mode: FailureMode = FailureMode.STRICT


@dataclass(frozen=True, slots=True)
class NormalizeStageConfig:
    n_jobs: int = 1
    polars_max_threads: int | None = None
    chunk_rows: int = 200_000
    failure_mode: FailureMode = FailureMode.STRICT
    apply_scaling: bool = True
    apply_resampling: bool = True
    resampling_profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingRunConfig:
    stages: tuple[ProcessingStage, ...] | None = None
    raw: RawStageConfig = field(default_factory=RawStageConfig)
    normalize: NormalizeStageConfig = field(default_factory=NormalizeStageConfig)
