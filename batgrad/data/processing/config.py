from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from batgrad.contracts.metadata import MetadataLayout, MetadataLayoutSpec

if TYPE_CHECKING:
    from batgrad.data.locations import DatasetSource


class ProcessingStage(StrEnum):
    TO_PARQUET = "raw_to_parquet"
    NORMALIZE = "normalize"


class FailureMode(StrEnum):
    CONTINUE = "continue"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class ProcessingStageSpec:
    stage: ProcessingStage
    input_source: DatasetSource
    output_source: DatasetSource
    processing_stage: str
    manifest_layout: MetadataLayoutSpec
    footer_layout: MetadataLayoutSpec


PROCESSING_STAGE_SPECS: dict[ProcessingStage, ProcessingStageSpec] = {
    ProcessingStage.TO_PARQUET: ProcessingStageSpec(
        stage=ProcessingStage.TO_PARQUET,
        input_source="raw",
        output_source="parquet",
        processing_stage="raw",
        manifest_layout=MetadataLayoutSpec(required=MetadataLayout().parquet_manifest),
        footer_layout=MetadataLayoutSpec(required=MetadataLayout().parquet_footer),
    ),
    ProcessingStage.NORMALIZE: ProcessingStageSpec(
        stage=ProcessingStage.NORMALIZE,
        input_source="parquet",
        output_source="normalized",
        processing_stage="normalize",
        manifest_layout=MetadataLayoutSpec(required=MetadataLayout().normalized_manifest),
        footer_layout=MetadataLayoutSpec(required=MetadataLayout().normalized_footer),
    ),
}


@dataclass(frozen=True, slots=True)
class RawStageConfig:
    n_jobs: int = 1
    polars_max_threads: int | None = None
    chunk_rows: int = 256_000
    failure_mode: FailureMode = FailureMode.STRICT
    compression: str = "zstd"
    use_content_defined_chunking: bool = True
    row_group_size: int = 256_000
    max_shard_size_bytes: int = 500 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NormalizeStageConfig:
    n_jobs: int = 1
    polars_max_threads: int | None = None
    chunk_rows: int = 200_000
    failure_mode: FailureMode = FailureMode.STRICT
    compression: str = "zstd"
    use_content_defined_chunking: bool = True
    row_group_size: int = 256_000
    max_shard_size_bytes: int = 500 * 1024 * 1024
    apply_scaling: bool = True
    apply_resampling: bool = True
    resampling_profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingRunConfig:
    stages: tuple[ProcessingStage, ...] | None = None
    raw: RawStageConfig = field(default_factory=RawStageConfig)
    normalize: NormalizeStageConfig = field(default_factory=NormalizeStageConfig)
