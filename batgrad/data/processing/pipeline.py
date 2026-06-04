from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batgrad.data.datasets.registry import DatasetIds, get_dataset
from batgrad.data.processing.config import ProcessingStage
from batgrad.data.processing.normalize import normalize_dataset
from batgrad.data.processing.raw import raw_to_parquet

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import DatasetSpec
    from batgrad.data.locations import DatasetSource
    from batgrad.data.processing.config import ProcessingRunConfig
    from batgrad.storage.store import DataStore


@dataclass(frozen=True, slots=True)
class ProcessingRun:
    dataset_id: DatasetIds
    input_store: DataStore
    output_store: DataStore
    config: ProcessingRunConfig

    @property
    def spec(self) -> DatasetSpec:
        return get_dataset(self.dataset_id)

    def list_input_files(
        self,
        source: DatasetSource,
        pattern: str = "*",
    ) -> tuple[str, ...]:
        return self.input_store.list_files(
            self.spec.location.source_root(source),
            pattern=pattern,
        )

    def list_output_files(
        self,
        source: DatasetSource,
        pattern: str = "*",
    ) -> tuple[str, ...]:
        return self.output_store.list_files(
            self.spec.location.source_root(source),
            pattern=pattern,
        )

    def raw_to_parquet(self) -> None:
        raw_to_parquet(
            spec=self.spec,
            input_store=self.input_store,
            output_store=self.output_store,
            config=self.config.raw,
        )

    def normalize(self) -> None:
        normalize_dataset(
            spec=self.spec,
            input_store=self.input_store,
            output_store=self.output_store,
            config=self.config.normalize,
        )

    def resolved_stages(self) -> tuple[ProcessingStage, ...]:
        if self.config.stages is None:
            return self.spec.default_stages
        return self.config.stages

    def validate_stages(self, stages: tuple[ProcessingStage, ...]) -> None:
        if ProcessingStage.RAW_TO_PARQUET in stages and self.spec.raw is None:
            raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support raw_to_parquet")

        if ProcessingStage.NORMALIZE in stages and self.spec.normalize is None:
            raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support normalize")

    def run(self) -> None:
        stages = self.resolved_stages()
        self.validate_stages(stages)

        for stage in stages:
            if stage == ProcessingStage.RAW_TO_PARQUET:
                self.raw_to_parquet()
            elif stage == ProcessingStage.NORMALIZE:
                self.normalize()
            else:
                raise ValueError(f"Unknown processing stage: {stage!r}")
