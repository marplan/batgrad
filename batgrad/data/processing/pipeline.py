from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batgrad.data.datasets.registry import DatasetIds, get_dataset
from batgrad.data.processing.config import ProcessingStage

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import Dataset, DatasetSpec
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
    def dataset(self) -> Dataset:
        return get_dataset(self.dataset_id)

    @property
    def spec(self) -> DatasetSpec:
        return self.dataset.spec

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

    def resolved_stages(self) -> tuple[ProcessingStage, ...]:
        if self.config.stages is None:
            return self.spec.default_stages
        return self.config.stages

    def validate_stages(self, stages: tuple[ProcessingStage, ...]) -> None:
        for stage in stages:
            if stage == ProcessingStage.TO_PARQUET:
                if self.spec.raw is None:
                    raise ValueError(
                        f"Dataset {self.spec.dataset_id!r} does not support {stage.value}",
                    )
            elif stage == ProcessingStage.NORMALIZE:
                if self.spec.normalize is None:
                    raise ValueError(
                        f"Dataset {self.spec.dataset_id!r} does not support {stage.value}",
                    )
            else:
                raise ValueError(f"Unknown processing stage: {stage!r}")

    def run_stage(self, stage: ProcessingStage) -> None:
        if stage == ProcessingStage.TO_PARQUET:
            self.dataset.raw_to_parquet(
                input_store=self.input_store,
                output_store=self.output_store,
                config=self.config.raw,
            )
            return
        if stage == ProcessingStage.NORMALIZE:
            self.dataset.normalize(
                input_store=self.input_store,
                output_store=self.output_store,
                config=self.config.normalize,
            )
            return
        raise ValueError(f"Unknown processing stage: {stage!r}")

    def run(self) -> None:
        stages = self.resolved_stages()
        self.validate_stages(stages)

        for stage in stages:
            self.run_stage(stage)
