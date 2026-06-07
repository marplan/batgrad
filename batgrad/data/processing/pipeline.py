from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batgrad.data.datasets.registry import DatasetIds, get_dataset
from batgrad.data.processing.config import PROCESSING_STAGE_SPECS, ProcessingStage

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
            stage_spec = PROCESSING_STAGE_SPECS.get(stage)
            if stage_spec is None:
                raise ValueError(f"Unknown processing stage: {stage!r}")

            if getattr(self.spec, stage_spec.dataset_spec_attr) is None:
                raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support {stage.value}")

    def run_stage(self, stage: ProcessingStage) -> None:
        stage_spec = PROCESSING_STAGE_SPECS.get(stage)
        if stage_spec is None:
            raise ValueError(f"Unknown processing stage: {stage!r}")

        stage_config = getattr(self.config, stage_spec.run_config_attr)
        stage_method = getattr(self.dataset, stage_spec.dataset_method)
        stage_method(
            input_store=self.input_store,
            output_store=self.output_store,
            config=stage_config,
        )

    def run(self) -> None:
        stages = self.resolved_stages()
        self.validate_stages(stages)

        for stage in stages:
            self.run_stage(stage)
