from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from batgrad.data.processing.interactive import InteractiveStageRun, run_load_interactive
from batgrad.data.processing.normalize import (
    NormalizeStageConfig,
    NormalizeStageSpec,
    run_normalize,
    run_normalize_interactive,
)
from batgrad.data.processing.raw import IngestStageConfig, IngestStageSpec, run_ingest

if TYPE_CHECKING:
    from batgrad.contracts.mapping import DatasetStageId, DatasetTypeId
    from batgrad.data.processing.normalize import NormalizeTask
    from batgrad.data.processing.raw import IngestTask, RawDatasetAdapter
    from batgrad.storage.store import DataProcessingStore

STAGE_SPECS = IngestStageSpec | NormalizeStageSpec
DatasetId = str


@dataclass(frozen=True)
class DatasetInfo:
    name: str | None = None
    year: int | None = None
    author: str | None = None
    parent_dataset_id: DatasetId | None = None
    misc: dict[str, object] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: DatasetId
    dataset_type: DatasetTypeId
    info: DatasetInfo | None = None
    processing_stages: dict[DatasetStageId, STAGE_SPECS] = field(default_factory=dict)

    @property
    def root(self) -> str:
        """Relative root location of the dataset.

        Absolute root location will be resolved by `DataStore`.
        """
        return f"type={self.dataset_type}/dataset={self.dataset_id}"

    def source_root(self, source: DatasetStageId) -> str:
        return f"{self.root}/source={source}"

    def source_file(self, source: DatasetStageId, file_name: str) -> str:
        return f"{self.source_root(source)}/{file_name}"

    def manifest(self, source: DatasetStageId) -> str:
        return self.source_file(source, "manifest.parquet")


@dataclass(frozen=True)
class Dataset:
    spec: DatasetSpec
    raw_adapter: RawDatasetAdapter | None = None

    def ingest(
        self,
        input_store: DataProcessingStore,
        output_store: DataProcessingStore,
        config: IngestStageConfig,
        *,
        scratch_store: DataProcessingStore | None = None,
        tasks: tuple[IngestTask, ...] | None = None,
    ) -> None:
        if self.raw_adapter is None:
            raise TypeError(f"Dataset {self.spec.dataset_id!r} does not support ingest")
        run_ingest(
            self.raw_adapter,
            input_store,
            output_store,
            config,
            scratch_store=scratch_store,
            tasks=tasks,
        )

    def normalize(
        self,
        input_store: DataProcessingStore,
        output_store: DataProcessingStore,
        config: NormalizeStageConfig,
        *,
        scratch_store: DataProcessingStore | None = None,
        tasks: tuple[NormalizeTask, ...] | None = None,
        dry_run: bool = False,
    ) -> None:
        run_normalize(
            self.spec,
            input_store,
            output_store,
            config,
            scratch_store=scratch_store,
            tasks=tasks,
            dry_run=dry_run,
        )

    def normalize_interactive(
        self,
        input_store: DataProcessingStore,
        scratch_store: DataProcessingStore,
        config: NormalizeStageConfig,
        *,
        protocols: object = None,
        group_values: object = None,
        annotate: bool = True,
        source_run: InteractiveStageRun | None = None,
        normalize_spec: NormalizeStageSpec | None = None,
    ) -> InteractiveStageRun:
        return run_normalize_interactive(
            self.spec,
            input_store,
            scratch_store,
            config,
            protocols=protocols,
            group_values=group_values,
            annotate=annotate,
            source_run=source_run,
            normalize_spec=normalize_spec,
        )

    def load_interactive(
        self,
        input_store: DataProcessingStore,
        *,
        source: DatasetStageId | str,
        protocols: object = None,
        group_values: object = None,
    ) -> InteractiveStageRun:
        return run_load_interactive(
            self.spec,
            input_store,
            source=source,
            protocols=protocols,
            group_values=group_values,
        )
