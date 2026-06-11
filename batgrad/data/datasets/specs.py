from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from batgrad.data.processing.config import ProcessingStage

if TYPE_CHECKING:
    from batgrad.contracts.columns import ColumnSpec
    from batgrad.data.datasets.registry import DatasetIds
    from batgrad.data.locations import DatasetLocation
    from batgrad.data.processing.config import NormalizeStageConfig, RawStageConfig
    from batgrad.data.processing.normalize import NormalizeInteractiveRun
    from batgrad.data.processing.normalize_spec import NormalizeSpec
    from batgrad.data.processing.raw_spec import RawIngestSpec
    from batgrad.storage.store import DataStore


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
    vals: type
    metadata: type
    info: DatasetInfo | None = None
    raw: RawIngestSpec | None = None
    normalize: NormalizeSpec | None = None

    @property
    def default_stages(self) -> tuple[ProcessingStage, ...]:
        stages: list[ProcessingStage] = []
        if self.raw is not None:
            stages.append(ProcessingStage.TO_PARQUET)
        if self.normalize is not None:
            stages.append(ProcessingStage.NORMALIZE)
        return tuple(stages)


class Dataset(Protocol):
    spec: DatasetSpec

    def raw_to_parquet(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: RawStageConfig,
    ) -> None: ...

    def normalize(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: NormalizeStageConfig,
    ) -> None: ...

    def normalize_interactive(
        self,
        input_store: DataStore,
        scratch_store: DataStore,
        protocol: str,
        group_values: dict[ColumnSpec, object],
        config: NormalizeStageConfig,
        *,
        annotate: bool = True,
    ) -> NormalizeInteractiveRun: ...
