from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, DatasetTypeId
from batgrad.contracts.metadata import INGEST_STAGE_METADATA
from batgrad.contracts.protocols import BatteryProtocols
from batgrad.data.datasets.config import DatasetSpec
from batgrad.data.processing.normalize import NormalizeProtocolSpec, NormalizeStageSpec
from batgrad.data.processing.raw import IngestBatch, IngestProtocolSpec, IngestStageSpec, IngestTask
from batgrad.data.transforms.checks import ColumnBoundsCheckSpec, MissingCheckSpec, TimeCheckSpec
from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec
from batgrad.data.transforms.transforms import CRateTransformSpec
from batgrad.storage.local import LocalDataProcessingStore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.storage.store import DataProcessingStore


@dataclass(frozen=True)
class SyntheticAdapter:
    spec: DatasetSpec

    def plan_raw_tasks(
        self,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> tuple[IngestTask, ...]:
        del input_store, raw_spec
        return (IngestTask("task-a", ("raw/a.csv",)), IngestTask("task-b", ("raw/b.csv",)))

    def load_raw_task(
        self,
        task: IngestTask,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> Iterator[IngestBatch]:
        del input_store, raw_spec
        cell = "cell-a" if task.task_id == "task-a" else "cell-b"
        offset = 0 if task.task_id == "task-a" else 10
        frame = synthetic_raw_frame(cell=cell, cycle=1, offset=offset)
        yield IngestBatch(
            data=frame,
            protocol_id=DatasetProtocolId.cycling,
            source_paths=task.source_paths,
            metadata={
                BaseColumns.proto: DatasetProtocolId.cycling,
                BaseColumns.cell_id: cell,
                BaseColumns.cidx: 1,
            },
        )


def store_at(path: Path, *, create: bool = True) -> LocalDataProcessingStore:
    return LocalDataProcessingStore(path.resolve(), create=create)


def raw_stage_spec() -> IngestStageSpec:
    return IngestStageSpec(
        metadata=INGEST_STAGE_METADATA,
        included_file_patterns=("*.csv",),
        excluded_file_patterns=("*_skip.csv",),
        protocol_specs=(
            IngestProtocolSpec(
                protocol=BatteryProtocols.cyc,
                columns=(
                    BaseColumns.time.with_alias("time_s"),
                    BaseColumns.curr.with_alias("current_a"),
                    BaseColumns.volt.with_alias("voltage_v"),
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                ),
            ),
        ),
    )


def normalize_stage_spec(*, resample_points: int | None = None) -> NormalizeStageSpec:
    resampling = (
        None
        if resample_points is None
        else MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=resample_points)
    )
    return NormalizeStageSpec(
        protocol_specs=(
            NormalizeProtocolSpec(
                protocol=BatteryProtocols.cyc,
                columns=(BaseColumns.time, BaseColumns.curr, BaseColumns.volt, BaseColumns.crate),
                constant_columns={BaseColumns.cap_nom: 2.0},
                transforms=(CRateTransformSpec(BaseColumns.curr, BaseColumns.crate, 2.0),),
                checks=(
                    MissingCheckSpec((BaseColumns.curr, BaseColumns.volt)),
                    TimeCheckSpec(BaseColumns.time, BaseColumns.dt, max_dt_s=5.0),
                    ColumnBoundsCheckSpec({BaseColumns.volt: (2.0, 5.0)}),
                ),
                resampling=resampling,
            ),
        ),
    )


def dataset_spec(*, resample_points: int | None = None) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="synthetic-test",
        dataset_type=DatasetTypeId.synthetic,
        processing_stages={
            DatasetStageId.ingested: raw_stage_spec(),
            DatasetStageId.normalized: normalize_stage_spec(resample_points=resample_points),
        },
    )


def synthetic_raw_frame(*, cell: str = "cell-a", cycle: int = 1, offset: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "current_a": [1.0 + offset, 1.1 + offset, 1.2 + offset, 1.3 + offset, 1.4 + offset],
            "voltage_v": [3.0, 3.1, 3.2, 3.3, 3.4],
            str(BaseColumns.cell_id): [cell] * 5,
            str(BaseColumns.cidx): [cycle] * 5,
        }
    )


def canonical_frame(*, cell: str = "cell-a", cycle: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            str(BaseColumns.time): [0.0, 1.0, 1.0, 10.0, 11.0],
            str(BaseColumns.curr): [1.0, 1.1, None, 1.3, 1.4],
            str(BaseColumns.volt): [3.0, 3.1, 6.0, 3.3, 3.4],
            str(BaseColumns.cell_id): [cell] * 5,
            str(BaseColumns.cidx): [cycle] * 5,
        }
    )


def clean_canonical_frame(*, cell: str = "cell-a", cycle: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            str(BaseColumns.time): [0.0, 1.0, 2.0, 3.0, 4.0],
            str(BaseColumns.curr): [1.0, 1.1, 1.2, 1.3, 1.4],
            str(BaseColumns.volt): [3.0, 3.1, 3.2, 3.3, 3.4],
            str(BaseColumns.cell_id): [cell] * 5,
            str(BaseColumns.cidx): [cycle] * 5,
        }
    )
