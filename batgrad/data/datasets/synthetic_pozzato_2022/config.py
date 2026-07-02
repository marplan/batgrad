from __future__ import annotations

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, DatasetTypeId
from batgrad.contracts.metadata import EIS_PROTOCOL_METADATA, INGEST_STAGE_METADATA
from batgrad.contracts.protocols import (
    EIS_IMPEDANCE_COLUMN_GROUPS,
    BatteryProtocols,
    BatteryProtocolSpec,
)
from batgrad.data.datasets.config import Dataset, DatasetInfo, DatasetSpec
from batgrad.data.datasets.synthetic_pozzato_2022.mapping import (
    NormalizedEisColumns,
    NormalizedTimeseriesColumns,
    RawEisColumns,
    RawTimeseriesColumns,
)
from batgrad.data.datasets.synthetic_pozzato_2022.raw import SyntheticPozzato2022RawAdapter
from batgrad.data.processing.normalize import NormalizeProtocolSpec, NormalizeStageSpec
from batgrad.data.processing.raw import IngestProtocolSpec, IngestStageSpec
from batgrad.data.transforms.checks import (
    CheckSpec,
    ColumnBoundsCheckSpec,
    DomainAxisCheckSpec,
    ImpedanceComponentsCheckSpec,
    MissingCheckSpec,
    TimeCheckSpec,
)
from batgrad.data.transforms.resampling import (
    LinearResamplingSpec,
    MinMaxLTTBResamplingSpec,
    ResamplingSpec,
)

RAW_TIMESERIES = RawTimeseriesColumns()
RAW_EIS = RawEisColumns()
NORMALIZED_TIMESERIES = NormalizedTimeseriesColumns()
NORMALIZED_EIS = NormalizedEisColumns()

SYNTHETIC_EIS_PROTOCOL = BatteryProtocolSpec(
    protocol_id=DatasetProtocolId.eis,
    axis_col=BaseColumns.freq,
    metadata=EIS_PROTOCOL_METADATA,
    task_key_group=(BaseColumns.soc_v,),
    one_of_col_groups=EIS_IMPEDANCE_COLUMN_GROUPS,
)

INGEST_STAGE_SPEC = IngestStageSpec(
    metadata=INGEST_STAGE_METADATA,
    included_file_patterns=("*.parquet",),
    protocol_specs=(
        IngestProtocolSpec(
            protocol=BatteryProtocols.cyc,
            columns=tuple(RAW_TIMESERIES),
        ),
        IngestProtocolSpec(
            protocol=BatteryProtocols.rpt,
            columns=tuple(RAW_TIMESERIES),
        ),
        IngestProtocolSpec(
            protocol=SYNTHETIC_EIS_PROTOCOL,
            columns=tuple(RAW_EIS),
        ),
    ),
)

TIME_NORMALIZE_CHECKS: tuple[CheckSpec, ...] = (
    MissingCheckSpec(
        columns=(
            NORMALIZED_TIMESERIES.crate,
            NORMALIZED_TIMESERIES.volt,
            NORMALIZED_TIMESERIES.temp,
            NORMALIZED_TIMESERIES.amb_temp,
            NORMALIZED_TIMESERIES.a_heat,
        ),
    ),
    TimeCheckSpec(BaseColumns.time, BaseColumns.dt),
    ColumnBoundsCheckSpec(
        {
            BaseColumns.crate: (-50.0, 50.0),
            BaseColumns.volt: (2.3, 4.6),
            BaseColumns.temp: (5.0, 65.0),
            BaseColumns.amb_temp: (0.0, 50.0),
            BaseColumns.a_heat: (1.0, 50.0),
        },
    ),
)


def time_normalize_protocol(
    protocol: BatteryProtocolSpec,
    *,
    resampling: ResamplingSpec | None = None,
) -> NormalizeProtocolSpec:
    return NormalizeProtocolSpec(
        protocol=protocol,
        columns=tuple(NORMALIZED_TIMESERIES),
        checks=TIME_NORMALIZE_CHECKS,
        resampling=resampling,
    )


NORMALIZE_STAGE_SPEC = NormalizeStageSpec(
    protocol_specs=(
        time_normalize_protocol(
            protocol=BatteryProtocols.cyc,
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.time,
                y_col=BaseColumns.volt,
                points_ratio=0.6,
            ),
        ),
        time_normalize_protocol(
            protocol=BatteryProtocols.rpt,
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.time,
                y_col=BaseColumns.volt,
                points=1024,
            ),
        ),
        NormalizeProtocolSpec(
            protocol=SYNTHETIC_EIS_PROTOCOL,
            columns=tuple(NORMALIZED_EIS),
            checks=(
                ImpedanceComponentsCheckSpec(),
                MissingCheckSpec(),
                DomainAxisCheckSpec(
                    axis_col=BaseColumns.freq,
                    zero_replacement=1e-7,
                    enforce_positive=True,
                ),
            ),
            resampling=LinearResamplingSpec(
                x_col=BaseColumns.freq,
                points=32,
                scale="log",
            ),
        ),
    ),
)

DATASET_SPEC = DatasetSpec(
    dataset_id="synthetic-pozzato-2022-m50t",
    dataset_type=DatasetTypeId.synthetic,
    info=DatasetInfo(
        name="Synthetic Pozzato 2022 M50T",
        year=2022,
        author="Pozzato synthetic",
        parent_dataset_id="pozzato-2022",
        misc={"Cell": "LG INR21700 M50T"},
    ),
    processing_stages={
        DatasetStageId.ingested: INGEST_STAGE_SPEC,
        DatasetStageId.normalized: NORMALIZE_STAGE_SPEC,
    },
)

DATASET = Dataset(
    spec=DATASET_SPEC,
    raw_adapter=SyntheticPozzato2022RawAdapter(DATASET_SPEC),
)
