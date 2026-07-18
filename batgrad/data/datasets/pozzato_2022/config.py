from __future__ import annotations

from batgrad.contracts.mapping import (
    BaseColumns,
    DatasetStageId,
    DatasetTypeId,
)
from batgrad.contracts.metadata import INGEST_STAGE_METADATA
from batgrad.contracts.protocols import BatteryProtocols, BatteryProtocolSpec
from batgrad.data.datasets.config import Dataset, DatasetInfo, DatasetSpec
from batgrad.data.datasets.pozzato_2022.mapping import (
    NormalizedEisColumns,
    NormalizedTimeseriesColumns,
    RawEisColumns,
    RawTimeseriesColumns,
    RawTimeseriesDroppedColumns,
)
from batgrad.data.datasets.pozzato_2022.raw import Pozzato2022RawAdapter
from batgrad.data.processing.normalize import (
    NormalizeProtocolSpec,
    NormalizeStageSpec,
)
from batgrad.data.processing.raw import (
    IngestProtocolSpec,
    IngestStageSpec,
)
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
from batgrad.data.transforms.transforms import CRateTransformSpec

RAW_TIMESERIES = RawTimeseriesColumns()
RAW_TIMESERIES_DROPPED = RawTimeseriesDroppedColumns()
RAW_EIS = RawEisColumns()
NORMALIZED_TIMESERIES = NormalizedTimeseriesColumns()
NORMALIZED_EIS = NormalizedEisColumns()
NOMINAL_CAPACITY_AH = 5.0
AMBIENT_TEMPERATURE_DEGC = 20.0
COOLING_ALPHA = 20.0


INGEST_STAGE_SPEC = IngestStageSpec(
    metadata=INGEST_STAGE_METADATA,
    included_file_patterns=("*.xlsx",),
    excluded_file_patterns=("**/README.xlsx",),
    protocol_specs=(
        IngestProtocolSpec(
            protocol=BatteryProtocols.cyc,
            columns=tuple(RAW_TIMESERIES),
            dropped_columns=tuple(RAW_TIMESERIES_DROPPED),
            flip_current_sign=True,
        ),
        IngestProtocolSpec(
            protocol=BatteryProtocols.hppc,
            columns=tuple(RAW_TIMESERIES),
            dropped_columns=tuple(RAW_TIMESERIES_DROPPED),
            flip_current_sign=True,
        ),
        IngestProtocolSpec(
            protocol=BatteryProtocols.rpt,
            columns=tuple(RAW_TIMESERIES),
            dropped_columns=tuple(RAW_TIMESERIES_DROPPED),
            flip_current_sign=True,
        ),
        IngestProtocolSpec(
            protocol=BatteryProtocols.eis,
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
        ),
    ),
    TimeCheckSpec(BaseColumns.time, BaseColumns.dt),
    ColumnBoundsCheckSpec(
        {
            BaseColumns.crate: (-50.0, 50.0),
            BaseColumns.volt: (2.3, 4.6),
            BaseColumns.temp: (15.0, 55.0),
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
        constant_columns={
            BaseColumns.amb_temp: AMBIENT_TEMPERATURE_DEGC,
            BaseColumns.a_heat: COOLING_ALPHA,
        },
        transforms=(
            CRateTransformSpec(
                source_col=BaseColumns.curr,
                target_col=BaseColumns.crate,
                nominal_capacity_ah=NOMINAL_CAPACITY_AH,
            ),
        ),
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
                points_ratio=0.1,
            ),
        ),
        time_normalize_protocol(
            protocol=BatteryProtocols.hppc,
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.time,
                y_col=BaseColumns.volt,
                points=16_384,
            ),
        ),
        time_normalize_protocol(
            protocol=BatteryProtocols.rpt,
            resampling=MinMaxLTTBResamplingSpec(
                x_col=BaseColumns.time,
                y_col=BaseColumns.volt,
                points=4096,
            ),
        ),
        NormalizeProtocolSpec(
            protocol=BatteryProtocols.eis,
            columns=tuple(NORMALIZED_EIS),
            constant_columns={
                BaseColumns.amb_temp: AMBIENT_TEMPERATURE_DEGC,
                BaseColumns.a_heat: COOLING_ALPHA,
            },
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
                points=48,
                scale="log",
            ),
        ),
    ),
)


DATASET_SPEC = DatasetSpec(
    dataset_id="pozzato-2022",
    dataset_type=DatasetTypeId.published,
    info=DatasetInfo(
        name="Pozzato 2022",
        year=2022,
        author="Pozzato",
        misc={
            "chemistry": "NMC/graphite",
            "authors": (
                "Gabriele Pozzato",
                "Anirudh Allam",
                "Simona Onori",
            ),
            "doi": "10.1016/j.dib.2022.107995",
            "license": "CC BY 4.0",
            "source_url": (
                "https://osf.io/qsabn/overview?view_only=2a03b6c78ef14922a3e244f3d549de78"
            ),
            "download_url": (
                "https://www.dropbox.com/scl/fo/3ss0age6ggfcm67okldhw/h?"
                "rlkey=tnczvb82gukfe2n4gol2uyo7x&dl=0"
            ),
        },
    ),
    processing_stages={
        DatasetStageId.ingested: INGEST_STAGE_SPEC,
        DatasetStageId.normalized: NORMALIZE_STAGE_SPEC,
    },
)

DATASET = Dataset(
    spec=DATASET_SPEC,
    raw_adapter=Pozzato2022RawAdapter(DATASET_SPEC),
)
