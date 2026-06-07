from __future__ import annotations

from batgrad.contracts.columns import ColumnSpec, MetadataColumns
from batgrad.contracts.domains import Domains
from batgrad.contracts.metadata import MetadataLayout, MetadataLayoutSpec
from batgrad.contracts.values import BaseValues
from batgrad.data.datasets.pozzato_2022.mapping import Pozzato2022Columns, Pozzato2022Values
from batgrad.data.datasets.specs import (
    DatasetInfo,
    DatasetSpec,
    NormalizeSpec,
    ProtocolNormalizeSpec,
    RawIngestSpec,
    RawProtocolSchema,
)
from batgrad.data.locations import DatasetLocation
from batgrad.data.transforms.checks import MissingCheckSpec, TimeCheckSpec
from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec

cols = Pozzato2022Columns
metadata = MetadataLayout()

RAW_TIMESERIES_COLUMNS = (
    cols.time,
    cols.date_time,
    cols.current,
    cols.c_rate,
    cols.voltage,
    cols.aux_temperature_1,
    cols.aux_temperature_2,
    cols.aux_temperature_3,
    cols.charge_capacity,
    cols.discharge_capacity,
    cols.charge_energy,
    cols.discharge_energy,
    cols.internal_resistance,
    cols.acr,
    cols.dv_dt,
    cols.device_id,
    cols.test_id,
    cols.channel_id,
    cols.cycle_index,
    cols.cycle_id,
    cols.step_id,
    cols.pt,
    cols.step_time,
    cols.step_index,
)

RAW_TIMESERIES_DROPPED_COLUMNS = (
    ColumnSpec("dropped Current", alias=("Current",)),
    ColumnSpec("dropped Voltage", alias=("Voltage",)),
)

RAW_EIS_COLUMNS = (
    cols.freq,
    cols.z_mag,
    cols.z_phase,
    cols.z_real,
    cols.z_imag,
    cols.device_id,
    cols.test_id,
    cols.channel_id,
    cols.cycle_id,
    cols.step_id,
    cols.pt,
    cols.step_time,
    cols.step_index,
)

RAW_INGEST_SPEC = RawIngestSpec(
    file_suffixes=(".xlsx",),
    excluded_file_patterns=("**/README.xlsx",),
    row_group_size=256 * 1024,
    max_shard_size_bytes=512 * 1024 * 1024,
    footer_layout=MetadataLayoutSpec(
        required=metadata.parquet_footer,
        optional=(MetadataColumns.nom_capa,),
    ),
    footer_metadata={MetadataColumns.nom_capa: 5.0},
    protocol_schemas=(
        RawProtocolSchema(
            protocol=BaseValues.cycling_protocol,
            domain=Domains.time,
            metadata=metadata.cycling,
            columns=RAW_TIMESERIES_COLUMNS,
            dropped_columns=RAW_TIMESERIES_DROPPED_COLUMNS,
        ),
        RawProtocolSchema(
            protocol=BaseValues.hppc_protocol,
            domain=Domains.time,
            metadata=metadata.hppc,
            columns=RAW_TIMESERIES_COLUMNS,
            dropped_columns=RAW_TIMESERIES_DROPPED_COLUMNS,
        ),
        RawProtocolSchema(
            protocol=BaseValues.rpt_protocol,
            domain=Domains.time,
            metadata=metadata.rpt,
            columns=RAW_TIMESERIES_COLUMNS,
            dropped_columns=RAW_TIMESERIES_DROPPED_COLUMNS,
        ),
        RawProtocolSchema(
            protocol=BaseValues.eis_protocol,
            domain=Domains.freq,
            metadata=metadata.eis,
            columns=RAW_EIS_COLUMNS,
        ),
    ),
)


NORMALIZE_SPEC = NormalizeSpec(
    spec_id="pozzato-2022-normalized-v1",
    protocol_specs={
        "Cycling": ProtocolNormalizeSpec(
            domain=Domains.time,
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                cols.temperature,
            ),
            checks=(
                MissingCheckSpec(),
                TimeCheckSpec(time_col=cols.time, dt_col=cols.dt),
            ),
            resampling=MinMaxLTTBResamplingSpec(
                x_col=cols.time,
                y_col=cols.voltage,
                points_ratio=0.1,
            ),
        ),
        "HPPC": ProtocolNormalizeSpec(
            domain=Domains.time,
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                cols.temperature,
            ),
            checks=(
                MissingCheckSpec(),
                TimeCheckSpec(time_col=cols.time, dt_col=cols.dt),
            ),
            resampling=MinMaxLTTBResamplingSpec(
                x_col=cols.time,
                y_col=cols.voltage,
                points=16_384,
            ),
        ),
    },
)


DATASET_SPEC = DatasetSpec(
    dataset_id="pozzato-2022",
    location=DatasetLocation(
        dataset_type="published",
        dataset_id="pozzato-2022",
    ),
    info=DatasetInfo(
        name="Pozzato 2022",
        year=2022,
        author="Pozzato",
        misc={"Chemistry": "NMC/graphite"},
    ),
    raw=RAW_INGEST_SPEC,
    normalize=NORMALIZE_SPEC,
    cols=Pozzato2022Columns,
    vals=Pozzato2022Values,
    metadata=MetadataLayout,
)
