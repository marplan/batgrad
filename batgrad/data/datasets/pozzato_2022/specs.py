from __future__ import annotations

from batgrad.contracts.columns import BaseColumns, ColumnSpec, MetadataColumns
from batgrad.contracts.domains import Domains
from batgrad.contracts.metadata import MetadataLayout
from batgrad.contracts.values import BaseValues
from batgrad.data.datasets.pozzato_2022.mapping import Pozzato2022Columns, Pozzato2022Values
from batgrad.data.datasets.specs import (
    DatasetInfo,
    DatasetSpec,
)
from batgrad.data.locations import DatasetLocation
from batgrad.data.processing.normalize_spec import NormalizeSpec, ProtocolNormalizeSpec
from batgrad.data.processing.raw_spec import RawIngestSpec, RawProtocolSchema
from batgrad.data.transforms.checks import (
    ColumnBoundsCheckSpec,
    DomainAxisCheckSpec,
    ImpedanceComponentsCheckSpec,
    MissingCheckSpec,
    TimeCheckSpec,
)
from batgrad.data.transforms.resampling import LinearResamplingSpec, MinMaxLTTBResamplingSpec
from batgrad.data.transforms.transforms import CRateTransformSpec

cols = Pozzato2022Columns
metadata = MetadataLayout()
normalized_temperature = cols.temperature.with_alias(cols.aux_temperature_2, cols.aux_temperature_1)

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
    footer_metadata={MetadataColumns.nom_capa: 5.0},
    protocol_schemas=(
        RawProtocolSchema(
            protocol=BaseValues.cycling_protocol,
            domain=Domains.time,
            metadata=metadata.cycling,
            columns=RAW_TIMESERIES_COLUMNS,
            dropped_columns=RAW_TIMESERIES_DROPPED_COLUMNS,
            flip_current_sign=True,
        ),
        RawProtocolSchema(
            protocol=BaseValues.hppc_protocol,
            domain=Domains.time,
            metadata=metadata.hppc,
            columns=RAW_TIMESERIES_COLUMNS,
            dropped_columns=RAW_TIMESERIES_DROPPED_COLUMNS,
            flip_current_sign=True,
        ),
        RawProtocolSchema(
            protocol=BaseValues.rpt_protocol,
            domain=Domains.time,
            metadata=metadata.rpt,
            columns=RAW_TIMESERIES_COLUMNS,
            dropped_columns=RAW_TIMESERIES_DROPPED_COLUMNS,
            flip_current_sign=True,
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
        BaseValues.cycling_protocol: ProtocolNormalizeSpec(
            domain=Domains.time,
            group_by=(BaseColumns.cell_id, BaseColumns.cycle_index),
            order_by=(cols.time,),
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                normalized_temperature,
            ),
            transforms=(
                CRateTransformSpec(
                    source_col=cols.current,
                    target_col=cols.c_rate,
                    nominal_capacity_ah=5.0,
                ),
            ),
            checks=(
                MissingCheckSpec(),
                TimeCheckSpec(time_col=cols.time, dt_col=cols.dt),
                ColumnBoundsCheckSpec(),
            ),
            resampling=MinMaxLTTBResamplingSpec(
                x_col=cols.time,
                y_col=cols.voltage,
                points_ratio=0.1,
            ),
        ),
        BaseValues.hppc_protocol: ProtocolNormalizeSpec(
            domain=Domains.time,
            group_by=(BaseColumns.cell_id, BaseColumns.cycle_index),
            order_by=(cols.time,),
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                normalized_temperature,
            ),
            transforms=(
                CRateTransformSpec(
                    source_col=cols.current,
                    target_col=cols.c_rate,
                    nominal_capacity_ah=5.0,
                ),
            ),
            checks=(
                MissingCheckSpec(),
                TimeCheckSpec(time_col=cols.time, dt_col=cols.dt),
                ColumnBoundsCheckSpec(),
            ),
            resampling=MinMaxLTTBResamplingSpec(
                x_col=cols.time,
                y_col=cols.voltage,
                points=16_384,
            ),
        ),
        BaseValues.rpt_protocol: ProtocolNormalizeSpec(
            domain=Domains.time,
            group_by=(BaseColumns.cell_id, BaseColumns.cycle_index),
            order_by=(cols.time,),
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                normalized_temperature,
            ),
            transforms=(
                CRateTransformSpec(
                    source_col=cols.current,
                    target_col=cols.c_rate,
                    nominal_capacity_ah=5.0,
                ),
            ),
            checks=(
                MissingCheckSpec(),
                TimeCheckSpec(time_col=cols.time, dt_col=cols.dt),
                ColumnBoundsCheckSpec(),
            ),
            resampling=MinMaxLTTBResamplingSpec(
                x_col=cols.time,
                y_col=cols.voltage,
                points=4096,
            ),
        ),
        BaseValues.eis_protocol: ProtocolNormalizeSpec(
            domain=Domains.freq,
            group_by=(BaseColumns.cell_id, BaseColumns.cycle_index),
            order_by=(cols.freq,),
            columns=(
                cols.freq,
                cols.z_mag,
                cols.z_phase,
                cols.z_real,
                cols.z_imag,
            ),
            checks=(
                ImpedanceComponentsCheckSpec(),
                MissingCheckSpec(),
                ColumnBoundsCheckSpec(),
                DomainAxisCheckSpec(
                    axis_col=cols.freq,
                    zero_replacement=1e-7,
                    enforce_positive=True,
                ),
            ),
            resampling=LinearResamplingSpec(
                x_col=cols.freq,
                points=48,
                scale="log",
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
