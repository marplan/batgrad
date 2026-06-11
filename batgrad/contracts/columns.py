from __future__ import annotations

from typing import Self

import polars as pl


class ColumnSpec(str):
    __slots__ = (
        "_alias_lookup",
        "alias",
        "col_max",
        "col_min",
        "description",
        "dtype",
    )

    alias: tuple[str, ...]
    dtype: pl.DataType | None
    description: str | None
    col_min: float | None
    col_max: float | None
    _alias_lookup: frozenset[str] | None

    def __new__(
        cls,
        name: str,
        alias: tuple[str, ...] = (),
        dtype: type[pl.DataType] | pl.DataType | None = None,
        description: str | None = None,
        col_min: float | None = None,
        col_max: float | None = None,
    ) -> Self:
        instance = super().__new__(cls, name)
        instance.alias = (name, *tuple(alias for alias in alias if alias != name))
        instance.dtype = dtype
        instance.description = description
        instance.col_min = col_min
        instance.col_max = col_max
        instance._alias_lookup = None
        return instance

    def with_alias(self, *alias: str) -> Self:
        return type(self)(
            str(self),
            alias=alias,
            dtype=self.dtype,
            description=self.description,
            col_min=self.col_min,
            col_max=self.col_max,
        )

    def matching_name(self, columns: set[str] | tuple[str, ...] | list[str]) -> str | None:
        available = {column.casefold(): column for column in columns}
        for alias in self.alias:
            match = available.get(alias.casefold())
            if match is not None:
                return match
        return None

    def has_match(self, column: str | set[str]) -> bool:
        if self._alias_lookup is None:
            self._alias_lookup = frozenset(alias.casefold() for alias in self.alias)

        if isinstance(column, set):
            return self.matching_name(column) is not None
        return column.casefold() in self._alias_lookup


class BaseColumns:
    cycle_index = ColumnSpec(
        "cycle index",
        dtype=pl.Int64,
        description="Logical cycle/test identifier used for grouping and split keys",
    )
    dataset_id = ColumnSpec("dataset id", dtype=pl.String)
    cell_id = ColumnSpec("cell id", dtype=pl.String)
    stream_id = ColumnSpec("stream id", dtype=pl.String)
    stream_part_idx = ColumnSpec("stream part idx", dtype=pl.Int64)
    type_token = ColumnSpec("type token", dtype=pl.String)
    file_path = ColumnSpec("file path", dtype=pl.String)
    split = ColumnSpec("split", dtype=pl.String)
    row_count = ColumnSpec("row count", dtype=pl.Int64)
    file_paths = ColumnSpec("file paths", dtype=pl.Struct)
    annotations = ColumnSpec(
        "annotations",
        dtype=pl.List(
            pl.Struct(
                {
                    "column": pl.String,
                    "reason": pl.String,
                },
            ),
        ),
    )
    annotation_columns = ColumnSpec("annotation columns", dtype=pl.String)
    annotation_reasons = ColumnSpec("annotation reasons", dtype=pl.String)

    device_id = ColumnSpec("Device ID", dtype=pl.String, description="Device identifier")
    test_id = ColumnSpec("Test ID", dtype=pl.Int64, description="Test identifier")
    channel_id = ColumnSpec("Channel ID", dtype=pl.Int64, description="Channel identifier")
    step_index = ColumnSpec("Step index", dtype=pl.Int64, description="Step index")
    step_time = ColumnSpec("Step time [s]", dtype=pl.Float64, description="Step time in seconds")
    step_id = ColumnSpec("Step ID", dtype=pl.Int64, description="Step identifier")
    cycle_id = ColumnSpec("Cycle ID", dtype=pl.Int64, description="Cycle identifier")
    pt = ColumnSpec("Pt", dtype=pl.Int64, description="Point identifier")

    axis_kind = ColumnSpec("axis kind", dtype=pl.String)
    axis_col = ColumnSpec("axis column", dtype=pl.String)


class MetadataColumns:
    chem = ColumnSpec("chemistry", dtype=pl.String)
    nom_capa = ColumnSpec("Nominal capacity [Ah]", dtype=pl.Float64)

    raw_file_paths = ColumnSpec("raw file paths", dtype=pl.List(pl.String))
    parquet_segments = ColumnSpec(
        "parquet segments",
        dtype=pl.List(
            pl.Struct(
                {
                    "file path": pl.String,
                    "row start": pl.Int64,
                    "row count": pl.Int64,
                },
            ),
        ),
    )
    normalized_segments = ColumnSpec(
        "normalized segments",
        dtype=pl.List(
            pl.Struct(
                {
                    "file path": pl.String,
                    "row start": pl.Int64,
                    "row count": pl.Int64,
                },
            ),
        ),
    )

    domain_id = ColumnSpec("domain id", dtype=pl.String)
    protocol = ColumnSpec("protocol", dtype=pl.String)
    role = ColumnSpec("role", dtype=pl.String)

    row_start = ColumnSpec("row start", dtype=pl.Int64)
    ingest_order = ColumnSpec("ingest order", dtype=pl.Int64)
    row_idx_in_stream = ColumnSpec("row idx in stream", dtype=pl.Int64)

    soc_pct = ColumnSpec("SOC [%]", dtype=pl.Float64)
    protocol_temperature = ColumnSpec("protocol temperature [degC]", dtype=pl.Float64)
    protocol_c_rate = ColumnSpec("protocol C-rate", dtype=pl.Float64)

    resampling_method = ColumnSpec("resampling method", dtype=pl.String)
    resampling_params = ColumnSpec("resampling params", dtype=pl.String)
    time_convention = ColumnSpec("time convention", dtype=pl.String)

    schema_version = ColumnSpec("schema version", dtype=pl.String)
    processing_stage = ColumnSpec("processing stage", dtype=pl.String)
    git_commit = ColumnSpec("git commit", dtype=pl.String)
    git_dirty = ColumnSpec("git dirty", dtype=pl.Boolean)
    manifest_path = ColumnSpec("manifest path", dtype=pl.String)
    protocols = ColumnSpec("protocols", dtype=pl.List(pl.String))
    domains = ColumnSpec("domains", dtype=pl.List(pl.String))


class BatteryColumns:
    time = ColumnSpec("Time [s]", dtype=pl.Float64, description="Test time in seconds")
    dt = ColumnSpec(
        "Time diff [s]",
        dtype=pl.Float64,
        description="Time difference in seconds",
        col_min=0.0,
        col_max=86_400.0,
    )
    date_time = ColumnSpec(
        "Date time",
        dtype=pl.Datetime,
        description="Date and time of measurement",
    )

    current = ColumnSpec(
        "Current [A]",
        dtype=pl.Float64,
        description="Current measurement in Amperes",
        col_min=-30.0,
        col_max=50.0,
    )
    c_rate = ColumnSpec(
        "Current [C-rate]",
        dtype=pl.Float64,
        description="Current normalized by nominal capacity",
        col_min=-50.0,
        col_max=50.0,
    )
    voltage = ColumnSpec(
        "Terminal voltage [V]",
        dtype=pl.Float64,
        description="Terminal voltage in Volts",
        col_min=2.3,
        col_max=4.6,
    )
    temperature = ColumnSpec(
        "Auxiliary temperature [degC]",
        dtype=pl.Float64,
        description="Auxiliary temperature in Celsius",
        col_min=15.0,
        col_max=55.0,
    )
    core_temperature = ColumnSpec(
        "Core temperature [degC]",
        dtype=pl.Float64,
        description="Core/cell temperature in Celsius",
        col_min=15.0,
        col_max=55.0,
    )
    surface_temperature = ColumnSpec(
        "Surface temperature [degC]",
        dtype=pl.Float64,
        description="Surface temperature in Celsius",
        col_min=15.0,
        col_max=55.0,
    )

    charge_capacity = ColumnSpec(
        "Charge capacity [Ah]",
        dtype=pl.Float64,
        description="Charge capacity in Ampere-hours",
    )
    discharge_capacity = ColumnSpec(
        "Discharge capacity [Ah]",
        dtype=pl.Float64,
        description="Discharge capacity in Ampere-hours",
    )
    charge_energy = ColumnSpec(
        "Charge energy [Wh]",
        dtype=pl.Float64,
        description="Charge energy in Watt-hours",
    )
    discharge_energy = ColumnSpec(
        "Discharge energy [Wh]",
        dtype=pl.Float64,
        description="Discharge energy in Watt-hours",
    )
    internal_resistance = ColumnSpec(
        "Internal resistance [Ohm]",
        dtype=pl.Float64,
        description="Internal resistance in Ohms",
    )
    acr = ColumnSpec("ACR [Ohm]", dtype=pl.Float64, description="AC resistance in Ohms")
    dv_dt = ColumnSpec(
        "dV/dt [V.s-1]",
        dtype=pl.Float64,
        description="Voltage derivative in Volts per second",
    )

    freq = ColumnSpec("Frequency [Hz]", dtype=pl.Float64, description="Frequency in Hertz")
    z_mag = ColumnSpec(
        "Impedance magnitude [Ohm]",
        dtype=pl.Float64,
        description="Impedance magnitude in Ohms",
        col_min=-1000.0,
        col_max=1000.0,
    )
    z_phase = ColumnSpec(
        "Impedance phase [deg]",
        dtype=pl.Float64,
        description="Impedance phase in degrees",
        col_min=-1000.0,
        col_max=1000.0,
    )
    z_real = ColumnSpec(
        "Impedance real [Ohm]",
        dtype=pl.Float64,
        description="Real part of impedance",
        col_min=-1000.0,
        col_max=1000.0,
    )
    z_imag = ColumnSpec(
        "Impedance imaginary [Ohm]",
        dtype=pl.Float64,
        description="Imaginary part of impedance",
        col_min=-1000.0,
        col_max=1000.0,
    )


class RawBatteryColumns:
    current = BatteryColumns.current
    voltage = BatteryColumns.voltage
    aux_temperature_1 = ColumnSpec(
        "Auxiliary temperature 1 [degC]",
        dtype=pl.Float64,
        description="First raw auxiliary temperature channel in Celsius",
    )
    aux_temperature_2 = ColumnSpec(
        "Auxiliary temperature 2 [degC]",
        dtype=pl.Float64,
        description="Second raw auxiliary temperature channel in Celsius",
    )
    aux_temperature_3 = ColumnSpec(
        "Auxiliary temperature 3 [degC]",
        dtype=pl.Float64,
        description="Third raw auxiliary temperature channel in Celsius",
    )


def collect_column_specs(columns: type) -> dict[str, ColumnSpec]:
    specs: dict[str, ColumnSpec] = {}
    for name, value in vars(columns).items():
        if name.startswith("_"):
            continue
        if isinstance(value, ColumnSpec):
            specs[name] = value
    return specs


def validate_unique_aliases(columns: type) -> None:
    owner_by_alias: dict[str, str] = {}
    for attr_name, spec in collect_column_specs(columns).items():
        for alias in spec.alias:
            key = alias.casefold()
            owner = owner_by_alias.get(key)
            if owner is not None and owner != attr_name:
                raise ValueError(
                    f"Duplicate alias {alias!r} in {columns.__name__}: {owner!r} and {attr_name!r}",
                )
            owner_by_alias[key] = attr_name


def dtype_map_from_specs(specs: tuple[ColumnSpec, ...]) -> dict[str, pl.DataType]:
    out: dict[str, pl.DataType] = {}
    for spec in specs:
        if spec.dtype is None:
            raise ValueError(f"Column {spec!r} has no dtype")
        out[spec] = spec.dtype
    return out
