from __future__ import annotations

from typing import Self

import polars as pl


class ColumnSpec(str):
    __slots__ = (
        "_alias_lookup",
        "aliases",
        "col_max",
        "col_min",
        "description",
        "dtype",
    )

    aliases: tuple[str, ...]
    dtype: pl.DataType | None
    description: str | None
    col_min: float | None
    col_max: float | None
    _alias_lookup: frozenset[str] | None

    def __new__(
        cls,
        name: str,
        aliases: tuple[str, ...] = (),
        dtype: type[pl.DataType] | pl.DataType | None = None,
        description: str | None = None,
        col_min: float | None = None,
        col_max: float | None = None,
    ) -> Self:
        instance = super().__new__(cls, name)
        instance.aliases = (name, *tuple(alias for alias in aliases if alias != name))
        instance.dtype = dtype
        instance.description = description
        instance.col_min = col_min
        instance.col_max = col_max
        instance._alias_lookup = None
        return instance

    def with_aliases(self, *aliases: str) -> Self:
        return type(self)(
            str(self),
            aliases=aliases,
            dtype=self.dtype,
            description=self.description,
            col_min=self.col_min,
            col_max=self.col_max,
        )

    def has_match(self, column: str | set[str]) -> bool:
        if self._alias_lookup is None:
            self._alias_lookup = frozenset(alias.casefold() for alias in self.aliases)

        if isinstance(column, set):
            return any(value.casefold() in self._alias_lookup for value in column)
        return column.casefold() in self._alias_lookup


class BaseColumns:
    dataset_id = ColumnSpec("dataset id", dtype=pl.String)
    cell_id = ColumnSpec("cell id", dtype=pl.String)
    stream_id = ColumnSpec("stream id", dtype=pl.String)
    stream_part_idx = ColumnSpec("stream part idx", dtype=pl.Int64)
    type_token = ColumnSpec("type token", dtype=pl.String)
    file_path = ColumnSpec("file path", dtype=pl.String)
    split = ColumnSpec("split", dtype=pl.String)
    row_count = ColumnSpec("row count", dtype=pl.Int64)
    mark = ColumnSpec("marked", dtype=pl.String)
    comment = ColumnSpec("comments", dtype=pl.String)


class BatteryColumns:
    time = ColumnSpec("Time [s]", dtype=pl.Float64)
    dt = ColumnSpec("Time diff [s]", dtype=pl.Float64)
    date_time = ColumnSpec("Date time", dtype=pl.Datetime)

    current = ColumnSpec(
        "Current [A]",
        dtype=pl.Float64,
        col_min=-30.0,
        col_max=50.0,
    )
    c_rate = ColumnSpec(
        "Current [C-rate]",
        dtype=pl.Float64,
        col_min=-50.0,
        col_max=50.0,
    )
    voltage = ColumnSpec(
        "Terminal voltage [V]",
        dtype=pl.Float64,
        col_min=2.3,
        col_max=4.6,
    )
    temperature = ColumnSpec(
        "Auxiliary temperature [degC]",
        dtype=pl.Float64,
        col_min=15.0,
        col_max=55.0,
    )

    freq = ColumnSpec("Frequency [Hz]", dtype=pl.Float64)
    z_mag = ColumnSpec("Impedance magnitude [Ohm]", dtype=pl.Float64)
    z_phase = ColumnSpec("Impedance phase [deg]", dtype=pl.Float64)
    z_real = ColumnSpec("Impedance real [Ohm]", dtype=pl.Float64)
    z_imag = ColumnSpec("Impedance imaginary [Ohm]", dtype=pl.Float64)


class MetadataColumns:
    chem = ColumnSpec("chemistry", dtype=pl.String)
    nom_capa = ColumnSpec("Nominal capacity [Ah]", dtype=pl.Float64)


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
        for alias in spec.aliases:
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
