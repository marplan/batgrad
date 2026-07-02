from __future__ import annotations

from typing import NamedTuple

import polars as pl

from batgrad.contracts.mapping import BaseColumns, MappingSpec


class RawTimeseriesColumns(NamedTuple):
    time: MappingSpec = BaseColumns.time
    curr: MappingSpec = BaseColumns.curr
    crate: MappingSpec = BaseColumns.crate
    volt: MappingSpec = BaseColumns.volt
    temp: MappingSpec = BaseColumns.temp.with_alias("Auxiliary temperature [degC]")
    core_temp: MappingSpec = MappingSpec("Core temperature [degC]", dtype=pl.Float64)
    throughput: MappingSpec = MappingSpec("Cumulative throughput [Ah]", dtype=pl.Float64)
    amb_temp: MappingSpec = BaseColumns.amb_temp
    a_heat: MappingSpec = BaseColumns.a_heat


class RawEisColumns(NamedTuple):
    freq: MappingSpec = BaseColumns.freq
    z_real: MappingSpec = BaseColumns.z_real
    z_imag: MappingSpec = BaseColumns.z_imag


class NormalizedTimeseriesColumns(NamedTuple):
    dt: MappingSpec = BaseColumns.dt
    crate: MappingSpec = BaseColumns.crate
    volt: MappingSpec = BaseColumns.volt
    temp: MappingSpec = BaseColumns.temp
    amb_temp: MappingSpec = BaseColumns.amb_temp
    a_heat: MappingSpec = BaseColumns.a_heat


class NormalizedEisColumns(NamedTuple):
    z_mag: MappingSpec = BaseColumns.z_mag
    z_phase: MappingSpec = BaseColumns.z_phase
    z_real: MappingSpec = BaseColumns.z_real
    z_imag: MappingSpec = BaseColumns.z_imag
