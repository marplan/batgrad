from __future__ import annotations

from typing import NamedTuple

import polars as pl

from batgrad.contracts.mapping import BaseColumns, MappingSpec


class RawTimeseriesDroppedColumns(NamedTuple):
    curr: MappingSpec = MappingSpec("dropped Current", dtype=pl.Float64, alias="Current")
    volt: MappingSpec = MappingSpec("dropped Voltage", dtype=pl.Float64, alias="Voltage")


def parse_pozzato_datetime(source: str) -> pl.Expr:
    return pl.col(source).str.strptime(
        pl.Datetime("us"),
        format="%Y-%m-%d %H:%M:%S%.3f",
    )


class RawTimeseriesColumns(NamedTuple):
    time: MappingSpec = BaseColumns.time.with_alias("Test_Time(s)", "Test Time (s)")
    dtime: MappingSpec = BaseColumns.dtime.with_alias("Date_Time").with_parser(
        parse_pozzato_datetime
    )
    curr: MappingSpec = BaseColumns.curr.with_alias("Current(A)", "Current (A)")
    volt: MappingSpec = BaseColumns.volt.with_alias("Voltage(V)", "Voltage (V)")
    temp_1: MappingSpec = BaseColumns.temp_1.with_alias("Aux_Temperature(¡æ)_1")
    temp_2: MappingSpec = BaseColumns.temp_2.with_alias("Aux_Temperature(¡æ)_2")
    temp_3: MappingSpec = BaseColumns.temp_3.with_alias("Aux_Temperature(¡æ)_3")
    cap_chg: MappingSpec = BaseColumns.cap_chg.with_alias(
        "Charge_Capacity(Ah)",
        "Charge Capacity (Ah)",
    )
    cap_dchg: MappingSpec = BaseColumns.cap_dchg.with_alias(
        "Discharge_Capacity(Ah)",
        "Discharge Capacity (Ah)",
    )
    eng_chg: MappingSpec = BaseColumns.eng_chg.with_alias(
        "Charge_Energy(Wh)",
        "Charge Energy (Wh)",
    )
    eng_dchg: MappingSpec = BaseColumns.eng_dchg.with_alias(
        "Discharge_Energy(Wh)",
        "Discharge Energy (Wh)",
    )
    res: MappingSpec = BaseColumns.res.with_alias(
        "Internal Resistance(Ohm)",
        "Internal Resistance (Ohm)",
    )
    acr: MappingSpec = BaseColumns.acr.with_alias("ACR (Ohm)", "ACR(Ohm)")
    dvdt: MappingSpec = BaseColumns.dvdt.with_alias("dV/dt(V/s)", "dV/dt (V/s)")
    dev: MappingSpec = BaseColumns.dev.with_alias("Device_ID")
    test: MappingSpec = BaseColumns.test.with_alias("Test_ID")
    chan: MappingSpec = BaseColumns.chan.with_alias("Channel_ID")
    ccidx: MappingSpec = BaseColumns.ccidx.with_alias("Cycle_Index", "Cycle Index")
    cyc: MappingSpec = BaseColumns.cyc.with_alias("Cycle_ID")
    step_id: MappingSpec = BaseColumns.step_id.with_alias("Step_ID")
    pt: MappingSpec = BaseColumns.pt
    step_t: MappingSpec = BaseColumns.step_t.with_alias("Step_Time(s)", "Step Time (s)")
    step: MappingSpec = BaseColumns.step.with_alias("Step_Index", "Step Index")


class RawEisColumns(NamedTuple):
    freq: MappingSpec = BaseColumns.freq.with_alias("Freq")
    z_mag: MappingSpec = BaseColumns.z_mag.with_alias("Zmod")
    z_phase: MappingSpec = BaseColumns.z_phase.with_alias("Zphz")
    z_real: MappingSpec = BaseColumns.z_real.with_alias("Zreal")
    z_imag: MappingSpec = BaseColumns.z_imag.with_alias("imaginary")
    dev: MappingSpec = BaseColumns.dev.with_alias("Device_ID")
    test: MappingSpec = BaseColumns.test.with_alias("Test_ID")
    chan: MappingSpec = BaseColumns.chan.with_alias("Channel_ID")
    cyc: MappingSpec = BaseColumns.cyc.with_alias("Cycle_ID")
    step_id: MappingSpec = BaseColumns.step_id.with_alias("Step_ID")
    pt: MappingSpec = BaseColumns.pt
    step_t: MappingSpec = BaseColumns.step_t.with_alias("Step_Time(s)", "Step Time (s)")
    step: MappingSpec = BaseColumns.step.with_alias("Step_Index", "Step Index")


class NormalizedTimeseriesColumns(NamedTuple):
    dt: MappingSpec = BaseColumns.dt
    crate: MappingSpec = BaseColumns.crate
    volt: MappingSpec = BaseColumns.volt
    temp: MappingSpec = BaseColumns.temp.with_alias(
        BaseColumns.temp_2,
        BaseColumns.temp_1,
        BaseColumns.temp_3,
    )


class NormalizedEisColumns(NamedTuple):
    z_mag: MappingSpec = BaseColumns.z_mag
    z_phase: MappingSpec = BaseColumns.z_phase
    z_real: MappingSpec = BaseColumns.z_real
    z_imag: MappingSpec = BaseColumns.z_imag
