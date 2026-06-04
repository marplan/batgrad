from __future__ import annotations

from batgrad.contracts.columns import BaseColumns, BatteryColumns


class Pozzato2022Columns:
    current = BatteryColumns.current.with_aliases("Current(A)", "Current (A)", "Current")
    c_rate = BatteryColumns.c_rate.with_aliases("Normalized Current (C-rate)", "C-rate")
    voltage = BatteryColumns.voltage.with_aliases("Voltage(V)", "Voltage (V)", "Voltage")
    temperature = BatteryColumns.temperature.with_aliases(
        "Aux_Temperature(¡æ)_1",
        "Aux_Temperature(¡æ)_2",
        "Aux_Temperature(¡æ)_3",
    )
    time = BatteryColumns.time.with_aliases("Test_Time(s)", "Test Time (s)")
    dt = BatteryColumns.dt
    date_time = BatteryColumns.date_time.with_aliases("Date_Time")

    freq = BatteryColumns.freq.with_aliases("Freq")
    z_mag = BatteryColumns.z_mag.with_aliases("Zmod")
    z_phase = BatteryColumns.z_phase.with_aliases("Zphz")

    dataset_id = BaseColumns.dataset_id
    cell_id = BaseColumns.cell_id
    stream_id = BaseColumns.stream_id
    type_token = BaseColumns.type_token
