from __future__ import annotations

from batgrad.contracts.columns import BaseColumns, BatteryColumns, RawBatteryColumns
from batgrad.contracts.values import BaseValues


class Pozzato2022Columns:
    current = RawBatteryColumns.current.with_alias("Current(A)", "Current (A)")
    c_rate = BatteryColumns.c_rate.with_alias(
        "Current [C-rate]",
        "Normalized Current (C-rate)",
        "C-rate",
    )
    voltage = RawBatteryColumns.voltage.with_alias("Voltage(V)", "Voltage (V)")
    temperature = BatteryColumns.temperature
    aux_temperature_1 = RawBatteryColumns.aux_temperature_1.with_alias("Aux_Temperature(¡æ)_1")
    aux_temperature_2 = RawBatteryColumns.aux_temperature_2.with_alias("Aux_Temperature(¡æ)_2")
    aux_temperature_3 = RawBatteryColumns.aux_temperature_3.with_alias("Aux_Temperature(¡æ)_3")
    time = BatteryColumns.time.with_alias("Test_Time(s)", "Test Time (s)")
    dt = BatteryColumns.dt
    date_time = BatteryColumns.date_time.with_alias("Date_Time")

    freq = BatteryColumns.freq.with_alias("Freq")
    z_mag = BatteryColumns.z_mag.with_alias("Zmod")
    z_phase = BatteryColumns.z_phase.with_alias("Zphz")
    z_real = BatteryColumns.z_real.with_alias("Zreal")
    z_imag = BatteryColumns.z_imag.with_alias("imaginary")

    charge_capacity = BatteryColumns.charge_capacity.with_alias(
        "Charge_Capacity(Ah)",
        "Charge Capacity (Ah)",
    )
    discharge_capacity = BatteryColumns.discharge_capacity.with_alias(
        "Discharge_Capacity(Ah)",
        "Discharge Capacity (Ah)",
    )
    charge_energy = BatteryColumns.charge_energy.with_alias(
        "Charge_Energy(Wh)",
        "Charge Energy (Wh)",
    )
    discharge_energy = BatteryColumns.discharge_energy.with_alias(
        "Discharge_Energy(Wh)",
        "Discharge Energy (Wh)",
    )
    internal_resistance = BatteryColumns.internal_resistance.with_alias(
        "Internal Resistance(Ohm)",
        "Internal Resistance (Ohm)",
    )
    acr = BatteryColumns.acr.with_alias("ACR (Ohm)", "ACR(Ohm)")
    dv_dt = BatteryColumns.dv_dt.with_alias("dV/dt(V/s)", "dV/dt (V/s)")

    cycle_index = BaseColumns.cycle_index.with_alias("Cycle_Index", "Cycle Index")
    device_id = BaseColumns.device_id.with_alias("Device_ID")
    test_id = BaseColumns.test_id.with_alias("Test_ID")
    channel_id = BaseColumns.channel_id.with_alias("Channel_ID")
    cycle_id = BaseColumns.cycle_id.with_alias("Cycle_ID")
    step_id = BaseColumns.step_id.with_alias("Step_ID")
    pt = BaseColumns.pt
    step_time = BaseColumns.step_time.with_alias("Step_Time(s)", "Step Time (s)")
    step_index = BaseColumns.step_index.with_alias("Step_Index", "Step Index")

    dataset_id = BaseColumns.dataset_id
    cell_id = BaseColumns.cell_id
    file_path = BaseColumns.file_path
    stream_id = BaseColumns.stream_id
    stream_part_idx = BaseColumns.stream_part_idx
    row_count = BaseColumns.row_count
    split = BaseColumns.split
    type_token = BaseColumns.type_token
    file_paths = BaseColumns.file_paths
    annotations = BaseColumns.annotations


class Pozzato2022Values(BaseValues):
    pass
