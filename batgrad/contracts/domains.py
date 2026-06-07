from __future__ import annotations

from dataclasses import dataclass

from batgrad.contracts.columns import BatteryColumns, ColumnSpec
from batgrad.contracts.values import BaseValues, ValueSpec


@dataclass(frozen=True, slots=True)
class DomainSpec:
    domain_id: ValueSpec
    axis_col: ColumnSpec
    one_of_col_groups: tuple[tuple[ColumnSpec, ...], ...] = ()


class Domains:
    time = DomainSpec(
        domain_id=BaseValues.time_domain,
        axis_col=BatteryColumns.time,
    )

    freq = DomainSpec(
        domain_id=BaseValues.freq_domain,
        axis_col=BatteryColumns.freq,
        one_of_col_groups=(
            (BatteryColumns.z_mag, BatteryColumns.z_phase),
            (BatteryColumns.z_real, BatteryColumns.z_imag),
        ),
    )
