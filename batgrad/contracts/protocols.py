from __future__ import annotations

from dataclasses import dataclass

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, MappingSpec
from batgrad.contracts.metadata import (
    CYCLING_PROTOCOL_METADATA,
    EIS_PROTOCOL_METADATA,
    HPPC_PROTOCOL_METADATA,
    RPT_PROTOCOL_METADATA,
    ProtocolMetadata,
)


@dataclass(frozen=True, slots=True)
class BatteryProtocolSpec:
    protocol_id: DatasetProtocolId
    axis_col: MappingSpec
    metadata: ProtocolMetadata
    one_of_col_groups: tuple[tuple[MappingSpec, ...], ...] = ()


class BatteryProtocols:
    cyc = BatteryProtocolSpec(
        protocol_id=DatasetProtocolId.cycling,
        axis_col=BaseColumns.time,
        metadata=CYCLING_PROTOCOL_METADATA,
    )

    hppc = BatteryProtocolSpec(
        protocol_id=DatasetProtocolId.hppc,
        axis_col=BaseColumns.time,
        metadata=HPPC_PROTOCOL_METADATA,
    )

    rpt = BatteryProtocolSpec(
        protocol_id=DatasetProtocolId.rpt,
        axis_col=BaseColumns.time,
        metadata=RPT_PROTOCOL_METADATA,
    )

    eis = BatteryProtocolSpec(
        protocol_id=DatasetProtocolId.eis,
        axis_col=BaseColumns.freq,
        metadata=EIS_PROTOCOL_METADATA,
        one_of_col_groups=(
            (BaseColumns.z_mag, BaseColumns.z_phase),
            (BaseColumns.z_real, BaseColumns.z_imag),
        ),
    )
