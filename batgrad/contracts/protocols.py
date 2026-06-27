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
    """Canonical configuration for one battery protocol.

    `axis_col` is the default domain column used for analysis and plots.
    `metadata` defines protocol-specific task grouping and metadata additions.
    `one_of_col_groups` lists alternative required column groups for protocols
    that accept more than one representation, such as EIS impedance columns.

    Attributes:
        protocol_id: Canonical protocol label used in manifests, paths, and
            dataset mappings.
        axis_col: Default domain/x-axis column for analysis and plots.
        metadata: Protocol-specific metadata extensions and task grouping keys.
        one_of_col_groups: Alternative accepted required column groups. When set,
            a dataset must provide at least one full group.
    """

    protocol_id: DatasetProtocolId
    axis_col: MappingSpec
    metadata: ProtocolMetadata
    one_of_col_groups: tuple[tuple[MappingSpec, ...], ...] = ()


class BatteryProtocols:
    """Shared protocol specs used by dataset configs and processing code."""

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
