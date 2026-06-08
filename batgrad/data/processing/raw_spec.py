from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from batgrad.contracts.columns import ColumnSpec
    from batgrad.contracts.domains import DomainSpec
    from batgrad.contracts.values import ValueSpec


@dataclass(frozen=True, slots=True)
class RawProtocolSchema:
    protocol: ValueSpec
    domain: DomainSpec
    metadata: tuple[ColumnSpec, ...]
    columns: tuple[ColumnSpec, ...]
    dropped_columns: tuple[ColumnSpec, ...] = ()
    flip_current_sign: bool = False

    @property
    def output_columns(self) -> tuple[ColumnSpec, ...]:
        return tuple(dict.fromkeys((*self.columns, *self.metadata)))


@dataclass(frozen=True, slots=True)
class RawIngestSpec:
    footer_metadata: Mapping[ColumnSpec, object] = field(default_factory=dict)
    file_suffixes: tuple[str, ...] = field(default_factory=tuple)
    excluded_file_patterns: tuple[str, ...] = field(default_factory=tuple)
    protocol_schemas: tuple[RawProtocolSchema, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "footer_metadata", dict(self.footer_metadata))

    def protocol_schema(self, protocol: object) -> RawProtocolSchema:
        for schema in self.protocol_schemas:
            if str(schema.protocol) == str(protocol):
                return schema
        raise ValueError(f"Protocol {protocol!r} is not declared in raw protocol schemas")
