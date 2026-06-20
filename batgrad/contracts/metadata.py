from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from batgrad.contracts.mapping import BaseColumns, DatasetStageId, MappingSpec

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class MetadataLayout:
    required: tuple[MappingSpec, ...] = ()
    optional: Mapping[MappingSpec, object | None] = field(default_factory=dict)

    @property
    def columns(self) -> tuple[MappingSpec, ...]:
        return (*self.required, *self.optional.keys())

    @property
    def values(self) -> dict[MappingSpec, object | None]:
        return dict(self.optional)

    def with_optional(self, optional: Mapping[MappingSpec, object | None]) -> MetadataLayout:
        values = dict(self.optional)
        values.update(optional)
        return MetadataLayout(
            required=self.required,
            optional=values,
        )


@dataclass(frozen=True)
class StageLayout:
    stage_id: DatasetStageId
    manifest: MetadataLayout
    footer: MetadataLayout

    def with_manifest(self, optional: Mapping[MappingSpec, object | None]) -> StageLayout:
        return StageLayout(
            stage_id=self.stage_id,
            manifest=self.manifest.with_optional(optional),
            footer=self.footer,
        )

    def with_footer(self, optional: Mapping[MappingSpec, object | None]) -> StageLayout:
        return StageLayout(
            stage_id=self.stage_id,
            manifest=self.manifest,
            footer=self.footer.with_optional(optional),
        )


INGEST_STAGE_METADATA = StageLayout(
    stage_id=DatasetStageId.ingested,
    manifest=MetadataLayout(
        required=(
            BaseColumns.raw_paths,
            BaseColumns.parq_segs,
            BaseColumns.row_n,
            BaseColumns.proto,
        ),
    ),
    footer=MetadataLayout(
        required=(
            BaseColumns.set_id,
            BaseColumns.stage,
            BaseColumns.git_commit,
            BaseColumns.git_status,
            BaseColumns.manifest,
        ),
    ),
)

NORMALIZE_STAGE_METADATA = StageLayout(
    stage_id=DatasetStageId.normalized,
    manifest=MetadataLayout(
        required=(
            BaseColumns.norm_segs,
            BaseColumns.row_n,
            BaseColumns.proto,
        ),
    ),
    footer=MetadataLayout(
        required=(
            BaseColumns.set_id,
            BaseColumns.stage,
            BaseColumns.git_commit,
            BaseColumns.git_status,
            BaseColumns.manifest,
        ),
    ),
)


@dataclass(frozen=True)
class ProtocolMetadata:
    manifest_extra: MetadataLayout
    footer_extra: MetadataLayout
    task_key: tuple[MappingSpec, ...] = ()


def cycle_key_protocol_metadata() -> ProtocolMetadata:
    return ProtocolMetadata(
        task_key=(
            BaseColumns.cell_id,
            BaseColumns.cidx,
        ),
        manifest_extra=MetadataLayout(required=(BaseColumns.proto,)),
        footer_extra=MetadataLayout(),
    )


CYCLING_PROTOCOL_METADATA = cycle_key_protocol_metadata()
HPPC_PROTOCOL_METADATA = cycle_key_protocol_metadata()
RPT_PROTOCOL_METADATA = cycle_key_protocol_metadata()

EIS_PROTOCOL_METADATA = ProtocolMetadata(
    task_key=(
        BaseColumns.cell_id,
        BaseColumns.cidx,
        BaseColumns.soc_pct,
    ),
    manifest_extra=MetadataLayout(required=(BaseColumns.proto,)),
    footer_extra=MetadataLayout(),
)
