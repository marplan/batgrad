from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from batgrad.contracts.mapping import BaseColumns, DatasetStageId, MappingSpec

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class MetadataLayout:
    """Required columns and optional default values for stage metadata.

    Layouts are used for both manifest rows and parquet footers. `required`
    columns must be present; `optional` entries provide default metadata values
    that can be added or overridden by dataset/protocol-specific configuration.

    Attributes:
        required: Metadata columns that must be present.
        optional: Metadata columns and default values that may be added by the
            stage, dataset, or protocol configuration.

    Examples:
        >>> layout = MetadataLayout(required=(BaseColumns.proto,))
        >>> expanded = layout.with_optional({BaseColumns.cell_id: None})
        >>> expanded.columns
        ('protocol', 'cell id')
        >>> expanded.values
        {'cell id': None}
    """

    required: tuple[MappingSpec, ...] = ()
    optional: Mapping[MappingSpec, object | None] = field(default_factory=dict)

    @property
    def columns(self) -> tuple[MappingSpec, ...]:
        """Required columns followed by optional columns.

        Returns:
            All columns declared by this layout.
        """
        return (*self.required, *self.optional.keys())

    @property
    def values(self) -> dict[MappingSpec, object | None]:
        """Optional metadata defaults as a mutable copy.

        Returns:
            A new dictionary containing optional columns and their defaults.
        """
        return dict(self.optional)

    def with_optional(self, optional: Mapping[MappingSpec, object | None]) -> MetadataLayout:
        """Return a layout with additional or overridden optional defaults.

        Args:
            optional: Optional metadata columns and default values to merge into
                this layout.

        Returns:
            A layout with the same required columns and merged optional defaults.
        """
        values = dict(self.optional)
        values.update(optional)
        return MetadataLayout(
            required=self.required,
            optional=values,
        )


@dataclass(frozen=True)
class StageLayout:
    """Manifest and footer metadata contract for one processing stage.

    Manifest metadata is validated against and written to manifest rows. Footer
    metadata is encoded as parquet key/value metadata when data files are written.
    Optional values set to `None` declare metadata that is expected from task
    metadata rather than a fixed stage-level default.

    Attributes:
        stage_id: Pipeline stage this layout describes.
        manifest: Metadata required or defaulted in the stage manifest rows.
        footer: Metadata required or defaulted in generated parquet footers.

    Examples:
        >>> layout = INGEST_STAGE_METADATA.with_manifest(
        ...     {BaseColumns.cell_id: None, BaseColumns.cidx: None}
        ... ).with_footer({BaseColumns.time_conv: "start_of_interval"})
        >>> BaseColumns.cell_id in layout.manifest.optional
        True
        >>> layout.footer.values[BaseColumns.time_conv]
        'start_of_interval'
    """

    stage_id: DatasetStageId
    manifest: MetadataLayout
    footer: MetadataLayout

    def with_manifest(self, optional: Mapping[MappingSpec, object | None]) -> StageLayout:
        """Return a stage layout with extra optional manifest metadata.

        Args:
            optional: Manifest metadata columns and default values to add or
                override.

        Returns:
            A stage layout with updated manifest metadata and unchanged footer
            metadata.
        """
        return StageLayout(
            stage_id=self.stage_id,
            manifest=self.manifest.with_optional(optional),
            footer=self.footer,
        )

    def with_footer(self, optional: Mapping[MappingSpec, object | None]) -> StageLayout:
        """Return a stage layout with extra optional footer metadata.

        Args:
            optional: Footer metadata columns and default values to add or
                override.

        Returns:
            A stage layout with unchanged manifest metadata and updated footer
            metadata.
        """
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
    """Protocol-specific metadata extensions and task grouping keys.

    `task_key` identifies the columns used to group manifest rows into protocol
    processing tasks, such as cell/cycle for cycling data or cell/cycle/SOC for
    EIS data. Processing stage specs expand stage metadata with protocol task
    keys and extras before manifests and parquet files are written.

    Attributes:
        manifest_extra: Protocol-specific metadata required or defaulted in
            manifest rows.
        footer_extra: Protocol-specific metadata required or defaulted in
            generated parquet footers.
        task_key: Columns used to group manifest rows into processing tasks.

    Examples:
        >>> metadata = cycle_key_protocol_metadata()
        >>> layout = INGEST_STAGE_METADATA.with_manifest(
        ...     {column: None for column in metadata.task_key}
        ... )
        >>> layout.manifest.values[BaseColumns.cell_id] is None
        True
    """

    manifest_extra: MetadataLayout
    footer_extra: MetadataLayout
    task_key: tuple[MappingSpec, ...] = ()


def cycle_key_protocol_metadata() -> ProtocolMetadata:
    """Return metadata for protocols grouped by cell and cycle.

    Returns:
        Protocol metadata whose task key is `cell id` and `cycle index`.
    """
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
