from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from batgrad.contracts.domains import Domains, DomainSpec
from batgrad.contracts.metadata import MetadataLayout, MetadataLayoutSpec
from batgrad.data.processing.config import (
    PROCESSING_STAGE_SPECS,
    NormalizeStageConfig,
    ProcessingStage,
    RawStageConfig,
)
from batgrad.data.transforms.checks import CHECK_REGISTRY, CheckSpecBase, TimeCheckSpec
from batgrad.data.transforms.resampling import RESAMPLING_REGISTRY, ResamplingSpecBase

if TYPE_CHECKING:
    from batgrad.contracts.columns import ColumnSpec
    from batgrad.contracts.values import ValueSpec
    from batgrad.data.datasets.registry import DatasetIds
    from batgrad.data.locations import DatasetLocation
    from batgrad.data.transforms.resampling import ResamplingSpecBase
    from batgrad.storage.store import DataStore

DOMAIN_REQUIRED_CHECKS: dict[ValueSpec, frozenset[type[CheckSpecBase]]] = {
    Domains.time.domain_id: frozenset({TimeCheckSpec}),
}


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str | None = None
    year: int | None = None
    author: str | None = None
    parent_dataset_id: str | None = None
    misc: dict[str, str] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: DatasetIds
    location: DatasetLocation
    cols: type
    vals: type
    metadata: type
    info: DatasetInfo | None = None
    raw: RawIngestSpec | None = None
    normalize: NormalizeSpec | None = None

    @property
    def default_stages(self) -> tuple[ProcessingStage, ...]:
        return tuple(
            stage
            for stage, stage_spec in PROCESSING_STAGE_SPECS.items()
            if getattr(self, stage_spec.dataset_spec_attr) is not None
        )


@dataclass(frozen=True, slots=True)
class RawProtocolSchema:
    protocol: ValueSpec
    domain: DomainSpec
    metadata: tuple[ColumnSpec, ...]
    columns: tuple[ColumnSpec, ...]
    dropped_columns: tuple[ColumnSpec, ...] = ()

    @property
    def output_columns(self) -> tuple[ColumnSpec, ...]:
        return tuple(dict.fromkeys((*self.columns, *self.metadata)))


@dataclass(frozen=True, slots=True)
class RawIngestSpec:
    input_source: Literal["raw"] = "raw"
    output_source: Literal["parquet"] = "parquet"
    schema_version: str = "raw-parquet-v1"
    processing_stage: str = "raw"
    manifest_layout: MetadataLayoutSpec = field(
        default_factory=lambda: MetadataLayoutSpec(required=MetadataLayout().parquet_manifest),
    )
    footer_layout: MetadataLayoutSpec = field(
        default_factory=lambda: MetadataLayoutSpec(required=MetadataLayout().parquet_footer),
    )
    footer_metadata: dict[ColumnSpec, object] = field(default_factory=dict)
    file_suffixes: tuple[str, ...] = field(default_factory=tuple)
    excluded_file_patterns: tuple[str, ...] = field(default_factory=tuple)
    compression: str = "zstd"
    use_content_defined_chunking: bool = True
    row_group_size: int = 256_000
    max_shard_size_bytes: int = 500 * 1024 * 1024
    protocol_schemas: tuple[RawProtocolSchema, ...] = field(default_factory=tuple)

    def protocol_schema(self, protocol: object) -> RawProtocolSchema:
        for schema in self.protocol_schemas:
            if str(schema.protocol) == str(protocol):
                return schema
        raise ValueError(f"Protocol {protocol!r} is not declared in raw protocol schemas")


@dataclass(frozen=True, slots=True)
class ProtocolNormalizeSpec:
    domain: DomainSpec
    columns: tuple[ColumnSpec, ...]
    checks: tuple[CheckSpecBase, ...] = field(default_factory=tuple)
    resampling: ResamplingSpecBase | None = None


@dataclass(frozen=True, slots=True)
class NormalizeSpec:
    spec_id: str
    input_source: Literal["parquet"] = "parquet"
    output_source: Literal["normalized"] = "normalized"
    time_convention: str = "start_of_interval"
    protocol_specs: dict[str, ProtocolNormalizeSpec] = field(default_factory=dict)
    row_group_size: int = 256_000
    max_shard_size_bytes: int = 500 * 1024 * 1024

    def __post_init__(self) -> None:
        self._validate_registered_required_checks()
        self._validate_required_domain_checks()
        self._validate_registered_resampling()

    def _validate_registered_required_checks(self) -> None:
        required_checks: set[type[CheckSpecBase]] = set()
        for checks in DOMAIN_REQUIRED_CHECKS.values():
            required_checks.update(checks)

        missing_handlers = required_checks - set(CHECK_REGISTRY)
        if missing_handlers:
            names = sorted(check.name for check in missing_handlers)
            raise ValueError(f"Required checks are missing handlers: {names}")

    def _validate_required_domain_checks(self) -> None:
        for protocol, protocol_spec in self.protocol_specs.items():
            required_checks = DOMAIN_REQUIRED_CHECKS.get(
                protocol_spec.domain.domain_id,
                frozenset(),
            )
            selected_checks = {type(check) for check in protocol_spec.checks}
            missing_checks = required_checks - selected_checks

            if missing_checks:
                names = sorted(check.name for check in missing_checks)
                raise ValueError(
                    f"Protocol {protocol!r} with domain "
                    f"{protocol_spec.domain.domain_id!r} is missing required checks: {names}",
                )

    def _validate_registered_resampling(self) -> None:
        for protocol, protocol_spec in self.protocol_specs.items():
            resampling = protocol_spec.resampling
            if resampling is None:
                continue

            if type(resampling) not in RESAMPLING_REGISTRY:
                raise ValueError(
                    f"Protocol {protocol!r} uses unregistered resampling "
                    f"{type(resampling).__name__}",
                )


class Dataset(Protocol):
    spec: DatasetSpec

    def raw_to_parquet(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: RawStageConfig,
    ) -> None: ...

    def normalize(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: NormalizeStageConfig,
    ) -> None: ...
