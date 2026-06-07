from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from batgrad import _loggers
from batgrad.contracts.columns import ColumnSpec, MetadataColumns
from batgrad.data.processing.metadata import (
    ManifestRow,
    build_raw_footer_metadata,
    build_raw_manifest,
    resolve_git_state,
)

if TYPE_CHECKING:
    import polars as pl

    from batgrad.data.datasets.specs import DatasetSpec, RawIngestSpec
    from batgrad.storage.store import DataStore, TableWriter

logger = _loggers.get_logger(__name__)


@dataclass(slots=True)
class ProtocolShardState:
    protocol: str
    path: str
    writer: TableWriter
    row_count: int = 0
    protocols: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)


class RawProtocolShardWriter:
    def __init__(self, output_store: DataStore, spec: DatasetSpec, raw_spec: RawIngestSpec) -> None:
        self.output_store = output_store
        self.spec = spec
        self.raw_spec = raw_spec
        self.git_state = resolve_git_state()
        if self.git_state.dirty:
            logger.warning("raw parquet footers will record git_dirty=true")
        self.manifest_path = spec.location.manifest(raw_spec.output_source)
        self._states: dict[str, ProtocolShardState] = {}
        self._next_part_idx: dict[str, int] = {}
        self._manifest_rows: list[ManifestRow] = []
        self._next_ingest_order = 0

    def append(
        self,
        data: pl.DataFrame,
        metadata: dict[ColumnSpec, object],
        source_paths: tuple[str, ...],
    ) -> None:
        if data.height == 0:
            return

        protocol_value = metadata.get(MetadataColumns.protocol)
        if protocol_value is None:
            raise ValueError("Raw batch metadata is missing protocol")
        protocol = str(protocol_value)

        state = self._state_for_protocol(protocol, data)
        row_start = state.row_count
        state.writer.write_table(data, row_group_size=self.raw_spec.row_group_size)
        state.row_count += data.height
        state.protocols.add(protocol)
        domain_value = metadata.get(MetadataColumns.domain_id)
        if domain_value is not None:
            state.domains.add(str(domain_value))

        self._manifest_rows.append(
            ManifestRow(
                file_path=state.path,
                row_start=row_start,
                row_count=data.height,
                ingest_order=self._next_ingest_order,
                source_paths=source_paths,
                metadata=metadata,
            ),
        )
        self._next_ingest_order += 1

        if self._should_roll(state):
            self._close_state(protocol)

    def close(self, manifest: Literal["write", "skip"] = "write") -> None:
        for protocol in tuple(self._states):
            self._close_state(protocol)
        if manifest == "write":
            self._write_manifest()

    def _state_for_protocol(self, protocol: str, data: pl.DataFrame) -> ProtocolShardState:
        state = self._states.get(protocol)
        if state is not None:
            return state

        part_idx = self._next_part_idx.get(protocol, 0)
        self._next_part_idx[protocol] = part_idx + 1

        protocol_dir = protocol.casefold()
        file_name = f"{protocol.casefold()}_part-{part_idx:06d}.parquet"
        source_root = self.spec.location.source_root(self.raw_spec.output_source)
        location = f"{source_root}/{protocol_dir}/{file_name}"
        writer = self.output_store.open_table_writer(
            location,
            data.to_arrow().schema,
            compression=self.raw_spec.compression,
            use_content_defined_chunking=self.raw_spec.use_content_defined_chunking,
        )
        state = ProtocolShardState(protocol=protocol, path=location, writer=writer)
        self._states[protocol] = state
        return state

    def _should_roll(self, state: ProtocolShardState) -> bool:
        max_size = self.raw_spec.max_shard_size_bytes
        if max_size <= 0:
            return False

        size_bytes = self.output_store.table_size_bytes(state.path)
        if size_bytes is None:
            return False

        return size_bytes >= max_size

    def _close_state(self, protocol: str) -> None:
        state = self._states.pop(protocol)
        footer_metadata = build_raw_footer_metadata(
            spec=self.spec,
            raw_spec=self.raw_spec,
            git_state=self.git_state,
            manifest_path=self.manifest_path,
            file_path=state.path,
            protocols=state.protocols,
            domains=state.domains,
            row_count=state.row_count,
        )
        state.writer.close(footer_metadata)

    def _write_manifest(self) -> None:
        manifest = build_raw_manifest(self.raw_spec, self._manifest_rows)
        self.output_store.write_table(
            manifest,
            self.manifest_path,
            row_group_size=self.raw_spec.row_group_size,
        )
