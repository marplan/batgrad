from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol

import polars as pl

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.processing.metadata import (
    as_int,
    encode_footer_values,
    git_state,
    hashable_manifest_value,
)
from batgrad.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from batgrad.contracts.mapping import DatasetStageId, MappingSpec
    from batgrad.contracts.metadata import MetadataLayout
    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.storage.store import DataProcessingStore, TableWriter


class ShardWriteConfig(Protocol):
    compression: str
    use_content_defined_chunking: bool
    row_group_size: int
    max_shard_size_bytes: int


@dataclass
class ShardState:
    key: str
    path: str
    writer: TableWriter
    row_count: int = 0
    footer_values: dict[MappingSpec, object | None] = field(default_factory=dict)
    protocols: set[str] = field(default_factory=set)


class ShardWriter:
    def __init__(
        self,
        output_store: DataProcessingStore,
        dataset_spec: DatasetSpec,
        stage_id: DatasetStageId,
        output_root: str,
        manifest_path: str,
        manifest_metadata: MetadataLayout,
        footer_metadata: MetadataLayout,
        shard_key_col: MappingSpec,
        segment_col: MappingSpec,
        source_paths_col: MappingSpec,
        config: ShardWriteConfig,
    ) -> None:
        self.output_store = output_store
        self.dataset_spec = dataset_spec
        self.stage_id = stage_id
        self.output_root = output_root
        self.manifest_path = manifest_path
        self.manifest_metadata = manifest_metadata
        self.footer_metadata = footer_metadata
        self.shard_key_col = shard_key_col
        self.segment_col = segment_col
        self.source_paths_col = source_paths_col
        self.config = config
        self.git_state = git_state()
        if self.git_state.dirty == "dirty":
            logger.warning("%s footers will record git dirty status", stage_id)
        self._states: dict[str, ShardState] = {}
        self._next_part_idx: dict[str, int] = {}
        self._manifest_rows: list[dict[MappingSpec, object]] = []

    def append(
        self,
        data: pl.DataFrame,
        metadata: dict[MappingSpec, object],
        source_paths: tuple[str, ...],
    ) -> None:
        if data.height == 0:
            return
        shard_key = str(metadata[self.shard_key_col])
        state = self._state_for_key(shard_key, data)
        for column in self.footer_metadata.columns:
            if column in metadata:
                state.footer_values[column] = metadata[column]
        protocol = metadata.get(BaseColumns.proto)
        if protocol is not None:
            state.protocols.add(str(protocol))
        row_start = state.row_count
        state.writer.write_table(data, row_group_size=self.config.row_group_size)
        state.row_count += data.height
        self._manifest_rows.append(
            manifest_row(
                self.segment_col,
                self.source_paths_col,
                state.path,
                row_start,
                data.height,
                source_paths,
                metadata,
            ),
        )
        if self._should_roll(state):
            self._close_state(shard_key)

    def close(self, *, manifest: Literal["write", "error", "skip"] = "write") -> None:
        for shard_key in tuple(self._states):
            self._close_state(shard_key)
        if manifest == "write":
            self.output_store.write_table(
                build_manifest(self.manifest_metadata, self.segment_col, self._manifest_rows),
                self.manifest_path,
                metadata=encode_footer_values(self._footer_values({})),
                row_group_size=self.config.row_group_size,
            )
        elif manifest == "error":
            self.output_store.write_table(
                build_manifest(self.manifest_metadata, self.segment_col, self._manifest_rows),
                _error_manifest_path(self.manifest_path),
                metadata=encode_footer_values(self._footer_values({})),
                row_group_size=self.config.row_group_size,
            )
        elif manifest != "skip":
            raise ValueError(f"manifest must be 'write', 'error', or 'skip', got {manifest!r}")

    def _state_for_key(self, shard_key: str, data: pl.DataFrame) -> ShardState:
        state = self._states.get(shard_key)
        if state is not None:
            return state
        part_idx = self._next_part_idx.get(shard_key, 0)
        self._next_part_idx[shard_key] = part_idx + 1
        file_name = f"{shard_key.casefold()}_part-{part_idx:06d}.parquet"
        path = f"{self.output_root}/{shard_key.casefold()}/{file_name}"
        writer = self.output_store.open_table_writer(
            path,
            data.to_arrow().schema,
            self.config.compression,
            use_content_defined_chunking=self.config.use_content_defined_chunking,
        )
        state = ShardState(shard_key, path, writer)
        self._states[shard_key] = state
        return state

    def _should_roll(self, state: ShardState) -> bool:
        if self.config.max_shard_size_bytes <= 0:
            return False
        size = self.output_store.table_size_bytes(state.path)
        return size is not None and size >= self.config.max_shard_size_bytes

    def _close_state(self, shard_key: str) -> None:
        state = self._states.pop(shard_key)
        state.writer.close(encode_footer_values(self._footer_values(state.footer_values)))

    def _footer_values(
        self,
        task_values: dict[MappingSpec, object | None],
    ) -> dict[MappingSpec, object | None]:
        return resolve_footer_values(
            self.footer_metadata,
            task_values,
            {
                BaseColumns.set_id: self.dataset_spec.dataset_id,
                BaseColumns.stage: str(self.stage_id),
                BaseColumns.git_commit: self.git_state.commit,
                BaseColumns.git_status: self.git_state.dirty,
                BaseColumns.manifest: self.manifest_path,
            },
        )


def resolve_footer_values(
    footer_metadata: MetadataLayout,
    task_values: dict[MappingSpec, object | None],
    runtime_values: dict[MappingSpec, object | None],
) -> dict[MappingSpec, object | None]:
    layout_values = footer_metadata.values
    values: dict[MappingSpec, object | None] = {}
    missing_required: list[MappingSpec] = []

    for column in footer_metadata.columns:
        if column in task_values and task_values[column] is not None:
            values[column] = task_values[column]
        elif column in runtime_values and runtime_values[column] is not None:
            values[column] = runtime_values[column]
        elif column in layout_values:
            values[column] = layout_values[column]
        elif column in footer_metadata.required:
            missing_required.append(column)
        else:
            values[column] = None

    if missing_required:
        raise ValueError(f"Footer metadata is missing required values: {missing_required}")
    return values


def manifest_row(
    segment_col: MappingSpec,
    source_paths_col: MappingSpec,
    file_path: str,
    row_start: int,
    row_count: int,
    source_paths: tuple[str, ...],
    metadata: dict[MappingSpec, object],
) -> dict[MappingSpec, object]:
    values: dict[MappingSpec, object] = {
        source_paths_col: list(source_paths),
        segment_col: [
            {
                str(BaseColumns.path): file_path,
                str(BaseColumns.row0): row_start,
                str(BaseColumns.row_n): row_count,
            },
        ],
        BaseColumns.row_n: row_count,
    }
    values.update(metadata)
    return values


def build_manifest(
    manifest_metadata: MetadataLayout,
    segment_col: MappingSpec,
    rows: list[dict[MappingSpec, object]],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={str(column): column.dtype for column in manifest_metadata.columns},
        )
    group_columns = [
        column
        for column in manifest_metadata.columns
        if column not in {segment_col, BaseColumns.row_n}
    ]
    grouped: dict[tuple[object, ...], dict[MappingSpec, object]] = {}
    for row in rows:
        key = tuple(hashable_manifest_value(row.get(column)) for column in group_columns)
        target = grouped.setdefault(key, {column: row.get(column) for column in group_columns})
        target[BaseColumns.row_n] = as_int(target.get(BaseColumns.row_n, 0)) + as_int(
            row[BaseColumns.row_n],
        )
        if BaseColumns.raw_paths in manifest_metadata.columns:
            raw_paths = _ensure_list(target, BaseColumns.raw_paths)
            for path in _list_value(row.get(BaseColumns.raw_paths)):
                if path not in raw_paths:
                    raw_paths.append(path)
        _ensure_list(target, segment_col).extend(_list_value(row[segment_col]))
    frame = pl.DataFrame(
        [{str(key): value for key, value in row.items()} for row in grouped.values()],
        infer_schema_length=None,
    )
    return frame.select(
        pl.col(str(column)).cast(column.dtype).alias(str(column))
        for column in manifest_metadata.columns
    )


def _error_manifest_path(manifest_path: str) -> str:
    path = PurePosixPath(manifest_path)
    return path.with_name(f"err_{path.name}").as_posix()


def _ensure_list(row: dict[MappingSpec, object], column: MappingSpec) -> list[object]:
    value = row.setdefault(column, [])
    if not isinstance(value, list):
        raise TypeError(f"Manifest column {column!r} must contain a list")
    resolved = list(value)
    row[column] = resolved
    return resolved


def _list_value(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected manifest list value, got {type(value).__name__}")
    return list(value)
