from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batgrad.contracts.mapping import BaseColumns
from batgrad.contracts.metadata import MetadataLayout

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batgrad.contracts.mapping import DatasetStageId, MappingSpec
    from batgrad.contracts.metadata import ProtocolMetadata, StageLayout
    from batgrad.data.datasets.config import DatasetSpec


def validate_metadata_columns(
    layout: MetadataLayout,
    columns: tuple[str, ...] | list[str],
    *,
    context: str,
) -> None:
    available = set(columns)
    missing = [column for column in layout.columns if str(column) not in available]
    if missing:
        raise ValueError(f"{context} is missing metadata columns: {missing}")


def stage_manifest_metadata(spec: DatasetSpec, stage_id: DatasetStageId) -> MetadataLayout:
    stage_spec = spec.processing_stages.get(stage_id)
    manifest_metadata = getattr(stage_spec, "manifest_metadata", None)
    if not isinstance(manifest_metadata, MetadataLayout):
        raise TypeError(f"Dataset {spec.dataset_id!r} has no manifest metadata for {stage_id}")
    return manifest_metadata


def stage_layout_with_protocol_metadata(
    layout: StageLayout,
    protocol_metadata: Iterable[ProtocolMetadata],
    *,
    manifest_extra: Iterable[MappingSpec] = (),
    footer_extra: Iterable[MappingSpec] = (),
) -> StageLayout:
    protocol_metadata = tuple(protocol_metadata)
    return layout.with_manifest(
        _new_optional_columns(
            layout.manifest,
            (
                *manifest_extra,
                *(column for item in protocol_metadata for column in item.task_key),
                *(column for item in protocol_metadata for column in item.manifest_extra.columns),
            ),
        ),
    ).with_footer(
        _new_optional_columns(
            layout.footer,
            (
                *footer_extra,
                *(column for item in protocol_metadata for column in item.footer_extra.columns),
            ),
        ),
    )


def _new_optional_columns(
    layout: MetadataLayout,
    columns: Iterable[MappingSpec],
) -> dict[MappingSpec, None]:
    existing = set(layout.columns)
    optional = {}
    for column in columns:
        if column in existing:
            continue
        optional[column] = None
        existing.add(column)
    return optional


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: str


def encode_footer_values(values: dict[MappingSpec, object | None]) -> dict[str, str]:
    encoded = {}
    for key, value in values.items():
        if value is None:
            encoded[str(key)] = "null"
        elif isinstance(value, bool):
            encoded[str(key)] = "true" if value else "false"
        elif isinstance(value, list | tuple | dict):
            encoded[str(key)] = json.dumps(value)
        else:
            encoded[str(key)] = str(value)
    return encoded


def git_state() -> GitState:
    git = shutil.which("git")
    if git is None:
        return GitState(commit="na", dirty="na")
    try:
        commit = subprocess.run(  # noqa: S603 - git executable is resolved with shutil.which.
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603 - git executable is resolved with shutil.which.
                [git, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        return GitState(commit="na", dirty="na")
    return GitState(commit=commit, dirty="dirty" if dirty else "clean")


def hashable_manifest_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(hashable_manifest_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, hashable_manifest_value(item)) for key, item in value.items()))
    return value


def as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}")


def merge_manifest_raw_paths(rows: list[dict[str, object]]) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        row_paths = row.get(str(BaseColumns.raw_paths))
        if not isinstance(row_paths, list):
            continue
        for source_path in row_paths:
            value = str(source_path)
            if value in seen:
                continue
            seen.add(value)
            paths.append(value)
    return tuple(paths)


def merge_manifest_segments(
    rows: list[dict[str, object]],
    column: MappingSpec,
) -> tuple[dict[str, object], ...]:
    segments: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        row_segments = row.get(str(column))
        if not isinstance(row_segments, list):
            continue
        for segment in row_segments:
            if not isinstance(segment, Mapping):
                continue
            values = dict(segment)
            file_path = str(values[str(BaseColumns.path)])
            row_start = as_int(values[str(BaseColumns.row0)])
            row_count = as_int(values[str(BaseColumns.row_n)])
            key = (file_path, row_start, row_count)
            if key in seen:
                continue
            seen.add(key)
            segments.append(
                {
                    str(BaseColumns.path): file_path,
                    str(BaseColumns.row0): row_start,
                    str(BaseColumns.row_n): row_count,
                },
            )
    return tuple(segments)


def safe_name(value: object, max_len: int = 96) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value))[:max_len]


def group_task_id(protocol: str, group_values: Mapping[MappingSpec, object]) -> str:
    suffix = "_".join(
        f"{safe_name(column)}-{safe_name(value)}" for column, value in group_values.items()
    )
    return f"{safe_name(protocol)}_{suffix}" if suffix else safe_name(protocol)
