from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.columns import BaseColumns, ColumnSpec, MetadataColumns
from batgrad.contracts.metadata import (
    MetadataLayout,
    schema_from_layout,
    validate_layout_values,
    validate_no_extra_layout_values,
)

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import DatasetSpec, RawIngestSpec


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    dirty: bool


@dataclass(slots=True)
class ManifestRow:
    file_path: str
    row_start: int
    row_count: int
    ingest_order: int
    source_paths: tuple[str, ...]
    metadata: dict[ColumnSpec, object]


def build_raw_footer_metadata(
    spec: DatasetSpec,
    raw_spec: RawIngestSpec,
    git_state: GitState,
    manifest_path: str,
    file_path: str,
    protocols: set[str],
    domains: set[str],
    row_count: int,
) -> dict[str, str]:
    values: dict[ColumnSpec, object] = {
        BaseColumns.dataset_id: spec.dataset_id,
        MetadataColumns.schema_version: raw_spec.schema_version,
        MetadataColumns.processing_stage: raw_spec.processing_stage,
        MetadataColumns.git_commit: git_state.commit,
        MetadataColumns.git_dirty: git_state.dirty,
        MetadataColumns.manifest_path: manifest_path,
        MetadataColumns.protocols: sorted(protocols),
        MetadataColumns.domains: sorted(domains),
        BaseColumns.row_count: row_count,
    }
    values.update(raw_spec.footer_metadata)
    validate_layout_values(
        values,
        raw_spec.footer_layout,
        context=f"Raw parquet footer for {file_path}",
    )
    validate_no_extra_layout_values(
        values,
        raw_spec.footer_layout,
        context=f"Raw parquet footer for {file_path}",
    )
    return encode_footer_values(values)


def build_raw_manifest(raw_spec: RawIngestSpec, rows: list[ManifestRow]) -> pl.DataFrame:
    return pl.DataFrame(
        [raw_manifest_row_values(raw_spec, row) for row in rows],
        schema=schema_from_layout(raw_spec.manifest_layout),
        orient="row",
    )


def raw_manifest_row_values(
    raw_spec: RawIngestSpec,
    row: ManifestRow,
) -> dict[ColumnSpec, object]:
    values: dict[ColumnSpec, object] = {
        BaseColumns.file_path: row.file_path,
        MetadataColumns.row_start: row.row_start,
        BaseColumns.row_count: row.row_count,
        MetadataColumns.ingest_order: row.ingest_order,
        MetadataColumns.source_file_paths: list(row.source_paths),
        MetadataColumns.protocol: row.metadata.get(MetadataColumns.protocol),
        MetadataColumns.domain_id: row.metadata.get(MetadataColumns.domain_id),
        BaseColumns.cell_id: row.metadata.get(BaseColumns.cell_id),
        BaseColumns.cycle_index: row.metadata.get(BaseColumns.cycle_index),
        MetadataColumns.soc_pct: row.metadata.get(MetadataColumns.soc_pct),
    }
    values.update(row.metadata)
    for column in raw_spec.manifest_layout.columns:
        values.setdefault(column, None)
    validate_layout_values(
        values,
        raw_spec.manifest_layout,
        context=f"Raw manifest row for {row.file_path}",
    )
    validate_no_extra_layout_values(
        values,
        raw_spec.manifest_layout,
        context=f"Raw manifest row for {row.file_path}",
    )
    return values


def manifest_schema() -> dict[str, pl.DataType]:
    return schema_from_layout(MetadataLayout().parquet_manifest)


def encode_footer_values(values: dict[ColumnSpec, object]) -> dict[str, str]:
    encoded: dict[str, str] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            encoded[key] = "true" if value else "false"
        elif isinstance(value, list | tuple):
            encoded[key] = json.dumps(list(value))
        else:
            encoded[key] = str(value)
    return encoded


def resolve_git_state() -> GitState:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required to write parquet footer metadata")

    commit = subprocess.run(  # noqa: S603 - git path is resolved with shutil.which.
        [git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # noqa: S603 - git path is resolved with shutil.which.
            [git, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    return GitState(commit=commit, dirty=dirty)
