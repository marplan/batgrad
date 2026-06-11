from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import polars as pl

from batgrad.contracts.columns import BaseColumns, ColumnSpec, MetadataColumns
from batgrad.contracts.metadata import (
    MetadataLayout,
    MetadataLayoutSpec,
    schema_from_layout,
    validate_layout_values,
    validate_no_extra_layout_values,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from batgrad.data.datasets.specs import DatasetSpec
    from batgrad.data.processing.config import ProcessingStageSpec


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
    raw_file_paths: tuple[str, ...]
    metadata: dict[ColumnSpec, object]


def build_stage_footer_metadata(
    spec: DatasetSpec,
    stage_spec: ProcessingStageSpec,
    footer_layout: MetadataLayoutSpec,
    footer_metadata: Mapping[ColumnSpec, object],
    git_state: GitState,
    manifest_path: str,
    file_path: str,
    protocols: set[str],
    domains: set[str],
    row_count: int,
) -> dict[str, str]:
    values: dict[ColumnSpec, object] = {
        BaseColumns.dataset_id: spec.dataset_id,
        MetadataColumns.processing_stage: stage_spec.processing_stage,
        MetadataColumns.git_commit: git_state.commit,
        MetadataColumns.git_dirty: git_state.dirty,
        MetadataColumns.manifest_path: manifest_path,
        MetadataColumns.protocols: sorted(protocols),
        MetadataColumns.domains: sorted(domains),
        BaseColumns.row_count: row_count,
    }
    values.update(footer_metadata)
    validate_layout_values(
        values,
        footer_layout,
        context=f"{stage_spec.processing_stage} parquet footer for {file_path}",
    )
    validate_no_extra_layout_values(
        values,
        footer_layout,
        context=f"{stage_spec.processing_stage} parquet footer for {file_path}",
    )
    return encode_footer_values(values)


def build_stage_manifest(
    stage_spec: ProcessingStageSpec,
    manifest_layout: MetadataLayoutSpec,
    rows: list[ManifestRow],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema_from_layout(manifest_layout))

    append_rows = [stage_manifest_row_values(row) for row in rows]
    return compact_stage_manifest(stage_spec, manifest_layout, pl.DataFrame(append_rows))


def compact_stage_manifest(
    stage_spec: ProcessingStageSpec,
    manifest_layout: MetadataLayoutSpec,
    append_manifest: pl.DataFrame,
) -> pl.DataFrame:
    segment_col = _stage_segment_column(stage_spec)
    is_normalized = segment_col == MetadataColumns.normalized_segments
    final_schema = schema_from_layout(manifest_layout)
    aggregate_columns = {
        BaseColumns.row_count,
        segment_col,
    }
    if is_normalized:
        aggregate_columns.add(MetadataColumns.raw_file_paths)
        aggregate_columns.add(MetadataColumns.parquet_segments)

    missing_exprs = [
        pl.lit(None).alias(column)
        for column in manifest_layout.columns
        if column not in append_manifest.columns and column != segment_col
    ]
    if missing_exprs:
        append_manifest = append_manifest.with_columns(missing_exprs)

    append_manifest = append_manifest.with_columns(
        pl.struct(
            BaseColumns.file_path,
            MetadataColumns.row_start,
            BaseColumns.row_count,
        ).alias("_stage_segment"),
    )
    group_columns = [
        column
        for column in manifest_layout.columns
        if column not in aggregate_columns and column in append_manifest.columns
    ]
    sort_columns = [MetadataColumns.ingest_order, *group_columns]
    aggregate_exprs: list[pl.Expr] = [
        pl.col(BaseColumns.row_count).sum().alias(BaseColumns.row_count),
        pl.col("_stage_segment").alias(segment_col),
    ]
    if is_normalized:
        aggregate_exprs.append(
            pl.col(MetadataColumns.raw_file_paths)
            .list.explode()
            .unique(maintain_order=True)
            .alias(MetadataColumns.raw_file_paths),
        )
        aggregate_exprs.append(
            pl.col(MetadataColumns.parquet_segments)
            .drop_nulls()
            .first()
            .alias(
                MetadataColumns.parquet_segments,
            ),
        )

    compacted = (
        append_manifest.sort(sort_columns)
        .group_by(group_columns, maintain_order=True)
        .agg(*aggregate_exprs)
    )
    missing_exprs = [
        pl.lit(None).alias(column)
        for column in manifest_layout.columns
        if column not in compacted.columns
    ]
    if missing_exprs:
        compacted = compacted.with_columns(missing_exprs)
    return compacted.select(
        pl.col(column).cast(dtype, strict=False).alias(column)
        for column, dtype in final_schema.items()
    )


def _stage_segment_column(stage_spec: ProcessingStageSpec) -> ColumnSpec:
    if stage_spec.output_source == "parquet":
        return MetadataColumns.parquet_segments
    if stage_spec.output_source == "normalized":
        return MetadataColumns.normalized_segments
    raise ValueError(f"No segment manifest column for output source {stage_spec.output_source!r}")


def extend_metadata_layout(
    layout: MetadataLayoutSpec,
    extra_columns: tuple[ColumnSpec, ...],
) -> MetadataLayoutSpec:
    optional = list(layout.optional)
    existing = set(layout.columns)
    for column in extra_columns:
        if column in existing:
            continue
        optional.append(column)
        existing.add(column)
    return MetadataLayoutSpec(required=layout.required, optional=tuple(optional))


def extend_layout_with_group_columns(
    layout: MetadataLayoutSpec,
    protocol_specs: Mapping[str, object],
) -> MetadataLayoutSpec:
    columns = tuple(
        dict.fromkeys(
            column
            for protocol_spec in protocol_specs.values()
            for column in getattr(protocol_spec, "group_by", ())
        ),
    )
    return extend_metadata_layout(layout, columns)


def merge_manifest_raw_file_paths(rows: list[dict[str, object]]) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        row_paths = row[MetadataColumns.raw_file_paths]
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
    column: ColumnSpec,
) -> tuple[dict[str, object], ...]:
    segments: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        row_segments = row[column]
        if not isinstance(row_segments, list):
            continue
        for segment in row_segments:
            if not isinstance(segment, dict):
                continue
            segment_values = cast("Mapping[str, object]", segment)
            file_path = str(segment_values[BaseColumns.file_path])
            row_start = int(cast("str | int", segment_values[MetadataColumns.row_start]))
            row_count = int(cast("str | int", segment_values[BaseColumns.row_count]))
            key = (file_path, row_start, row_count)
            if key in seen:
                continue
            seen.add(key)
            segments.append(
                {
                    BaseColumns.file_path: file_path,
                    MetadataColumns.row_start: row_start,
                    BaseColumns.row_count: row_count,
                },
            )
    return tuple(segments)


def manifest_segment_file_paths(segments: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(segment[BaseColumns.file_path]) for segment in segments))


def group_task_id(protocol: str, group_values: Mapping[ColumnSpec, object]) -> str:
    parts = [protocol]
    parts.extend(f"{column}={value}" for column, value in group_values.items())
    return "|".join(parts)


def stage_manifest_row_values(row: ManifestRow) -> dict[ColumnSpec, object]:
    values: dict[ColumnSpec, object] = {
        BaseColumns.file_path: row.file_path,
        MetadataColumns.row_start: row.row_start,
        BaseColumns.row_count: row.row_count,
        MetadataColumns.ingest_order: row.ingest_order,
        MetadataColumns.raw_file_paths: list(row.raw_file_paths),
        MetadataColumns.protocol: row.metadata.get(MetadataColumns.protocol),
        MetadataColumns.domain_id: row.metadata.get(MetadataColumns.domain_id),
    }
    values.update(row.metadata)
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
        return GitState(commit="missing", dirty=False)

    try:
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
    except (OSError, subprocess.CalledProcessError):
        return GitState(commit="missing", dirty=False)
    return GitState(commit=commit, dirty=dirty)
