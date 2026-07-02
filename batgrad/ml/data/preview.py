from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.mapping import BaseColumns
from batgrad.contracts.paths import dataset_id_from_manifest_path

if TYPE_CHECKING:
    from batgrad.storage.store import DatasetStoreReader


def load_manifest_preview(
    store: DatasetStoreReader | None,
    manifest_commits: dict[str, str],
) -> pl.DataFrame:
    if store is None or not manifest_commits:
        return pl.DataFrame()
    frames = []
    for manifest_path in manifest_commits:
        try:
            frame = (
                store.scan_table(manifest_path)
                .collect()
                .with_columns(pl.lit(manifest_path).alias(BaseColumns.manifest))
            )
        except FileNotFoundError:
            continue
        if BaseColumns.set_id not in frame.columns:
            frame = frame.with_columns(
                pl.lit(dataset_id_from_manifest_path(manifest_path)).alias(BaseColumns.set_id)
            )
        frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def available_protocols(raw_manifest: pl.DataFrame) -> tuple[str, ...]:
    if raw_manifest.height and BaseColumns.proto in raw_manifest.columns:
        return tuple(sorted(str(value) for value in raw_manifest[BaseColumns.proto].unique()))
    return ()


def validation_group_options(raw_manifest: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in raw_manifest.columns
        if column not in {BaseColumns.norm_segs, BaseColumns.raw_paths}
    )


def default_validation_group_by(group_options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in (BaseColumns.set_id, BaseColumns.cell_id, BaseColumns.cidx)
        if column in group_options
    )


def shard_columns_for_protocols(
    schema_by_protocol: dict[object, tuple[str, ...]],
) -> tuple[str, ...]:
    if not schema_by_protocol:
        return ()
    column_sets = [set(columns) for columns in schema_by_protocol.values()]
    return tuple(sorted(set.intersection(*column_sets))) if column_sets else ()


def default_input_columns(shard_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in (BaseColumns.time, BaseColumns.curr, BaseColumns.volt)
        if column in shard_columns
    )


def default_target_columns(shard_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column) for column in (BaseColumns.curr, BaseColumns.volt) if column in shard_columns
    )


def active_protocol_options(schema_by_protocol: dict[object, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(str(protocol) for protocol in schema_by_protocol) or ("cycling",)
