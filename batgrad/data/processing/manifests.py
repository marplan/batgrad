from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts import row_ids
from batgrad.storage.segments import collect_frame

MANIFEST_ROW_ID_COLUMN = row_ids.MANIFEST_ROW_ID_COLUMN
MANIFEST_ROW_IDS_COLUMN = row_ids.MANIFEST_ROW_IDS_COLUMN

if TYPE_CHECKING:
    from batgrad.contracts.mapping import DatasetStageId
    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.storage.store import DataProcessingStore


def available_manifest_stages(
    dataset_spec: DatasetSpec,
    store: DataProcessingStore,
) -> tuple[DatasetStageId, ...]:
    stages = []
    for stage_id in dataset_spec.processing_stages:
        try:
            store.scan_table(dataset_spec.manifest(stage_id)).collect()
        except FileNotFoundError:
            continue
        stages.append(stage_id)
    return tuple(stages)


def load_stage_manifest(
    dataset_spec: DatasetSpec,
    store: DataProcessingStore,
    stage_id: DatasetStageId,
) -> pl.DataFrame:
    return collect_frame(store.scan_table(dataset_spec.manifest(stage_id)))


def sort_manifest(frame: pl.DataFrame) -> pl.DataFrame:
    sort_columns = [
        column for column in ("cycle index", "cell id", "protocol") if column in frame.columns
    ]
    return frame.sort(*sort_columns) if sort_columns else frame


def with_manifest_row_id(
    frame: pl.DataFrame,
    column: str = MANIFEST_ROW_ID_COLUMN,
) -> pl.DataFrame:
    return frame.with_row_index(column) if frame.height else frame


def selected_manifest_rows(
    manifest: pl.DataFrame,
    selected_row_ids: tuple[int, ...],
    *,
    row_id_col: str = MANIFEST_ROW_ID_COLUMN,
) -> pl.DataFrame:
    if row_id_col not in manifest.columns or not selected_row_ids:
        return manifest.limit(0)
    return manifest.filter(pl.col(row_id_col).is_in(selected_row_ids))
