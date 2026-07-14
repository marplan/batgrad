from __future__ import annotations

import polars as pl

from batgrad.contracts.row_ids import ML_INDEX_ROW_ID_COLUMN
from batgrad.ml.data.index import available_manifest_paths
from batgrad.notebook_helpers import selected_row_ids_from_table


def discover_normalized_manifest_status(store: object) -> tuple[tuple[str, ...], str | None]:
    if store is None:
        return (), None
    try:
        manifests = available_manifest_paths(store)
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        return (), f"Could not search for normalized manifests: {exc}"
    if not manifests:
        return (
            (),
            "No normalized manifests found. Expected files matching "
            "type=*/dataset=*/source=normalized/manifest.parquet.",
        )
    return manifests, None


def selected_index_rows(index_frame: pl.DataFrame, index_table: object | None) -> pl.DataFrame:
    if index_table is None or not index_frame.height:
        return pl.DataFrame()
    selected_row_ids = selected_row_ids_from_table(index_table.value)
    if not selected_row_ids:
        return pl.DataFrame()
    return index_frame.filter(pl.col(ML_INDEX_ROW_ID_COLUMN).is_in(selected_row_ids))
