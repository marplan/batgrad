from __future__ import annotations

from typing import Any

import marimo as mo
import polars as pl

from batgrad.contracts.mapping import BaseColumns
from batgrad.ml.data.config import ScalingRule, ValidationConfig
from batgrad.ml.data.index import MlDatasetIndex, available_manifest_paths
from batgrad.ml.data.loader import create_index
from batgrad.ml.data.materialization import resolve_index_schema_by_protocol
from batgrad.notebook_helpers import selected_row_ids_from_table, wrap_anywidget_blocks
from batgrad.viz.ml import (
    MlBatchPreview,
    MlBatchPreviewSubmission,
    build_ml_batch_preview,
    count_ml_batch_preview_groups,
    ml_batch_preview_unavailable_message,
    update_ml_batch_preview,
)

PREVIEW_SCALING: tuple[ScalingRule, ...] = (
    ScalingRule(BaseColumns.crate, -6.0, 6.0),
    ScalingRule(BaseColumns.volt, 2.3, 4.6),
    ScalingRule(BaseColumns.temp, 5.0, 65.0),
    ScalingRule(BaseColumns.amb_temp, 0.0, 50.0),
    ScalingRule(BaseColumns.a_heat, 1.0, 50.0),
    ScalingRule(BaseColumns.dt, 0.0, 10_000.0, transform="log1p"),
)


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


def build_index_frame(
    *,
    store: object,
    selected_manifest_commits: dict[str, str],
    protocols: tuple[object, ...] | None,
    protocol_mode: str,
    validation_fraction: float,
    validation_seed: int,
    group_by: tuple[str, ...],
) -> tuple[str | None, pl.DataFrame]:
    if store is None or not selected_manifest_commits:
        return None, pl.DataFrame()
    try:
        ml_index = create_index(
            store=store,
            manifest_paths=selected_manifest_commits,
            protocols=protocols,
            protocol_mode=protocol_mode,
            validation=ValidationConfig.sample(
                fraction=validation_fraction,
                seed=validation_seed,
                group_by=group_by,
            ),
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return str(exc), pl.DataFrame()
    return None, ml_index.frame


def selected_index_rows(index_frame: pl.DataFrame, index_table: object | None) -> pl.DataFrame:
    if index_table is None or not index_frame.height:
        return pl.DataFrame()
    selected_row_ids = selected_row_ids_from_table(index_table.value)
    if not selected_row_ids:
        return pl.DataFrame()
    from batgrad.contracts.row_ids import ML_INDEX_ROW_ID_COLUMN

    return index_frame.filter(pl.col(ML_INDEX_ROW_ID_COLUMN).is_in(selected_row_ids))


def selected_schema_by_protocol(
    store: object,
    selected_index_frame: pl.DataFrame,
) -> tuple[dict[object, tuple[str, ...]], str | None]:
    if store is None or not selected_index_frame.height:
        return {}, None
    try:
        return resolve_index_schema_by_protocol(
            store,
            MlDatasetIndex(selected_index_frame),
        ), None
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        return {}, str(exc)


def preview_group_count(
    selected_index_frame: pl.DataFrame,
    *,
    strategy: str,
    active_protocol: str,
    batch_size: int,
    seq_len: int,
    stateful_n_windows: int,
) -> tuple[str | None, int]:
    warning = ml_batch_preview_unavailable_message(
        strategy=strategy,
        active_protocol=active_protocol,
    )
    if warning is not None or selected_index_frame.is_empty():
        return warning, 0
    return warning, count_ml_batch_preview_groups(
        MlDatasetIndex(selected_index_frame),
        strategy=strategy,
        active_protocol=active_protocol,
        batch_size=batch_size,
        seq_len=seq_len,
        stateful_n_windows=stateful_n_windows,
    )


def make_batch_preview_submission(
    *,
    submit_id: int,
    selected_index_frame: pl.DataFrame,
    batch_warning: str | None,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    batch_size: int,
    seq_len: int,
    batch_group_index: int,
    sample_index: int,
    consecutive_step: int,
    max_preview_group: int,
    max_sample_index: int,
    max_consecutive_index: int,
    strategy: str,
    stateful_n_windows: int,
    active_protocol: str,
    enable_scaling: bool,
) -> MlBatchPreviewSubmission | None:
    if batch_warning is not None or selected_index_frame.is_empty() or not input_columns or not target_columns:
        return None
    scaling = selected_preview_scaling((*input_columns, *target_columns)) if enable_scaling else ()
    return MlBatchPreviewSubmission(
        submit_id=submit_id,
        input_columns=input_columns,
        target_columns=target_columns,
        batch_size=batch_size,
        seq_len=seq_len,
        batch_group_index=min(max(0, batch_group_index), max_preview_group),
        sample_index=min(max(0, sample_index), max_sample_index),
        consecutive_step=min(max(0, consecutive_step), max_consecutive_index),
        strategy=strategy,
        stateful_n_windows=stateful_n_windows,
        active_protocol=active_protocol,
        scaling=scaling,
    )


def selected_preview_scaling(columns: tuple[str, ...]) -> tuple[ScalingRule, ...]:
    selected = tuple(dict.fromkeys(columns))
    rules = {rule.name: rule for rule in PREVIEW_SCALING}
    missing = tuple(column for column in selected if column not in rules)
    if missing:
        raise ValueError(f"Missing preview scaling rules for selected ML columns: {missing}")
    return tuple(rules[column] for column in selected)


def build_batch_preview(
    *,
    store: object,
    selected_index_frame: pl.DataFrame,
    submission: MlBatchPreviewSubmission | None,
) -> tuple[str | None, MlBatchPreview | None, object | None]:
    if store is None or not selected_index_frame.height or submission is None:
        return None, None, None
    try:
        preview = build_ml_batch_preview(
            store,
            MlDatasetIndex(selected_index_frame),
            submission,
        )
    except (
        FileNotFoundError,
        StopIteration,
        ValueError,
        TypeError,
        RuntimeError,
        NotImplementedError,
    ) as exc:
        return str(exc), None, None
    return None, preview, wrap_anywidget_blocks((preview.widget,))[0]


def update_batch_preview(
    *,
    preview: MlBatchPreview | None,
    batch_group_index: int,
    sample_index: int,
    consecutive_step: int,
) -> tuple[MlBatchPreview | None, object | None]:
    if preview is None:
        return None, None
    if (
        batch_group_index == preview.submission.batch_group_index
        and sample_index == preview.submission.sample_index
        and consecutive_step == preview.submission.consecutive_step
    ):
        return preview, None
    updated = update_ml_batch_preview(
        preview,
        batch_group_index,
        sample_index,
        consecutive_step,
    )
    view = wrap_anywidget_blocks((updated.widget,))[0] if updated.widget is not preview.widget else None
    return updated, view


def batch_preview_view(
    *,
    batch_warning: str | None,
    schema_error: str | None,
    submission_error: str | None,
    batch_error: str | None,
    preview: MlBatchPreview | None,
    preview_view: object | None,
    stored_preview_view: object | None,
) -> object:
    if batch_warning is not None:
        return mo.callout(batch_warning, kind="warn")
    if schema_error is not None:
        return mo.callout(schema_error, kind="danger")
    if submission_error is not None:
        return mo.callout(submission_error, kind="danger")
    if batch_error is not None:
        return mo.callout(batch_error, kind="danger")
    if preview is not None and (preview_view is not None or stored_preview_view is not None):
        return mo.vstack(
            [
                preview_view or stored_preview_view,
                mo.md("### Batch metadata"),
                mo.ui.table(preview.metadata),
            ]
        )
    return mo.md("Select one or more ML index rows, choose input/target columns, then click Plot.")
