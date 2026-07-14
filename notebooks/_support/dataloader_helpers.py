from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import marimo as mo

from batgrad.contracts.mapping import BaseColumns
from batgrad.ml.data.index import MlDatasetIndex
from batgrad.ml.data.materialization import resolve_index_schema_by_protocol
from batgrad.ml.data.preview import (
    MlBatchPreviewSpec,
    count_ml_batch_preview_groups,
    ml_batch_preview_unavailable_message,
)
from batgrad.notebook_helpers import wrap_anywidget_blocks
from batgrad.viz.ml import (
    MlBatchPreview,
    build_ml_batch_preview,
    update_ml_batch_preview,
)

if TYPE_CHECKING:
    import polars as pl

    from batgrad.ml.data.config import ScalingRule

BATCH_STRATEGY_OPTIONS = {
    "Shuffled protocol groups": "shuffled_protocol_groups",
    "Sequential debug": "sequential",
}


@dataclass(frozen=True, slots=True)
class BatchPreviewSubmission:
    submit_id: int
    spec: MlBatchPreviewSpec


@dataclass(frozen=True, slots=True)
class BatchPreviewDisplay:
    preview: MlBatchPreview
    view: object


def protocol_requires_resubmit(
    submission: BatchPreviewSubmission | None,
    active_protocol: str,
) -> bool:
    return submission is not None and submission.spec.active_protocol != active_protocol


def default_input_columns(shard_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in (
            BaseColumns.dt,
            BaseColumns.crate,
            BaseColumns.volt,
            BaseColumns.temp,
            BaseColumns.amb_temp,
            BaseColumns.a_heat,
        )
        if column in shard_columns
    )


def default_target_columns(shard_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column) for column in (BaseColumns.volt, BaseColumns.temp) if column in shard_columns
    )


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
    scaling: tuple[ScalingRule, ...],
) -> BatchPreviewSubmission | None:
    if (
        batch_warning is not None
        or selected_index_frame.is_empty()
        or not input_columns
        or not target_columns
    ):
        return None
    return BatchPreviewSubmission(
        submit_id=submit_id,
        spec=MlBatchPreviewSpec(
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
        ),
    )


def build_batch_preview(
    *,
    store: object,
    selected_index_frame: pl.DataFrame,
    submission: BatchPreviewSubmission | None,
) -> tuple[str | None, MlBatchPreview | None, object | None]:
    if store is None or not selected_index_frame.height or submission is None:
        return None, None, None
    try:
        preview = build_ml_batch_preview(
            store,
            MlDatasetIndex(selected_index_frame),
            submission.spec,
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


def close_batch_preview(preview: MlBatchPreview | None) -> None:
    if preview is None:
        return
    close = getattr(preview.widget, "close", None)
    if callable(close):
        close()


def update_batch_preview(
    *,
    preview: MlBatchPreview | None,
    batch_group_index: int,
    sample_index: int,
    consecutive_step: int,
) -> tuple[str | None, MlBatchPreview | None, object | None]:
    if preview is None:
        return None, None, None
    if (
        batch_group_index == preview.spec.batch_group_index
        and sample_index == preview.spec.sample_index
        and consecutive_step == preview.spec.consecutive_step
    ):
        return None, preview, None
    try:
        updated = update_ml_batch_preview(
            preview,
            batch_group_index,
            sample_index,
            consecutive_step,
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
    view = (
        wrap_anywidget_blocks((updated.widget,))[0]
        if updated.widget is not preview.widget
        else None
    )
    return None, updated, view


def updated_batch_preview_display(
    *,
    previous: BatchPreviewDisplay | None,
    preview: MlBatchPreview | None,
    view: object | None,
) -> BatchPreviewDisplay | None:
    if preview is None:
        return previous
    if view is not None:
        return BatchPreviewDisplay(preview, view)
    if previous is not None and previous.preview.widget is preview.widget:
        return BatchPreviewDisplay(preview, previous.view)
    return previous


def batch_preview_view(
    *,
    batch_warning: str | None,
    schema_error: str | None,
    submission_error: str | None,
    batch_error: str | None,
    preview: MlBatchPreview | None,
    preview_view: object | None,
) -> object:
    if batch_warning is not None:
        return mo.callout(batch_warning, kind="warn")
    if schema_error is not None:
        return mo.callout(schema_error, kind="danger")
    if submission_error is not None:
        return mo.callout(submission_error, kind="danger")
    if batch_error is not None:
        return mo.callout(batch_error, kind="danger")
    if preview is not None and preview_view is not None:
        return mo.vstack(
            [
                preview_view,
                mo.md("### Batch metadata"),
                mo.ui.table(preview.metadata),
            ]
        )
    return mo.md("Select one or more ML index rows, choose input/target columns, then click Plot.")
