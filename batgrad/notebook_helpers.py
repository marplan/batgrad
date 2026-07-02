from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, SupportsInt

import marimo as mo
import polars as pl
import pyarrow.parquet as pq

from batgrad.contracts.mapping import BaseColumns
from batgrad.contracts.row_ids import (
    MANIFEST_ROW_ID_COLUMN,
    MANIFEST_ROW_IDS_COLUMN,
    ML_INDEX_ROW_ID_COLUMN,
)
from batgrad.storage.local import LocalDataProcessingStore

if TYPE_CHECKING:
    from typing import Any

    from anywidget import AnyWidget

    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True)
class PlotInspectionResult:
    data: pl.DataFrame
    table: Any
    total_rows: int
    offset: int
    end: int
    view: Any


class ShowableWidget(Protocol):
    def show(self) -> AnyWidget: ...


def hidden_row_id_columns(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    return [
        column
        for column in (ML_INDEX_ROW_ID_COLUMN, MANIFEST_ROW_ID_COLUMN, MANIFEST_ROW_IDS_COLUMN)
        if column in available
    ]


def open_local_store_status(value: object) -> tuple[LocalDataProcessingStore | None, str | None]:
    if not isinstance(value, str | Path):
        root = value or "<empty>"
        return None, f"Store root is not available: {root!r}. expected str or Path"
    try:
        return LocalDataProcessingStore(value), None
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        root = value or "<empty>"
        return None, f"Store root is not available: {root!r}. {exc}"


def make_selectable_table(
    frame: pl.DataFrame,
    *,
    selection: Literal["single", "multi", "single-cell", "multi-cell"] | None = "multi",
) -> object:
    return mo.ui.table(
        frame,
        selection=selection,
        hidden_columns=hidden_row_id_columns(frame.columns),
    )


def wrap_anywidget_blocks(widgets: Iterable[ShowableWidget]) -> tuple[object, ...]:
    return tuple(mo.ui.anywidget(widget.show()) for widget in widgets)


def manifest_footer_value(store: DatasetStoreReader | None, manifest_path: str, key: str) -> str:
    if store is None:
        return ""
    try:
        with store.local_file(manifest_path) as local_path:
            metadata = pq.ParquetFile(local_path).metadata.metadata
    except (FileNotFoundError, OSError):
        return ""
    if metadata is None:
        return ""
    value = metadata.get(key.encode())
    return "" if value is None else value.decode()


def manifest_commit_lines(store: DatasetStoreReader | None, manifest_paths: tuple[str, ...]) -> str:
    lines = []
    for manifest_path in manifest_paths:
        commit = manifest_footer_value(store, manifest_path, BaseColumns.git_commit)
        lines.append(f"{manifest_path}={commit[:7]}")
    return "\n".join(lines)


def parse_manifest_commits(value: str) -> dict[str, str]:
    mapping = {}
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        path, commit = stripped.rsplit("=", 1)
        mapping[path.strip()] = commit.strip()
    return mapping


def selected_row_ids_from_table(value: object) -> tuple[int, ...]:
    rows = _table_rows(value)
    row_ids: list[int] = []

    def extend_row_ids(values: object) -> None:
        if values is None:
            return
        if isinstance(values, str):
            row_ids.extend(int(row_id) for row_id in values.split(", ") if row_id)
            return
        if isinstance(values, Iterable):
            row_ids.extend(_row_id(row_id) for row_id in values)
            return
        row_ids.append(_row_id(values))

    for row in rows:
        if ML_INDEX_ROW_ID_COLUMN in row and row[ML_INDEX_ROW_ID_COLUMN] is not None:
            row_ids.append(_row_id(row[ML_INDEX_ROW_ID_COLUMN]))
        elif MANIFEST_ROW_IDS_COLUMN in row:
            extend_row_ids(row[MANIFEST_ROW_IDS_COLUMN])
        elif MANIFEST_ROW_ID_COLUMN in row and row[MANIFEST_ROW_ID_COLUMN] is not None:
            row_ids.append(_row_id(row[MANIFEST_ROW_ID_COLUMN]))
    return tuple(row_ids)


def make_plot_inspection_result(
    *,
    plot_widgets: tuple[Any, ...],
    offset: int,
    limit: int,
    controls: tuple[Any, ...],
) -> PlotInspectionResult:
    frames = []
    total_rows = 0
    remaining = limit
    plot_widget_values = tuple(plot_widget.value for plot_widget in plot_widgets)
    for widget_index, (plot_widget, widget_value) in enumerate(
        zip(plot_widgets, plot_widget_values, strict=True),
    ):
        selection = widget_value.get("selection", {})
        if remaining <= 0:
            frame, widget_rows = plot_widget.widget.selected_data(
                selection=selection,
                widget_index=widget_index,
                offset=0,
                limit=1,
            )
        else:
            frame, widget_rows = plot_widget.widget.selected_data(
                selection=selection,
                widget_index=widget_index,
                offset=max(0, offset - total_rows),
                limit=remaining,
            )
        total_rows += widget_rows
        if frame.height:
            frames.append(frame)
            remaining -= frame.height

    data = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    end = min(offset + data.height, total_rows)
    table = mo.ui.table(data, selection=None)
    view = mo.vstack(
        [
            mo.md("## Inspection").style(margin_top="1em"),
            mo.hstack(list(controls), justify="start"),
            mo.md(f"Rows `{offset:,}` to `{end:,}` of `{total_rows:,}` selected plot rows."),
            table,
        ]
    )
    return PlotInspectionResult(
        data=data, table=table, total_rows=total_rows, offset=offset, end=end, view=view
    )


def _table_rows(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, pl.DataFrame):
        return list(value.iter_rows(named=True))
    if isinstance(value, Iterable) and not isinstance(value, str):
        return [
            {str(key): item for key, item in row.items()}
            for row in value
            if isinstance(row, Mapping)
        ]
    return []


def _row_id(value: object) -> int:
    if isinstance(value, str | bytes | bytearray | SupportsInt):
        return int(value)
    raise TypeError(f"Expected row id to be int-like or string, got {type(value).__name__}")
