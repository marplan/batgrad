# ruff: noqa: ANN401, INP001, TC001

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, SupportsInt

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, MappingSpec
from batgrad.data.datasets.config import Dataset
from batgrad.data.processing.interactive import selected_manifest_inputs
from batgrad.data.processing.manifests import (
    MANIFEST_ROW_ID_COLUMN,
    MANIFEST_ROW_IDS_COLUMN,
    selected_manifest_rows,
)
from batgrad.data.processing.normalize import (
    NormalizeStageConfig,
    NormalizeStageSpec,
    normalize_spec_with_resampling,
)
from batgrad.data.transforms.resampling import LinearResamplingSpec, MinMaxLTTBResamplingSpec
from batgrad.storage.store import DataProcessingStore
from batgrad.viz.interactive import make_widgets

CANONICAL_PROTOCOL_ORDER = (
    DatasetProtocolId.cycling,
    DatasetProtocolId.hppc,
    DatasetProtocolId.rpt,
    DatasetProtocolId.eis,
)
DEFAULT_MANIFEST_GROUP_COLUMNS = (BaseColumns.cidx, BaseColumns.cell_id)
DEFAULT_MANIFEST_COLUMN_ORDER = (
    BaseColumns.cidx,
    BaseColumns.cell_id,
    BaseColumns.proto,
    "protocols",
    BaseColumns.soc_pct,
    BaseColumns.row_n,
)
DEFAULT_WIDGET_COLUMNS = {
    BaseColumns.volt,
    BaseColumns.crate,
    BaseColumns.curr,
    BaseColumns.temp,
    BaseColumns.temp_1,
    BaseColumns.temp_2,
    BaseColumns.temp_3,
    BaseColumns.z_real,
    BaseColumns.z_imag,
    BaseColumns.z_mag,
    BaseColumns.z_phase,
}


@dataclass(frozen=True)
class EtlSubmission:
    submit_id: int
    stage: DatasetStageId | None
    effective_stage: DatasetStageId | None
    selected_row_ids: tuple[int, ...]
    group_manifest: bool
    interactive_normalization: bool
    overlay_ingested: bool
    x_col: str
    cycling_points: float
    hppc_points: float
    rpt_points: float
    eis_points: int
    widget_cols: tuple[str, ...]


@dataclass(frozen=True)
class EtlControls:
    group_manifest: Any
    selection_table: Any
    interactive_normalization: Any
    overlay_ingested: Any
    cycling_points: Any
    hppc_points: Any
    rpt_points: Any
    eis_points: Any
    widget_cols: Any
    x_col: Any
    submit_button: Any
    default_widget_columns: tuple[str, ...]
    view: Any


@dataclass(frozen=True)
class EtlPlotResult:
    output: Any = None
    view: Any = None
    run: Any = None
    widgets: tuple[Any, ...] = ()
    plot_widgets: tuple[Any, ...] = ()
    selected_manifest: pl.DataFrame | None = None
    group_values: list[dict[MappingSpec, object]] | None = None
    protocols: tuple[object, ...] = ()
    status_text: str | None = None


@dataclass(frozen=True)
class EtlInspectionResult:
    data: pl.DataFrame
    table: Any
    total_rows: int
    offset: int
    end: int
    view: Any


def scratch_state() -> dict[str, object]:
    return {
        "source_key": None,
        "source_run": None,
        "widget_run": None,
        "plot_widgets": (),
        "widget_view": None,
        "plot_result": EtlPlotResult(),
    }


def clear_committed_runs(state: dict[str, object]) -> None:
    previous_widget_run = state.get("widget_run")
    if previous_widget_run is not None:
        previous_widget_run.clean()
    previous_source_run = state.get("source_run")
    if previous_source_run is not None:
        previous_source_run.clean()
    state["source_key"] = None
    state["source_run"] = None
    state["widget_run"] = None
    state["plot_widgets"] = ()
    state["widget_view"] = None
    state["plot_result"] = EtlPlotResult()


def stage_plot_columns(
    dataset: Dataset,
) -> tuple[
    dict[str, tuple[MappingSpec, ...]],
    dict[str, tuple[MappingSpec, ...]],
    dict[str, MappingSpec],
    dict[str, MappingSpec],
]:
    stage_widget_columns = {}
    stage_x_columns = {}
    for stage_key, stage_spec in dataset.spec.processing_stages.items():
        axis_columns = {
            protocol_spec.protocol.axis_col for protocol_spec in stage_spec.protocol_specs
        }
        output_columns = tuple(
            dict.fromkeys(
                column
                for protocol_spec in stage_spec.protocol_specs
                for column in protocol_spec.output_columns
            )
        )
        stage_widget_columns[str(stage_key)] = tuple(
            dict.fromkeys(column for column in output_columns if column not in axis_columns)
        )
        stage_x_columns[str(stage_key)] = output_columns

    widget_col_by_label = {
        str(column): column for columns in stage_widget_columns.values() for column in columns
    }
    x_col_by_label = {
        str(column): column for columns in stage_x_columns.values() for column in columns
    }
    return stage_widget_columns, stage_x_columns, widget_col_by_label, x_col_by_label


def default_stage_widget_columns(
    stage_widget_columns: dict[str, tuple[MappingSpec, ...]],
) -> dict[str, tuple[str, ...]]:
    return {
        stage_key: tuple(str(column) for column in columns if column in DEFAULT_WIDGET_COLUMNS)
        for stage_key, columns in stage_widget_columns.items()
    }


def order_manifest_columns(frame: pl.DataFrame) -> list[str]:
    preferred = tuple(str(column) for column in DEFAULT_MANIFEST_COLUMN_ORDER)
    return [column for column in preferred if column in frame.columns] + [
        column for column in frame.columns if column not in preferred
    ]


def grouped_manifest(frame: pl.DataFrame) -> pl.DataFrame:
    group_keys = [
        str(column) for column in DEFAULT_MANIFEST_GROUP_COLUMNS if str(column) in frame.columns
    ]
    if not group_keys and str(BaseColumns.proto) in frame.columns:
        group_keys = [str(BaseColumns.proto)]
    aggregations = [
        pl.col(column).alias(MANIFEST_ROW_IDS_COLUMN)
        if column == MANIFEST_ROW_ID_COLUMN
        else pl.col(column)
        for column in frame.columns
        if column not in group_keys
    ]
    return frame.group_by(*group_keys).agg(aggregations).sort(*group_keys)


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
        if MANIFEST_ROW_IDS_COLUMN in row:
            extend_row_ids(row[MANIFEST_ROW_IDS_COLUMN])
        elif MANIFEST_ROW_ID_COLUMN in row and row[MANIFEST_ROW_ID_COLUMN] is not None:
            row_ids.append(_row_id(row[MANIFEST_ROW_ID_COLUMN]))
    return tuple(row_ids)


def make_controls(
    mo: Any,
    *,
    manifest: pl.DataFrame,
    selected_stage: DatasetStageId | None,
    interactive_normalization: Any,
    group_manifest: Any,
    set_submission: Any,
    stage_widget_columns: dict[str, tuple[MappingSpec, ...]],
    stage_x_columns: dict[str, tuple[MappingSpec, ...]],
    default_widget_columns_by_stage: dict[str, tuple[str, ...]],
) -> EtlControls:
    table_data = (
        grouped_manifest(manifest) if group_manifest.value and manifest.height else manifest
    )
    table_display = table_data.select(order_manifest_columns(table_data))
    selection_table = mo.ui.table(
        table_display,
        selection="multi",
        hidden_columns=[
            column
            for column in (MANIFEST_ROW_ID_COLUMN, MANIFEST_ROW_IDS_COLUMN)
            if column in table_display.columns
        ],
    )
    effective_stage = (
        DatasetStageId.normalized
        if selected_stage == DatasetStageId.ingested and interactive_normalization.value
        else selected_stage
    )
    overlay_ingested = mo.ui.checkbox(
        value=False,
        label="Overlay ingested data",
        disabled=effective_stage != DatasetStageId.normalized,
    )
    widget_columns = stage_widget_columns.get(str(effective_stage), ())
    default_widget_columns = default_widget_columns_by_stage.get(str(effective_stage), ())
    cycling_points = mo.ui.number(
        value=1000,
        start=0.001,
        step=100,
        label="cycling LTTB points or ratio (<1)",
    )
    hppc_points = mo.ui.number(
        value=16_384,
        start=0.001,
        step=512,
        label="HPPC LTTB points or ratio (<1)",
    )
    rpt_points = mo.ui.number(
        value=4096,
        start=0.001,
        step=256,
        label="RPT LTTB points or ratio (<1)",
    )
    eis_points = mo.ui.number(value=48, start=3, step=1, label="EIS points")
    widget_cols = mo.ui.multiselect(
        options=[str(column) for column in widget_columns],
        value=list(default_widget_columns),
        label="Plot columns",
    )
    x_columns = stage_x_columns.get(str(effective_stage), ())
    x_col = mo.ui.dropdown(
        options=[str(column) for column in x_columns] or [""],
        value=str(x_columns[0]) if x_columns else "",
        label="Plot x-axis",
    )

    def commit_submission(value: object) -> int:
        next_value = int(value or 0) + 1
        set_submission(
            EtlSubmission(
                submit_id=next_value,
                stage=selected_stage,
                effective_stage=effective_stage,
                selected_row_ids=selected_row_ids_from_table(selection_table.value),
                group_manifest=bool(group_manifest.value),
                interactive_normalization=bool(interactive_normalization.value),
                overlay_ingested=bool(overlay_ingested.value),
                x_col=str(x_col.value),
                cycling_points=float(cycling_points.value),
                hppc_points=float(hppc_points.value),
                rpt_points=float(rpt_points.value),
                eis_points=int(eis_points.value),
                widget_cols=tuple(widget_cols.value),
            )
        )
        return next_value

    submit_button = mo.ui.button(value=0, on_click=commit_submission, label="Plot")
    processing_controls = []
    if effective_stage == DatasetStageId.normalized:
        processing_controls.append(overlay_ingested)
    if selected_stage == DatasetStageId.ingested and interactive_normalization.value:
        processing_controls.extend(
            [
                mo.md("### Resampling"),
                mo.hstack([cycling_points, hppc_points, rpt_points]),
                eis_points,
            ]
        )
    view = mo.vstack(
        [
            mo.md("## Selection").style(margin_top="1em"),
            group_manifest,
            selection_table,
            mo.hstack(
                [mo.md("## Interactive Normalization"), interactive_normalization],
                justify="start",
                align="center",
            ).style(margin_top="1em"),
            *processing_controls,
            mo.md("## Visualization").style(margin_top="1em"),
            *([x_col] if x_columns else []),
            widget_cols,
            submit_button,
        ]
    )
    return EtlControls(
        group_manifest=group_manifest,
        selection_table=selection_table,
        interactive_normalization=interactive_normalization,
        overlay_ingested=overlay_ingested,
        cycling_points=cycling_points,
        hppc_points=hppc_points,
        rpt_points=rpt_points,
        eis_points=eis_points,
        widget_cols=widget_cols,
        x_col=x_col,
        submit_button=submit_button,
        default_widget_columns=default_widget_columns,
        view=view,
    )


def run_submission(
    mo: Any,
    *,
    dataset: Dataset,
    input_store: DataProcessingStore,
    manifest: pl.DataFrame,
    state: dict[str, object],
    submission: EtlSubmission | None,
    default_widget_columns: tuple[str, ...],
    widget_col_by_label: dict[str, MappingSpec],
    x_col_by_label: dict[str, MappingSpec],
) -> EtlPlotResult:
    widget_run = state.get("widget_run")
    widget_view = state.get("widget_view")
    if submission is None:
        output = (
            mo.md("Pending UI changes are not applied until **Plot** is clicked.")
            if widget_run is not None
            else None
        )
        return EtlPlotResult(output=output, view=widget_view)

    selected_source_stage = submission.stage or DatasetStageId.ingested
    selected_stage_spec = dataset.spec.processing_stages[selected_source_stage]
    selected_manifest = selected_manifest_rows(manifest, submission.selected_row_ids)
    group_values, selected_protocols = selected_manifest_inputs(
        selected_manifest,
        selected_stage_spec,
        CANONICAL_PROTOCOL_ORDER,
    )
    if not group_values:
        clear_committed_runs(state)
        return EtlPlotResult(output=mo.md("Select at least one manifest row."))
    if not selected_protocols:
        clear_committed_runs(state)
        return EtlPlotResult(
            output=mo.md("Selected manifest rows do not contain any protocols."),
            selected_manifest=selected_manifest,
            group_values=group_values,
        )

    protocols = selected_protocols
    selected_widget_cols = tuple(
        widget_col_by_label[label]
        for label in submission.widget_cols or default_widget_columns
        if label in widget_col_by_label
    )
    selected_x_col = x_col_by_label.get(str(submission.x_col or ""))
    source_normalize_spec, widget_normalize_spec = etl_normalize_specs(dataset, submission)
    effective_stage = submission.effective_stage or selected_source_stage
    use_interactive_normalization = (
        selected_source_stage == DatasetStageId.ingested
        and effective_stage == DatasetStageId.normalized
        and submission.interactive_normalization
    )
    overlay_sources = (
        (DatasetStageId.ingested,)
        if effective_stage == DatasetStageId.normalized and submission.overlay_ingested
        else ()
    )
    previous_widget_run = state.get("widget_run")
    if previous_widget_run is not None:
        previous_widget_run.clean()
        state["widget_run"] = None

    if use_interactive_normalization:
        source_key = etl_source_cache_key(dataset, protocols, group_values)
        source_run = state.get("source_run")
        if state.get("source_key") != source_key:
            if source_run is not None:
                source_run.clean()
            source_run = dataset.normalize_interactive(
                input_store,
                input_store,
                etl_normalize_config(apply_resampling=False),
                protocols=protocols,
                group_values=group_values,
                annotate=True,
                normalize_spec=source_normalize_spec,
            )
            state["source_key"] = source_key
            state["source_run"] = source_run

        widget_run = dataset.normalize_interactive(
            input_store,
            input_store,
            etl_normalize_config(apply_resampling=True),
            protocols=protocols,
            group_values=group_values,
            annotate=True,
            source_run=source_run,
            normalize_spec=widget_normalize_spec,
        )
    else:
        source_run = state.get("source_run")
        if source_run is not None:
            source_run.clean()
        state["source_key"] = None
        state["source_run"] = None
        widget_run = dataset.load_interactive_manifest(
            input_store,
            source=effective_stage,
            manifest=selected_manifest,
            protocols=protocols,
            group_values=group_values,
        )

    state["widget_run"] = widget_run
    widgets = make_widgets(
        widget_run,
        cols=selected_widget_cols,
        overlay_sources=overlay_sources,
        x_col=selected_x_col,
        max_points_per_trace=1_000,
        max_points_per_figure=100_000,
        max_batch_rows=500_000,
    )
    plot_widgets = tuple(mo.ui.anywidget(widget.show()) for widget in widgets)
    widget_view = mo.vstack(list(plot_widgets))
    result = EtlPlotResult(
        view=widget_view,
        run=widget_run,
        widgets=widgets,
        plot_widgets=plot_widgets,
        selected_manifest=selected_manifest,
        group_values=group_values,
        protocols=protocols,
    )
    state["plot_widgets"] = plot_widgets
    state["widget_view"] = widget_view
    state["plot_result"] = result
    return result


def make_inspection_result(
    mo: Any,
    *,
    plot_widgets: tuple[Any, ...],
    offset: int,
    limit: int,
    controls: tuple[Any, ...],
) -> EtlInspectionResult:
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
    return EtlInspectionResult(
        data=data, table=table, total_rows=total_rows, offset=offset, end=end, view=view
    )


def etl_lttb_resampling(
    x_col: MappingSpec,
    y_col: MappingSpec,
    value: float,
) -> MinMaxLTTBResamplingSpec:
    if value < 1.0:
        return MinMaxLTTBResamplingSpec(x_col=x_col, y_col=y_col, points_ratio=value)
    return MinMaxLTTBResamplingSpec(x_col=x_col, y_col=y_col, points=int(value))


def etl_normalize_specs(
    dataset: Dataset,
    submission: EtlSubmission,
) -> tuple[NormalizeStageSpec, NormalizeStageSpec]:
    base_normalize_spec = dataset.spec.processing_stages[DatasetStageId.normalized]
    if not isinstance(base_normalize_spec, NormalizeStageSpec):
        raise TypeError("Dataset normalized stage must be a NormalizeStageSpec")

    no_resampling = {spec.protocol_id: None for spec in base_normalize_spec.protocol_specs}
    resampling_by_protocol = {
        DatasetProtocolId.cycling: etl_lttb_resampling(
            BaseColumns.time,
            BaseColumns.volt,
            submission.cycling_points,
        ),
        DatasetProtocolId.hppc: etl_lttb_resampling(
            BaseColumns.time,
            BaseColumns.volt,
            submission.hppc_points,
        ),
        DatasetProtocolId.rpt: etl_lttb_resampling(
            BaseColumns.time,
            BaseColumns.volt,
            submission.rpt_points,
        ),
        DatasetProtocolId.eis: LinearResamplingSpec(
            x_col=BaseColumns.freq,
            points=submission.eis_points,
            scale="log",
        ),
    }
    return (
        normalize_spec_with_resampling(base_normalize_spec, no_resampling),
        normalize_spec_with_resampling(base_normalize_spec, resampling_by_protocol),
    )


def etl_normalize_config(*, apply_resampling: bool) -> NormalizeStageConfig:
    return NormalizeStageConfig(
        n_jobs=-1,
        apply_resampling=apply_resampling,
        max_batch_rows=500_000,
    )


def etl_source_cache_key(
    dataset: Dataset,
    protocols: object,
    group_values: list[dict[MappingSpec, object]],
) -> tuple[object, ...]:
    return (
        dataset.spec.dataset_id,
        tuple(protocols)
        if isinstance(protocols, Iterable) and not isinstance(protocols, str)
        else protocols,
        tuple(
            tuple((str(column), value) for column, value in selector.items())
            for selector in group_values
        ),
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
