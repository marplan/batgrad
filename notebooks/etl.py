# ruff: noqa: ANN001, ANN202, C901, I001, I002, INP001, N803, PLC0415, PLR1711

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from batgrad.contracts.mapping import BaseColumns, DatasetStageId
    from batgrad.contracts.protocols import BatteryProtocols
    from batgrad.data.datasets.registry import get_dataset
    from batgrad.data.datasets.pozzato_2022.config import (
        AMBIENT_TEMPERATURE_DEGC,
        NORMALIZED_EIS,
        NORMALIZED_TIMESERIES,
        TIME_NORMALIZE_CHECKS,
        CRateTransformSpec,
        DomainAxisCheckSpec,
        ImpedanceComponentsCheckSpec,
        LinearResamplingSpec,
        MinMaxLTTBResamplingSpec,
        MissingCheckSpec,
        NOMINAL_CAPACITY_AH,
    )
    from batgrad.data.processing.normalize import (
        NormalizeProtocolSpec,
        NormalizeStageConfig,
        NormalizeStageSpec,
    )
    from batgrad.logging import configure
    from batgrad.storage.local import LocalDataProcessingStore
    from batgrad.viz.interactive import make_widgets

    configure(level="INFO")
    return (
        AMBIENT_TEMPERATURE_DEGC,
        BaseColumns,
        BatteryProtocols,
        CRateTransformSpec,
        DatasetStageId,
        DomainAxisCheckSpec,
        ImpedanceComponentsCheckSpec,
        LinearResamplingSpec,
        LocalDataProcessingStore,
        MinMaxLTTBResamplingSpec,
        MissingCheckSpec,
        NOMINAL_CAPACITY_AH,
        NORMALIZED_EIS,
        NORMALIZED_TIMESERIES,
        NormalizeProtocolSpec,
        NormalizeStageConfig,
        NormalizeStageSpec,
        TIME_NORMALIZE_CHECKS,
        get_dataset,
        make_widgets,
        mo,
        pl,
    )


@app.cell
def _(mo):

    dataset_id = mo.ui.dropdown(
        options=["pozzato-2022"],
        value="pozzato-2022",
        label="Dataset",
    )
    store_root = mo.ui.text(
        value="/data/loc_datasets/",
        label="Store root",
    )
    return dataset_id, store_root


@app.cell
def _(LocalDataProcessingStore, dataset_id, mo, store_root):
    input_store = LocalDataProcessingStore(store_root.value)
    mo.vstack([dataset_id, store_root])
    return (input_store,)


@app.cell
def _(BaseColumns, DatasetStageId, dataset_id, get_dataset, input_store, pl):
    dataset = get_dataset(dataset_id.value)
    manifest_path = dataset.spec.manifest(DatasetStageId.ingested)
    manifest = input_store.scan_table(manifest_path).collect()
    canonical_protocol_order = ("cycling", "HPPC", "RPT", "EIS")
    normalized_spec = dataset.spec.processing_stages[DatasetStageId.normalized]
    axis_columns = {
        protocol_spec.protocol.axis_col for protocol_spec in normalized_spec.protocol_specs
    }
    widget_columns = tuple(
        dict.fromkeys(
            column
            for protocol_spec in normalized_spec.protocol_specs
            for column in protocol_spec.output_columns
            if column not in axis_columns
        )
    )
    widget_col_by_label = {str(column): column for column in widget_columns}
    eis_default_columns = {
        BaseColumns.z_real,
        BaseColumns.z_imag,
        BaseColumns.z_mag,
        BaseColumns.z_phase,
    }
    default_widget_columns = tuple(
        str(column)
        for column in widget_columns
        if column in {BaseColumns.volt, BaseColumns.crate, BaseColumns.temp}
        or column in eis_default_columns
    )
    manifest_summary = (
        manifest.group_by("cycle index", "cell id")
        .agg(pl.col("protocol").unique().sort().alias("protocols"))
        .with_columns(
            pl.col("protocols").map_elements(
                lambda protocols: [
                    protocol
                    for protocol in canonical_protocol_order
                    if protocol in protocols
                ],
                return_dtype=pl.List(pl.String),
            )
        )
        .sort("cycle index", "cell id")
    )
    manifest_summary_display = manifest_summary.with_columns(
        pl.col("protocols").list.join(", ")
    )
    return (
        canonical_protocol_order,
        dataset,
        default_widget_columns,
        manifest_summary,
        manifest_summary_display,
        widget_col_by_label,
        widget_columns,
    )


@app.cell
def _(default_widget_columns, manifest_summary_display, mo, widget_columns):
    get_submission, set_submission = mo.state(None)
    selection_table = mo.ui.table(
        manifest_summary_display,
        selection="multi",
        label="Ingested manifest",
    )
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
        label="Widget columns",
    )

    def commit_submission(value):
        next_value = (value or 0) + 1
        set_submission(
            {
                "submit_id": next_value,
                "selection": selection_table.value,
                "cycling_points": cycling_points.value,
                "hppc_points": hppc_points.value,
                "rpt_points": rpt_points.value,
                "eis_points": eis_points.value,
                "widget_cols": widget_cols.value,
            }
        )
        return next_value

    submit_button = mo.ui.button(
        value=0,
        on_click=commit_submission,
        label="Normalize / resample",
    )
    mo.vstack(
        [
            mo.md("### Selection"),
            mo.md("Select one or more rows. Use the table search/filter to narrow the manifest."),
            selection_table,
            mo.md("### Sampling"),
            mo.hstack([cycling_points, hppc_points, rpt_points]),
            eis_points,
            widget_cols,
            submit_button,
        ]
    )
    return (get_submission,)


@app.cell
def _(
    BaseColumns,
    canonical_protocol_order,
    default_widget_columns,
    get_submission,
    manifest_summary,
    widget_col_by_label,
):
    protocols_by_pair = {
        (row["cycle index"], row["cell id"]): row["protocols"]
        for row in manifest_summary.iter_rows(named=True)
    }
    committed_values = get_submission() or {}
    selected = committed_values.get("selection")
    if selected is None:
        selected_rows = []
    elif hasattr(selected, "iter_rows"):
        selected_rows = list(selected.iter_rows(named=True))
    else:
        selected_rows = list(selected)

    group_values = [
        {
            BaseColumns.cidx: int(row["cycle index"]),
            BaseColumns.cell_id: row["cell id"],
        }
        for row in selected_rows
    ]
    selected_protocol_set = {
        protocol
        for row in selected_rows
        for protocol in protocols_by_pair.get((row["cycle index"], row["cell id"]), ())
    }
    selected_protocols = tuple(
        protocol for protocol in canonical_protocol_order if protocol in selected_protocol_set
    )
    selected_widget_cols = tuple(
        widget_col_by_label[label]
        for label in committed_values.get("widget_cols", default_widget_columns)
        if label in widget_col_by_label
    )
    return (
        committed_values,
        group_values,
        selected_protocols,
        selected_widget_cols,
    )


@app.cell
def _(
    AMBIENT_TEMPERATURE_DEGC,
    BaseColumns,
    BatteryProtocols,
    CRateTransformSpec,
    DomainAxisCheckSpec,
    ImpedanceComponentsCheckSpec,
    LinearResamplingSpec,
    MinMaxLTTBResamplingSpec,
    MissingCheckSpec,
    NOMINAL_CAPACITY_AH,
    NORMALIZED_EIS,
    NORMALIZED_TIMESERIES,
    NormalizeProtocolSpec,
    NormalizeStageSpec,
    TIME_NORMALIZE_CHECKS,
    committed_values,
):
    def time_protocol(protocol, *, resampling=None):
        return NormalizeProtocolSpec(
            protocol=protocol,
            columns=tuple(NORMALIZED_TIMESERIES),
            constant_columns={BaseColumns.amb_temp: AMBIENT_TEMPERATURE_DEGC},
            transforms=(
                CRateTransformSpec(
                    source_col=BaseColumns.curr,
                    target_col=BaseColumns.crate,
                    nominal_capacity_ah=NOMINAL_CAPACITY_AH,
                ),
            ),
            checks=TIME_NORMALIZE_CHECKS,
            resampling=resampling,
        )

    def lttb_resampling(x_col, y_col, value):
        value = float(value)
        if value < 1.0:
            return MinMaxLTTBResamplingSpec(
                x_col=x_col,
                y_col=y_col,
                points_ratio=value,
            )
        return MinMaxLTTBResamplingSpec(
            x_col=x_col,
            y_col=y_col,
            points=int(value),
        )

    def normalize_spec(*, resample):
        cycling_resampling = hppc_resampling = rpt_resampling = eis_resampling = None
        if resample:
            cycling_resampling = lttb_resampling(
                BaseColumns.time,
                BaseColumns.volt,
                committed_values.get("cycling_points", 1000),
            )
            hppc_resampling = lttb_resampling(
                BaseColumns.time,
                BaseColumns.volt,
                committed_values.get("hppc_points", 16_384),
            )
            rpt_resampling = lttb_resampling(
                BaseColumns.time,
                BaseColumns.volt,
                committed_values.get("rpt_points", 4096),
            )
            eis_resampling = LinearResamplingSpec(
                x_col=BaseColumns.freq,
                points=int(committed_values.get("eis_points", 48)),
                scale="log",
            )

        return NormalizeStageSpec(
            protocol_specs=(
                time_protocol(BatteryProtocols.cyc, resampling=cycling_resampling),
                time_protocol(BatteryProtocols.hppc, resampling=hppc_resampling),
                time_protocol(BatteryProtocols.rpt, resampling=rpt_resampling),
                NormalizeProtocolSpec(
                    protocol=BatteryProtocols.eis,
                    columns=tuple(NORMALIZED_EIS),
                    constant_columns={BaseColumns.amb_temp: AMBIENT_TEMPERATURE_DEGC},
                    checks=(
                        ImpedanceComponentsCheckSpec(),
                        MissingCheckSpec(),
                        DomainAxisCheckSpec(
                            axis_col=BaseColumns.freq,
                            zero_replacement=1e-7,
                            enforce_positive=True,
                        ),
                    ),
                    resampling=eis_resampling,
                ),
            )
        )

    source_normalize_spec = normalize_spec(resample=False)
    widget_normalize_spec = normalize_spec(resample=True)
    return source_normalize_spec, widget_normalize_spec


@app.cell
def _():
    scratch_state = {
        "source_key": None,
        "source_run": None,
        "widget_run": None,
        "widget_view": None,
    }
    return (scratch_state,)


@app.cell
def _(
    BaseColumns,
    NormalizeStageConfig,
    dataset,
    get_submission,
    group_values,
    input_store,
    make_widgets,
    mo,
    scratch_state,
    selected_protocols,
    selected_widget_cols,
    source_normalize_spec,
    widget_normalize_spec,
):
    widget_run = scratch_state.get("widget_run")
    widget_view = scratch_state.get("widget_view")
    _output = None
    def clear_committed_runs():
        previous_widget_run = scratch_state.get("widget_run")
        if previous_widget_run is not None:
            previous_widget_run.clean()
        previous_source_run = scratch_state.get("source_run")
        if previous_source_run is not None:
            previous_source_run.clean()
        scratch_state["source_key"] = None
        scratch_state["source_run"] = None
        scratch_state["widget_run"] = None
        scratch_state["widget_view"] = None

    if get_submission() is None:
        if widget_run is None:
            widget_view = mo.md(
                "Select manifest rows, tune sampling, then click **Normalize / resample**."
            )
        else:
            _output = mo.md(
                "Pending UI changes are not applied until **Normalize / resample** is clicked."
            )
    elif not group_values:
        widget_view = clear_committed_runs()
        _output = mo.md("Select at least one manifest row.")
    elif not selected_protocols:
        widget_view = clear_committed_runs()
        _output = mo.md("Selected manifest rows do not contain any protocols.")
    else:
        protocols = selected_protocols
        previous_widget_run = scratch_state.get("widget_run")
        if previous_widget_run is not None:
            previous_widget_run.clean()
            scratch_state["widget_run"] = None

        source_key = (
            dataset.spec.dataset_id,
            tuple(protocols),
            tuple(
                (selector[BaseColumns.cidx], selector[BaseColumns.cell_id])
                for selector in group_values
            ),
        )
        source_run = scratch_state.get("source_run")
        if scratch_state.get("source_key") != source_key:
            if source_run is not None:
                source_run.clean()
            source_run = dataset.normalize_interactive(
                input_store,
                input_store,
                NormalizeStageConfig(
                    n_jobs=-1,
                    apply_resampling=False,
                    max_batch_rows=500_000,
                ),
                protocols=protocols,
                group_values=group_values,
                annotate=True,
                normalize_spec=source_normalize_spec,
            )
            scratch_state["source_key"] = source_key
            scratch_state["source_run"] = source_run

        widget_run = dataset.normalize_interactive(
            input_store,
            input_store,
            NormalizeStageConfig(
                n_jobs=-1,
                apply_resampling=True,
                max_batch_rows=500_000,
            ),
            protocols=protocols,
            group_values=group_values,
            annotate=True,
            source_run=source_run,
            normalize_spec=widget_normalize_spec,
        )
        scratch_state["widget_run"] = widget_run
        widgets = make_widgets(
            widget_run,
            cols=selected_widget_cols,
            max_points_per_trace=1_000,
            max_points_per_figure=100_000,
            max_batch_rows=500_000,
        )
        widget_view = mo.vstack([mo.ui.anywidget(widget.show()) for widget in widgets])
        scratch_state["widget_view"] = widget_view
        _output = mo.md("Resampled interactive run is ready.")
    mo.vstack([item for item in (_output, widget_view) if item is not None])
    return


if __name__ == "__main__":
    app.run()
