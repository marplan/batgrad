# ruff: noqa: ANN001, ANN202, I002, INP001, PLR1711, S603, S607

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import os
    import subprocess
    import sys
    from pathlib import Path

    def is_batgrad_root(root: Path) -> bool:
        return (
            (root / "pyproject.toml").is_file()
            and (root / "batgrad" / "__init__.py").is_file()
            and (root / "notebooks" / "_support" / "etl_helpers.py").is_file()
        )

    local_root = Path(__file__).resolve().parents[1]
    if not is_batgrad_root(local_root):
        local_root = Path("/marimo/batgrad")
        if not is_batgrad_root(local_root):
            local_root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/marplan/batgrad.git",
                    str(local_root),
                ],
                check=True,
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--editable",
                str(local_root),
            ],
            check=True,
        )

    project_root = local_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    os.environ.setdefault("DATA_ROOT", "/marimo/data")

    import marimo as mo
    import polars as pl

    from batgrad.contracts.mapping import DatasetStageId
    from batgrad.data.datasets.registry import dataset_ids, get_dataset
    from batgrad.data.processing.manifests import (
        available_manifest_stages,
        load_stage_manifest,
        sort_manifest,
        with_manifest_row_id,
    )
    from batgrad.logging import configure_logging
    from batgrad.notebook_helpers import make_plot_inspection_result, open_local_store_status
    from notebooks._support.etl_helpers import (
        EtlPlotResult,
        default_stage_widget_columns,
        make_controls,
        run_submission,
        scratch_state as make_scratch_state,
        stage_plot_columns,
    )
    from notebooks._support.logging_helpers import capture_log_lines
    from scripts.hf_assets import download_datasets

    configure_logging(level="INFO")


@app.cell
def _():
    download_examples = mo.ui.run_button(label="Download example datasets (~7.2 GiB)")
    store_root = mo.ui.text(
        value=os.getenv("DATA_ROOT"),
        label="Store root",
    )
    return download_examples, store_root


@app.cell
def _(download_examples):
    mo.vstack([mo.md("# ETL"), download_examples])
    return


@app.cell
def _(download_examples, store_root):
    downloaded_root = None
    if download_examples.value:
        _log_lines = []
        with mo.status.spinner(
            title="Downloading example datasets"
        ) as _spinner, capture_log_lines(
            _log_lines,
            lambda line: _spinner.update(subtitle=line),
        ):
            downloaded_root = download_datasets(dataset_ids(), store_root.value)
    return (downloaded_root,)


@app.cell
def _(downloaded_root, store_root):
    _ = downloaded_root
    input_store, store_error = open_local_store_status(store_root.value)
    return input_store, store_error


@app.cell
def _(input_store):
    available_stages_by_dataset = {}
    if input_store is not None:
        for registered_id in dataset_ids():
            stages = available_manifest_stages(get_dataset(registered_id).spec, input_store)
            if stages:
                available_stages_by_dataset[registered_id] = stages
    available_dataset_ids = tuple(available_stages_by_dataset)
    dataset_id = mo.ui.dropdown(
        options=available_dataset_ids,
        value=available_dataset_ids[0] if available_dataset_ids else None,
        label="Dataset",
        disabled=not available_dataset_ids,
    )
    return available_stages_by_dataset, dataset_id


@app.cell
def _(
    available_stages_by_dataset,
    dataset_id,
    store_error,
    store_root,
):
    selected_dataset_id = dataset_id.value
    dataset = get_dataset(selected_dataset_id) if selected_dataset_id is not None else None
    available_stages = available_stages_by_dataset.get(selected_dataset_id, ())
    available_stage_options = [str(stage) for stage in available_stages]
    stage_id = mo.ui.dropdown(
        options=available_stage_options or [""],
        value=available_stage_options[0] if available_stage_options else "",
        label="Stage",
    )
    dataset_controls_view = mo.vstack(
        [
            mo.hstack([store_root, dataset_id], justify="start", wrap=True),
            stage_id,
            *(
                [
                    mo.md(
                        """
                        - Select `Dataset` and `Store root` -> choose stage -> select manifest
                        rows.<br>
                        - Enable interactive normalization if needed -> adjust resampling -> click
                        `Plot`.<br>
                        - Select regions in plots -> click `Sync selection` -> inspect rows in the
                        table.<br>
                        - Use `plot_result` for widgets/runs and `inspection_result.data` for
                        selected data.
                        """
                    )
                ]
                if available_stage_options
                else [
                    mo.md(
                        store_error
                        or """No manifests found for this `Dataset/Store root`. Please check your
                        `DATA_ROOT` in the **.env** file or manually enter a valid `Store root`."""
                    )
                ]
            ),
        ]
    )
    return available_stages, dataset, dataset_controls_view, stage_id


@app.cell
def _(dataset, input_store, stage_id):
    selected_stage = DatasetStageId(stage_id.value) if stage_id.value else None
    if dataset is None or input_store is None or selected_stage is None:
        manifest = pl.DataFrame()
    else:
        try:
            manifest = load_stage_manifest(dataset.spec, input_store, selected_stage)
        except FileNotFoundError:
            manifest = pl.DataFrame()
    if dataset is None:
        stage_widget_columns, stage_x_columns = {}, {}
        widget_col_by_label, x_col_by_label = {}, {}
    else:
        (
            stage_widget_columns,
            stage_x_columns,
            widget_col_by_label,
            x_col_by_label,
        ) = stage_plot_columns(dataset)
    default_widget_columns_by_stage = default_stage_widget_columns(stage_widget_columns)
    manifest = with_manifest_row_id(sort_manifest(manifest))
    return (
        default_widget_columns_by_stage,
        manifest,
        selected_stage,
        stage_widget_columns,
        stage_x_columns,
        widget_col_by_label,
        x_col_by_label,
    )


@app.cell
def _(selected_stage):
    get_submission, set_submission = mo.state(None)
    interactive_normalization = mo.ui.checkbox(
        value=False,
        disabled=selected_stage != DatasetStageId.ingested,
    )
    return get_submission, interactive_normalization, set_submission


@app.cell
def _(interactive_normalization):
    group_manifest = mo.ui.checkbox(
        value=interactive_normalization.value,
        label="Group manifest rows into streams",
    )
    return (group_manifest,)


@app.cell
def _(
    available_stages,
    default_widget_columns_by_stage,
    group_manifest,
    interactive_normalization,
    manifest,
    selected_stage,
    set_submission,
    stage_widget_columns,
    stage_x_columns,
):
    etl_controls = make_controls(
        manifest=manifest,
        selected_stage=selected_stage,
        interactive_normalization=interactive_normalization,
        group_manifest=group_manifest,
        set_submission=set_submission,
        stage_widget_columns=stage_widget_columns,
        stage_x_columns=stage_x_columns,
        default_widget_columns_by_stage=default_widget_columns_by_stage,
        ingested_available=DatasetStageId.ingested in available_stages,
    )
    default_widget_columns = etl_controls.default_widget_columns
    return default_widget_columns, etl_controls


@app.cell
def _():
    sync_selection = mo.ui.button(label="Sync selection")
    inspection_offset = mo.ui.number(
        value=0,
        start=0,
        step=100_000,
        label="Inspection row offset",
    )
    inspection_limit = mo.ui.number(
        value=100_000,
        start=1,
        step=10_000,
        label="Inspection max rows",
    )
    return inspection_limit, inspection_offset, sync_selection


@app.cell
def _():
    scratch_state = make_scratch_state()
    return (scratch_state,)


@app.cell
def _(
    dataset,
    default_widget_columns,
    get_submission,
    input_store,
    manifest,
    scratch_state,
    widget_col_by_label,
    x_col_by_label,
):
    plot_result = (
        EtlPlotResult()
        if dataset is None or input_store is None
        else run_submission(
            dataset=dataset,
            input_store=input_store,
            manifest=manifest,
            state=scratch_state,
            submission=get_submission(),
            default_widget_columns=default_widget_columns,
            widget_col_by_label=widget_col_by_label,
            x_col_by_label=x_col_by_label,
        )
    )
    plot_items = [item for item in (plot_result.output, plot_result.view) if item is not None]
    plot_view = mo.vstack(plot_items) if plot_items else None
    return plot_result, plot_view


@app.cell
def _(inspection_limit, inspection_offset, plot_result, sync_selection):
    _ = sync_selection.value
    _offset = int(inspection_offset.value or 0)
    _limit = int(inspection_limit.value or 100_000)
    inspection_result = make_plot_inspection_result(
        plot_widgets=plot_result.plot_widgets,
        offset=_offset,
        limit=_limit,
        controls=(sync_selection, inspection_offset, inspection_limit),
    )
    return (inspection_result,)


@app.cell
def _(dataset_controls_view, etl_controls, inspection_result, plot_view):
    mo.vstack(
        [
            dataset_controls_view,
            etl_controls.view,
            *([plot_view] if plot_view is not None else []),
            inspection_result.view,
        ]
    ).style(gap="0")
    return


if __name__ == "__main__":
    app.run()
