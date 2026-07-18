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
            and (root / "notebooks" / "_support" / "dataloader_helpers.py").is_file()
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
        # NOTE: Molab has no NVCC, so keep this notebook on the Torch-only runtime.
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--editable",
                str(local_root),
                "torch>=2.10,<2.11",
            ],
            check=True,
            cwd=local_root,
        )

    project_root = local_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    os.environ.setdefault("DATA_ROOT", "/marimo/data")

    import marimo as mo
    import polars as pl

    from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
    from batgrad.data.datasets.registry import dataset_ids
    from batgrad.logging import configure_logging
    from batgrad.ml.data.config import (
        LoaderConfig,
        ScalingRule,
        ValidationConfig,
        WindowConfig,
    )
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.data.loader import create_dataloader_from_index, create_index
    from batgrad.ml.data.preview import (
        active_protocol_options,
        available_protocols,
        default_validation_group_by,
        load_manifest_preview,
        shard_columns_for_protocols,
        validation_group_options,
    )
    from batgrad.notebook_helpers import (
        make_selectable_table,
        manifest_commit_lines,
        open_local_store_status,
        parse_manifest_commits,
    )
    from notebooks._support.dataloader_helpers import (
        BATCH_STRATEGY_OPTIONS,
        BatchPreviewDisplay,
        batch_preview_view as render_batch_preview_view,
        build_batch_preview,
        close_batch_preview,
        default_input_columns,
        default_target_columns,
        make_batch_preview_submission,
        preview_group_count,
        protocol_requires_resubmit,
        selected_schema_by_protocol,
        update_batch_preview,
        updated_batch_preview_display,
    )
    from notebooks._support.logging_helpers import capture_log_lines
    from notebooks._support.ml_data_helpers import (
        discover_normalized_manifest_status,
        selected_index_rows,
    )
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
    store, store_error = open_local_store_status(store_root.value)
    return store, store_error


@app.cell
def _(store):
    manifests, manifest_error = discover_normalized_manifest_status(store)
    manifest_select = mo.ui.multiselect(
        options=manifests,
        value=manifests[:1],
        label="Normalized manifests",
    )
    return manifest_error, manifest_select


@app.cell
def _(manifest_select, store):
    manifest_commits = mo.ui.text_area(
        value=manifest_commit_lines(store, tuple(manifest_select.value)),
        label="Manifest path = git commit prefix",
        full_width=True,
    )
    return (manifest_commits,)


@app.cell
def _(manifest_commits):
    selected_manifest_commits = parse_manifest_commits(manifest_commits.value)
    return (selected_manifest_commits,)


@app.cell
def _(selected_manifest_commits, store):
    raw_manifest = load_manifest_preview(store, selected_manifest_commits)
    _protocol_options = available_protocols(raw_manifest)
    protocol_select = mo.ui.multiselect(
        options=_protocol_options,
        value=_protocol_options,
        label="Protocols",
    )
    return protocol_select, raw_manifest


@app.cell
def _(raw_manifest):
    group_options = validation_group_options(raw_manifest)
    default_group_by = default_validation_group_by(group_options)
    group_by = mo.ui.multiselect(
        options=group_options,
        value=default_group_by,
        label="Validation group by",
    )
    validation_fraction = mo.ui.number(
        value=0.2,
        start=0.0,
        stop=0.95,
        step=0.05,
        label="Validation fraction",
    )
    validation_seed = mo.ui.number(
        value=69,
        start=0,
        step=1,
        label="Validation seed",
    )
    protocol_mode = mo.ui.dropdown(
        options=("available", "strict"),
        value="available",
        label="Protocol mode",
    )
    return group_by, protocol_mode, validation_fraction, validation_seed


@app.cell
def _(
    group_by,
    protocol_mode,
    protocol_select,
    selected_manifest_commits,
    store,
    validation_fraction,
    validation_seed,
):
    if store is None or not selected_manifest_commits:
        ml_index = None
        index_error = None
        index_frame = pl.DataFrame()
    else:
        try:
            ml_index = create_index(
                store=store,
                manifest_paths=selected_manifest_commits,
                protocols=tuple(protocol_select.value) or None,
                protocol_mode=protocol_mode.value,
                validation=ValidationConfig.sample(
                    fraction=float(validation_fraction.value),
                    seed=int(validation_seed.value),
                    group_by=tuple(group_by.value),
                ),
            )
            index_error = None
            index_frame = ml_index.frame
        except (FileNotFoundError, TypeError, ValueError) as exc:
            ml_index = None
            index_error = str(exc)
            index_frame = pl.DataFrame()
    return index_error, index_frame


@app.cell
def _(index_error, index_frame):
    index_table = (
        make_selectable_table(index_frame)
        if index_error is None and index_frame.height
        else None
    )
    return (index_table,)


@app.cell
def _(index_frame, index_table):
    selected_index_frame = selected_index_rows(index_frame, index_table)
    selected_index = (
        MlDatasetIndex(selected_index_frame) if selected_index_frame.height else None
    )
    return selected_index, selected_index_frame


@app.cell
def _(selected_index_frame, store):
    schema_by_protocol, schema_error = selected_schema_by_protocol(store, selected_index_frame)
    return schema_by_protocol, schema_error


@app.cell
def _(schema_by_protocol):
    shard_columns = shard_columns_for_protocols(schema_by_protocol)
    default_inputs = default_input_columns(shard_columns)
    default_targets = default_target_columns(shard_columns)
    input_columns = mo.ui.multiselect(
        options=shard_columns,
        value=default_inputs,
        label="Input columns",
    )
    target_columns = mo.ui.multiselect(
        options=shard_columns,
        value=default_targets,
        label="Target columns",
    )
    batch_size = mo.ui.number(value=4, start=1, step=1, label="Batch size")
    seq_len = mo.ui.number(value=128, start=1, step=1, label="Sequence length")
    strategy = mo.ui.dropdown(
        options=BATCH_STRATEGY_OPTIONS,
        value="Shuffled protocol groups",
        label="Strategy",
    )
    _active_protocol_options = active_protocol_options(schema_by_protocol)
    active_protocol = mo.ui.dropdown(
        options=_active_protocol_options,
        value=_active_protocol_options[0],
        label="Selected protocol",
    )
    stateful_n_windows = mo.ui.number(
        value=1,
        start=1,
        step=1,
        label="Consecutive batches",
    )
    enable_scaling = mo.ui.checkbox(value=False, label="Use notebook scaling rules")
    return (
        active_protocol,
        batch_size,
        enable_scaling,
        input_columns,
        seq_len,
        stateful_n_windows,
        strategy,
        target_columns,
    )


@app.cell
def _():
    scaling_defaults = (
        ScalingRule(BaseColumns.dt, 0.0, 10_000.0, transform="log1p"),
        ScalingRule(BaseColumns.crate, -6.0, 6.0),
        ScalingRule(BaseColumns.volt, 2.3, 4.6),
        ScalingRule(BaseColumns.temp, 5.0, 65.0),
        ScalingRule(BaseColumns.amb_temp, 0.0, 50.0),
        ScalingRule(BaseColumns.a_heat, 0.0, 55.0),
    )
    return (scaling_defaults,)


@app.cell
def _(enable_scaling, input_columns, scaling_defaults, target_columns):
    scaling_rules = ()
    scaling_view = None
    if bool(enable_scaling.value):
        selected_columns = tuple(dict.fromkeys((*input_columns.value, *target_columns.value)))
        defaults_by_column = {rule.name: rule for rule in scaling_defaults}
        missing = tuple(column for column in selected_columns if column not in defaults_by_column)
        if missing:
            scaling_view = mo.callout(
                f"No notebook scaling rules for selected ML columns: {missing}. "
                "Add rules to `scaling_defaults` or disable scaling.",
                kind="warn",
            )
        else:
            scaling_rules = tuple(defaults_by_column[column] for column in selected_columns)
            if scaling_rules:
                scaling_view = mo.ui.table(
                    [
                        {
                            "column": rule.name,
                            "input_min": rule.input_min,
                            "input_max": rule.input_max,
                            "output_min": rule.output_min,
                            "output_max": rule.output_max,
                            "transform": rule.transform,
                            "clip": rule.clip,
                        }
                        for rule in scaling_rules
                    ]
                )
    return scaling_rules, scaling_view


@app.cell
def _(
    active_protocol,
    batch_size,
    input_columns,
    scaling_rules,
    selected_index,
    seq_len,
    stateful_n_windows,
    store,
    strategy,
    target_columns,
):
    loader_config = None
    dataloader = None
    batch = None
    dataloader_error = None
    if (
        store is not None
        and selected_index is not None
        and input_columns.value
        and target_columns.value
    ):
        try:
            split = str(selected_index.frame[0, BaseColumns.split])
            protocol = DatasetProtocolId(str(active_protocol.value))
            loader_config = LoaderConfig(
                split=split,
                default_window=WindowConfig(
                    batch_size=int(batch_size.value),
                    seq_len=int(seq_len.value),
                ),
                strategy=str(strategy.value),
                protocol_order=(protocol,),
                stateful_n_windows=int(stateful_n_windows.value),
            )
            dataloader = create_dataloader_from_index(
                store,
                selected_index,
                tuple(input_columns.value),
                tuple(target_columns.value),
                protocols=(protocol,),
                scaling=scaling_rules,
                config=loader_config,
            )
            batch = next(iter(dataloader), None)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            dataloader_error = str(exc)
    return batch, dataloader, dataloader_error, loader_config


@app.cell
def _(
    active_protocol,
    batch_size,
    enable_scaling,
    input_columns,
    scaling_rules,
    selected_index_frame,
    seq_len,
    set_batch_submission,
    set_batch_submission_error,
    stateful_n_windows,
    strategy,
    target_columns,
):
    max_batch_index = max(0, int(batch_size.value) - 1)
    max_consecutive_index = max(0, int(stateful_n_windows.value) - 1)
    batch_warning, preview_group_count_ = preview_group_count(
        selected_index_frame,
        strategy=str(strategy.value),
        active_protocol=str(active_protocol.value),
        batch_size=int(batch_size.value),
        seq_len=int(seq_len.value),
        stateful_n_windows=int(stateful_n_windows.value),
    )
    max_preview_group = max(0, preview_group_count_ - 1)
    batch_group_index = mo.ui.number(
        value=0, start=0, stop=max_preview_group, step=1, label="Batch group"
    )
    batch_index = mo.ui.number(
        value=0,
        start=0,
        stop=max_batch_index,
        step=1,
        label="Batch index",
    )
    consecutive_index = mo.ui.number(
        value=0,
        start=0,
        stop=max_consecutive_index,
        step=1,
        label="Consecutive index",
    )

    def submit_current_preview(submit_id):
        if bool(enable_scaling.value) and not scaling_rules:
            set_batch_submission_error(
                "Scaling is enabled but no complete notebook rules are selected."
            )
            return None
        try:
            submission = make_batch_preview_submission(
                submit_id=submit_id,
                selected_index_frame=selected_index_frame,
                batch_warning=batch_warning,
                input_columns=tuple(input_columns.value),
                target_columns=tuple(target_columns.value),
                batch_size=int(batch_size.value),
                seq_len=int(seq_len.value),
                batch_group_index=int(batch_group_index.value),
                sample_index=int(batch_index.value),
                consecutive_step=int(consecutive_index.value),
                max_preview_group=max_preview_group,
                max_sample_index=max_batch_index,
                max_consecutive_index=max_consecutive_index,
                strategy=str(strategy.value),
                stateful_n_windows=int(stateful_n_windows.value),
                active_protocol=str(active_protocol.value),
                scaling=scaling_rules,
            )
        except (TypeError, ValueError) as exc:
            set_batch_submission_error(str(exc))
            return None
        if submission is not None:
            set_batch_submission_error(None)
            set_batch_submission(submission)
        return submission

    def commit_preview(value):
        next_value = int(value or 0) + 1
        submit_current_preview(next_value)
        return next_value

    plot_batch = mo.ui.button(value=0, on_click=commit_preview, label="Plot")
    return (
        batch_group_index,
        batch_index,
        batch_warning,
        consecutive_index,
        plot_batch,
        submit_current_preview,
    )


@app.cell
def _():
    get_batch_submission, set_batch_submission = mo.state(None)
    get_batch_submission_error, set_batch_submission_error = mo.state(None)
    get_batch_display, set_batch_display = mo.state(None)
    return (
        get_batch_display,
        get_batch_submission,
        get_batch_submission_error,
        set_batch_display,
        set_batch_submission,
        set_batch_submission_error,
    )


@app.cell
def _(active_protocol, get_batch_submission, submit_current_preview):
    current_submission = get_batch_submission()
    selected_protocol = str(active_protocol.value)
    if protocol_requires_resubmit(current_submission, selected_protocol):
        submit_current_preview(current_submission.submit_id + 1)
    return


@app.cell
def _(get_batch_submission, selected_index_frame, set_batch_display, store):
    submission = get_batch_submission()
    batch_error, batch_preview, batch_preview_view = build_batch_preview(
        store=store,
        selected_index_frame=selected_index_frame,
        submission=submission,
    )
    if batch_preview is not None and batch_preview_view is not None:
        def _replace_display(previous):
            if previous is not None and previous.preview.widget is not batch_preview.widget:
                close_batch_preview(previous.preview)
            return BatchPreviewDisplay(batch_preview, batch_preview_view)

        set_batch_display(_replace_display)
    elif batch_error is not None or submission is not None:
        def _clear_display(previous):
            if previous is not None:
                close_batch_preview(previous.preview)

        set_batch_display(_clear_display)
    return batch_error, batch_preview, batch_preview_view


@app.cell
def _(
    batch_group_index,
    batch_index,
    consecutive_index,
    get_batch_display,
    set_batch_display,
):
    previous_display = get_batch_display()
    previous_preview = None if previous_display is None else previous_display.preview
    preview_update_error, preview, preview_view = update_batch_preview(
        preview=previous_preview,
        batch_group_index=int(batch_group_index.value),
        sample_index=int(batch_index.value),
        consecutive_step=int(consecutive_index.value),
    )
    next_display = updated_batch_preview_display(
        previous=previous_display,
        preview=preview,
        view=preview_view,
    )
    if preview is not None and preview is not previous_preview and next_display is not None:
        if previous_preview is not None and previous_preview.widget is not preview.widget:
            close_batch_preview(previous_preview)
        set_batch_display(next_display)
    return preview, preview_update_error, preview_view


@app.cell
def _(
    batch_error,
    batch_preview,
    batch_preview_view,
    batch_warning,
    get_batch_display,
    get_batch_submission_error,
    preview,
    preview_update_error,
    preview_view,
    schema_error,
):
    _ = batch_preview, batch_preview_view, preview, preview_view
    current_display = get_batch_display()
    batch_view = render_batch_preview_view(
        batch_warning=batch_warning,
        schema_error=schema_error,
        submission_error=get_batch_submission_error(),
        batch_error=batch_error or preview_update_error,
        preview=None if current_display is None else current_display.preview,
        preview_view=None if current_display is None else current_display.view,
    )
    return (batch_view,)


@app.cell
def _(
    download_examples,
    group_by,
    index_error,
    index_table,
    manifest_commits,
    manifest_error,
    manifest_select,
    protocol_mode,
    protocol_select,
    store_error,
    store_root,
    validation_fraction,
    validation_seed,
):
    messages = []
    if store_error is not None:
        messages.append(mo.callout(store_error, kind="danger"))
    elif manifest_error is not None:
        messages.append(mo.callout(manifest_error, kind="warn"))
    mo.vstack(
        [
            mo.md("# ML Data Loader"),
            download_examples,
            mo.hstack([store_root], justify="start"),
            *messages,
            manifest_select,
            manifest_commits,
            mo.md("## ML index"),
            mo.hstack([protocol_select, group_by], justify="start"),
            protocol_mode,
            mo.hstack([validation_fraction, validation_seed], justify="start"),
            mo.callout(index_error, kind="danger")
            if index_error is not None
            else index_table
            if index_table is not None
            else mo.md("No ML index available. Select a valid store root and manifest."),
        ]
    )
    return


@app.cell
def _(
    active_protocol,
    batch,
    batch_group_index,
    batch_index,
    batch_size,
    batch_view,
    consecutive_index,
    dataloader,
    dataloader_error,
    enable_scaling,
    input_columns,
    loader_config,
    plot_batch,
    scaling_view,
    seq_len,
    stateful_n_windows,
    strategy,
    target_columns,
):
    _ = batch, dataloader, loader_config
    optional_messages = []
    if scaling_view is not None:
        optional_messages.append(scaling_view)
    if dataloader_error is not None:
        optional_messages.append(mo.callout(dataloader_error, kind="danger"))
    mo.vstack(
        [
            mo.md("## Batch preview"),
            mo.hstack([strategy, input_columns, target_columns], justify="start"),
            mo.hstack([batch_size, seq_len, stateful_n_windows], justify="start"),
            mo.hstack(
                [
                    active_protocol,
                    enable_scaling,
                    batch_group_index,
                    batch_index,
                    consecutive_index,
                ],
                justify="start",
            ),
            *optional_messages,
            plot_batch,
            batch_view,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
