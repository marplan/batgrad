# ruff: noqa: ANN001, ANN202, I002, INP001, PLR1711

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import os

    import marimo as mo
    import polars as pl

    from batgrad.logging import configure_logger
    from batgrad.ml.data.preview import (
        active_protocol_options,
        available_protocols,
        default_input_columns,
        default_target_columns,
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
    from notebooks.ml_helpers import (
        batch_preview_view as render_batch_preview_view,
        build_batch_preview,
        build_index_frame,
        close_batch_preview,
        discover_normalized_manifest_status,
        make_batch_preview_submission,
        preview_group_count,
        selected_index_rows,
        selected_schema_by_protocol,
        update_batch_preview,
    )

    configure_logger(level="INFO")


@app.cell
def _():
    store_root = mo.ui.text(
        value=os.getenv("DATA_ROOT"),
        label="Store root",
    )
    return (store_root,)


@app.cell
def _(store_root):
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
    index_error, index_frame = build_index_frame(
        store=store,
        selected_manifest_commits=selected_manifest_commits,
        protocols=tuple(protocol_select.value) or None,
        protocol_mode=protocol_mode.value,
        validation_fraction=float(validation_fraction.value),
        validation_seed=int(validation_seed.value),
        group_by=tuple(group_by.value),
    )
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
    return (selected_index_frame,)


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
        options={
            "shuffled_protocol_groups": "Shuffled protocol groups",
            "sequential": "Sequential debug",
        },
        value="shuffled_protocol_groups",
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
    enable_scaling = mo.ui.checkbox(value=False, label="Enable min-max scaling")
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
def _(
    active_protocol,
    batch_size,
    enable_scaling,
    input_columns,
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
    batch = mo.ui.number(value=0, start=0, stop=max_preview_group, step=1, label="Batch")
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

    def commit_preview(value):
        next_value = int(value or 0) + 1
        try:
            submission = make_batch_preview_submission(
                submit_id=next_value,
                selected_index_frame=selected_index_frame,
                batch_warning=batch_warning,
                input_columns=tuple(input_columns.value),
                target_columns=tuple(target_columns.value),
                batch_size=int(batch_size.value),
                seq_len=int(seq_len.value),
                batch_group_index=int(batch.value),
                sample_index=int(batch_index.value),
                consecutive_step=int(consecutive_index.value),
                max_preview_group=max_preview_group,
                max_sample_index=max_batch_index,
                max_consecutive_index=max_consecutive_index,
                strategy=str(strategy.value),
                stateful_n_windows=int(stateful_n_windows.value),
                active_protocol=str(active_protocol.value),
                enable_scaling=bool(enable_scaling.value),
            )
        except (TypeError, ValueError) as exc:
            set_batch_submission_error(str(exc))
            return next_value
        if submission is not None:
            set_batch_submission_error(None)
            set_batch_submission(submission)
        return next_value

    plot_batch = mo.ui.button(value=0, on_click=commit_preview, label="Plot")
    return batch, batch_index, batch_warning, consecutive_index, plot_batch


@app.cell
def _():
    get_batch_submission, set_batch_submission = mo.state(None)
    get_batch_submission_error, set_batch_submission_error = mo.state(None)
    get_batch_preview, set_batch_preview = mo.state(None)
    get_batch_preview_view, set_batch_preview_view = mo.state(None)
    return (
        get_batch_preview,
        get_batch_preview_view,
        get_batch_submission,
        get_batch_submission_error,
        set_batch_preview,
        set_batch_preview_view,
        set_batch_submission,
        set_batch_submission_error,
    )


@app.cell
def _():
    batch_preview_state = {"preview": None}
    return (batch_preview_state,)


@app.cell
def _(
    batch_preview_state,
    get_batch_submission,
    selected_index_frame,
    set_batch_preview,
    set_batch_preview_view,
    store,
):
    submission = get_batch_submission()
    batch_error, batch_preview, batch_preview_view = build_batch_preview(
        store=store,
        selected_index_frame=selected_index_frame,
        submission=submission,
    )
    _previous_preview = batch_preview_state.get("preview")
    if batch_preview is not None:
        if _previous_preview is not None and _previous_preview.widget is not batch_preview.widget:
            close_batch_preview(_previous_preview)
        batch_preview_state["preview"] = batch_preview
        set_batch_preview(batch_preview)
        set_batch_preview_view(batch_preview_view)
    elif _previous_preview is not None and (batch_error is not None or submission is not None):
        close_batch_preview(_previous_preview)
        batch_preview_state["preview"] = None
        set_batch_preview(None)
        set_batch_preview_view(None)
    return batch_error, batch_preview, batch_preview_view


@app.cell
def _(
    batch,
    batch_index,
    batch_preview_state,
    consecutive_index,
    get_batch_preview,
    set_batch_preview,
    set_batch_preview_view,
):
    _previous_preview = get_batch_preview()
    preview, preview_view = update_batch_preview(
        preview=_previous_preview,
        batch_group_index=int(batch.value),
        sample_index=int(batch_index.value),
        consecutive_step=int(consecutive_index.value),
    )
    if (
        _previous_preview is not None
        and preview is not None
        and _previous_preview.widget is not preview.widget
    ):
        close_batch_preview(_previous_preview)
        batch_preview_state["preview"] = preview
    if preview_view is not None:
        set_batch_preview_view(preview_view)
    if preview is not None:
        batch_preview_state["preview"] = preview
        set_batch_preview(preview)
    return preview, preview_view


@app.cell
def _(
    batch_error,
    batch_preview,
    batch_preview_view,
    batch_warning,
    get_batch_preview_view,
    get_batch_submission_error,
    preview,
    preview_view,
    schema_error,
):
    batch_view = render_batch_preview_view(
        batch_warning=batch_warning,
        schema_error=schema_error,
        submission_error=get_batch_submission_error(),
        batch_error=batch_error,
        preview=preview or batch_preview,
        preview_view=preview_view or batch_preview_view,
        stored_preview_view=get_batch_preview_view(),
    )
    return (batch_view,)


@app.cell
def _(
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
    batch_index,
    batch_size,
    batch_view,
    consecutive_index,
    enable_scaling,
    input_columns,
    plot_batch,
    seq_len,
    stateful_n_windows,
    strategy,
    target_columns,
):
    mo.vstack(
        [
            mo.md("## Batch preview"),
            mo.hstack([strategy, input_columns, target_columns], justify="start"),
            mo.hstack([batch_size, seq_len, stateful_n_windows], justify="start"),
            mo.hstack(
                [active_protocol, enable_scaling, batch, batch_index, consecutive_index],
                justify="start",
            ),
            plot_batch,
            batch_view,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
