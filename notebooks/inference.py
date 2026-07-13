# ruff: noqa: ANN001, ANN202, I002, INP001, PLR1711

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import os

    import marimo as mo

    from batgrad.logging import configure_logger
    from batgrad.ml.data.preview import (
        available_protocols,
        default_validation_group_by,
        load_manifest_preview,
        validation_group_options,
    )
    from batgrad.ml.inference import available_devices
    from batgrad.notebook_helpers import (
        make_selectable_table,
        manifest_commit_lines,
        open_local_store_status,
        parse_manifest_commits,
    )
    from batgrad.viz.ml import inference_group_label
    from notebooks.inference_helpers import (
        build_batch_inference,
        checkpoint_frame,
        checkpoint_options,
        inference_view as render_inference_view,
        make_checkpoint_table,
        make_inference_request,
        make_inference_submission,
        render_batch_result,
        selected_checkpoints_from_table,
    )
    from notebooks.ml_helpers import (
        build_index_frame,
        discover_normalized_manifest_status,
        selected_index_rows,
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
        make_selectable_table(index_frame) if index_error is None and index_frame.height else None
    )
    return (index_table,)


@app.cell
def _(index_frame, index_table):
    selected_index_frame = selected_index_rows(index_frame, index_table)
    return (selected_index_frame,)


@app.cell
def _():
    checkpoints = checkpoint_options()
    checkpoint_frame_ = checkpoint_frame(checkpoints)
    checkpoint_table = make_checkpoint_table(checkpoint_frame_)
    _devices = available_devices()
    device_select = mo.ui.dropdown(
        options=_devices,
        value=_devices[1] if len(_devices) > 1 else _devices[0],
        label="Device",
    )
    return checkpoint_frame_, checkpoint_table, device_select


@app.cell
def _(checkpoint_frame_, checkpoint_table):
    selected_checkpoints = selected_checkpoints_from_table(
        checkpoint_table.value,
        checkpoint_frame_,
    )
    return (selected_checkpoints,)


@app.cell
def _():
    get_inference_request, set_inference_request = mo.state(None)
    get_inference_submission_error, set_inference_submission_error = mo.state(None)
    get_inference_result, set_inference_result = mo.state(None)
    return (
        get_inference_request,
        get_inference_result,
        get_inference_submission_error,
        set_inference_request,
        set_inference_result,
        set_inference_submission_error,
    )


@app.cell
def _(
    device_select,
    selected_checkpoints,
    selected_index_frame,
    set_inference_request,
    set_inference_submission_error,
    store,
):
    masked_suffix_steps = mo.ui.text(
        value="128",
        label="Masked suffix steps (comma-separated; 0 = classic)",
    )
    rollout_steps = mo.ui.number(
        value=200,
        start=1,
        step=1,
        label="Rollout steps",
    )

    def commit_inference(value):
        next_value = int(value or 0) + 1
        try:
            submission = make_inference_submission(
                submit_id=next_value,
                checkpoints=selected_checkpoints,
                device=str(device_select.value),
                masked_suffix_steps=str(masked_suffix_steps.value),
                rollout_steps=int(rollout_steps.value),
            )
        except (TypeError, ValueError) as exc:
            set_inference_submission_error(str(exc))
            return next_value
        try:
            request = make_inference_request(
                store=store,
                selected_index_frame=selected_index_frame,
                submission=submission,
            )
        except (TypeError, ValueError) as exc:
            set_inference_submission_error(str(exc))
            return next_value
        set_inference_submission_error(None)
        set_inference_request(request)
        return next_value

    run_inference = mo.ui.button(value=0, on_click=commit_inference, label="Run inference")
    return masked_suffix_steps, rollout_steps, run_inference


@app.cell
def _(get_inference_request, set_inference_result):
    inference_error, inference_result = build_batch_inference(get_inference_request())
    if inference_result is not None:
        set_inference_result(inference_result)
    return inference_error, inference_result


@app.cell
def _(get_inference_result, inference_result):
    result = inference_result or get_inference_result()
    row_options = (
        [
            f"{idx}: {inference_group_label(group_key)}"
            for idx, group_key in enumerate(result.group_keys)
        ]
        if result is not None
        else ["No inference result"]
    )
    batch_row_select = mo.ui.dropdown(
        options=row_options,
        value=row_options[0],
        label="Selected row",
    )
    return batch_row_select, result


@app.cell
def _(batch_row_select, result):
    batch_result_view = (
        render_batch_result(result, int(str(batch_row_select.value).split(":", 1)[0]))
        if result is not None
        else None
    )
    return (batch_result_view,)


@app.cell
def _(batch_result_view, get_inference_submission_error, inference_error):
    inference_view = render_inference_view(
        submission_error=get_inference_submission_error(),
        inference_error=inference_error,
        result_view=batch_result_view,
    )
    return (inference_view,)


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
            mo.md("# ML Inference"),
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
    batch_row_select,
    checkpoint_table,
    device_select,
    inference_view,
    masked_suffix_steps,
    rollout_steps,
    run_inference,
):
    mo.vstack(
        [
            mo.md("## Batch inference"),
            mo.hstack([device_select], justify="start"),
            checkpoint_table,
            mo.hstack([masked_suffix_steps, rollout_steps], justify="start"),
            run_inference,
            batch_row_select,
            inference_view,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
