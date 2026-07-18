# ruff: noqa: ANN001, ANN202, B018, I002, INP001, PLR1711, S603, S607

import marimo

__generated_with = "0.23.14"
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
            and (root / "notebooks" / "_support" / "inference_helpers.py").is_file()
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
        # NOTE: Molab has no NVCC, so use the published Mamba wheel.
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--editable",
                str(local_root),
                "--group",
                "ml",
                "torch>=2.10,<2.11",
                "mamba-ssm==2.3.2.post1",
            ],
            check=True,
            cwd=local_root,
            env={**os.environ, "MAMBA_SKIP_CUDA_BUILD": "TRUE"},
        )

    project_root = local_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    os.environ.setdefault("DATA_ROOT", "/marimo/data")

    import marimo as mo
    import polars as pl

    from batgrad.data.datasets.registry import dataset_ids
    from batgrad.logging import configure_logging
    from batgrad.ml.data.config import ValidationConfig
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.data.loader import create_index
    from batgrad.ml.data.preview import (
        available_protocols,
        default_validation_group_by,
        load_manifest_preview,
        validation_group_options,
    )
    from batgrad.ml.inference import (
        available_devices,
        evaluate_checkpoints,
        resolve_device,
    )
    from batgrad.notebook_helpers import (
        make_selectable_table,
        manifest_commit_lines,
        open_local_store_status,
        parse_manifest_commits,
    )
    from batgrad.viz.ml import inference_group_label
    from notebooks._support.inference_helpers import (
        checkpoint_discovery_status,
        checkpoint_frame,
        inference_view as render_inference_view,
        make_checkpoint_table,
        make_inference_request,
        make_inference_submission,
        render_batch_result,
        selected_checkpoints_from_table,
    )
    from notebooks._support.logging_helpers import capture_log_lines, log_view
    from notebooks._support.ml_data_helpers import (
        discover_normalized_manifest_status,
        selected_index_rows,
    )
    from scripts.hf_assets import CHECKPOINTS, download_checkpoints, download_datasets

    configure_logging(level="INFO")


@app.cell
def _():
    download_examples = mo.ui.run_button(
        label="Download example datasets and checkpoint (~7.2 GiB)"
    )
    store_root = mo.ui.text(
        value=os.getenv("DATA_ROOT"),
        label="Store root",
    )
    return download_examples, store_root


@app.cell
def _(download_examples, store_root):
    downloaded_assets = None
    if download_examples.value:
        _log_lines = []
        with mo.status.spinner(
            title="Downloading example datasets and checkpoint"
        ) as _spinner, capture_log_lines(
            _log_lines,
            lambda line: _spinner.update(subtitle=line),
        ):
            _data_root = download_datasets(dataset_ids(), store_root.value)
            _outputs_root = download_checkpoints(CHECKPOINTS, "outputs")
            downloaded_assets = (_data_root, _outputs_root)
    return (downloaded_assets,)


@app.cell
def _(downloaded_assets, store_root):
    _ = downloaded_assets
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
    return index_error, index_frame, ml_index


@app.cell
def _(index_error, index_frame):
    index_table = (
        make_selectable_table(index_frame) if index_error is None and index_frame.height else None
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
def _():
    checkpoint_root = mo.ui.text(value=".", label="Checkpoint search root")
    return (checkpoint_root,)


@app.cell
def _(checkpoint_root, downloaded_assets):
    _ = downloaded_assets
    checkpoints, checkpoint_error = checkpoint_discovery_status(checkpoint_root.value)
    checkpoint_frame_ = checkpoint_frame(checkpoints)
    checkpoint_table = make_checkpoint_table(checkpoint_frame_) if checkpoints else None
    _devices = available_devices()
    device_select = mo.ui.dropdown(
        options=_devices,
        value=_devices[1] if len(_devices) > 1 else _devices[0],
        label="Device",
    )
    return checkpoint_error, checkpoint_frame_, checkpoint_table, device_select


@app.cell
def _(checkpoint_frame_, checkpoint_table):
    selected_checkpoints = (
        selected_checkpoints_from_table(checkpoint_table.value, checkpoint_frame_)
        if checkpoint_table is not None
        else ()
    )
    return (selected_checkpoints,)


@app.cell
def _():
    get_inference_request, set_inference_request = mo.state(None)
    get_inference_submission_error, set_inference_submission_error = mo.state(None)
    get_inference_result, set_inference_result = mo.state(None)
    get_inference_error, set_inference_error = mo.state(None)
    get_inference_logs, set_inference_logs = mo.state(())
    return (
        get_inference_error,
        get_inference_logs,
        get_inference_request,
        get_inference_result,
        get_inference_submission_error,
        set_inference_request,
        set_inference_result,
        set_inference_submission_error,
        set_inference_error,
        set_inference_logs,
    )


@app.cell
def _(
    device_select,
    selected_checkpoints,
    selected_index_frame,
    set_inference_error,
    set_inference_logs,
    set_inference_request,
    set_inference_result,
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
        set_inference_result(None)
        set_inference_error(None)
        set_inference_logs(())
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

    run_inference = mo.ui.button(
        value=0,
        on_click=commit_inference,
        label="Run inference",
        disabled=not selected_checkpoints,
    )
    return masked_suffix_steps, rollout_steps, run_inference


@app.cell
def _(get_inference_result):
    result = get_inference_result()
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
def _(batch_result_view, get_inference_error, get_inference_submission_error):
    inference_view = render_inference_view(
        submission_error=get_inference_submission_error(),
        inference_error=get_inference_error(),
        result_view=batch_result_view,
    )
    return (inference_view,)


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
    selected_index,
    store_error,
    store_root,
    validation_fraction,
    validation_seed,
):
    _ = selected_index
    messages = []
    if store_error is not None:
        messages.append(mo.callout(store_error, kind="danger"))
    elif manifest_error is not None:
        messages.append(mo.callout(manifest_error, kind="warn"))
    mo.vstack(
        [
            mo.md("# ML Inference"),
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
    batch_row_select,
    checkpoint_error,
    checkpoint_root,
    checkpoint_table,
    device_select,
    get_inference_request,
    inference_view,
    masked_suffix_steps,
    rollout_steps,
    run_inference,
):
    inference_request_ready = get_inference_request()
    checkpoint_view = (
        mo.callout(checkpoint_error, kind="warn")
        if checkpoint_error is not None
        else checkpoint_table
    )
    items = [
        mo.md("## Batch inference"),
        mo.hstack([checkpoint_root, device_select], justify="start"),
        checkpoint_view,
        mo.hstack([masked_suffix_steps, rollout_steps], justify="start"),
        run_inference,
    ]
    if inference_request_ready is None:
        items.extend((batch_row_select, inference_view))
    mo.vstack(items)
    return (inference_request_ready,)


@app.cell
def _(
    inference_request_ready,
    set_inference_error,
    set_inference_logs,
    set_inference_request,
    set_inference_result,
):
    if inference_request_ready is not None:
        _log_lines = []
        try:
            with mo.status.spinner(
                title="Running inference"
            ) as _spinner, capture_log_lines(
                _log_lines,
                lambda line: _spinner.update(subtitle=line),
            ):
                _submission = inference_request_ready.submission
                _device = resolve_device(_submission.device)
                _result = evaluate_checkpoints(
                    inference_request_ready.store,
                    inference_request_ready.selected_index_frame,
                    _submission.checkpoints,
                    device=_device,
                    suffix_steps=_submission.masked_suffix_steps,
                    rollout_steps=_submission.rollout_steps,
                )
        except (
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
            NotImplementedError,
        ) as exc:
            set_inference_error(str(exc))
        else:
            set_inference_result(_result)
            set_inference_error(None)
        finally:
            set_inference_logs(tuple(_log_lines))
            set_inference_request(None)
    return


@app.cell
def _(get_inference_logs):
    inference_log = log_view(
        get_inference_logs(),
        title="Inference log",
        empty="No inference run yet",
    )
    inference_log
    return


if __name__ == "__main__":
    app.run()
