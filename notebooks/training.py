# ruff: noqa: ANN001, ANN202, C901, FBT003, I002, INP001, PLR0915, PLR1711

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo

    from batgrad.ml.inference import available_devices, resolve_device
    from notebooks._support.training_helpers import (
        TrainingPlotRequest,
        build_training_plot_display,
        close_training_plot,
        create_training_session,
    )
    from notebooks._support.logging_helpers import log_view


@app.cell
def _():
    config_paths = tuple(str(path) for path in sorted(Path("configs").rglob("*.json")))
    default_config = "configs/ml_dry_run_cpu.json"
    config_options = config_paths or (default_config,)
    config_path = mo.ui.dropdown(
        options=config_options,
        value=default_config if default_config in config_options else config_options[0],
        label="Experiment config",
    )
    devices = available_devices()
    device_select = mo.ui.dropdown(options=devices, value="cpu", label="Device")
    return config_path, device_select


@app.cell
def _():
    get_session, set_session = mo.state(None)
    get_training_error, set_training_error = mo.state(None)
    get_plot_request, set_plot_request = mo.state(None)
    get_plot_display, set_plot_display = mo.state(None)
    get_operation, set_operation = mo.state(None)
    get_batch_index, set_batch_index = mo.state(0)
    get_rescale, set_rescale = mo.state(True)
    get_steps_per_click, set_steps_per_click = mo.state(1)
    get_validation_label, set_validation_label = mo.state(None)
    get_version, set_version = mo.state(0)
    return (
        get_batch_index,
        get_operation,
        get_plot_display,
        get_plot_request,
        get_rescale,
        get_session,
        get_steps_per_click,
        get_training_error,
        get_validation_label,
        get_version,
        set_batch_index,
        set_operation,
        set_plot_display,
        set_plot_request,
        set_rescale,
        set_session,
        set_steps_per_click,
        set_training_error,
        set_validation_label,
        set_version,
    )


@app.cell
def _(
    get_plot_display,
    set_operation,
    set_plot_display,
    set_plot_request,
    set_session,
    set_training_error,
):
    def prepare_initialize(value):
        next_value = int(value or 0) + 1
        close_training_plot(get_plot_display())
        set_plot_display(None)
        set_plot_request(None)
        set_session(None)
        set_training_error(None)
        set_operation(("initialize", next_value))
        return next_value

    initialize_training = mo.ui.button(
        value=0,
        label="Initialize / reset",
        on_click=prepare_initialize,
    )
    return (initialize_training,)


@app.cell
def _(get_session, get_validation_label, get_version, set_validation_label):
    get_version()
    current = get_session()
    validation_record = None
    if (
        current is not None
        and current.latest_record is not None
        and current.latest_record.phase.startswith("validation")
    ):
        labels = current.record_labels()
        if labels:
            selected_validation = get_validation_label()
            selected_validation = (
                selected_validation if selected_validation in labels else labels[0]
            )
            validation_record = mo.ui.dropdown(
                options=labels,
                value=selected_validation,
                label="Validation result",
                on_change=set_validation_label,
            )
    return (validation_record,)


@app.cell
def _(
    get_batch_index,
    get_plot_display,
    get_rescale,
    get_session,
    get_steps_per_click,
    get_version,
    set_batch_index,
    set_operation,
    set_plot_display,
    set_plot_request,
    set_rescale,
    set_steps_per_click,
    set_training_error,
    validation_record,
):
    get_version()
    session = get_session()
    selected_label = None if validation_record is None else str(validation_record.value)
    record = None if session is None else session.record(selected_label)
    max_batch_index = max(
        (() if record is None else (int(record.trace.inputs.shape[0]) - 1,)),
        default=0,
    )
    selected_batch_index = min(max(0, int(get_batch_index())), max_batch_index)
    batch_index = mo.ui.dropdown(
        options=tuple(range(max_batch_index + 1)),
        value=selected_batch_index,
        label="Batch index",
        on_change=lambda value: set_batch_index(int(value)),
    )
    steps_per_click = mo.ui.number(
        value=int(get_steps_per_click()),
        start=1,
        step=1,
        label="Training steps",
        on_change=lambda value: set_steps_per_click(int(value)),
    )
    rescale = mo.ui.checkbox(
        value=bool(get_rescale()),
        label="Rescale to physical units",
        on_change=lambda value: set_rescale(bool(value)),
    )

    def clear_plot():
        close_training_plot(get_plot_display())
        set_plot_display(None)
        set_plot_request(None)

    def prepare_run(kind):
        def run(value):
            next_value = int(value or 0) + 1
            clear_plot()
            set_training_error(None)
            set_operation((kind, next_value))
            return next_value

        return run

    def plot_latest(value):
        next_value = int(value or 0) + 1
        current = get_session()
        selected = None if current is None else current.record(selected_label)
        if current is None or selected is None:
            set_training_error("No training or validation batch is available to plot")
            return next_value
        set_plot_request(
            TrainingPlotRequest(
                next_value,
                selected_label,
                int(batch_index.value),
                bool(rescale.value),
            )
        )
        set_training_error(None)
        return next_value

    train_steps = mo.ui.button(
        value=0,
        label="Train",
        on_click=prepare_run("train"),
    )
    validate = mo.ui.button(
        value=0,
        label="Validate now",
        on_click=prepare_run("validate"),
    )
    plot = mo.ui.button(value=0, on_click=plot_latest, label="Plot latest")
    return batch_index, plot, rescale, steps_per_click, train_steps, validate


@app.cell
def _(
    get_plot_display,
    get_plot_request,
    get_session,
    set_plot_display,
    set_training_error,
):
    _request = get_plot_request()
    _previous_display = get_plot_display()
    if _request is None:
        close_training_plot(_previous_display)
        set_plot_display(None)
    else:
        _current = get_session()
        if _current is None:
            close_training_plot(_previous_display)
            set_plot_display(None)
            set_training_error("Initialize training before plotting")
        else:
            try:
                _next_display = build_training_plot_display(_current, _request)
            except (RuntimeError, TypeError, ValueError) as exc:
                close_training_plot(_previous_display)
                set_plot_display(None)
                set_training_error(str(exc))
            else:
                close_training_plot(_previous_display)
                set_plot_display(_next_display)
                set_training_error(None)
    return


@app.cell
def _(get_plot_display, get_session, get_training_error, get_version):
    get_version()
    training_session = get_session()
    config = None if training_session is None else training_session.config
    device = None if training_session is None else training_session.device
    model = None if training_session is None else training_session.model
    optimizer = None if training_session is None else training_session.optimizer
    scheduler = None if training_session is None else training_session.scheduler
    train_dataloader = None if training_session is None else training_session.train_loader
    validation_dataloader = None if training_session is None else training_session.val_loader
    dataset_index = None if training_session is None else training_session.index
    latest_batch = None if training_session is None else training_session.latest_batch
    latest_step_result = None if training_session is None else training_session.latest_step_result
    latest_metrics = None if latest_step_result is None else latest_step_result.metrics
    latest_loss = None if latest_metrics is None else latest_metrics.loss
    latest_record = None if training_session is None else training_session.record()
    validation_result = None if training_session is None else training_session.validation_result
    latest_trace = (
        None
        if training_session is None or training_session.latest_train is None
        else training_session.latest_train.trace
    )
    training_error = get_training_error()
    plot_display = get_plot_display()
    plot_view = None if plot_display is None else plot_display.view
    session_log = log_view(
        () if training_session is None else (training_session.log_lines or ()),
        title="Session log",
        empty="Not initialized",
    )
    return (
        config,
        dataset_index,
        device,
        latest_batch,
        latest_loss,
        latest_metrics,
        latest_record,
        latest_step_result,
        latest_trace,
        model,
        optimizer,
        plot_view,
        scheduler,
        session_log,
        train_dataloader,
        training_error,
        training_session,
        validation_dataloader,
        validation_result,
    )


@app.cell
def _(
    batch_index,
    config,
    config_path,
    dataset_index,
    device,
    device_select,
    get_operation,
    initialize_training,
    latest_batch,
    latest_loss,
    latest_metrics,
    latest_record,
    latest_step_result,
    latest_trace,
    model,
    optimizer,
    plot,
    plot_view,
    rescale,
    scheduler,
    steps_per_click,
    train_dataloader,
    train_steps,
    training_error,
    training_session,
    validate,
    validation_dataloader,
    validation_record,
    validation_result,
):
    operation_ready = get_operation()
    _ = (
        config,
        dataset_index,
        device,
        latest_batch,
        latest_loss,
        latest_metrics,
        latest_record,
        latest_step_result,
        latest_trace,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        training_session,
        validation_dataloader,
        validation_result,
    )
    items = [
        mo.md("# Interactive ML Training"),
        mo.hstack([config_path, device_select, initialize_training], justify="start"),
    ]
    if training_error is not None:
        items.append(mo.callout(training_error, kind="danger"))
    items.extend(
        [
            mo.hstack([steps_per_click, train_steps, validate], justify="start"),
        ]
    )
    if operation_ready is None:
        items.append(mo.hstack([batch_index, rescale, plot], justify="start"))
        if validation_record is not None:
            items.append(validation_record)
        if plot_view is not None:
            items.append(plot_view)
    mo.vstack(items)
    return (operation_ready,)


@app.cell
def _(
    config_path,
    device_select,
    get_session,
    operation_ready,
    set_operation,
    set_session,
    set_training_error,
    set_version,
    steps_per_click,
):
    if operation_ready is not None:
        _operation_kind, _request_id = operation_ready
        _ = _request_id
        try:
            if _operation_kind == "initialize":
                with mo.status.spinner(
                    title="Initializing training session",
                ) as _spinner:
                    _device = resolve_device(str(device_select.value))
                    _session = create_training_session(
                        str(config_path.value),
                        _device,
                        progress_callback=lambda message: _spinner.update(subtitle=message),
                    )
                set_session(_session)
            elif _operation_kind == "train":
                _current = get_session()
                if _current is None:
                    raise RuntimeError("Initialize training before stepping")
                _requested_steps = int(steps_per_click.value)
                _available_steps = max(0, _current.max_steps - _current.step)
                _total_steps = min(_requested_steps, _available_steps)
                if _total_steps <= 0:
                    raise RuntimeError(f"Training complete at step {_current.step}")
                with mo.status.progress_bar(
                    total=_total_steps,
                    title="Training model",
                    completion_title="Training request complete",
                    remove_on_exit=True,
                ) as _progress:
                    for _step_index in range(_total_steps):
                        _ = _step_index
                        _current.train_steps(
                            1,
                            progress_callback=lambda line: _progress.update(
                                increment=0,
                                subtitle=line,
                            ),
                        )
                        _progress.update()
            elif _operation_kind == "validate":
                _current = get_session()
                if _current is None:
                    raise RuntimeError("Initialize training before validation")
                with mo.status.spinner(
                    title="Validating model",
                ) as _spinner:
                    _current.validate_now(
                        progress_callback=lambda line: _spinner.update(subtitle=line)
                    )
            else:
                raise ValueError(f"Unknown training operation: {_operation_kind}")
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            set_training_error(str(exc))
        else:
            set_training_error(None)
        finally:
            set_operation(None)
            set_version(lambda version: version + 1)
    return


@app.cell
def _(session_log):
    session_log
    return


if __name__ == "__main__":
    app.run()
