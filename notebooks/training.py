# ruff: noqa: ANN001, ANN202, C901, FBT003, I002, INP001, PLR0915, PLR1711

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    from html import escape
    from pathlib import Path

    import marimo as mo

    from batgrad.ml.inference import available_devices, resolve_device
    from notebooks._support.training_helpers import (
        TrainingPlotRequest,
        TrainingRunState,
        build_training_plot_display,
        close_training_plot,
        create_training_session,
    )


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
    get_training_run, set_training_run = mo.state(
        TrainingRunState(),
        allow_self_loops=True,
    )
    get_batch_index, set_batch_index = mo.state(0)
    get_rescale, set_rescale = mo.state(True)
    get_steps_per_click, set_steps_per_click = mo.state(1)
    get_validation_label, set_validation_label = mo.state(None)
    get_version, set_version = mo.state(0)
    return (
        get_batch_index,
        get_plot_display,
        get_plot_request,
        get_rescale,
        get_session,
        get_steps_per_click,
        get_training_error,
        get_training_run,
        get_validation_label,
        get_version,
        set_batch_index,
        set_plot_display,
        set_plot_request,
        set_rescale,
        set_session,
        set_steps_per_click,
        set_training_error,
        set_training_run,
        set_validation_label,
        set_version,
    )


@app.cell
def _(
    config_path,
    device_select,
    set_plot_request,
    set_session,
    set_training_error,
    set_training_run,
    set_version,
):
    def initialize(value):
        next_value = int(value or 0) + 1
        try:
            device = resolve_device(str(device_select.value))
            session = create_training_session(str(config_path.value), device)
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            set_training_error(str(exc))
            return next_value
        set_plot_request(None)
        set_session(session)
        set_training_error(None)
        set_training_run(TrainingRunState())
        set_version(lambda current: current + 1)
        return next_value

    initialize_training = mo.ui.button(value=0, on_click=initialize, label="Initialize / reset")
    return (initialize_training,)


@app.cell
def _(
    get_session,
    get_training_run,
    get_validation_label,
    get_version,
    set_validation_label,
):
    get_version()
    current = get_session()
    validation_record = None
    if (
        not get_training_run().active
        and current is not None
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
    get_rescale,
    get_session,
    get_steps_per_click,
    get_training_run,
    get_version,
    set_batch_index,
    set_plot_request,
    set_rescale,
    set_steps_per_click,
    set_training_error,
    set_training_run,
    set_version,
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
        set_plot_request(None)

    def train(value):
        next_value = int(value or 0) + 1
        current = get_session()
        if current is None:
            set_training_error("Initialize training before stepping")
            return next_value
        if get_training_run().active:
            set_training_error("Training is already running")
            return next_value
        requested_steps = int(steps_per_click.value)
        available_steps = max(0, current.max_steps - current.step)
        total_steps = min(requested_steps, available_steps)
        clear_plot()
        if total_steps <= 0:
            set_training_error(f"Training complete at step {current.step}")
            return next_value
        set_training_error(None)
        set_training_run(TrainingRunState(total_steps, 0, "running"))
        return next_value

    def validate_now(value):
        next_value = int(value or 0) + 1
        current = get_session()
        if current is None:
            set_training_error("Initialize training before validation")
            return next_value
        if get_training_run().active:
            set_training_error("Stop training before validation")
            return next_value
        try:
            current.validate_now()
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            set_training_error(str(exc))
            return next_value
        clear_plot()
        set_training_error(None)
        set_version(lambda version: version + 1)
        return next_value

    def plot_latest(value):
        next_value = int(value or 0) + 1
        current = get_session()
        selected = None if current is None else current.record(selected_label)
        if current is None or selected is None:
            set_training_error("No training or validation batch is available to plot")
            return next_value
        if get_training_run().active:
            set_training_error("Stop training before plotting")
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

    def stop_training(value):
        next_value = int(value or 0) + 1
        run = get_training_run()
        if run.active:
            set_training_run(TrainingRunState(run.total_steps, run.completed_steps, "stopped"))
            set_version(lambda version: version + 1)
        return next_value

    train_steps = mo.ui.button(value=0, on_click=train, label="Train")
    stop = mo.ui.button(value=0, on_click=stop_training, label="Stop")
    validate = mo.ui.button(value=0, on_click=validate_now, label="Validate now")
    plot = mo.ui.button(value=0, on_click=plot_latest, label="Plot latest")
    return (
        batch_index,
        plot,
        rescale,
        steps_per_click,
        stop,
        train_steps,
        validate,
    )


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
def _(
    get_plot_display,
    get_session,
    get_training_error,
    get_training_run,
    get_version,
):
    get_version()
    _training_run = get_training_run()
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
    log_text = (
        "Not initialized"
        if training_session is None
        else "\n".join(training_session.log_lines or ())
    )
    session_log = mo.Html(
        '<div style="font-size: 0.875rem;">'
        '<div style="font-weight: 600; margin-bottom: 0.25rem;">Session log</div>'
        '<div style="max-height: 420px; min-height: 180px; overflow-y: auto; '
        "display: flex; flex-direction: column-reverse; border: 1px solid var(--slate-6); "
        'border-radius: 4px; background: var(--slate-1);">'
        '<pre style="flex: 0 0 auto; margin: 0; padding: 0.75rem; white-space: pre-wrap; '
        f'overflow-wrap: anywhere;">{escape(log_text)}</pre></div></div>'
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
    session_log,
    steps_per_click,
    stop,
    train_dataloader,
    train_steps,
    training_error,
    training_session,
    validate,
    validation_dataloader,
    validation_record,
    validation_result,
):
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
            mo.hstack([steps_per_click, train_steps, stop, validate], justify="start"),
        ]
    )
    items.append(mo.hstack([batch_index, rescale, plot], justify="start"))
    if validation_record is not None:
        items.append(validation_record)
    if plot_view is not None:
        items.append(plot_view)
    items.append(session_log)
    mo.vstack(items)
    return


@app.cell
def _(
    get_session,
    get_training_run,
    set_training_error,
    set_training_run,
    set_version,
):
    _run = get_training_run()
    if _run.active:
        _current = get_session()
        if _current is None:
            set_training_error("Initialize training before stepping")
            _next_run = TrainingRunState(_run.total_steps, _run.completed_steps, "error")
        else:
            try:
                _previous_step = _current.step
                _current.train_steps(1)
                _completed_steps = _run.completed_steps + int(_current.step > _previous_step)
                _status = (
                    "completed"
                    if _completed_steps >= _run.total_steps or _current.step >= _current.max_steps
                    else "running"
                )
                _next_run = TrainingRunState(
                    _run.total_steps,
                    _completed_steps,
                    _status,
                )
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                set_training_error(str(exc))
                _next_run = TrainingRunState(
                    _run.total_steps,
                    _run.completed_steps,
                    "error",
                )
        set_training_run(_next_run)
        if not _next_run.active:
            set_version(lambda version: version + 1)
    return


if __name__ == "__main__":
    app.run()
