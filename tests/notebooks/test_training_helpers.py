from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from batgrad.ml.loggers import StdoutRunLogger
from batgrad.ml.objective import ObjectiveTrace
from batgrad.ml.validation import RolloutExample, ValidationResult
from notebooks._support import training_helpers
from notebooks._support.training_helpers import (
    InteractiveTrainingSession,
    TrainingPlotRecord,
    build_training_widget,
)
from tests.ml.conftest import RecordingModel, make_batch, make_config, make_index, make_store


class _Dataset:
    def __init__(self, batches):
        self.batches = batches

    def set_epoch(self, _epoch_idx: int) -> None:
        pass

    def steps_per_epoch(self, _epoch_idx: int = 0) -> int:
        return len(self.batches)


class _Loader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset.batches)


def test_training_widget_includes_standard_traces_and_mask_boundaries() -> None:
    config = make_config(batch_size=2, seq_len=10, suffix_steps=3, roll_forward_steps=4)
    batch = make_batch(config)
    trace = ObjectiveTrace(
        batch.inputs,
        batch.targets,
        batch.targets.clone(),
        batch.mask,
        (7, 10, 13),
        10,
        4,
    )
    record = TrainingPlotRecord("train", 5, trace, batch.state.group_keys)
    session = SimpleNamespace(config=config)

    widget = build_training_widget(session, record, 0, rescale=False)

    assert {trace.name for trace in widget._fig.data} == {"input", "target", "prediction"}
    assert {annotation.text for annotation in widget._fig.layout.annotations} >= {"mask_pred"}
    assert widget._fig.layout.title.text == "train | step=5"


def test_interactive_session_executes_steps_and_logs_each_one() -> None:
    config = make_config(batch_size=2, seq_len=10, suffix_steps=3, roll_forward_steps=4)
    config = replace(config, train=replace(config.train, log_every_steps=100))
    batch = make_batch(config)
    dataset = _Dataset([batch])
    loader = _Loader(dataset)
    model = RecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    session = InteractiveTrainingSession(
        config=config,
        device=torch.device("cpu"),
        store=make_store(),
        index=make_index(),
        train_loader=loader,
        val_loader=loader,
        train_dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        run_logger=StdoutRunLogger(),
        max_steps=2,
    )

    session.train_steps(2)

    assert session.step == 2
    assert session.latest_train is not None
    assert session.latest_batch is batch
    assert session.latest_step_result is not None
    assert session.latest_step_result.metrics is not None
    assert session.latest_step_result.metrics.loss.device.type == "cpu"
    assert session.latest_step_result.metrics.loss.grad_fn is None
    assert session.latest_train.trace.roll_forward_steps == 4
    assert session.log_lines
    assert any("Running first training step" in line for line in session.log_lines)
    assert sum("train/loss_ce=" in line for line in session.log_lines or ()) == 2


def test_capture_logs_reports_records_incrementally() -> None:
    lines = []
    updates = []

    with training_helpers._capture_logs(lines, updates.append):
        training_helpers.train_module.logger.info("Preparing test session")

    assert any("Preparing test session" in line for line in lines)
    assert updates == lines


def test_validation_exposes_teacher_forced_batches_and_rollouts(monkeypatch) -> None:
    config = make_config(batch_size=2, seq_len=10, suffix_steps=3)
    batch = make_batch(config)
    dataset = _Dataset([batch])
    loader = _Loader(dataset)
    model = RecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trace = ObjectiveTrace(
        batch.inputs,
        batch.targets,
        batch.targets.clone(),
        batch.mask,
        (7,),
        10,
        0,
    )
    rollout = RolloutExample(
        inputs=batch.inputs[0],
        context_prediction=batch.targets[0, :6],
        prediction=batch.targets[0, 5:],
        target=batch.targets[0],
        target_start=5,
        match={"cell id": "cell-b", "cycle index": 2, "protocol": "cycling"},
        anchor=14,
    )

    def fake_validate(*_args, trace_callback=None, **_kwargs):
        assert trace_callback is not None
        trace_callback(trace)
        trace_callback(trace)
        return ValidationResult(rollout_examples=(rollout,))

    monkeypatch.setattr(training_helpers, "validate", fake_validate)
    session = InteractiveTrainingSession(
        config=config,
        device=torch.device("cpu"),
        store=make_store(),
        index=make_index(),
        train_loader=loader,
        val_loader=loader,
        train_dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        run_logger=StdoutRunLogger(),
        max_steps=2,
    )

    session.validate_now()

    assert session.record_labels() == (
        "Teacher-forced batch 1",
        "Teacher-forced batch 2",
        "Rollout 1: cell-b cycle=2 anchor=14",
    )
    rollout_record = session.record(session.record_labels()[-1])
    assert rollout_record is not None
    assert rollout_record.boundary_label == "rollout_pred"
    assert rollout_record.trace.inputs.shape[0] == 1
    assert rollout_record.trace.predictions.shape == rollout_record.trace.targets.shape
    assert torch.equal(rollout_record.trace.predictions[0, :5], rollout.context_prediction[:5])
