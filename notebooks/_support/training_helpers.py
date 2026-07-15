from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from batgrad.logging import LogFormatter, capture_output
from batgrad.ml import train as train_module
from batgrad.ml.config import (
    ExperimentConfig,
    load_experiment_config,
    resolve_store_root,
    resolved_validation_masked_suffix,
)
from batgrad.ml.data.scaling import inverse_scale_tensor
from batgrad.ml.distributed import DistributedContext
from batgrad.ml.experiment import amp_enabled, scaling_rules
from batgrad.ml.inference import InferencePrediction, InferenceResult
from batgrad.ml.loggers import StdoutRunLogger
from batgrad.ml.metrics import loss_metrics_to_cpu
from batgrad.ml.nn import build_model
from batgrad.ml.objective import ObjectiveTrace
from batgrad.ml.validation import RolloutExample, ValidationResult, validate
from batgrad.notebook_helpers import wrap_anywidget_blocks
from batgrad.storage.local import LocalDataProcessingStore
from batgrad.viz.ml import (
    _rollout_target_time_axis,
    build_inference_widget,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from batgrad.ml.data.batch import Batch
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.data.loader import MlDataIterable
    from batgrad.ml.nn import MambaCarryState
    from batgrad.storage.store import DatasetStoreReader
    from batgrad.viz.widgets.plotly_trace_resampler import PlotlyTraceResampler


@dataclass(frozen=True, slots=True)
class TrainingPlotRecord:
    phase: str
    step: int
    trace: ObjectiveTrace
    group_keys: tuple[tuple[object, ...], ...]
    label: str = ""
    boundary_label: str = "mask_pred"


@dataclass(frozen=True, slots=True)
class TrainingPlotRequest:
    submit_id: int
    record_label: str | None
    batch_index: int
    rescale: bool


@dataclass(frozen=True, slots=True)
class TrainingPlotDisplay:
    widget: PlotlyTraceResampler
    view: object


@dataclass(frozen=True, slots=True)
class TrainingRunState:
    total_steps: int = 0
    completed_steps: int = 0
    status: str = "idle"

    @property
    def active(self) -> bool:
        return self.status == "running"


@dataclass(slots=True)
class InteractiveTrainingSession:
    config: ExperimentConfig
    device: torch.device
    store: DatasetStoreReader
    index: MlDatasetIndex
    train_loader: Iterable[Batch]
    val_loader: Iterable[Batch]
    train_dataset: MlDataIterable
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    run_logger: StdoutRunLogger
    max_steps: int
    step: int = 0
    epoch_idx: int = 0
    epoch_step: int = 0
    train_iterator: Iterator[Batch] | None = None
    carried_mamba_states: dict[str, MambaCarryState] | None = None
    carried_group_idx: int | None = None
    carried_step_idx: int | None = None
    latest_train: TrainingPlotRecord | None = None
    latest_validation: TrainingPlotRecord | None = None
    validation_records: tuple[TrainingPlotRecord, ...] = ()
    latest_record: TrainingPlotRecord | None = None
    latest_records: tuple[TrainingPlotRecord, ...] = ()
    latest_batch: Batch | None = None
    latest_step_result: train_module.OptimizerStepResult | None = None
    validation_result: ValidationResult | None = None
    log_lines: list[str] | None = None
    clip_trigger_count: int = 0
    clip_observed_count: int = 0
    log_token_count: int = 0
    log_time: float = 0.0
    first_train_step: bool = True

    def __post_init__(self) -> None:
        if self.log_lines is None:
            self.log_lines = []
        if self.log_time == 0.0:
            self.log_time = time.perf_counter()

    def train_steps(self, count: int) -> None:
        if self.log_lines is None:
            self.log_lines = []
        with _capture_logs(self.log_lines):
            self._train_steps(count)

    def _train_steps(self, count: int) -> None:
        for _ in range(max(0, int(count))):
            if self.step >= self.max_steps:
                train_module.logger.info("Training complete at step %d", self.step)
                return
            batch = self._next_batch()
            steps_per_epoch = train_module._local_steps_per_epoch(
                self.train_dataset,
                self.epoch_idx,
                _single_process_context(self.device),
            )
            next_step = self.step + 1
            self.log_token_count += train_module._model_compute_token_count(self.config, batch)
            if self.first_train_step:
                train_module.logger.info("Running first training step")
            traces: list[ObjectiveTrace] = []
            initial_states = train_module._initial_mamba_states_for_batch(
                batch,
                self.carried_mamba_states,
                self.carried_group_idx,
                self.carried_step_idx,
            )
            result = train_module.optimize_batch(
                self.config,
                self.model,
                batch,
                self.device,
                self.optimizer,
                self.scheduler,
                self.scaler,
                collect_metrics=True,
                initial_mamba_states=initial_states,
                trace_callback=traces.append,
            )
            self.latest_batch = batch
            retained_metrics = loss_metrics_to_cpu(result.metrics)
            if retained_metrics is None:
                raise RuntimeError("optimizer step did not return loss metrics")
            self.latest_step_result = replace(
                result,
                metrics=retained_metrics,
                total_grad_norm=result.total_grad_norm.detach().cpu(),
            )
            self.step = next_step
            self.carried_mamba_states = result.mamba_states
            self.carried_group_idx = batch.state.stateful_group_idx
            self.carried_step_idx = batch.state.stateful_step_idx
            if traces:
                self.latest_train = TrainingPlotRecord(
                    "train",
                    self.step,
                    traces[-1],
                    batch.state.group_keys,
                    f"Training step {self.step}",
                )
                self.latest_record = self.latest_train
                self.latest_records = (self.latest_train,)
            self.clip_observed_count += 1
            if float(result.total_grad_norm.detach().cpu()) > self.config.train.grad_clip_norm:
                self.clip_trigger_count += 1
            if self.first_train_step:
                train_module.logger.info("First training step complete")
                self.first_train_step = False
            now = time.perf_counter()
            elapsed = max(now - self.log_time, 1e-9)
            tokens_per_sec = self.log_token_count / elapsed
            self.log_time = now
            self.log_token_count = 0
            epoch, epoch_pct = train_module._epoch_progress(
                self.epoch_idx, self.epoch_step, steps_per_epoch
            )
            self.run_logger.log_metrics(
                self.step,
                {
                    "train/loss_ce": float(result.metrics.loss.detach().cpu()),
                    "train/lr": float(self.scheduler.get_last_lr()[0]),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/epoch": epoch,
                    "train/epoch_pct": epoch_pct,
                    "train/grad_norm/model": float(result.total_grad_norm.detach().cpu()),
                    "train/grad_clip/trigger_fraction": self.clip_trigger_count
                    / self.clip_observed_count,
                    **result.grad_metrics,
                },
                epoch=epoch,
                epoch_pct=epoch_pct,
            )

    def validate_now(self) -> None:
        if self.log_lines is None:
            self.log_lines = []
        with _capture_logs(self.log_lines):
            self._validate_now()

    def _validate_now(self) -> None:
        traces: list[ObjectiveTrace] = []
        self.latest_validation = None
        self.validation_records = ()
        self.validation_result = validate(
            self.config,
            self.model,
            self.val_loader,
            self.index,
            self.store,
            self.device,
            trace_callback=traces.append,
        )
        teacher_forced_records = tuple(
            TrainingPlotRecord(
                "validation teacher-forced",
                self.step,
                trace,
                tuple(
                    ("validation", batch_idx, lane, self.config.data.protocols[0])
                    for lane in range(int(trace.inputs.shape[0]))
                ),
                f"Teacher-forced batch {batch_idx + 1}",
            )
            for batch_idx, trace in enumerate(traces)
        )
        rollout_records = tuple(
            _rollout_plot_record(self.config, self.step, rollout, rollout_idx)
            for rollout_idx, rollout in enumerate(self.validation_result.rollout_examples)
        )
        self.validation_records = (*teacher_forced_records, *rollout_records)
        self.latest_validation = self.validation_records[0] if self.validation_records else None
        self.latest_record = self.latest_validation
        self.latest_records = self.validation_records
        metrics = train_module._validation_metric_payload(self.config, self.validation_result)
        if metrics:
            steps_per_epoch = train_module._local_steps_per_epoch(
                self.train_dataset,
                self.epoch_idx,
                _single_process_context(self.device),
            )
            epoch, epoch_pct = train_module._epoch_progress(
                self.epoch_idx, max(1, self.epoch_step), steps_per_epoch
            )
            self.run_logger.log_metrics(
                self.step,
                metrics,
                epoch=epoch,
                epoch_pct=epoch_pct,
            )

    def record(self, label: str | None = None) -> TrainingPlotRecord | None:
        if label is not None:
            return next((record for record in self.latest_records if record.label == label), None)
        return self.latest_record

    def record_labels(self) -> tuple[str, ...]:
        return tuple(record.label for record in self.latest_records)

    def _next_batch(self) -> Batch:
        if self.train_iterator is None:
            self.train_dataset.set_epoch(self.epoch_idx)
            self.train_iterator = iter(self.train_loader)
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.epoch_idx += 1
            self.epoch_step = 0
            self.train_dataset.set_epoch(self.epoch_idx)
            self.train_iterator = iter(self.train_loader)
            try:
                batch = next(self.train_iterator)
            except StopIteration as exc:
                raise ValueError("training split produced no batches") from exc
            else:
                self.epoch_step = 1
                return batch
        else:
            self.epoch_step += 1
            return batch


def create_training_session(
    config_path: str | Path,
    device: torch.device,
) -> InteractiveTrainingSession:
    config = load_experiment_config(config_path)
    config = replace(
        config,
        train=replace(config.train, log_every_steps=1),
        validation=replace(config.validation, log_rollout_plots=True),
        run=replace(
            config.run,
            device=str(device),
            output_dir=None,
            compile_model=False,
        ),
    )
    log_lines: list[str] = []
    with _capture_logs(log_lines):
        torch.manual_seed(config.run.seed)
        store = LocalDataProcessingStore(resolve_store_root(config.data.store_root))
        train_module.logger.info("Creating data loaders")
        train_loader, val_loader, train_dataset, index = train_module._create_loaders(config, store)
        train_module.logger.info("Data loaders ready")
        distributed = _single_process_context(device)
        max_steps = config.train.max_steps or train_module._max_steps_for_epochs(
            train_dataset, config.train.epochs, distributed
        )
        train_module.logger.info("Creating model on %s", device)
        model = build_model(config, device)
        train_module.logger.info("Model ready on %s", device)
        train_module._init_model_from_checkpoint(config, model, device)
        optimizer = train_module.build_optimizer(config, model)
        scheduler = train_module._build_scheduler(config, optimizer, max_steps)
        train_module.logger.info("Initializing run logger")
        run_logger = StdoutRunLogger()
        train_module.logger.info("Run logger ready")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled(config, device))
        train_module.logger.info("Starting training loop")
    return InteractiveTrainingSession(
        config=config,
        device=device,
        store=store,
        index=index,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        run_logger=run_logger,
        max_steps=max_steps,
        log_lines=log_lines,
    )


def build_training_widget(
    session: InteractiveTrainingSession,
    record: TrainingPlotRecord,
    batch_index: int,
    *,
    rescale: bool,
) -> PlotlyTraceResampler:
    trace = record.trace
    config = session.config if rescale else _identity_scaling_config(session.config)
    prediction = InferencePrediction(
        checkpoint_alias=f"{record.phase} step {record.step}",
        checkpoint_path="",
        suffix_steps=session.config.train.masked_suffix.suffix_steps
        if session.config.train.masked_suffix.enabled
        else 0,
        context_predictions=trace.predictions[:, :0],
        predictions=trace.predictions,
        metrics=None,
        target_start=0,
    )
    group_keys = record.group_keys
    if len(group_keys) != int(trace.inputs.shape[0]):
        group_keys = tuple(
            (record.phase, lane, session.config.data.protocols[0])
            for lane in range(int(trace.inputs.shape[0]))
        )
    result = InferenceResult(
        config,
        trace.inputs,
        trace.targets,
        (prediction,),
        trace.context_len,
        trace.roll_forward_steps,
        group_keys,
        None,
    )
    lane = min(max(0, int(batch_index)), int(trace.inputs.shape[0]) - 1)
    widget = build_inference_widget(
        result,
        lane,
        include_inputs=True,
        index_axis=not rescale,
        standard_role_labels=True,
    )
    suffix = (
        session.config.train.masked_suffix
        if record.phase == "train"
        else resolved_validation_masked_suffix(session.config)
    )
    suffix_steps = suffix.suffix_steps if suffix.enabled else 0
    widget._fig.update_layout(
        title=(
            f"{record.phase} | step={record.step} | context={trace.context_len} | "
            f"masked_suffix_steps={suffix_steps} | "
            f"effective_seq_len={int(trace.inputs.shape[1])} | "
            f"roll_forward_steps={trace.roll_forward_steps}"
        )
    )
    if rescale:
        display_inputs = inverse_scale_tensor(
            trace.inputs, session.config.data.input_columns, scaling_rules(session.config)
        )
        x_values = _rollout_target_time_axis(config, display_inputs[lane])
    else:
        x_values = [float(idx) for idx in range(int(trace.inputs.shape[1]))]
    for boundary in trace.mask_boundaries:
        if not x_values:
            break
        x_value = x_values[min(max(0, boundary), len(x_values) - 1)]
        widget._fig.add_vline(
            x=x_value,
            line_dash="dot",
            line_color="orange",
            row="all",
            col=1,
        )
        widget._fig.add_annotation(
            x=x_value,
            y=1.0,
            xref="x",
            yref="paper",
            text=record.boundary_label,
            showarrow=False,
            yshift=8,
        )
    return widget


def build_training_plot_display(
    session: InteractiveTrainingSession,
    request: TrainingPlotRequest,
) -> TrainingPlotDisplay:
    record = session.record(request.record_label)
    if record is None:
        raise ValueError("No training or validation batch is available to plot")
    widget = build_training_widget(
        session,
        record,
        request.batch_index,
        rescale=request.rescale,
    )
    return TrainingPlotDisplay(widget, wrap_anywidget_blocks((widget,))[0])


def close_training_plot(display: TrainingPlotDisplay | None) -> None:
    if display is not None:
        display.widget.close()


def _rollout_plot_record(
    config: ExperimentConfig,
    step: int,
    example: RolloutExample,
    rollout_idx: int,
) -> TrainingPlotRecord:
    inputs = example.inputs.unsqueeze(0)
    targets = example.target.unsqueeze(0)
    predictions = torch.full_like(targets, float("nan"))
    context_steps = min(int(example.context_prediction.shape[0]), int(targets.shape[1]))
    predictions[:, :context_steps, :] = example.context_prediction[:context_steps].unsqueeze(0)
    rollout_end = min(
        example.target_start + int(example.prediction.shape[0]),
        int(targets.shape[1]),
    )
    rollout_steps = max(0, rollout_end - example.target_start)
    predictions[:, example.target_start : rollout_end, :] = example.prediction[
        :rollout_steps
    ].unsqueeze(0)
    protocol = example.match.get("protocol", config.data.protocols[0])
    group_key = (
        example.match.get("dataset id", "validation"),
        example.match.get("cell id", f"rollout-{rollout_idx + 1}"),
        example.match.get("cycle index", example.anchor),
        protocol,
    )
    cell = example.match.get("cell id", "unknown-cell")
    cycle = example.match.get("cycle index", "unknown-cycle")
    trace = ObjectiveTrace(
        inputs=inputs,
        targets=targets,
        predictions=predictions,
        mask=torch.isfinite(targets).all(dim=-1),
        mask_boundaries=(example.target_start,),
        context_len=context_steps,
        roll_forward_steps=rollout_steps,
    )
    return TrainingPlotRecord(
        "validation rollout",
        step,
        trace,
        (group_key,),
        f"Rollout {rollout_idx + 1}: {cell} cycle={cycle} anchor={example.anchor}",
        "rollout_pred",
    )


def _identity_scaling_config(config: ExperimentConfig) -> ExperimentConfig:
    identity = tuple(
        replace(
            rule,
            input_min=rule.output_min,
            input_max=rule.output_max,
            transform="linear",
            clip=False,
        )
        for rule in config.data.scaling
    )
    return replace(config, data=replace(config.data, scaling=identity))


def _single_process_context(device: torch.device) -> DistributedContext:
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )


@contextmanager
def _capture_logs(destination: list[str]) -> Iterator[None]:
    formatter = LogFormatter(use_colors=False)
    with capture_output(logging.DEBUG) as (stdout, stderr, records):
        try:
            yield
        finally:
            destination.extend(formatter.format(record) for record in records)
            destination.extend(line for line in stdout.getvalue().splitlines() if line)
            destination.extend(line for line in stderr.getvalue().splitlines() if line)
            del destination[:-2000]
