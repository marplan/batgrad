from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.checkpoint import LoadedCheckpoint, load_checkpoint
from batgrad.ml.config import ExperimentConfig, resolved_validation_masked_suffix
from batgrad.ml.data.config import LoaderConfig, WindowConfig
from batgrad.ml.data.index import MlDatasetIndex, sort_index_frame
from batgrad.ml.data.materialization import materialize_batch_plan
from batgrad.ml.data.planning import BatchPlan, WindowRef, build_stream_plans
from batgrad.ml.experiment import scaling_rules
from batgrad.ml.metrics import LossMetrics, loss_metrics_to_cpu
from batgrad.ml.rollout import rollout_batch

if TYPE_CHECKING:
    import polars as pl

    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    alias: str
    path: str


@dataclass(frozen=True, slots=True)
class InferencePrediction:
    checkpoint_alias: str
    checkpoint_path: str
    suffix_steps: int
    predictions: torch.Tensor
    metrics: LossMetrics | None
    target_start: int


@dataclass(frozen=True, slots=True)
class InferenceResult:
    config: ExperimentConfig
    inputs: torch.Tensor
    targets: torch.Tensor
    predictions: tuple[InferencePrediction, ...]
    context_len: int
    rollout_len: int
    group_keys: tuple[tuple[object, ...], ...]
    warning: str | None


def available_devices() -> tuple[str, ...]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{idx}" for idx in range(torch.cuda.device_count()))
    return tuple(devices)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but torch.cuda.is_available() is false")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device is not available: {value}")
    return device


def discover_checkpoints(root: str | Path = ".") -> tuple[Path, ...]:
    return tuple(sorted(Path(root).glob("**/checkpoints/*.pt")))


@torch.no_grad()
def evaluate_checkpoints(  # noqa: C901
    store: DatasetStoreReader,
    selected_index_frame: pl.DataFrame,
    selections: tuple[CheckpointSelection, ...],
    *,
    device: torch.device,
    suffix_steps: tuple[int, ...],
    rollout_steps: int,
) -> InferenceResult:
    if selected_index_frame.is_empty():
        raise ValueError("Select one or more ML index rows before running inference")
    if not selections:
        raise ValueError("Select at least one checkpoint before running inference")
    if any(not selection.alias.strip() or not selection.path.strip() for selection in selections):
        raise ValueError("Checkpoint aliases and paths must not be empty")
    if rollout_steps <= 0:
        raise ValueError("Rollout steps must be > 0")
    if not suffix_steps or any(step < 0 for step in suffix_steps):
        raise ValueError("Suffix steps must contain one or more non-negative values")
    checkpoints = tuple(load_checkpoint(selection.path, device) for selection in selections)
    _validate_compatible_checkpoints(checkpoints)
    reference = checkpoints[0].config
    context_len = reference.loader.seq_len
    invalid_suffix_steps = tuple(step for step in suffix_steps if step >= context_len and step > 0)
    if invalid_suffix_steps:
        raise ValueError(
            "Enabled suffix steps must be smaller than checkpoint seq_len, "
            f"got suffix_steps={invalid_suffix_steps} seq_len={context_len}"
        )
    selected_protocols = _selected_protocols(selected_index_frame, checkpoints)
    index = MlDatasetIndex(sort_index_frame(selected_index_frame))
    loader = _inference_loader(reference, selected_protocols, selected_index_frame.height, device)
    streams = tuple(
        stream
        for protocol in selected_protocols
        for stream in build_stream_plans(index, protocol, loader)
    )
    if not streams:
        raise ValueError("No selected ML index rows match checkpoint protocols")
    if len(streams) != selected_index_frame.height:
        raise ValueError(
            "Selected rows must all match checkpoint-configured protocols; "
            f"got {len(streams)} matching streams for {selected_index_frame.height} rows"
        )
    max_rollout = min(max(0, int(stream.row_count) - context_len - 1) for stream in streams)
    if max_rollout <= 0:
        raise ValueError(
            "Selected rows are too short for checkpoint context length "
            f"seq_len={context_len}. Shortest selected row_count="
            f"{min(stream.row_count for stream in streams)}"
        )
    effective_rollout = min(rollout_steps, max_rollout)
    warning = None
    if effective_rollout < rollout_steps:
        warning = (
            f"Requested {rollout_steps:,} rollout steps, clipped to "
            f"{effective_rollout:,} by the shortest selected file."
        )
    inference_loader = replace(
        loader,
        default_window=WindowConfig(
            batch_size=len(streams),
            seq_len=context_len + effective_rollout,
            drop_incomplete=False,
        ),
    )
    batch = materialize_batch_plan(
        store,
        BatchPlan(refs=tuple(WindowRef(stream, 0) for stream in streams)),
        reference.data.input_columns,
        reference.data.target_columns,
        scaling_rules(reference),
        inference_loader,
        batch_idx=0,
    )
    inputs = batch.inputs.to(device=device)
    targets = batch.targets.to(device=device)
    mask = batch.mask.to(device=device)
    predictions = tuple(
        _prediction_series(
            selection,
            checkpoint,
            inputs,
            targets,
            mask,
            context_len,
            effective_rollout,
            requested_suffix_steps,
            device,
        )
        for selection, checkpoint in zip(selections, checkpoints, strict=True)
        for requested_suffix_steps in suffix_steps
    )
    return InferenceResult(
        config=reference,
        inputs=inputs.detach().cpu(),
        targets=targets.detach().cpu(),
        predictions=predictions,
        context_len=context_len,
        rollout_len=effective_rollout,
        group_keys=batch.state.group_keys,
        warning=warning,
    )


def _prediction_series(
    selection: CheckpointSelection,
    checkpoint: LoadedCheckpoint,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    rollout_steps: int,
    suffix_steps: int,
    device: torch.device,
) -> InferencePrediction:
    suffix = replace(
        resolved_validation_masked_suffix(checkpoint.config),
        enabled=suffix_steps > 0,
        suffix_steps=suffix_steps or checkpoint.config.train.masked_suffix.suffix_steps,
    )
    result = rollout_batch(
        checkpoint.config,
        checkpoint.model,
        inputs,
        context_len=context_len,
        rollout_steps=rollout_steps,
        suffix=suffix,
        device=device,
        targets=targets,
        mask=mask,
    )
    return InferencePrediction(
        checkpoint_alias=selection.alias,
        checkpoint_path=selection.path,
        suffix_steps=suffix_steps,
        predictions=result.prediction.detach().cpu(),
        metrics=loss_metrics_to_cpu(result.metrics),
        target_start=result.target_start,
    )


def _selected_protocols(
    selected_index_frame: pl.DataFrame,
    checkpoints: tuple[LoadedCheckpoint, ...],
) -> tuple[DatasetProtocolId, ...]:
    protocols = tuple(
        DatasetProtocolId(value)
        for value in selected_index_frame[BaseColumns.proto].unique(maintain_order=True).to_list()
    )
    if DatasetProtocolId.eis in protocols:
        raise ValueError("EIS inference is not supported yet")
    for checkpoint_idx, checkpoint in enumerate(checkpoints, start=1):
        unsupported = sorted(
            str(protocol)
            for protocol in protocols
            if str(protocol) not in checkpoint.config.data.protocols
        )
        if unsupported:
            raise ValueError(
                f"Selected protocols are not configured in checkpoint {checkpoint_idx}: "
                f"{unsupported}"
            )
    return protocols


def _inference_loader(
    config: ExperimentConfig,
    protocols: tuple[DatasetProtocolId, ...],
    batch_size: int,
    device: torch.device,
) -> LoaderConfig:
    return LoaderConfig(
        split=BaseColumns.split.values.train,
        default_window=WindowConfig(batch_size=max(1, batch_size), seq_len=config.loader.seq_len),
        seed=config.run.seed,
        strategy="sequential",
        protocol_order=protocols,
        stateful_n_windows=1,
        drop_incomplete_batches=False,
        data_access="windowed",
        num_workers=0,
        multiprocessing_context=None,
        device=str(device),
    )


def _validate_compatible_checkpoints(checkpoints: tuple[LoadedCheckpoint, ...]) -> None:
    reference = checkpoints[0].config
    for idx, checkpoint in enumerate(checkpoints[1:], start=2):
        config = checkpoint.config
        differences = []
        if config.data.input_columns != reference.data.input_columns:
            differences.append("input_columns")
        if config.data.target_columns != reference.data.target_columns:
            differences.append("target_columns")
        if config.loader.seq_len != reference.loader.seq_len:
            differences.append("seq_len")
        if scaling_rules(config) != scaling_rules(reference):
            differences.append("scaling")
        if differences:
            raise ValueError(
                "Selected checkpoints must use the same input columns, target columns, "
                "seq_len, and scaling. "
                f"Checkpoint {idx} differs: {', '.join(differences)}"
            )
