from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.config import resolved_validation_masked_suffix
from batgrad.ml.data.config import GROUP_KEY_CELL_CYCLE_PROTOCOL, WindowConfig
from batgrad.ml.data.materialization import materialize_window_ref
from batgrad.ml.data.planning import WindowRef, build_stream_plans
from batgrad.ml.data.scaling import scale_data
from batgrad.ml.distributed import all_reduce_loss_metrics, unwrap_model
from batgrad.ml.experiment import loader_config, scaling_rules
from batgrad.ml.metrics import LossMetrics, add_loss_metrics
from batgrad.ml.objective import batch_loss_with_metrics
from batgrad.ml.rollout import predict_context, rollout_batch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batgrad.ml.config import ExperimentConfig
    from batgrad.ml.data.batch import Batch
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.ml.distributed import DistributedContext
    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class RolloutExample:
    inputs: torch.Tensor
    context_prediction: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    target_start: int
    match: dict[str, object]
    anchor: int


@dataclass(frozen=True, slots=True)
class ValidationResult:
    teacher_forced_metrics: LossMetrics | None = None
    rollout_metrics: LossMetrics | None = None
    rollout_examples: tuple[RolloutExample, ...] = ()


@torch.no_grad()
def validate(
    config: ExperimentConfig,
    model: torch.nn.Module,
    val_loader: Iterable[Batch],
    index: MlDatasetIndex,
    store: DatasetStoreReader,
    device: torch.device,
    dist_ctx: DistributedContext | None = None,
) -> ValidationResult:
    model.eval()
    teacher_forced_metrics = None
    if config.validation.max_tf_batches > 0:
        teacher_forced_metrics = _validate_batches(
            config,
            model,
            val_loader,
            device,
        )
    rollout_metrics = None
    rollout_examples: tuple[RolloutExample, ...] = ()
    if config.validation.rollout_steps > 0 and (dist_ctx is None or dist_ctx.is_main):
        rollout_result = run_rollouts(
            config,
            unwrap_model(model),
            index,
            store,
            device,
        )
        rollout_metrics = rollout_result.rollout_metrics
        rollout_examples = rollout_result.rollout_examples
    return ValidationResult(teacher_forced_metrics, rollout_metrics, rollout_examples)


def _validate_batches(
    config: ExperimentConfig,
    model: torch.nn.Module,
    val_loader: Iterable[Batch],
    device: torch.device,
) -> LossMetrics | None:
    suffix = resolved_validation_masked_suffix(config)
    metrics: LossMetrics | None = None
    for idx, batch in enumerate(val_loader):
        metrics = add_loss_metrics(
            metrics,
            batch_loss_with_metrics(
                config,
                model,
                batch.inputs,
                batch.targets,
                batch.mask,
                device,
                suffix=suffix,
                include_rmse=True,
                mask_all_valid=batch.all_valid,
            ),
        )
        if idx + 1 >= config.validation.max_tf_batches:
            break
    if metrics is None:
        return None
    return all_reduce_loss_metrics(metrics)


@torch.no_grad()
def run_rollouts(
    config: ExperimentConfig,
    model: torch.nn.Module,
    index: MlDatasetIndex,
    store: DatasetStoreReader,
    device: torch.device,
) -> ValidationResult:
    val_index = index.filter_split(BaseColumns.split.values.val)
    context_len = config.loader.seq_len
    stored_rollout_len = config.validation.rollout_steps
    extension_steps = (
        config.validation.rollout_extension.steps
        if config.validation.rollout_extension.enabled
        else 0
    )
    rollout_len = stored_rollout_len + extension_steps
    window_config = replace(
        loader_config(config, BaseColumns.split.values.val),
        default_window=WindowConfig(batch_size=1, seq_len=context_len + stored_rollout_len),
    )
    stream_plans_by_protocol = {}
    scaling = scaling_rules(config)
    metrics: LossMetrics | None = None
    plot_series: list[RolloutExample] = []
    suffix = resolved_validation_masked_suffix(config)
    for group in config.validation.split.groups:
        if not group.rollout_start_offsets:
            continue
        protocol = _rollout_protocol(config, group.match)
        if protocol not in stream_plans_by_protocol:
            stream_plans_by_protocol[protocol] = build_stream_plans(
                val_index, protocol, window_config
            )
        matches = [
            stream
            for stream in stream_plans_by_protocol[protocol]
            if _stream_matches(stream.group_key, GROUP_KEY_CELL_CYCLE_PROTOCOL, group.match)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"rollout selector must match exactly one stream, got {len(matches)}: {group.match}"
            )
        stream = matches[0]
        for anchor in group.rollout_start_offsets:
            window_offset = _rollout_window_offset(
                anchor,
                context_len=context_len,
                stored_rollout_len=stored_rollout_len,
                row_count=stream.row_count,
            )
            batch = materialize_window_ref(
                store,
                WindowRef(stream, window_offset),
                config.data.input_columns,
                config.data.target_columns,
                scaling,
                window_config,
                batch_idx=0,
            )
            inputs = batch.inputs.to(device=device)
            targets = batch.targets.to(device=device)
            rollout_mask = batch.mask.to(device=device)
            if extension_steps:
                inputs, targets, rollout_mask = _append_rollout_extension(
                    config,
                    inputs,
                    targets,
                    rollout_mask,
                    context_len,
                    stored_rollout_len,
                )
            result = rollout_batch(
                config,
                model,
                inputs,
                context_len=context_len,
                rollout_steps=rollout_len,
                suffix=suffix,
                device=device,
                targets=targets,
                mask=rollout_mask,
            )
            if result.metrics is not None:
                metrics = add_loss_metrics(metrics, result.metrics)
            if result.prediction.shape[1] and config.validation.log_rollout_plots:
                plot_series.append(
                    RolloutExample(
                        inputs=inputs[:, : context_len + result.prediction.shape[1], :].cpu()[0],
                        context_prediction=predict_context(
                            config, model, inputs, context_len, device
                        ).cpu()[0],
                        prediction=result.prediction.cpu()[0],
                        target=targets[:, : context_len + result.prediction.shape[1], :].cpu()[0],
                        target_start=result.target_start,
                        match=group.match,
                        anchor=anchor,
                    )
                )
    return ValidationResult(
        rollout_metrics=metrics,
        rollout_examples=tuple(plot_series),
    )


def _rollout_window_offset(
    anchor: int,
    *,
    context_len: int,
    stored_rollout_len: int,
    row_count: int,
) -> int:
    minimum_anchor = context_len - 1
    if anchor < minimum_anchor:
        raise ValueError(
            "rollout anchor does not have a complete context window: "
            f"anchor={anchor} minimum={minimum_anchor} context_len={context_len}"
        )
    if anchor + stored_rollout_len >= row_count:
        raise ValueError(
            "rollout anchor does not have enough observed future rows: "
            f"anchor={anchor} rollout_steps={stored_rollout_len} row_count={row_count}"
        )
    return anchor - context_len + 1


def _append_rollout_extension(
    config: ExperimentConfig,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    stored_rollout_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extension = config.validation.rollout_extension
    suffix = inputs[:, -1:, :].clone().repeat(1, extension.steps, 1)
    scaling = scaling_rules(config)
    for column, physical_value in extension.input_values.items():
        column_idx = config.data.input_columns.index(column)
        value = torch.tensor([[[float(physical_value)]]], dtype=inputs.dtype, device=inputs.device)
        rule = tuple(item for item in scaling if item.name == column)
        if not rule:
            raise ValueError(f"Missing scaling rule for rollout extension column: {column}")
        suffix[:, :, column_idx] = scale_data(value, rule).reshape(())
    inputs = torch.cat((inputs, suffix), dim=1)
    targets = torch.cat(
        (
            targets,
            torch.full(
                (targets.shape[0], extension.steps, targets.shape[2]),
                float("nan"),
                dtype=targets.dtype,
                device=targets.device,
            ),
        ),
        dim=1,
    )
    mask = torch.cat(
        (
            mask,
            torch.zeros((mask.shape[0], extension.steps), dtype=torch.bool, device=mask.device),
        ),
        dim=1,
    )
    mask[:, context_len + stored_rollout_len - 1 :] = False
    return inputs, targets, mask


def _stream_matches(
    group_key: tuple[object, ...], group_by: tuple[str, ...], match: dict[str, object]
) -> bool:
    key_map = dict(zip(group_by, group_key, strict=True))
    return all(key_map.get(key) == value for key, value in match.items())


def _rollout_protocol(config: ExperimentConfig, match: dict[str, object]) -> DatasetProtocolId:
    value = match.get("protocol")
    if value is None:
        raise ValueError(f"rollout selector must include protocol: {match}")
    protocol = DatasetProtocolId(value)
    enabled = {DatasetProtocolId(item) for item in config.data.protocols}
    if protocol not in enabled:
        raise ValueError(
            f"rollout selector protocol {str(protocol)!r} is not in data.protocols: "
            f"{tuple(str(item) for item in enabled)}"
        )
    return protocol
