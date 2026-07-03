from __future__ import annotations

import json
import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Literal, Union, cast, get_args, get_origin, get_type_hints

from batgrad.contracts.mapping import DatasetProtocolId
from batgrad.ml.loggers import LoggingConfig  # noqa: TC001 - needed by get_type_hints at runtime
from batgrad.ml.nn import SequenceMixerConfig  # noqa: TC001 - needed by get_type_hints at runtime

type ScalingTransform = Literal["linear", "log1p"]


@dataclass(frozen=True, slots=True)
class ScalingRuleConfig:
    column: str
    input_min: float
    input_max: float
    output_min: float = -1.0
    output_max: float = 1.0
    clip: bool = False
    transform: ScalingTransform = "linear"

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("data.scaling[].column must not be empty")
        if self.input_min >= self.input_max:
            raise ValueError("data.scaling input_min must be < input_max")
        if self.output_min >= self.output_max:
            raise ValueError("data.scaling output_min must be < output_max")
        if self.transform not in {"linear", "log1p"}:
            raise ValueError("data.scaling transform must be 'linear' or 'log1p'")
        if self.transform == "log1p" and self.input_min < 0.0:
            raise ValueError("data.scaling log1p requires input_min >= 0")


@dataclass(frozen=True, slots=True)
class DataConfig:
    manifest_paths: dict[str, str]
    protocols: tuple[str, ...]
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    store_root: str | None = None
    feedback_columns: tuple[str, ...] = ()
    scaling: tuple[ScalingRuleConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.store_root is not None and not self.store_root.strip():
            raise ValueError("data.store_root must not be empty when provided")
        if not self.manifest_paths:
            raise ValueError("data.manifest_paths must not be empty")
        if not self.protocols:
            raise ValueError("data.protocols must not be empty")
        if not self.input_columns or not self.target_columns:
            raise ValueError("data.input_columns and data.target_columns must not be empty")
        for protocol in self.protocols:
            try:
                DatasetProtocolId(protocol)
            except ValueError as error:
                raise ValueError(f"Unsupported data protocol: {protocol!r}") from error
        _validate_data_columns(self)


def _validate_data_columns(config: DataConfig) -> None:
    input_columns = set(config.input_columns)
    target_columns = set(config.target_columns)
    scaling_columns = [rule.column for rule in config.scaling]
    duplicates = sorted(
        column for column in set(scaling_columns) if scaling_columns.count(column) > 1
    )
    if duplicates:
        raise ValueError(f"data.scaling contains duplicate columns: {duplicates}")
    unknown_scaling = sorted(set(scaling_columns) - input_columns - target_columns)
    if unknown_scaling:
        raise ValueError(
            f"data.scaling contains columns not selected for training: {unknown_scaling}"
        )
    missing_targets = sorted(target_columns - set(scaling_columns))
    if missing_targets:
        raise ValueError(f"Every target column needs a data.scaling rule: {missing_targets}")
    missing_feedback_inputs = sorted(set(config.feedback_columns) - input_columns)
    missing_feedback_targets = sorted(set(config.feedback_columns) - target_columns)
    if missing_feedback_inputs or missing_feedback_targets:
        raise ValueError(
            "data.feedback_columns must be present in both input_columns and target_columns. "
            f"Missing inputs={missing_feedback_inputs} missing targets={missing_feedback_targets}"
        )


@dataclass(frozen=True, slots=True)
class ValidationGroupConfig:
    match: dict[str, object]
    rollout_start_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.match:
            raise ValueError("validation.split.groups[].match must not be empty")
        if any(offset < 0 for offset in self.rollout_start_offsets):
            raise ValueError("validation rollout_start_offsets must be >= 0")


@dataclass(frozen=True, slots=True)
class ValidationSplitConfig:
    strategy: Literal["sample", "provide", "merge"] = "sample"
    fraction: float = 0.2
    group_by: tuple[str, ...] = ("dataset id", "cell id", "cycle index")
    groups: tuple[ValidationGroupConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in {"sample", "provide", "merge"}:
            raise ValueError(f"Unsupported validation split strategy: {self.strategy!r}")
        if not (0.0 <= self.fraction < 1.0):
            raise ValueError("validation.split.fraction must be in [0, 1)")
        if not self.group_by:
            raise ValueError("validation.split.group_by must not be empty")
        if self.strategy == "provide" and not self.groups:
            raise ValueError("validation.split.strategy='provide' requires groups")
        if self.strategy == "sample" and self.groups:
            raise ValueError("validation.split.strategy='sample' does not accept explicit groups")


@dataclass(frozen=True, slots=True)
class RolloutExtensionConfig:
    enabled: bool = False
    steps: int = 0
    input_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("validation.rollout_extension.steps must be >= 0")
        if self.enabled and self.steps <= 0:
            raise ValueError("validation.rollout_extension.steps must be > 0 when enabled")
        if self.enabled and not self.input_values:
            raise ValueError(
                "validation.rollout_extension.input_values must not be empty when enabled"
            )


@dataclass(frozen=True, slots=True)
class ValidationMaskedSuffixConfig:
    enabled: bool | None = None
    suffix_steps: int | None = None
    carry_mamba_state: bool | None = None

    def __post_init__(self) -> None:
        if self.suffix_steps is not None and self.suffix_steps <= 0:
            raise ValueError("validation.masked_suffix.suffix_steps must be > 0")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    split: ValidationSplitConfig = field(default_factory=ValidationSplitConfig)
    max_tf_batches: int = 1
    rollout_steps: int = 0
    log_rollout_plots: bool = True
    masked_suffix: ValidationMaskedSuffixConfig = field(
        default_factory=ValidationMaskedSuffixConfig
    )
    rollout_extension: RolloutExtensionConfig = field(default_factory=RolloutExtensionConfig)

    def __post_init__(self) -> None:
        if self.max_tf_batches < 0:
            raise ValueError("validation.max_tf_batches must be >= 0")
        if self.rollout_steps < 0:
            raise ValueError("validation.rollout_steps must be >= 0")
        has_rollout_offsets = any(group.rollout_start_offsets for group in self.split.groups)
        if self.rollout_steps > 0 and not has_rollout_offsets:
            raise ValueError(
                "validation.rollout_steps > 0 requires at least one split group with "
                "rollout_start_offsets"
            )


@dataclass(frozen=True, slots=True)
class LoaderTrainConfig:
    batch_size: int = 32
    seq_len: int = 1024
    strategy: Literal["sequential", "shuffled_protocol_groups"] = "shuffled_protocol_groups"
    stateful_n_windows: int = 1
    data_access: Literal["windowed", "full_in_mem"] = "windowed"
    num_workers: int = 0
    prefetch_to_device: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError("loader.batch_size and loader.seq_len must be > 0")
        if self.strategy not in {"sequential", "shuffled_protocol_groups"}:
            raise ValueError(f"Unsupported loader strategy: {self.strategy!r}")
        if self.stateful_n_windows == 0 or self.stateful_n_windows < -1:
            raise ValueError("loader.stateful_n_windows must be -1 or a positive integer")
        if self.data_access not in {"windowed", "full_in_mem"}:
            raise ValueError("loader.data_access must be 'windowed' or 'full_in_mem'")
        if self.num_workers < 0:
            raise ValueError("loader.num_workers must be >= 0")


@dataclass(frozen=True, slots=True)
class MaskedSuffixConfig:
    enabled: bool = True
    channels: tuple[str, ...] = ()
    suffix_steps: int = 128
    fill_value: float = 0.0
    loss_on_masked_only: bool = True
    carry_mamba_state: bool = True
    detach_between_windows: bool = True
    roll_forward_steps: int = 0

    def __post_init__(self) -> None:
        if self.enabled and not self.channels:
            raise ValueError("train.masked_suffix.channels must not be empty when enabled")
        if self.suffix_steps <= 0:
            raise ValueError("train.masked_suffix.suffix_steps must be > 0")
        if self.roll_forward_steps < 0:
            raise ValueError("train.masked_suffix.roll_forward_steps must be >= 0")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: float = 1.0
    log_per_epoch: int = 10
    log_every_steps: int | None = None
    validate_per_epoch: int = 1
    validate_every_steps: int | None = None
    loss: Literal["categorical_ce"] = "categorical_ce"
    grad_clip_norm: float = 1.0
    max_steps: int | None = None
    masked_suffix: MaskedSuffixConfig = field(default_factory=MaskedSuffixConfig)

    def __post_init__(self) -> None:
        if self.epochs <= 0.0:
            raise ValueError("train.epochs must be > 0")
        if self.log_per_epoch <= 0 or self.validate_per_epoch < 0:
            raise ValueError("train.log_per_epoch must be > 0 and validate_per_epoch must be >= 0")
        if self.log_every_steps is not None and self.log_every_steps <= 0:
            raise ValueError("train.log_every_steps must be > 0 when provided")
        if self.validate_every_steps is not None and self.validate_every_steps <= 0:
            raise ValueError("train.validate_every_steps must be > 0 when provided")
        if self.loss != "categorical_ce":
            raise ValueError("train.loss currently only supports 'categorical_ce'")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("train.grad_clip_norm must be > 0")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    kind: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    kind: Literal["linear_warmup_cosine", "none"] = "linear_warmup_cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.01


@dataclass(frozen=True, slots=True)
class RunConfig:
    device: str = "cuda"
    seed: int = 69
    use_amp: bool = True
    compile_model: bool = False
    init_from: str | None = None
    output_dir: str | None = "ml/runs"
    name: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    save_latest: bool = False
    save_best: bool = False
    save_final: bool = False
    monitors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.save_best and not self.monitors:
            raise ValueError("checkpoint.monitors must not be empty when save_best=true")
        duplicates = sorted(name for name in set(self.monitors) if self.monitors.count(name) > 1)
        if duplicates:
            raise ValueError(f"checkpoint.monitors contains duplicates: {duplicates}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    data: DataConfig
    loader: LoaderTrainConfig
    model: SequenceMixerConfig
    train: TrainConfig
    validation: ValidationConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    run: RunConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        _validate_protocol_strategy(self)
        _validate_mamba_state(self)
        _validate_masked_suffix_channels(self)
        _validate_rollout_extension(self)
        _validate_prefetch(self)
        _validate_output_mode(self)


def _validate_protocol_strategy(config: ExperimentConfig) -> None:
    if config.loader.strategy != "shuffled_protocol_groups":
        return
    supported = {
        str(DatasetProtocolId.cycling),
        str(DatasetProtocolId.hppc),
        str(DatasetProtocolId.rpt),
    }
    unsupported = sorted(set(config.data.protocols) - supported)
    if unsupported:
        raise ValueError(
            "loader.strategy='shuffled_protocol_groups' does not support protocols "
            f"{unsupported}. Supported: {sorted(supported)}"
        )
    if len(config.data.protocols) != 1:
        raise ValueError(
            "loader.strategy='shuffled_protocol_groups' currently supports exactly one "
            f"training protocol, got {list(config.data.protocols)}"
        )


def _validate_mamba_state(config: ExperimentConfig) -> None:
    if config.train.masked_suffix.carry_mamba_state and any(
        layer.kind == "mamba" and layer.mamba_config(config.model.mamba).is_mimo
        for layer in (*config.model.layers, *config.model.head_layers)
    ):
        raise ValueError("Mamba state carry currently requires Mamba layers with is_mimo=false")


def _validate_masked_suffix_channels(config: ExperimentConfig) -> None:
    suffix = config.train.masked_suffix
    if not suffix.enabled:
        return
    input_columns = set(config.data.input_columns)
    target_columns = set(config.data.target_columns)
    feedback_columns = set(config.data.feedback_columns)
    missing_inputs = sorted(set(suffix.channels) - input_columns)
    missing_targets = sorted(set(suffix.channels) - target_columns)
    missing_feedback = sorted(set(suffix.channels) - feedback_columns)
    if missing_inputs or missing_targets or missing_feedback:
        raise ValueError(
            "train.masked_suffix.channels must be present in data.input_columns, "
            "data.target_columns, and data.feedback_columns. "
            f"Missing inputs={missing_inputs} targets={missing_targets} "
            f"feedback={missing_feedback}"
        )


def _validate_rollout_extension(config: ExperimentConfig) -> None:
    if not config.validation.rollout_extension.enabled:
        return
    extension_columns = set(config.validation.rollout_extension.input_values)
    input_columns = set(config.data.input_columns)
    scaling_columns = {rule.column for rule in config.data.scaling}
    unknown = sorted(extension_columns - input_columns)
    missing_scaling = sorted(extension_columns - scaling_columns)
    if unknown:
        raise ValueError(
            f"validation.rollout_extension.input_values keys must be input columns: {unknown}"
        )
    if missing_scaling:
        raise ValueError(
            "validation.rollout_extension.input_values columns need data.scaling rules: "
            f"{missing_scaling}"
        )


def _validate_prefetch(config: ExperimentConfig) -> None:
    if config.loader.prefetch_to_device and not config.run.device.startswith("cuda"):
        raise ValueError("loader.prefetch_to_device requires run.device='cuda'")


def _validate_output_mode(config: ExperimentConfig) -> None:
    if config.run.output_dir is not None:
        return
    if config.logging.backend != "stdout":
        raise ValueError("run.output_dir=null requires logging.backend='stdout'")
    if _checkpoint_enabled(config.checkpoint):
        raise ValueError("run.output_dir=null requires checkpoint save flags to be false")


def _checkpoint_enabled(config: CheckpointConfig) -> bool:
    return config.save_latest or config.save_best or config.save_final


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise TypeError("experiment config JSON must be an object")
    return _coerce_dataclass(ExperimentConfig, raw, "config")


def resolve_store_root(configured: str | None) -> str:
    value = configured if configured is not None else os.getenv("DATA_ROOT")
    if value is None or not value.strip():
        raise ValueError("data.store_root is missing and DATA_ROOT is not set")
    return value


def _coerce_dataclass[T](cls: type[T], raw: object, path: str) -> T:
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must be an object")
    raw_dict = cast("dict[str, object]", raw)
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(raw_dict) - names)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {unknown}")
    values: dict[str, object] = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name in raw_dict:
            values[item.name] = _coerce_value(
                type_hints[item.name], raw_dict[item.name], f"{path}.{item.name}"
            )
        elif item.default is not MISSING:
            values[item.name] = item.default
        elif item.default_factory is not MISSING:
            values[item.name] = item.default_factory()
        else:
            raise ValueError(f"{path}.{item.name} is required")
    return cls(**values)


def _coerce_value(expected: object, value: object, path: str) -> object:
    origin = get_origin(expected)
    args = get_args(expected)
    result = value
    if origin is Literal:
        result = _coerce_literal(value, args, path)
    elif origin is tuple:
        result = _coerce_tuple(value, args, path)
    elif origin is dict:
        result = _coerce_dict(value, args, path)
    elif origin is UnionType or origin is Union:
        result = _coerce_union(value, args, path)
    elif isinstance(expected, type) and is_dataclass(expected):
        result = _coerce_dataclass(expected, value, path)
    elif expected in {str, int, float, bool}:
        result = _coerce_scalar(expected, value, path)
    return result


def _coerce_literal(value: object, args: tuple[object, ...], path: str) -> object:
    if value not in args:
        raise ValueError(f"{path}={value!r} is not supported. Supported: {list(args)}")
    return value


def _coerce_tuple(value: object, args: tuple[object, ...], path: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{path} must be an array")
    item_type = args[0] if args else object
    return tuple(_coerce_value(item_type, item, f"{path}[]") for item in value)


def _coerce_dict(value: object, args: tuple[object, ...], path: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    key_type, val_type = args or (object, object)
    return {
        _coerce_value(key_type, key, f"{path}.key"): _coerce_value(val_type, val, f"{path}.{key}")
        for key, val in value.items()
    }


def _coerce_union(value: object, args: tuple[object, ...], path: str) -> object:
    for option in args:
        if option is type(None) and value is None:
            return None
        try:
            return _coerce_value(option, value, path)
        except (TypeError, ValueError):
            continue
    raise TypeError(f"{path} does not match expected type")


def _coerce_scalar(expected: object, value: object, path: str) -> object:
    if expected is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if expected is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected is bool and isinstance(value, bool):
        return value
    if expected is str and isinstance(value, str):
        return value
    raise TypeError(f"{path} must be {getattr(expected, '__name__', expected)}")


def config_to_dict(value: object) -> object:
    if is_dataclass(value):
        return {item.name: config_to_dict(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [config_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): config_to_dict(item) for key, item in value.items()}
    return value
