from __future__ import annotations

import json
import math
import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Literal, Union, cast, get_args, get_origin, get_type_hints

from batgrad.contracts.mapping import DatasetProtocolId
from batgrad.ml.nn import SequenceMixerConfig  # noqa: TC001 - needed by get_type_hints at runtime

type ScalingTransform = Literal["linear", "log1p"]


@dataclass(frozen=True, slots=True)
class WandbConfig:
    """Weights & Biases run metadata.

    Attributes:
        project: W&B project name. The W&B client default is used when omitted.
        entity: Optional team or user account.
        group: Optional run group.
        name: Optional display name.
        tags: Searchable run tags.
    """

    project: str | None = None
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Metric logging configuration.

    Attributes:
        backend: Output backend. JSONL and W&B require a run output directory.
        mode: W&B synchronization mode; ignored by other backends.
        mirror_stdout: Whether file and W&B backends also emit concise console metrics.
        wandb: W&B-specific metadata.
    """

    backend: Literal["stdout", "jsonl", "wandb"] = "jsonl"
    mode: Literal["offline", "online"] = "offline"
    mirror_stdout: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if self.backend not in {"stdout", "jsonl", "wandb"}:
            raise ValueError(f"Unsupported logging backend: {self.backend!r}")
        if self.mode not in {"offline", "online"}:
            raise ValueError(f"Unsupported logging mode: {self.mode!r}")


@dataclass(frozen=True, slots=True)
class ScalingRuleConfig:
    """Serializable scaling rule for one selected column.

    Linear scaling maps `[input_min, input_max]` to
    `[output_min, output_max]`. The `log1p` transform is applied before the
    linear map and consequently requires non-negative input bounds.

    Attributes:
        column: Canonical selected column name.
        input_min: Lower bound in physical units.
        input_max: Upper bound in physical units.
        output_min: Lower model-space bound.
        output_max: Upper model-space bound.
        clip: Whether forward scaling clips to the output bounds.
        transform: Pre-scaling transform.

    Note:
        Manifest validation still rejects observed values outside the configured
        input bounds when `clip` is enabled. Clipping protects runtime values;
        it does not relax the dataset contract.
    """

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
        if not all(
            math.isfinite(value)
            for value in (self.input_min, self.input_max, self.output_min, self.output_max)
        ):
            raise ValueError("data.scaling bounds must be finite")
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
    """Dataset selection and model column contract.

    Attributes:
        manifest_paths: Mapping from canonical normalized manifest paths to the
            expected Git commit or commit prefix.
        protocols: Protocol values included in the experiment.
        input_columns: Ordered model input columns.
        target_columns: Ordered model target columns.
        protocol_mode: `"strict"` requires every selected dataset to contain
            every protocol; `"available"` retains the requested protocols that
            exist and warns about missing combinations.
        store_root: Data-store root, or `None` to use `DATA_ROOT`.
        feedback_columns: Columns that are both inputs and targets and may receive
            model predictions during rollout.
        scaling: Scaling contracts for selected columns. Every input and target
            requires exactly one rule.
    """

    manifest_paths: dict[str, str]
    protocols: tuple[str, ...]
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    protocol_mode: Literal["strict", "available"] = "available"
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
        duplicate_protocols = sorted(
            protocol for protocol in set(self.protocols) if self.protocols.count(protocol) > 1
        )
        if duplicate_protocols:
            raise ValueError(f"data.protocols contains duplicates: {duplicate_protocols}")
        if self.protocol_mode not in {"strict", "available"}:
            raise ValueError("data.protocol_mode must be 'strict' or 'available'")
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
    for field_name, columns in (
        ("input_columns", config.input_columns),
        ("target_columns", config.target_columns),
        ("feedback_columns", config.feedback_columns),
    ):
        duplicates = sorted(column for column in set(columns) if columns.count(column) > 1)
        if duplicates:
            raise ValueError(f"data.{field_name} contains duplicates: {duplicates}")
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
    missing_scaling = sorted((input_columns | target_columns) - set(scaling_columns))
    if missing_scaling:
        raise ValueError(
            f"Every input and target column needs a data.scaling rule: {missing_scaling}"
        )
    missing_feedback_inputs = sorted(set(config.feedback_columns) - input_columns)
    missing_feedback_targets = sorted(set(config.feedback_columns) - target_columns)
    if missing_feedback_inputs or missing_feedback_targets:
        raise ValueError(
            "data.feedback_columns must be present in both input_columns and target_columns. "
            f"Missing inputs={missing_feedback_inputs} missing targets={missing_feedback_targets}"
        )


@dataclass(frozen=True, slots=True)
class ValidationGroupConfig:
    """Explicit held-out group and optional rollout anchors.

    Attributes:
        match: Values matched against the validation split's grouping columns.
        rollout_start_offsets: Zero-based source-row offsets identifying the last
            observed input row at which anchored rollouts begin.
    """

    match: dict[str, object]
    rollout_start_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.match:
            raise ValueError("validation.split.groups[].match must not be empty")
        if any(offset < 0 for offset in self.rollout_start_offsets):
            raise ValueError("validation rollout_start_offsets must be >= 0")


@dataclass(frozen=True, slots=True)
class ValidationSplitConfig:
    """Group-aware validation split policy.

    `sample` deterministically hashes groups, `provide` uses only explicit
    selectors, and `merge` combines both sets. Splitting groups rather than rows
    reduces leakage between related measurements. Sampling selects
    `int(group_count * fraction)` groups, so small datasets can produce no sampled
    validation groups.

    Attributes:
        strategy: Group selection strategy.
        fraction: Fraction selected by deterministic sampling.
        group_by: Manifest columns defining one indivisible group.
        groups: Explicit group selectors used by `provide` and `merge`.
    """

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
        duplicate_group_by = sorted(
            column for column in set(self.group_by) if self.group_by.count(column) > 1
        )
        if duplicate_group_by:
            raise ValueError(f"validation.split.group_by contains duplicates: {duplicate_group_by}")
        if self.strategy == "provide" and not self.groups:
            raise ValueError("validation.split.strategy='provide' requires groups")
        if self.strategy == "sample" and self.groups:
            raise ValueError("validation.split.strategy='sample' does not accept explicit groups")


@dataclass(frozen=True, slots=True)
class RolloutExtensionConfig:
    """Unscored rollout continuation beyond observed rows.

    Attributes:
        enabled: Whether to append synthetic future control rows.
        steps: Number of unobserved rows to generate.
        input_values: Physical-unit values for known future input controls.

    Note:
        Extension rows have no targets and never contribute to validation metrics.
        Inputs omitted from `input_values` repeat their final observed value.
    """

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
    """Validation overrides for masked-suffix execution.

    `None` values inherit their training counterparts. Validation always
    disables training roll-forward, regardless of the training configuration.

    Attributes:
        enabled: Override masked-suffix validation.
        suffix_steps: Override the number of predicted suffix rows per call.
        carry_mamba_state: Override recurrent state carry during validation.
    """

    enabled: bool | None = None
    suffix_steps: int | None = None
    carry_mamba_state: bool | None = None

    def __post_init__(self) -> None:
        if self.suffix_steps is not None and self.suffix_steps <= 0:
            raise ValueError("validation.masked_suffix.suffix_steps must be > 0")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Held-out-window and anchored-rollout validation configuration.

    Attributes:
        split: Group-aware train/validation split.
        max_tf_batches: Maximum held-out batches evaluated at each validation;
            zero disables this pass.
        rollout_steps: Number of observed future rows scored per anchor.
        log_rollout_plots: Whether supported loggers receive trajectory plots.
        masked_suffix: Validation-time masked-suffix overrides.
        rollout_extension: Optional unscored continuation after observed rows.
    """

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
    """Training-loader settings stored in an experiment configuration.

    Attributes:
        batch_size: Number of stream lanes per batch.
        seq_len: Input context length before any roll-forward extension.
        strategy: Sequential windows or shuffled protocol-group streams.
        stateful_n_windows: Consecutive windows per stateful group, or `-1` for
            whole-stream mode.
        cross_protocol_state_carry: `"chain"` permits aligned protocol streams
            to share recurrent state.
        data_access: Windowed parquet reads or a full CPU tensor cache.
        num_workers: PyTorch data-loader worker count.
        prefetch_to_device: Whether to asynchronously prefetch to the CUDA device.

    Note:
        `full_in_mem` trades RAM for throughput and requires `num_workers=0`.
        Whole-stream batches are limited by the shortest lane and can omit longer
        lane tails. Finite stateful groups discard a final group shorter than
        `stateful_n_windows`. Shuffled protocol groups support cycling, HPPC, and
        RPT but not EIS.
    """

    batch_size: int = 32
    seq_len: int = 1024
    strategy: Literal["sequential", "shuffled_protocol_groups"] = "shuffled_protocol_groups"
    stateful_n_windows: int = 1
    cross_protocol_state_carry: Literal["chain"] | None = None
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
        if self.cross_protocol_state_carry not in {None, "chain"}:
            raise ValueError("loader.cross_protocol_state_carry must be null or 'chain'")
        if self.data_access not in {"windowed", "full_in_mem"}:
            raise ValueError("loader.data_access must be 'windowed' or 'full_in_mem'")
        if self.num_workers < 0:
            raise ValueError("loader.num_workers must be >= 0")


@dataclass(frozen=True, slots=True)
class MaskedSuffixConfig:
    """Autoregressive masked-suffix objective and rollout settings.

    Attributes:
        enabled: Whether feedback inputs in a suffix are predicted rather than
            supplied.
        channels: Feedback columns to mask and regenerate.
        suffix_steps: Maximum destination rows generated per model call.
        loss_on_masked_only: Score only feedback targets in the suffix when true.
        carry_mamba_state: Carry recurrent Mamba state between aligned windows.
        detach_between_windows: Truncate gradients between roll-forward windows.
        roll_forward_steps: Additional training positions traversed with generated
            feedback.

    Note:
        Generated feedback is detached before reuse. Enabled suffixes must be
        shorter than the loader sequence length.
    """

    enabled: bool = True
    channels: tuple[str, ...] = ()
    suffix_steps: int = 128
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
    """Optimization-loop controls.

    Attributes:
        epochs: Fractional or whole passes over planned training batches.
        log_per_epoch: Derived logging frequency when `log_every_steps` is unset.
        log_every_steps: Optional explicit logging interval.
        validate_per_epoch: Derived validation frequency; zero disables validation.
        validate_every_steps: Optional explicit validation interval.
        loss: Training objective. Only `"categorical_ce"` is supported.
        grad_clip_norm: Maximum global gradient norm after AMP unscaling.
        max_steps: Optional hard step limit overriding the epoch-derived count.
        masked_suffix: Autoregressive training objective.
    """

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
    """AdamW optimizer parameters.

    Attributes:
        kind: Optimizer implementation; currently only `"adamw"`.
        lr: Peak learning rate.
        weight_decay: Decoupled weight decay.
        beta1: First-moment coefficient.
        beta2: Second-moment coefficient.
        eps: Numerical stability term.
    """

    kind: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Learning-rate scheduler parameters.

    Attributes:
        kind: Cosine decay with linear warmup, or no scheduler.
        warmup_ratio: Fraction of total steps spent warming up.
        min_lr_ratio: Final cosine learning rate as a fraction of peak rate.
    """

    kind: Literal["linear_warmup_cosine", "none"] = "linear_warmup_cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.01


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Runtime and output settings.

    Attributes:
        device: PyTorch device string.
        seed: Random seed used by model initialization and data planning.
        use_amp: Enable CUDA automatic mixed precision.
        compile_model: Wrap the model with `torch.compile`.
        init_from: Optional checkpoint used to initialize compatible model weights.
        output_dir: Parent run directory, or `None` for output-free stdout runs.
        name: Stable run directory name. Existing directories with this name are
            replaced before training starts.

    Warning:
        `init_from` is weight initialization, not training resume: optimizer,
        scheduler, scaler, and cursor state are not restored.
    """

    device: str = "cuda"
    seed: int = 69
    use_amp: bool = True
    compile_model: bool = False
    init_from: str | None = None
    output_dir: str | None = "ml/runs"
    name: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint persistence policy.

    Attributes:
        save_latest: Replace the latest-step checkpoint after validation events.
        save_best: Keep the lowest value observed for each monitored metric at
            validation events. Unknown or unavailable metrics only emit a warning.
        save_final: Save model and training state after the last step.
        monitors: Metric names minimized by best-checkpoint tracking.
    """

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
    """Validated, frozen dataclass contract for one ML experiment.

    Constructing this object validates relationships spanning data, loader,
    model, objective, validation, device, output, and checkpoint settings. JSON
    callers should normally use `load_experiment_config` or
    `parse_experiment_config` so nested objects are coerced strictly.

    Attributes:
        data: Dataset revisions, protocols, columns, feedback, and scaling.
        loader: Training-loader behavior.
        model: Sequence-mixer architecture.
        train: Optimization-loop and objective controls.
        validation: Held-out and rollout validation.
        optim: AdamW parameters.
        scheduler: Learning-rate schedule.
        run: Device and output behavior.
        logging: Metric logging backend.
        checkpoint: Checkpoint persistence policy.

    Note:
        Frozen dataclasses prevent field reassignment, but mapping-valued fields
        remain ordinary dictionaries and should be treated as read-only.
    """

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
        _validate_masked_suffix_context(self)
        _validate_rollout_selectors(self)
        _validate_rollout_extension(self)
        _validate_prefetch(self)
        _validate_mamba_device(self)
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


def _validate_mamba_state(config: ExperimentConfig) -> None:
    validation_suffix = resolved_validation_masked_suffix(config)
    validation_carries_state = validation_suffix.carry_mamba_state and (
        (config.validation.max_tf_batches > 0 and validation_suffix.enabled)
        or config.validation.rollout_steps > 0
    )
    carries_state = (
        config.train.masked_suffix.enabled and config.train.masked_suffix.carry_mamba_state
    ) or validation_carries_state
    if carries_state and any(
        layer.kind == "mamba" and layer.mamba_config(config.model.mamba).is_mimo
        for layer in (*config.model.layers, *config.model.head_layers)
    ):
        raise ValueError("Mamba state carry currently requires Mamba layers with is_mimo=false")


def _validate_masked_suffix_channels(config: ExperimentConfig) -> None:
    suffix = config.train.masked_suffix
    if not suffix.enabled and not resolved_validation_masked_suffix(config).enabled:
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
    missing_masked = sorted(feedback_columns - set(suffix.channels))
    if missing_masked:
        raise ValueError(
            "train.masked_suffix.channels must equal data.feedback_columns when masked suffix "
            f"is enabled. Missing masked feedback columns={missing_masked}"
        )


def _validate_masked_suffix_context(config: ExperimentConfig) -> None:
    train_suffix = config.train.masked_suffix
    if train_suffix.enabled and train_suffix.suffix_steps >= config.loader.seq_len:
        raise ValueError(
            "train.masked_suffix.suffix_steps must be smaller than loader.seq_len when enabled"
        )
    validation_suffix = resolved_validation_masked_suffix(config)
    if validation_suffix.enabled and validation_suffix.suffix_steps >= config.loader.seq_len:
        raise ValueError(
            "effective validation.masked_suffix.suffix_steps must be smaller than loader.seq_len"
        )


def resolved_validation_masked_suffix(config: ExperimentConfig) -> MaskedSuffixConfig:
    """Resolve validation suffix overrides against training defaults.

    Args:
        config: Validated experiment configuration.

    Returns:
        A complete suffix configuration with `roll_forward_steps` forced to zero.
    """
    validation = config.validation.masked_suffix
    training = config.train.masked_suffix
    return MaskedSuffixConfig(
        enabled=training.enabled if validation.enabled is None else validation.enabled,
        channels=training.channels,
        suffix_steps=validation.suffix_steps or training.suffix_steps,
        loss_on_masked_only=training.loss_on_masked_only,
        carry_mamba_state=(
            training.carry_mamba_state
            if validation.carry_mamba_state is None
            else validation.carry_mamba_state
        ),
        detach_between_windows=training.detach_between_windows,
        roll_forward_steps=0,
    )


def _validate_rollout_selectors(config: ExperimentConfig) -> None:
    group_columns = set(config.validation.split.group_by)
    enabled_protocols = set(config.data.protocols)
    for group in config.validation.split.groups:
        unknown_columns = sorted(set(group.match) - group_columns)
        if unknown_columns:
            raise ValueError(
                "validation.split.groups[].match contains columns outside "
                "validation.split.group_by: "
                f"{unknown_columns}"
            )
        if not group.rollout_start_offsets:
            continue
        protocol = group.match.get("protocol")
        if protocol is None:
            raise ValueError(
                "validation split groups with rollout_start_offsets must include protocol in match"
            )
        if str(protocol) not in enabled_protocols:
            raise ValueError(
                f"validation rollout selector protocol must be in data.protocols: {protocol!r}"
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
    if config.loader.data_access == "full_in_mem" and config.loader.num_workers != 0:
        raise ValueError("loader.data_access='full_in_mem' requires loader.num_workers=0")


def _validate_mamba_device(config: ExperimentConfig) -> None:
    has_mamba = any(
        layer.kind == "mamba" for layer in (*config.model.layers, *config.model.head_layers)
    )
    if has_mamba and not config.run.device.startswith("cuda"):
        raise ValueError("Mamba layers require run.device to be a CUDA device")


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
    """Load and validate an experiment JSON file.

    Args:
        path: UTF-8 JSON file containing the complete experiment configuration.

    Returns:
        The strictly typed and cross-field-validated configuration.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If a value has the wrong JSON type.
        ValueError: If fields are missing, unknown, or mutually incompatible.

    Examples:
        Load the CPU dry-run configuration and inspect its model context:

        ```python
        from batgrad.ml.config import load_experiment_config

        config = load_experiment_config("configs/ml_dry_run_cpu.json")
        print(config.run.device, config.loader.seq_len)
        ```
    """
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return parse_experiment_config(raw)


def parse_experiment_config(raw: object) -> ExperimentConfig:
    """Strictly parse a decoded JSON-compatible experiment object.

    Unknown fields are rejected rather than ignored, and nested lists and objects
    are converted to their immutable dataclass representations.

    Args:
        raw: Decoded JSON-compatible root object.

    Returns:
        A validated experiment configuration.

    Raises:
        TypeError: If the root or a nested value has the wrong type.
        ValueError: If required fields are absent or validation fails.
    """
    if not isinstance(raw, dict):
        raise TypeError("experiment config JSON must be an object")
    return _coerce_dataclass(ExperimentConfig, raw, "config")


def resolve_store_root(configured: str | None) -> str:
    """Resolve the data-store root from configuration or `DATA_ROOT`.

    Args:
        configured: Explicit root. When `None`, the `DATA_ROOT` environment
            variable is used.

    Returns:
        A non-empty data-store root.

    Raises:
        ValueError: If neither source provides a non-empty value.
    """
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
