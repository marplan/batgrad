from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from batgrad.ml.config import (
    ValidationMaskedSuffixConfig,
    ValidationSplitConfig,
    config_to_dict,
    load_experiment_config,
    parse_experiment_config,
    resolved_validation_masked_suffix,
)
from batgrad.ml.experiment import data_validation_config, train_loader_config, val_loader_config
from batgrad.ml.nn import LayerConfig
from tests.ml.conftest import make_config


def test_train_loader_expands_sequence_for_roll_forward_but_val_does_not() -> None:
    config = make_config(seq_len=10, roll_forward_steps=4)

    assert train_loader_config(config).default_window.seq_len == 14
    assert val_loader_config(config).default_window.seq_len == 10


def test_validation_masked_suffix_inherits_train_defaults() -> None:
    config = make_config(suffix_steps=3, carry_mamba_state=True)

    suffix = resolved_validation_masked_suffix(config)
    assert suffix.enabled is True
    assert suffix.suffix_steps == 3
    assert suffix.carry_mamba_state is True


def test_validation_masked_suffix_overrides_train_defaults() -> None:
    config = make_config(
        suffix_steps=3,
        carry_mamba_state=True,
        validation_masked_suffix=ValidationMaskedSuffixConfig(
            enabled=False,
            suffix_steps=2,
            carry_mamba_state=False,
        ),
    )

    suffix = resolved_validation_masked_suffix(config)
    assert suffix.enabled is False
    assert suffix.suffix_steps == 2
    assert suffix.carry_mamba_state is False


def test_validation_suffix_disables_roll_forward() -> None:
    config = make_config(suffix_steps=3, roll_forward_steps=6)

    suffix = resolved_validation_masked_suffix(config)

    assert suffix.enabled is True
    assert suffix.suffix_steps == 3
    assert suffix.roll_forward_steps == 0


def test_baseline_config_loads_with_expected_training_wiring() -> None:
    config = load_experiment_config(Path("configs/ml_baseline.json"))

    assert config.loader.strategy == "shuffled_protocol_groups"
    assert config.loader.cross_protocol_state_carry == "chain"
    assert config.loader.data_access == "full_in_mem"
    assert config.train.masked_suffix.channels == config.data.feedback_columns
    assert config.validation.split.strategy == "merge"
    assert config.validation.rollout_extension.enabled is True
    assert set(config.validation.rollout_extension.input_values) <= {
        rule.column for rule in config.data.scaling
    }
    assert config.checkpoint.monitors == ("val/rollout/loss_ce",)


def test_in_memory_config_parser_matches_file_loader() -> None:
    loaded = load_experiment_config(Path("configs/ml_baseline.json"))
    raw = config_to_dict(loaded)

    assert parse_experiment_config(raw) == loaded


def test_config_rejects_full_in_mem_with_workers() -> None:
    raw = config_to_dict(load_experiment_config(Path("configs/ml_baseline.json")))
    assert isinstance(raw, dict)
    loader = raw["loader"]
    assert isinstance(loader, dict)
    loader["num_workers"] = 1

    with pytest.raises(ValueError, match=r"full_in_mem.*num_workers=0"):
        parse_experiment_config(raw)


def test_config_rejects_duplicate_protocols_and_validation_group_columns() -> None:
    raw = config_to_dict(load_experiment_config(Path("configs/ml_baseline.json")))
    assert isinstance(raw, dict)
    data = raw["data"]
    assert isinstance(data, dict)
    protocols = data["protocols"]
    assert isinstance(protocols, list)
    data["protocols"] = [*protocols, protocols[0]]

    with pytest.raises(ValueError, match=r"data\.protocols contains duplicates"):
        parse_experiment_config(raw)

    raw = config_to_dict(load_experiment_config(Path("configs/ml_baseline.json")))
    assert isinstance(raw, dict)
    validation = raw["validation"]
    assert isinstance(validation, dict)
    split = validation["split"]
    assert isinstance(split, dict)
    group_by = split["group_by"]
    assert isinstance(group_by, list)
    split["group_by"] = [*group_by, group_by[0]]

    with pytest.raises(ValueError, match=r"validation\.split\.group_by contains duplicates"):
        parse_experiment_config(raw)


def test_config_rejects_mamba_on_non_cuda_device() -> None:
    raw = config_to_dict(load_experiment_config(Path("configs/ml_baseline.json")))
    assert isinstance(raw, dict)
    run = raw["run"]
    assert isinstance(run, dict)
    run["device"] = "cpu"

    with pytest.raises(ValueError, match="Mamba layers require"):
        parse_experiment_config(raw)


@pytest.mark.parametrize("path", ["configs/ml_dry_run_cpu.json", "configs/ml_dry_run_gpu.json"])
def test_dry_run_configs_load(path: str) -> None:
    config = load_experiment_config(Path(path))

    assert config.loader.strategy == "sequential"
    assert config.train.masked_suffix.roll_forward_steps == 128
    assert config.run.output_dir is None
    assert config.logging.backend == "stdout"


def test_config_rejects_masked_suffix_channels_outside_feedback_columns() -> None:
    config = make_config()

    with pytest.raises(ValueError, match=r"train.masked_suffix.channels"):
        replace(
            config,
            data=replace(config.data, feedback_columns=("voltage",)),
        )


def test_config_requires_masked_suffix_to_cover_all_feedback_columns() -> None:
    config = make_config()

    with pytest.raises(ValueError, match=r"must equal data\.feedback_columns"):
        replace(
            config,
            train=replace(
                config.train,
                masked_suffix=replace(config.train.masked_suffix, channels=("voltage",)),
            ),
        )


def test_config_rejects_train_or_validation_suffix_without_context() -> None:
    config = make_config(seq_len=10, suffix_steps=3)

    with pytest.raises(ValueError, match=r"train\.masked_suffix\.suffix_steps"):
        replace(config, loader=replace(config.loader, seq_len=3))
    with pytest.raises(ValueError, match=r"effective validation\.masked_suffix\.suffix_steps"):
        replace(
            config,
            validation=replace(
                config.validation,
                masked_suffix=ValidationMaskedSuffixConfig(enabled=True, suffix_steps=10),
            ),
        )


def test_config_rejects_validation_mamba_carry_for_mimo_rollouts() -> None:
    config = make_config()

    with pytest.raises(ValueError, match="Mamba state carry"):
        replace(
            config,
            model=replace(
                config.model,
                mamba=replace(config.model.mamba, is_mimo=True),
                layers=(LayerConfig(kind="mamba"),),
            ),
            train=replace(
                config.train,
                masked_suffix=replace(config.train.masked_suffix, carry_mamba_state=False),
            ),
            validation=replace(
                config.validation,
                masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=True),
            ),
        )


def test_config_rejects_validation_mamba_carry_for_teacher_forced_suffix() -> None:
    config = make_config()

    with pytest.raises(ValueError, match="Mamba state carry"):
        replace(
            config,
            model=replace(
                config.model,
                mamba=replace(config.model.mamba, is_mimo=True),
                layers=(LayerConfig(kind="mamba"),),
            ),
            train=replace(
                config.train,
                masked_suffix=replace(config.train.masked_suffix, carry_mamba_state=False),
            ),
            validation=replace(
                config.validation,
                rollout_steps=0,
                max_tf_batches=1,
                masked_suffix=ValidationMaskedSuffixConfig(
                    enabled=True,
                    carry_mamba_state=True,
                ),
            ),
        )


def test_config_requires_rollout_selectors_to_include_enabled_protocol() -> None:
    config = make_config()
    group = config.validation.split.groups[0]

    with pytest.raises(ValueError, match="must include protocol"):
        replace(
            config,
            validation=replace(
                config.validation,
                split=replace(
                    config.validation.split,
                    groups=(replace(group, match={"cell id": "cell-b"}),),
                ),
            ),
        )
    with pytest.raises(ValueError, match=r"must be in data\.protocols"):
        replace(
            config,
            validation=replace(
                config.validation,
                split=replace(
                    config.validation.split,
                    groups=(
                        replace(
                            group,
                            match={
                                "dataset id": "synthetic-ml",
                                "cell id": "cell-b",
                                "cycle index": 2,
                                "protocol": "RPT",
                            },
                        ),
                    ),
                ),
            ),
        )


def test_config_rejects_selector_columns_outside_split_group_by() -> None:
    config = make_config()
    group = config.validation.split.groups[0]

    with pytest.raises(ValueError, match=r"outside validation\.split\.group_by"):
        replace(
            config,
            validation=replace(
                config.validation,
                split=ValidationSplitConfig(
                    strategy="provide",
                    group_by=("dataset id", "cell id"),
                    groups=(replace(group, match={"cell id": "cell-b", "protocol": "cycling"}),),
                ),
            ),
        )


@pytest.mark.parametrize("strategy", ["sample", "merge"])
def test_data_validation_sampling_uses_run_seed(strategy: str) -> None:
    config = make_config()
    groups = config.validation.split.groups if strategy == "merge" else ()
    config = replace(
        config,
        run=replace(config.run, seed=123),
        validation=replace(
            config.validation,
            rollout_steps=0,
            split=ValidationSplitConfig(
                strategy=strategy,  # type: ignore[arg-type]
                fraction=0.5,
                group_by=config.validation.split.group_by,
                groups=groups,
            ),
        ),
    )

    assert data_validation_config(config).seed == 123
