from __future__ import annotations

import math
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch

from batgrad.contracts.mapping import BaseColumns
from batgrad.ml import train as train_module, validation as validation_module
from batgrad.ml.config import (
    ScalingRuleConfig,
    ValidationGroupConfig,
    ValidationMaskedSuffixConfig,
    ValidationSplitConfig,
)
from batgrad.ml.nn import LayerConfig
from tests.ml.conftest import (
    INPUT_COLUMNS,
    TINY_GIT_COMMIT,
    TINY_MANIFEST_PATH,
    MetricLogger,
    StateProbeModel,
    make_config,
    make_memory_manifest_store,
)


def _real_loader_config(*, max_steps: int):
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3)
    return replace(
        config,
        data=replace(
            config.data,
            manifest_paths={TINY_MANIFEST_PATH: TINY_GIT_COMMIT},
            store_root="memory",
            scaling=tuple(
                ScalingRuleConfig(column=column, input_min=0.0, input_max=2000.0)
                for column in INPUT_COLUMNS
            ),
        ),
        train=replace(config.train, max_steps=max_steps, validate_every_steps=None),
        validation=replace(
            config.validation,
            split=ValidationSplitConfig(strategy="sample", fraction=0.0),
            rollout_steps=0,
            max_tf_batches=0,
        ),
        run=replace(config.run, output_dir=None),
    )


def test_training_builds_one_index_for_both_split_loaders(monkeypatch) -> None:
    config = _real_loader_config(max_steps=1)
    store = make_memory_manifest_store()
    original = train_module.create_index
    calls = 0

    def capture_index(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "create_index", capture_index)

    train_loader, val_loader, _train_dataset, index = train_module._create_loaders(config, store)

    assert calls == 1
    assert train_loader.dataset.full_index is index
    assert val_loader.dataset.full_index is index


def _patch_real_loader_training(monkeypatch, config, store, model: StateProbeModel) -> None:
    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: store)
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: model)


def test_training_uses_real_loader_batches_for_masking_and_state_carry(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=3)
    model = StateProbeModel()
    _patch_real_loader_training(monkeypatch, config, store, model)

    train_module.train_from_config(Path("tiny-config.json"))

    # A real materialized batch reaches the suffix path: selected feedback bins
    # are hidden while the preceding context remains visible.
    first_input = model.calls[0]["x"]
    assert tuple(first_input.shape) == (3, 10, 4, 5)
    assert torch.equal(first_input[:, -3:, 2:4, :], torch.zeros_like(first_input[:, -3:, 2:4, :]))
    assert bool(first_input[:, :-3, 2:4, :].abs().sum().item() > 0)

    # Each train batch has a suffix pass and a final refreshed-state pass. The
    # second stateful step receives that refreshed 2x2 state; a new plan group resets it.
    train_forwards = [call for call in model.calls if call["grad_enabled"]]
    assert [call["received_state_value"] for call in train_forwards] == [None, 1.0, None]
    assert model.bias.grad is not None


def test_training_updates_a_real_cpu_ffn_model_from_memory_manifest(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=1)
    config = replace(
        config,
        loader=replace(config.loader, data_access="full_in_mem", stateful_n_windows=1),
        model=replace(
            config.model,
            layers=(LayerConfig(kind="reduce", mode="sum_pool"), LayerConfig(kind="ffn")),
        ),
        train=replace(
            config.train,
            masked_suffix=replace(config.train.masked_suffix, carry_mamba_state=False),
        ),
        validation=replace(
            config.validation,
            split=ValidationSplitConfig(strategy="sample", fraction=0.5),
        ),
    )
    build_model = train_module.build_model
    created: list[tuple[torch.nn.Module, tuple[torch.Tensor, ...]]] = []

    def capture_model(*args, **kwargs):
        model = build_model(*args, **kwargs)
        initial_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
        created.append((model, initial_parameters))
        return model

    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: store)
    monkeypatch.setattr(train_module, "build_model", capture_model)
    run_logger = MetricLogger()
    monkeypatch.setattr(train_module, "build_logger", lambda *_args: run_logger)

    train_module.train_from_config(Path("tiny-config.json"))

    model, initial_parameters = created[0]
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(
        not torch.equal(initial, current)
        for initial, current in zip(initial_parameters, model.parameters(), strict=True)
    )
    assert any(
        math.isfinite(float(metrics["train/loss_ce"]))
        for _step, metrics in run_logger.metrics
        if "train/loss_ce" in metrics
    )


def test_validation_override_uses_real_loader_without_training_roll_forward(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=1)
    validation_group = ValidationGroupConfig(
        match={BaseColumns.set_id: "tiny-ml", BaseColumns.cell_id: "cell-b"},
    )
    config = replace(
        config,
        train=replace(config.train, validate_every_steps=1),
        validation=replace(
            config.validation,
            split=ValidationSplitConfig(
                strategy="provide",
                group_by=(
                    BaseColumns.set_id,
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                    BaseColumns.proto,
                ),
                groups=(validation_group,),
            ),
            max_tf_batches=1,
            rollout_steps=0,
            masked_suffix=ValidationMaskedSuffixConfig(enabled=False),
        ),
    )
    model = StateProbeModel()
    _patch_real_loader_training(monkeypatch, config, store, model)

    train_module.train_from_config(Path("tiny-config.json"))

    # Training has a suffix state-refresh pair. Validation is teacher forced,
    # still uses the configured 10-step context, and never inherits roll-forward.
    assert len(model.calls) == 3
    assert model.calls[-1]["x"].shape[1] == 10
    assert bool(model.calls[-1]["x"][:, -3:, 2:4, :].abs().sum().item() > 0)


def test_training_roll_forward_keeps_context_size_across_real_loader_windows(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=1)
    config = replace(
        config,
        loader=replace(config.loader, stateful_n_windows=1),
        train=replace(
            config.train,
            masked_suffix=replace(
                config.train.masked_suffix,
                roll_forward_steps=6,
                carry_mamba_state=False,
            ),
        ),
    )
    model = StateProbeModel()
    _patch_real_loader_training(monkeypatch, config, store, model)

    train_module.train_from_config(Path("tiny-config.json"))

    assert [call["x"].shape[1] for call in model.calls] == [10, 10, 10]
    assert all(call["states"] is None for call in model.calls)


def test_validation_suffix_overrides_length_and_carry_with_real_batch(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=1)
    validation_group = ValidationGroupConfig(
        match={BaseColumns.set_id: "tiny-ml", BaseColumns.cell_id: "cell-b"},
    )
    config = replace(
        config,
        train=replace(config.train, validate_every_steps=1),
        validation=replace(
            config.validation,
            split=ValidationSplitConfig(
                strategy="provide",
                group_by=(
                    BaseColumns.set_id,
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                    BaseColumns.proto,
                ),
                groups=(validation_group,),
            ),
            max_tf_batches=1,
            rollout_steps=0,
            masked_suffix=ValidationMaskedSuffixConfig(
                enabled=True,
                suffix_steps=2,
                carry_mamba_state=False,
            ),
        ),
    )
    model = StateProbeModel()
    _patch_real_loader_training(monkeypatch, config, store, model)

    train_module.train_from_config(Path("tiny-config.json"))

    validation_call = model.calls[-1]
    assert torch.equal(
        validation_call["x"][:, -2:, 2:4, :],
        torch.zeros_like(validation_call["x"][:, -2:, 2:4, :]),
    )
    assert bool(validation_call["x"][:, -3, 2:4, :].abs().sum().item() > 0)
    assert validation_call["return_states"] is False


def test_training_resolves_validation_cell_selector_with_real_loader(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=1)
    selector = {
        BaseColumns.set_id: "tiny-ml",
        BaseColumns.cell_id: "cell-b",
        BaseColumns.cidx: 2,
        BaseColumns.proto: "cycling",
    }
    config = replace(
        config,
        loader=replace(config.loader, batch_size=1, stateful_n_windows=1),
        train=replace(config.train, validate_every_steps=1),
        validation=replace(
            config.validation,
            split=ValidationSplitConfig(
                strategy="provide",
                group_by=(
                    BaseColumns.set_id,
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                    BaseColumns.proto,
                ),
                groups=(ValidationGroupConfig(match=selector),),
            ),
            max_tf_batches=1,
            rollout_steps=0,
        ),
    )
    model = StateProbeModel()
    _patch_real_loader_training(monkeypatch, config, store, model)
    observed_groups: list[tuple[object, ...]] = []

    def capture_validation(_config, _model, val_loader, *_args, **_kwargs):
        observed_groups.append(next(iter(val_loader)).state.group_keys[0])
        return validation_module.ValidationResult()

    monkeypatch.setattr(train_module, "validate", capture_validation)

    train_module.train_from_config(Path("tiny-config.json"))

    assert observed_groups == [("tiny-ml", "cell-b", 2, "cycling")]


def test_training_carries_probe_state_across_a_real_protocol_chain(monkeypatch) -> None:
    store = make_memory_manifest_store()
    config = _real_loader_config(max_steps=5)
    config = replace(
        config,
        loader=replace(config.loader, batch_size=1, stateful_n_windows=-1),
    )
    model = StateProbeModel()
    _patch_real_loader_training(monkeypatch, config, store, model)
    created_datasets = []
    create_loaders = train_module._create_loaders

    def capture_loader_creation(*args, **kwargs):
        loaders = create_loaders(*args, **kwargs)
        created_datasets.append(loaders[2])
        return loaders

    monkeypatch.setattr(train_module, "_create_loaders", capture_loader_creation)

    train_module.train_from_config(Path("tiny-config.json"))

    dataset = created_datasets[0]
    dataset.set_epoch(0)
    batches = list(islice(dataset, 5))
    assert [str(batch.state.protocols[0]) for batch in batches] == [
        "cycling",
        "cycling",
        "cycling",
        "cycling",
        "HPPC",
    ]
    train_forwards = [call for call in model.calls if call["grad_enabled"]]
    assert [call["received_state_value"] for call in train_forwards] == [
        None,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
