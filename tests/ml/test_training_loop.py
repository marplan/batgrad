from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from batgrad.ml import (
    checkpoint as checkpoint_module,
    train as train_module,
    validation as validation_module,
)
from batgrad.ml.config import CheckpointConfig
from batgrad.ml.distributed import DistributedContext
from batgrad.ml.metrics import LossMetrics
from tests.ml.conftest import (
    MetricLogger,
    RecordingModel,
    StateProbeModel,
    make_batch,
    make_config,
    make_index,
    make_store,
)


class TinyDataset:
    def __init__(self, batches: list[object]) -> None:
        self.batches = batches
        self.epochs: list[int] = []

    def set_epoch(self, epoch_idx: int) -> None:
        self.epochs.append(epoch_idx)

    def steps_per_epoch(self, epoch_idx: int = 0) -> int:
        del epoch_idx
        return len(self.batches)


class TinyLoader:
    def __init__(self, dataset: TinyDataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset.batches)


def test_ddp_rejects_cross_batch_state_carry(monkeypatch) -> None:
    cleanup_calls: list[None] = []
    monkeypatch.setattr(train_module, "cleanup_distributed", lambda: cleanup_calls.append(None))
    distributed = DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="cross-batch Mamba state carry"):
        train_module._validate_distributed_state_carry(make_config(), distributed)

    stateless_config = make_config()
    stateless_config = replace(
        stateless_config,
        loader=replace(stateless_config.loader, stateful_n_windows=1),
    )
    train_module._validate_distributed_state_carry(stateless_config, distributed)
    assert cleanup_calls == [None]


def test_training_setup_failure_cleans_up_distributed_context(monkeypatch) -> None:
    cleanup_calls: list[None] = []
    config = make_config()
    monkeypatch.setattr(
        train_module,
        "init_distributed",
        lambda _device: DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        ),
    )
    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: object())
    monkeypatch.setattr(
        train_module,
        "_create_loaders",
        lambda _config, _store: (_ for _ in ()).throw(RuntimeError("loader failed")),
    )
    monkeypatch.setattr(train_module, "cleanup_distributed", lambda: cleanup_calls.append(None))

    with pytest.raises(RuntimeError, match="loader failed"):
        train_module.train_from_config(Path("memory-config.json"))

    assert cleanup_calls == [None]


def test_training_loop_rolls_into_next_epoch_until_max_steps(monkeypatch) -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        train=replace(config.train, max_steps=3),
        validation=replace(config.validation, rollout_steps=0, max_tf_batches=0),
        run=replace(config.run, output_dir=None),
    )
    batches = [
        make_batch(config, stateful_group_idx=0, stateful_step_idx=0, stateful_steps=2),
        make_batch(config, stateful_group_idx=0, stateful_step_idx=1, stateful_steps=2),
    ]
    dataset = TinyDataset(batches)
    model = StateProbeModel()
    run_logger = MetricLogger()

    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: object())
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(train_module, "build_logger", lambda *_args: run_logger)
    monkeypatch.setattr(
        train_module,
        "_create_loaders",
        lambda _config, _store: (
            TinyLoader(dataset),
            TinyLoader(TinyDataset([])),
            dataset,
            make_index(rows=40, split_cell_b=True),
        ),
    )

    train_module.train_from_config(Path("memory-config.json"))

    assert dataset.epochs == [0, 1]
    assert [step for step, metrics in run_logger.metrics if "train/loss_ce" in metrics] == [1, 2, 3]
    train_forwards = [call for call in model.calls if call["grad_enabled"]]
    assert [call["received_state_value"] for call in train_forwards] == [None, 1.0, None]


def test_training_loop_passes_exact_refreshed_state_to_next_consecutive_batch(monkeypatch) -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, roll_forward_steps=0)
    config = replace(
        config,
        validation=replace(config.validation, rollout_steps=0, max_tf_batches=0),
        run=replace(config.run, output_dir=None),
    )
    batches = [
        make_batch(config, stateful_group_idx=0, stateful_step_idx=0, stateful_steps=2),
        make_batch(config, stateful_group_idx=0, stateful_step_idx=1, stateful_steps=2),
    ]
    dataset = TinyDataset(batches)
    train_loader = TinyLoader(dataset)
    val_loader = TinyLoader(TinyDataset([]))
    model = RecordingModel()

    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: object())
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        train_module,
        "_create_loaders",
        lambda _config, _store: (
            train_loader,
            val_loader,
            dataset,
            make_index(rows=40, split_cell_b=True),
        ),
    )

    train_module.train_from_config(Path("memory-config.json"))

    carried = model.calls[2]["states"]
    assert carried is not None
    assert carried["layer"].angle_state.tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert carried["layer"].angle_state.requires_grad is False
    assert carried["layer"].angle_state.grad_fn is None


def test_training_loop_does_not_carry_state_when_disabled(monkeypatch) -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, carry_mamba_state=False)
    config = replace(
        config,
        validation=replace(config.validation, rollout_steps=0, max_tf_batches=0),
        run=replace(config.run, output_dir=None),
    )
    batches = [
        make_batch(config, stateful_group_idx=0, stateful_step_idx=0, stateful_steps=2),
        make_batch(config, stateful_group_idx=0, stateful_step_idx=1, stateful_steps=2),
    ]
    dataset = TinyDataset(batches)
    train_loader = TinyLoader(dataset)
    val_loader = TinyLoader(TinyDataset([]))
    model = RecordingModel()

    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: object())
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        train_module,
        "_create_loaders",
        lambda _config, _store: (
            train_loader,
            val_loader,
            dataset,
            make_index(rows=40, split_cell_b=True),
        ),
    )

    train_module.train_from_config(Path("memory-config.json"))

    assert all(call["states"] is None for call in model.calls)
    assert not any(call["return_states"] for call in model.calls)


def test_training_loop_drops_state_for_non_consecutive_step_in_same_group(monkeypatch) -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, roll_forward_steps=0)
    config = replace(
        config,
        validation=replace(config.validation, rollout_steps=0, max_tf_batches=0),
        run=replace(config.run, output_dir=None),
    )
    batches = [
        make_batch(config, stateful_group_idx=0, stateful_step_idx=0, stateful_steps=3),
        make_batch(config, stateful_group_idx=0, stateful_step_idx=2, stateful_steps=3),
    ]
    dataset = TinyDataset(batches)
    train_loader = TinyLoader(dataset)
    val_loader = TinyLoader(TinyDataset([]))
    model = RecordingModel()

    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: object())
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        train_module,
        "_create_loaders",
        lambda _config, _store: (
            train_loader,
            val_loader,
            dataset,
            make_index(rows=40, split_cell_b=True),
        ),
    )

    train_module.train_from_config(Path("memory-config.json"))

    assert model.calls[2]["states"] is None


def test_training_loop_skips_teacher_forced_validation_when_disabled(monkeypatch) -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        train=replace(config.train, max_steps=1, validate_every_steps=1),
        validation=replace(config.validation, rollout_steps=0, max_tf_batches=0),
        run=replace(config.run, output_dir=None),
    )
    dataset = TinyDataset([make_batch(config)])
    train_loader = TinyLoader(dataset)
    val_loader = TinyLoader(TinyDataset([make_batch(config)]))
    model = RecordingModel()

    monkeypatch.setattr(train_module, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(train_module, "resolve_store_root", lambda _root: "memory")
    monkeypatch.setattr(train_module, "LocalDataProcessingStore", lambda _root: object())
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        train_module,
        "_create_loaders",
        lambda _config, _store: (
            train_loader,
            val_loader,
            dataset,
            make_index(rows=40, split_cell_b=True),
        ),
    )

    train_module.train_from_config(Path("memory-config.json"))

    assert len(model.calls) == 1


def test_validate_returns_teacher_forced_and_rollout_metrics() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        validation=replace(
            config.validation,
            max_tf_batches=1,
            rollout_steps=4,
            log_rollout_plots=True,
        ),
    )
    result = validation_module.validate(
        config,
        RecordingModel(),
        TinyLoader(TinyDataset([make_batch(config)])),
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert result.teacher_forced_metrics is not None
    assert result.rollout_metrics is not None
    assert result.rollout_examples

    payload = train_module._validation_metric_payload(config, result)

    assert "val/tf/loss_ce" in payload
    assert "val/rollout/loss_ce" in payload


def test_validate_weights_teacher_forced_loss_by_valid_target_count(monkeypatch) -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        validation=replace(config.validation, max_tf_batches=2, rollout_steps=0),
    )
    metrics = iter(
        (
            LossMetrics(
                loss=torch.tensor(1.0),
                feature_loss_sum=torch.tensor([2.0, 2.0]),
                feature_loss_count=torch.tensor([2.0, 2.0]),
            ),
            LossMetrics(
                loss=torch.tensor(9.0),
                feature_loss_sum=torch.tensor([162.0, 162.0]),
                feature_loss_count=torch.tensor([18.0, 18.0]),
            ),
        )
    )

    def next_loss_metrics(*_args, **_kwargs):
        return next(metrics)

    monkeypatch.setattr(validation_module, "batch_loss_with_metrics", next_loss_metrics)
    result = validation_module.validate(
        config,
        RecordingModel(),
        TinyLoader(TinyDataset([make_batch(config), make_batch(config)])),
        object(),
        object(),
        torch.device("cpu"),
    )

    assert result.teacher_forced_metrics is not None
    payload = train_module._validation_metric_payload(config, result)
    assert payload["val/tf/loss_ce"] == pytest.approx(8.2)
    assert payload["val/tf/loss_ce/voltage"] == pytest.approx(8.2)
    assert payload["val/tf/loss_ce/temperature"] == pytest.approx(8.2)


def test_validation_checkpoint_monitor_saves_only_improved_best(tmp_path: Path) -> None:
    config = make_config()
    config = replace(
        config,
        run=replace(config.run, output_dir=str(tmp_path / "runs")),
        checkpoint=CheckpointConfig(
            save_latest=True,
            save_best=True,
            monitors=("val/rollout/loss_ce",),
        ),
    )
    model = RecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    best: dict[str, float] = {}

    checkpoint_module.save_validation_checkpoints(
        config,
        model,
        optimizer,
        scheduler,
        scaler,
        tmp_path,
        step=1,
        epoch_idx=0,
        epoch_step=1,
        metrics={"val/rollout/loss_ce": 2.0},
        best=best,
    )
    first_best = (tmp_path / "best_val_rollout_loss_ce.pt").stat().st_mtime_ns
    checkpoint_module.save_validation_checkpoints(
        config,
        model,
        optimizer,
        scheduler,
        scaler,
        tmp_path,
        step=2,
        epoch_idx=0,
        epoch_step=2,
        metrics={"val/rollout/loss_ce": 3.0},
        best=best,
    )

    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best_val_rollout_loss_ce.pt").is_file()
    assert best == {"val/rollout/loss_ce": 2.0}
    assert (tmp_path / "best_val_rollout_loss_ce.pt").stat().st_mtime_ns == first_best
