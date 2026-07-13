from __future__ import annotations

from dataclasses import replace

import torch

from batgrad.ml.objective import (
    backward_batch_loss_with_metrics,
    batch_loss_count,
    batch_loss_with_metrics,
    masked_suffix_loss_with_metrics,
)
from tests.ml.conftest import RecordingModel, StateProbeModel, make_batch, make_config


def test_loss_on_masked_only_counts_only_suffix_target_channels() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, loss_on_masked_only=True)
    batch = make_batch(config)

    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.feature_loss_count is not None
    assert metrics.feature_loss_count.tolist() == [9.0, 9.0]


def test_loss_on_masked_only_counts_only_configured_suffix_channel() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, loss_on_masked_only=True)
    config = replace(
        config,
        data=replace(config.data, feedback_columns=("voltage",)),
        train=replace(
            config.train,
            masked_suffix=replace(config.train.masked_suffix, channels=("voltage",)),
        ),
    )
    batch = make_batch(config)

    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.feature_loss_count is not None
    assert metrics.feature_loss_count.tolist() == [9.0, 0.0]


def test_loss_on_masked_only_false_counts_all_valid_targets() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, loss_on_masked_only=False)
    batch = make_batch(config)

    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.feature_loss_count is not None
    assert metrics.feature_loss_count.tolist() == [30.0, 30.0]


def test_roll_forward_loss_counts_all_masked_windows() -> None:
    config = make_config(
        batch_size=3,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=7,
        loss_on_masked_only=True,
        carry_mamba_state=False,
    )
    batch = make_batch(config, seq_len=17)

    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.feature_loss_count is not None
    assert metrics.feature_loss_count.tolist() == [30.0, 30.0]


def test_roll_forward_backward_count_excludes_non_finite_targets() -> None:
    config = make_config(
        batch_size=3,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=7,
        loss_on_masked_only=True,
        carry_mamba_state=False,
    )
    batch = make_batch(config, seq_len=17)
    targets = batch.targets.clone()
    targets[:, :, 0] = float("nan")

    count = batch_loss_count(
        config,
        targets,
        batch.mask,
        suffix=config.train.masked_suffix,
    )
    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
    )

    assert metrics.feature_loss_count is not None
    assert count.item() == metrics.feature_loss_count.sum().item() == 30.0


def test_all_invalid_targets_produce_zero_gradients_without_backward_failure() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3, roll_forward_steps=3)
    batch = make_batch(config, seq_len=13)
    model = RecordingModel()

    metrics = backward_batch_loss_with_metrics(
        config,
        model,
        batch.inputs,
        torch.full_like(batch.targets, float("nan")),
        batch.mask,
        torch.device("cpu"),
        torch.amp.GradScaler("cuda", enabled=False),
        suffix=config.train.masked_suffix,
        collect_metrics=True,
    )

    assert metrics.loss.item() == 0.0
    assert model.bias.grad is not None
    assert model.bias.grad.item() == 0.0


def test_padded_suffix_rows_do_not_contribute_to_batch_loss() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3)
    batch = make_batch(config, all_valid=False)

    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.feature_loss_count is not None
    assert metrics.feature_loss_count.tolist() == [8.0, 8.0]


def test_model_receives_attention_mask_only_for_padded_batches() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3)
    all_valid = make_batch(config)
    padded = make_batch(config, all_valid=False)
    model = RecordingModel()

    batch_loss_with_metrics(
        config,
        model,
        all_valid.inputs,
        all_valid.targets,
        all_valid.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=all_valid.all_valid,
    )
    batch_loss_with_metrics(
        config,
        model,
        padded.inputs,
        padded.targets,
        padded.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=padded.all_valid,
    )

    assert model.calls[0]["mask"] is None
    assert torch.equal(model.calls[1]["mask"], padded.mask)


def test_masked_feedback_channels_are_zeroed_after_encoding() -> None:
    config = make_config(batch_size=3, seq_len=10, suffix_steps=3)
    batch = make_batch(config)
    model = RecordingModel()

    batch_loss_with_metrics(
        config,
        model,
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    encoded = model.calls[0]["x"]
    assert torch.equal(encoded[:, -3:, 2:4, :], torch.zeros_like(encoded[:, -3:, 2:4, :]))
    assert bool(encoded[:, :7, 2:4, :].abs().sum().item() > 0)


def test_roll_forward_reuses_probability_feedback_overrides() -> None:
    config = make_config(
        batch_size=3,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=3,
        carry_mamba_state=False,
    )
    batch = make_batch(config, seq_len=13)
    model = RecordingModel()

    masked_suffix_loss_with_metrics(
        config,
        model,
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert len(model.calls) >= 2
    second_encoded = model.calls[1]["x"]
    # First window predictions are written to global positions 7:10; in the next
    # window, whose global slice starts at 3, that appears at local positions 4:7.
    assert bool(second_encoded[:, 4:7, 2:4, :].sum().item() > 0)


def test_roll_forward_passes_prefix_state_to_next_training_window() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=3,
        carry_mamba_state=True,
    )
    batch = make_batch(config, seq_len=13)
    model = StateProbeModel()

    masked_suffix_loss_with_metrics(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=True,
    )

    assert [call["received_state_value"] for call in model.calls] == [None, None, 1.0]


def test_detach_between_windows_controls_training_state_graph() -> None:
    base = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=3,
        carry_mamba_state=True,
    )
    batch = make_batch(base, seq_len=13)

    received_state_requires_grad = []
    for detach in (True, False):
        config = replace(
            base,
            train=replace(
                base.train,
                masked_suffix=replace(base.train.masked_suffix, detach_between_windows=detach),
            ),
        )
        model = StateProbeModel()
        masked_suffix_loss_with_metrics(
            config,
            model,
            batch.inputs[:1],
            batch.targets[:1],
            batch.mask[:1],
            torch.device("cpu"),
            suffix=config.train.masked_suffix,
            mask_all_valid=True,
        )
        next_window_state = model.calls[-1]["states"]
        assert next_window_state is not None
        received_state_requires_grad.append(next_window_state["layer"].angle_state.requires_grad)

    assert received_state_requires_grad == [False, True]


def test_roll_forward_feedback_uses_logits_before_masked_input_suffix() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=3,
        carry_mamba_state=False,
    )
    batch = make_batch(config, seq_len=13)
    model = RecordingModel(position_bins=True)

    masked_suffix_loss_with_metrics(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    second_encoded = model.calls[1]["x"]
    # The first window writes logits at positions 6:9 into masked input rows 7:10.
    # The next window starts at row 3, so those predictions appear at local 4:7.
    bins = second_encoded[0, 4:7, 2:4, :].argmax(dim=-1)
    assert torch.equal(bins, torch.tensor([[1, 1], [2, 2], [3, 3]]))


def test_masked_suffix_loss_scores_logits_that_predict_masked_input_rows() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3, carry_mamba_state=False)
    inputs = torch.zeros((1, 10, 4))
    targets = torch.full((1, 10, 2), -1.0)
    # Logits 6:9 predict the masked input rows 7:10 and therefore target indices 6:9.
    targets[:, 6:9, :] = 1.0
    mask = torch.ones((1, 10), dtype=torch.bool)

    metrics = batch_loss_with_metrics(
        config,
        RecordingModel(),
        inputs,
        targets,
        mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=True,
    )

    assert metrics.loss.item() < 1.0


def test_roll_forward_reuses_decoded_scalar_feedback() -> None:
    config = make_config(
        batch_size=3,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=3,
        carry_mamba_state=False,
        feedback_mode="decoded_scalar",
    )
    batch = make_batch(config, seq_len=13)
    model = RecordingModel()

    masked_suffix_loss_with_metrics(
        config,
        model,
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert len(model.calls) >= 2
    second_encoded = model.calls[1]["x"]
    assert not torch.equal(second_encoded[:, 4:7, 2:4, :], model.calls[0]["x"][:, 4:7, 2:4, :])


def test_roll_forward_backward_path_accumulates_gradient() -> None:
    config = make_config(
        batch_size=3,
        seq_len=10,
        suffix_steps=3,
        roll_forward_steps=3,
        carry_mamba_state=False,
    )
    batch = make_batch(config, seq_len=13)
    model = RecordingModel()
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    metrics = backward_batch_loss_with_metrics(
        config,
        model,
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        scaler,
        suffix=config.train.masked_suffix,
        collect_metrics=True,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.loss.item() > 0
    assert model.bias.grad is not None
    assert len(model.calls) == 2


def test_teacher_forced_path_when_masked_suffix_disabled() -> None:
    config = make_config(batch_size=3, seq_len=10)
    config = replace(
        config,
        train=replace(
            config.train,
            masked_suffix=replace(config.train.masked_suffix, enabled=False),
        ),
    )
    batch = make_batch(config)
    model = RecordingModel()

    metrics = batch_loss_with_metrics(
        config,
        model,
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        mask_all_valid=batch.all_valid,
    )

    assert metrics.feature_loss_count is not None
    assert metrics.feature_loss_count.tolist() == [30.0, 30.0]
    assert len(model.calls) == 1
