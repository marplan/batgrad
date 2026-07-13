from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml import validation as validation_module
from batgrad.ml.config import (
    RolloutExtensionConfig,
    ValidationMaskedSuffixConfig,
    resolved_validation_masked_suffix,
)
from batgrad.ml.data.index import MlDatasetIndex
from batgrad.ml.experiment import encode_inputs
from batgrad.ml.rollout import RolloutResult, rollout_batch
from batgrad.ml.validation import run_rollouts
from tests.ml.conftest import (
    RecordingModel,
    StateProbeModel,
    make_batch,
    make_config,
    make_index,
    make_store,
)


def _rollout(
    config,
    model,
    inputs,
    targets,
    mask,
    context_len,
    rollout_len,
    device,
    *,
    masked_suffix,
):
    suffix = replace(
        resolved_validation_masked_suffix(config),
        enabled=masked_suffix,
    )
    return rollout_batch(
        config,
        model,
        inputs,
        context_len=context_len,
        rollout_steps=rollout_len,
        suffix=suffix,
        device=device,
        targets=targets,
        mask=mask,
    )


def _loss_count(result: RolloutResult) -> float:
    assert result.metrics is not None
    assert result.metrics.feature_loss_count is not None
    return float(result.metrics.feature_loss_count.sum().item())


def test_one_step_rollout_returns_requested_prediction_length() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=False),
    )
    batch = make_batch(config, seq_len=14)
    model = RecordingModel()

    result = _rollout(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        context_len=10,
        rollout_len=4,
        device=torch.device("cpu"),
        masked_suffix=False,
    )

    assert tuple(result.prediction.shape) == (1, 4, 2)
    assert _loss_count(result) == 8.0
    assert len(model.calls) == 4


def test_one_step_rollout_reuses_decoded_scalar_feedback() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        feedback_mode="decoded_scalar",
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=False),
    )
    batch = make_batch(config, seq_len=12)
    model = RecordingModel()

    _rollout(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        context_len=10,
        rollout_len=2,
        device=torch.device("cpu"),
        masked_suffix=False,
    )

    known_future = encode_inputs(config, batch.inputs[:1, 10:11, :], torch.device("cpu"))
    assert len(model.calls) == 2
    assert not torch.equal(model.calls[1]["x"][:, -1:, 2:4, :], known_future[:, :, 2:4, :])


def test_rollout_without_targets_returns_predictions_without_metrics() -> None:
    config = make_config(batch_size=1, seq_len=10)
    batch = make_batch(config, seq_len=13)
    suffix = replace(resolved_validation_masked_suffix(config), enabled=False)

    result = rollout_batch(
        config,
        RecordingModel(),
        batch.inputs[:1],
        context_len=10,
        rollout_steps=3,
        suffix=suffix,
        device=torch.device("cpu"),
    )

    assert tuple(result.prediction.shape) == (1, 3, 2)
    assert result.metrics is None


def test_rollout_requires_targets_and_mask_together() -> None:
    config = make_config(batch_size=1, seq_len=10)
    batch = make_batch(config, seq_len=11)

    with pytest.raises(ValueError, match="both be provided or both omitted"):
        rollout_batch(
            config,
            RecordingModel(),
            batch.inputs[:1],
            context_len=10,
            rollout_steps=1,
            suffix=replace(resolved_validation_masked_suffix(config), enabled=False),
            device=torch.device("cpu"),
            targets=batch.targets[:1],
        )


def test_rollout_rejects_suffix_that_covers_the_context() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    batch = make_batch(config, seq_len=14)

    with pytest.raises(ValueError, match="suffix_steps must be smaller"):
        rollout_batch(
            config,
            RecordingModel(),
            batch.inputs,
            context_len=10,
            rollout_steps=4,
            suffix=replace(config.train.masked_suffix, suffix_steps=10),
            device=torch.device("cpu"),
        )


def test_one_step_rollout_scores_the_target_of_the_last_context_logit() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=False),
    )
    inputs = torch.zeros((1, 11, 4))
    targets = torch.full((1, 11, 2), -1.0)
    # The final context logit at index 9 predicts target index 9, not index 10.
    targets[:, 9, :] = 1.0
    mask = torch.ones((1, 11), dtype=torch.bool)

    result = _rollout(
        config,
        RecordingModel(),
        inputs,
        targets,
        mask,
        context_len=10,
        rollout_len=1,
        device=torch.device("cpu"),
        masked_suffix=False,
    )

    assert _loss_count(result) == 2.0
    assert result.metrics is not None
    assert result.metrics.loss.item() < 1.0


def test_masked_suffix_rollout_advances_in_suffix_chunks() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=False),
    )
    batch = make_batch(config, seq_len=17)
    model = RecordingModel()

    result = _rollout(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        context_len=10,
        rollout_len=7,
        device=torch.device("cpu"),
        masked_suffix=True,
    )

    assert tuple(result.prediction.shape) == (1, 7, 2)
    assert _loss_count(result) == 14.0
    assert len(model.calls) == 3
    assert [call["x"].shape[1] for call in model.calls] == [10, 10, 10]


def test_masked_suffix_rollout_scores_logits_that_predict_masked_input_rows() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=False),
    )
    inputs = torch.zeros((1, 13, 4))
    targets = torch.full((1, 13, 2), -1.0)
    # The first chunk masks input rows 10:13 and scores logits/targets 9:12.
    targets[:, 9:12, :] = 1.0
    mask = torch.ones((1, 13), dtype=torch.bool)

    result = _rollout(
        config,
        RecordingModel(),
        inputs,
        targets,
        mask,
        context_len=10,
        rollout_len=3,
        device=torch.device("cpu"),
        masked_suffix=True,
    )

    assert result.metrics is not None
    assert result.metrics.loss.item() < 1.0


def test_masked_suffix_rollout_carries_state_between_chunks() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3, carry_mamba_state=True)
    batch = make_batch(config, seq_len=17)
    model = StateProbeModel()

    _rollout(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        context_len=10,
        rollout_len=7,
        device=torch.device("cpu"),
        masked_suffix=True,
    )

    # Each suffix chunk is followed by a prefix advance. The next chunk must
    # receive the state generated by that exact prefix, not merely any state.
    assert [call["received_state_value"] for call in model.calls] == [
        None,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
    ]


def test_masked_suffix_rollout_advances_state_by_the_next_chunk_shift() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3, carry_mamba_state=True)
    batch = make_batch(config, seq_len=17)
    model = StateProbeModel(advance_by_sequence_length=True)

    _rollout(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        context_len=10,
        rollout_len=7,
        device=torch.device("cpu"),
        masked_suffix=True,
    )

    # The final one-step suffix window starts one row after the prior window,
    # so its carried state must advance by one, not the previous three-step chunk.
    assert [call["received_state_value"] for call in model.calls] == [
        None,
        3.0,
        3.0,
        6.0,
        6.0,
        7.0,
    ]


def test_one_step_rollout_carries_state_between_steps() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=True),
    )
    batch = make_batch(config, seq_len=13)
    model = StateProbeModel()

    _rollout(
        config,
        model,
        batch.inputs[:1],
        batch.targets[:1],
        batch.mask[:1],
        context_len=10,
        rollout_len=3,
        device=torch.device("cpu"),
        masked_suffix=False,
    )

    assert [call["received_state_value"] for call in model.calls[::2]] == [None, 1.0, 2.0]


def test_masked_suffix_rollout_scores_all_targets_when_configured() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3, loss_on_masked_only=False)
    config = replace(
        config,
        data=replace(config.data, target_columns=(*config.data.target_columns, "time")),
    )
    batch = make_batch(config, seq_len=13)
    targets = torch.cat((batch.targets[:1], batch.inputs[:1, :, :1]), dim=-1)

    result = _rollout(
        config,
        RecordingModel(target_count=3),
        batch.inputs[:1],
        targets,
        batch.mask[:1],
        context_len=10,
        rollout_len=3,
        device=torch.device("cpu"),
        masked_suffix=True,
    )

    assert _loss_count(result) == 9.0


def test_masked_suffix_rollout_ignores_invalid_future_targets() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        validation_masked_suffix=ValidationMaskedSuffixConfig(carry_mamba_state=False),
    )
    batch = make_batch(config, seq_len=14, all_valid=False)
    mask = batch.mask[:1].clone()
    mask[:, 12:] = False

    result = _rollout(
        config,
        RecordingModel(),
        batch.inputs[:1],
        batch.targets[:1],
        mask,
        context_len=10,
        rollout_len=4,
        device=torch.device("cpu"),
        masked_suffix=True,
    )

    assert _loss_count(result) == 6.0


def test_run_rollouts_extension_marks_every_extension_target_unscored(monkeypatch) -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        validation_masked_suffix=ValidationMaskedSuffixConfig(
            enabled=False,
            carry_mamba_state=False,
        ),
    )
    config = replace(
        config,
        validation=replace(
            config.validation,
            rollout_extension=RolloutExtensionConfig(
                enabled=True,
                steps=2,
                input_values={"time": 0.0, "current": 1.0},
            ),
        ),
    )
    captured_masks: list[torch.Tensor] = []

    def capture_rollout(
        _config,
        _model,
        inputs,
        *,
        context_len,
        rollout_steps,
        suffix,
        device,
        targets=None,
        mask=None,
    ) -> RolloutResult:
        del context_len, suffix, targets
        assert mask is not None
        captured_masks.append(mask.clone())
        target_count = len(_config.data.target_columns)
        return RolloutResult(
            prediction=torch.zeros((inputs.shape[0], rollout_steps, target_count), device=device),
            metrics=None,
            target_start=inputs.shape[1] - rollout_steps - 1,
        )

    monkeypatch.setattr(validation_module, "rollout_batch", capture_rollout)
    run_rollouts(
        config,
        RecordingModel(),
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert len(captured_masks) == 1
    assert captured_masks[0][0, :13].all()
    assert not captured_masks[0][0, 13:].any()


def test_run_rollouts_resolves_validation_selector_and_returns_metrics() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    result = run_rollouts(
        config,
        RecordingModel(),
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert result.rollout_metrics is not None


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        (8, "complete context window"),
        (36, "enough observed future rows"),
    ],
)
def test_run_rollouts_rejects_anchor_outside_observed_horizon(
    anchor: int,
    message: str,
) -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    group = replace(config.validation.split.groups[0], rollout_start_offsets=(anchor,))
    config = replace(
        config,
        validation=replace(
            config.validation,
            split=replace(config.validation.split, groups=(group,)),
        ),
    )

    with pytest.raises(ValueError, match=message):
        run_rollouts(
            config,
            RecordingModel(),
            make_index(rows=40, split_cell_b=True),
            make_store(rows=40),
            torch.device("cpu"),
        )


def test_run_rollouts_matches_selector_by_column_name_not_group_by_order() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        validation=replace(
            config.validation,
            split=replace(
                config.validation.split,
                group_by=(
                    BaseColumns.proto,
                    BaseColumns.cidx,
                    BaseColumns.cell_id,
                    BaseColumns.set_id,
                ),
            ),
        ),
    )

    result = run_rollouts(
        config,
        RecordingModel(),
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert result.rollout_metrics is not None


def test_run_rollouts_requires_selector_to_match_one_stream() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    index = make_index(rows=40, split_cell_b=True)
    missing = replace(
        config.validation.split.groups[0],
        match={
            BaseColumns.cell_id: "missing-cell",
            BaseColumns.proto: str(DatasetProtocolId.cycling),
        },
    )
    config = replace(
        config,
        validation=replace(
            config.validation,
            split=replace(config.validation.split, groups=(missing,)),
        ),
    )

    with pytest.raises(ValueError, match="must match exactly one stream"):
        run_rollouts(
            config,
            RecordingModel(),
            index,
            make_store(rows=40),
            torch.device("cpu"),
        )


def test_run_rollouts_rejects_selector_matching_multiple_validation_streams() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    index = MlDatasetIndex(
        make_index(rows=40, split_cell_b=False).frame.with_columns(
            pl.when(pl.col(BaseColumns.cell_id) == "cell-b")
            .then(pl.lit(BaseColumns.split.values.val))
            .otherwise(pl.lit(BaseColumns.split.values.train))
            .alias(BaseColumns.split)
        )
    )
    ambiguous = replace(
        config.validation.split.groups[0],
        match={BaseColumns.cell_id: "cell-b", BaseColumns.proto: str(DatasetProtocolId.cycling)},
    )
    config = replace(
        config,
        validation=replace(
            config.validation,
            split=replace(config.validation.split, groups=(ambiguous,)),
        ),
    )

    with pytest.raises(ValueError, match="must match exactly one stream"):
        run_rollouts(
            config,
            RecordingModel(),
            index,
            make_store(rows=40),
            torch.device("cpu"),
        )


def test_run_rollouts_validation_override_selects_one_step_mode() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        suffix_steps=3,
        validation_masked_suffix=ValidationMaskedSuffixConfig(
            enabled=False,
            carry_mamba_state=False,
        ),
    )
    model = RecordingModel()

    run_rollouts(
        config,
        model,
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert len(model.calls) == config.validation.rollout_steps


def test_run_rollouts_extension_increases_one_step_prediction_horizon() -> None:
    config = make_config(
        batch_size=1,
        seq_len=10,
        validation_masked_suffix=ValidationMaskedSuffixConfig(
            enabled=False,
            carry_mamba_state=False,
        ),
    )
    config = replace(
        config,
        validation=replace(
            config.validation,
            rollout_extension=RolloutExtensionConfig(
                enabled=True,
                steps=2,
                input_values={"time": 0.0, "current": 1.0},
            ),
        ),
    )
    model = RecordingModel()

    run_rollouts(
        config,
        model,
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert len(model.calls) == config.validation.rollout_steps + 2


def test_run_rollouts_extension_increases_masked_suffix_prediction_horizon() -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        validation=replace(
            config.validation,
            rollout_steps=4,
            rollout_extension=RolloutExtensionConfig(
                enabled=True,
                steps=2,
                input_values={"time": 0.0, "current": 1.0},
            ),
        ),
    )
    model = RecordingModel()

    result = run_rollouts(
        config,
        model,
        make_index(rows=40, split_cell_b=True),
        make_store(rows=40),
        torch.device("cpu"),
    )

    assert result.rollout_metrics is not None
    assert len(model.calls) == 4
