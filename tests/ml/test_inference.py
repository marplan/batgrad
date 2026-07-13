from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import torch

from batgrad.contracts.mapping import BaseColumns
from batgrad.ml.config import config_to_dict
from batgrad.ml.inference import CheckpointSelection, evaluate_checkpoints
from batgrad.ml.nn import LayerConfig, build_model
from batgrad.viz.ml import inference_metrics_frame
from tests.ml.conftest import make_config, make_index, make_store


def test_inference_loads_checkpoint_and_runs_batched_rollout(tmp_path: Path) -> None:
    config = make_config(batch_size=2, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        model=replace(
            config.model,
            layers=(LayerConfig(kind="reduce", mode="sum_pool"), LayerConfig(kind="ffn")),
        ),
    )
    model = build_model(config, torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": config_to_dict(config), "model": model.state_dict(), "step": 4},
        checkpoint_path,
    )
    selected = (
        make_index(rows=40, split_cell_b=False)
        .frame.filter(pl.col(BaseColumns.proto) == "cycling")
        .head(2)
    )

    result = evaluate_checkpoints(
        make_store(rows=40),
        selected,
        (CheckpointSelection(alias="candidate", path=str(checkpoint_path)),),
        device=torch.device("cpu"),
        suffix_steps=(0, 3),
        rollout_steps=4,
    )

    assert tuple(result.inputs.shape) == (2, 14, 4)
    assert tuple(result.targets.shape) == (2, 14, 2)
    assert len(result.predictions) == 2
    assert all(tuple(series.predictions.shape) == (2, 4, 2) for series in result.predictions)
    assert {series.suffix_steps for series in result.predictions} == {0, 3}
    assert all(series.target_start == 9 for series in result.predictions)
    assert all(series.metrics is not None for series in result.predictions)

    metrics = inference_metrics_frame(result)

    assert metrics.height == 2
    assert metrics["strategy"].to_list() == ["classic", "masked_suffix"]
    for row, series in zip(metrics.iter_rows(named=True), result.predictions, strict=True):
        assert series.metrics is not None
        assert row["loss_ce"] == pytest.approx(float(series.metrics.loss))


@pytest.mark.parametrize(
    ("suffix_steps", "rollout_steps", "message"),
    [
        ((), 4, "Suffix steps"),
        ((-1,), 4, "Suffix steps"),
        ((0,), 0, "Rollout steps"),
    ],
)
def test_checkpoint_evaluation_validates_runtime_options(
    suffix_steps: tuple[int, ...],
    rollout_steps: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_checkpoints(
            make_store(rows=40),
            make_index(rows=40, split_cell_b=False).frame.head(1),
            (CheckpointSelection(alias="candidate", path="missing.pt"),),
            device=torch.device("cpu"),
            suffix_steps=suffix_steps,
            rollout_steps=rollout_steps,
        )


def test_checkpoint_evaluation_rejects_empty_index_before_loading() -> None:
    with pytest.raises(ValueError, match="index rows"):
        evaluate_checkpoints(
            make_store(rows=40),
            pl.DataFrame(),
            (CheckpointSelection(alias="candidate", path="missing.pt"),),
            device=torch.device("cpu"),
            suffix_steps=(0,),
            rollout_steps=4,
        )


def test_checkpoint_evaluation_rejects_suffix_covering_context(tmp_path: Path) -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    config = replace(
        config,
        model=replace(
            config.model,
            layers=(LayerConfig(kind="reduce", mode="sum_pool"), LayerConfig(kind="ffn")),
        ),
    )
    model = build_model(config, torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": config_to_dict(config), "model": model.state_dict()},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="smaller than checkpoint seq_len"):
        evaluate_checkpoints(
            make_store(rows=40),
            make_index(rows=40, split_cell_b=False).frame.head(1),
            (CheckpointSelection(alias="candidate", path=str(checkpoint_path)),),
            device=torch.device("cpu"),
            suffix_steps=(10,),
            rollout_steps=4,
        )
