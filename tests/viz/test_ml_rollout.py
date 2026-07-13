from __future__ import annotations

from dataclasses import replace

import torch

from batgrad.ml.inference import InferenceResult
from batgrad.viz.ml import _inference_protocol, _rollout_target_time_axis
from tests.ml.conftest import make_config


def test_rollout_target_time_axis_uses_next_row_timestamps() -> None:
    config = make_config()
    config = replace(
        config,
        data=replace(
            config.data,
            input_columns=("Time diff [s]", "current", "voltage", "temperature"),
            scaling=tuple(
                replace(rule, column="Time diff [s]") if rule.column == "time" else rule
                for rule in config.data.scaling
            ),
        ),
    )
    inputs = torch.tensor([[0.0], [2.0], [3.0]])

    assert _rollout_target_time_axis(config, inputs) == [2.0, 5.0, 8.0]


def test_inference_protocol_comes_from_selected_batch_lane() -> None:
    config = make_config()
    result = InferenceResult(
        config=config,
        inputs=torch.empty((1, 0, len(config.data.input_columns))),
        targets=torch.empty((1, 0, len(config.data.target_columns))),
        predictions=(),
        context_len=config.loader.seq_len,
        rollout_len=0,
        group_keys=(("dataset", "cell", 1, "HPPC"),),
        warning=None,
    )

    assert str(_inference_protocol(result, 0)) == "HPPC"
