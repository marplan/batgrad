from __future__ import annotations

from dataclasses import replace

import torch

from batgrad.ml.inference import InferencePrediction, InferenceResult
from batgrad.viz.ml import (
    _inference_input_col,
    _inference_plot_frame,
    _inference_protocol,
    _rollout_target_time_axis,
    build_inference_widget,
)
from batgrad.viz.plotting import COLORWAY
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


def test_default_inference_colors_start_with_visible_roles() -> None:
    config = make_config()
    sequence_len = 3
    prediction = InferencePrediction(
        checkpoint_alias="checkpoint",
        checkpoint_path="checkpoint.pt",
        suffix_steps=0,
        context_predictions=torch.ones((1, sequence_len, len(config.data.target_columns))),
        predictions=torch.zeros((1, 1, len(config.data.target_columns))),
        metrics=None,
        target_start=2,
    )
    result = InferenceResult(
        config=config,
        inputs=torch.zeros((1, sequence_len, len(config.data.input_columns))),
        targets=torch.zeros((1, sequence_len, len(config.data.target_columns))),
        predictions=(prediction,),
        context_len=sequence_len,
        rollout_len=0,
        group_keys=(("dataset", "cell", 1, "cycling"),),
        warning=None,
    )

    widget = build_inference_widget(result, 0)

    colors_by_name = {trace.name: trace.line.color for trace in widget._fig.data}
    assert colors_by_name["ground truth"] == COLORWAY[0]
    prediction_name = next(name for name in colors_by_name if name.startswith("prediction"))
    assert colors_by_name[prediction_name] == COLORWAY[1]
    assert {annotation.text for annotation in widget._fig.layout.annotations} >= {"rollout_pred"}


def test_inference_plot_hides_future_feedback_inputs_but_keeps_controls() -> None:
    config = make_config()
    sequence_len = 4
    inputs = torch.ones((1, sequence_len, len(config.data.input_columns)))
    targets = torch.ones((1, sequence_len, len(config.data.target_columns)))
    columns = tuple(dict.fromkeys((*config.data.input_columns, *config.data.target_columns)))

    frame = _inference_plot_frame(
        config,
        inputs,
        targets,
        (),
        0,
        columns,
        "sequence index",
        index_axis=True,
        feedback_input_end=2,
    )

    feedback_column = config.data.feedback_columns[0]
    control_column = next(
        column for column in config.data.input_columns if column not in config.data.feedback_columns
    )
    assert frame[_inference_input_col(feedback_column)].to_list() == [1.0, 1.0, None, None]
    assert frame[_inference_input_col(control_column)].to_list() == [1.0] * sequence_len
