from __future__ import annotations

from dataclasses import replace

import torch

from batgrad.ml.objective import ObjectiveTrace, batch_loss_with_metrics
from tests.ml.conftest import RecordingModel, make_batch, make_config


def test_teacher_forced_objective_captures_exact_predictions() -> None:
    config = make_config(batch_size=2, seq_len=10)
    suffix = replace(config.train.masked_suffix, enabled=False, roll_forward_steps=0)
    batch = make_batch(config, seq_len=10)
    traces: list[ObjectiveTrace] = []

    batch_loss_with_metrics(
        config,
        RecordingModel(),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=suffix,
        trace_callback=traces.append,
    )

    assert len(traces) == 1
    trace = traces[0]
    assert trace.predictions.shape == batch.targets.shape
    assert trace.context_len == 10
    assert trace.roll_forward_steps == 0
    assert trace.mask_boundaries == ()
    assert len(trace.windows) == 1
    assert trace.windows[0].logits.shape[:3] == trace.predictions.shape


def test_masked_suffix_trace_spans_effective_sequence_and_boundaries() -> None:
    config = make_config(batch_size=2, seq_len=10, suffix_steps=3, roll_forward_steps=4)
    batch = make_batch(config)
    traces: list[ObjectiveTrace] = []

    batch_loss_with_metrics(
        config,
        RecordingModel(position_bins=True),
        batch.inputs,
        batch.targets,
        batch.mask,
        torch.device("cpu"),
        suffix=config.train.masked_suffix,
        trace_callback=traces.append,
    )

    trace = traces[0]
    assert trace.predictions.shape == batch.targets.shape
    assert trace.context_len == 10
    assert trace.roll_forward_steps == 4
    assert trace.mask_boundaries == (7, 10, 13)
    assert bool(torch.isfinite(trace.predictions).all())
    assert len(trace.windows) == 3
    for window in trace.windows:
        assert torch.equal(
            trace.predictions[:, window.target_slice, :],
            window.predictions[:, window.prediction_slice, :],
        )
