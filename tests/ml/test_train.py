from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from batgrad.ml.config import load_experiment_config
from batgrad.ml.train import _model_compute_token_count


def test_model_compute_token_count_includes_roll_forward_windows() -> None:
    base = load_experiment_config("configs/ml_baseline.json")
    config = replace(
        base,
        train=replace(
            base.train,
            masked_suffix=replace(base.train.masked_suffix, roll_forward_steps=896),
        ),
    )
    batch = SimpleNamespace(
        active=SimpleNamespace(inputs=torch.zeros((96, 1920, 1), dtype=torch.float32))
    )

    assert _model_compute_token_count(config, batch) == 96 * 8 * 1024
