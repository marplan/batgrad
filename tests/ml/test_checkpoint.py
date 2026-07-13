from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from batgrad.ml.checkpoint import load_checkpoint, load_model_weights, read_checkpoint_config
from batgrad.ml.config import config_to_dict
from batgrad.ml.nn import LayerConfig, build_model
from tests.ml.conftest import make_config


def _checkpoint_config():
    config = make_config()
    return replace(
        config,
        model=replace(
            config.model,
            layers=(LayerConfig(kind="reduce", mode="sum_pool"), LayerConfig(kind="ffn")),
        ),
    )


def test_checkpoint_loads_config_model_and_step(tmp_path: Path) -> None:
    config = _checkpoint_config()
    device = torch.device("cpu")
    model = build_model(config, device)
    path = tmp_path / "model.pt"
    torch.save(
        {"config": config_to_dict(config), "model": model.state_dict(), "step": 7},
        path,
    )

    loaded = load_checkpoint(path, device)

    assert read_checkpoint_config(path) == config
    assert loaded.config == config
    assert loaded.step == 7
    assert loaded.model.training is False
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(model.parameters(), loaded.model.parameters(), strict=True)
    )


def test_checkpoint_requires_config_and_model_payloads(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save({}, path)

    with pytest.raises(TypeError, match="config"):
        read_checkpoint_config(path)
    with pytest.raises(TypeError, match="model"):
        load_checkpoint(path, torch.device("cpu"))


def test_weight_initialization_rejects_reordered_input_columns(tmp_path: Path) -> None:
    checkpoint_config = _checkpoint_config()
    device = torch.device("cpu")
    checkpoint_model = build_model(checkpoint_config, device)
    path = tmp_path / "model.pt"
    torch.save(
        {"config": config_to_dict(checkpoint_config), "model": checkpoint_model.state_dict()},
        path,
    )
    current_config = replace(
        checkpoint_config,
        data=replace(
            checkpoint_config.data,
            input_columns=tuple(reversed(checkpoint_config.data.input_columns)),
        ),
    )

    with pytest.raises(ValueError, match="input_columns"):
        load_model_weights(build_model(current_config, device), path, device, current_config)
