from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from batgrad.contracts.mapping import DatasetProtocolId
from batgrad.ml.config import load_experiment_config
from batgrad.ml.masked_suffix import attention_mask_or_none
from batgrad.ml.rollout import rollout_protocol
from batgrad.ml.train import _checkpoint_dir, _model_compute_token_count


class DummyRunLogger:
    def __init__(self, run_id: str | None) -> None:
        self._run_id = run_id

    def run_id(self) -> str | None:
        return self._run_id


def test_model_compute_token_count_includes_roll_forward_windows() -> None:
    base = load_experiment_config("configs/ml_baseline.json")
    config = replace(
        base,
        train=replace(
            base.train,
            masked_suffix=replace(base.train.masked_suffix, roll_forward_steps=896),
        ),
    )
    batch = SimpleNamespace(inputs=torch.zeros((96, 1920, 1), dtype=torch.float32))

    assert _model_compute_token_count(config, batch) == 96 * 8 * 1024


def test_attention_mask_or_none_uses_all_valid_metadata() -> None:
    mask = torch.tensor([[True, True], [True, False]])

    assert attention_mask_or_none(mask, all_valid=True) is None
    assert attention_mask_or_none(mask, all_valid=False) is mask
    assert attention_mask_or_none(torch.ones((2, 2), dtype=torch.bool)) is None
    assert attention_mask_or_none(mask) is mask


def test_rollout_protocol_uses_selector_protocol() -> None:
    config = load_experiment_config("configs/ml_baseline.json")

    assert rollout_protocol(config, {"protocol": "HPPC"}) == DatasetProtocolId.hppc


def test_rollout_protocol_requires_selector_protocol() -> None:
    config = load_experiment_config("configs/ml_baseline.json")

    with pytest.raises(ValueError, match="must include protocol"):
        rollout_protocol(config, {"dataset id": "example"})


def test_rollout_protocol_must_be_enabled() -> None:
    base = load_experiment_config("configs/ml_baseline.json")
    config = replace(base, data=replace(base.data, protocols=("cycling",)))

    with pytest.raises(ValueError, match="not in data.protocols"):
        rollout_protocol(config, {"protocol": "HPPC"})


def test_checkpoint_dir_uses_run_id() -> None:
    assert _checkpoint_dir(Path("runs/20260706-154616"), DummyRunLogger("wlw39sey")) == Path(
        "runs/20260706-154616/checkpoints/wlw39sey"
    )


def test_checkpoint_dir_falls_back_to_run_dir_name() -> None:
    assert _checkpoint_dir(Path("runs/20260706-154616"), DummyRunLogger(None)) == Path(
        "runs/20260706-154616/checkpoints/20260706-154616"
    )
