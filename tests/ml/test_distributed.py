from __future__ import annotations

import pytest
import torch

from batgrad.ml import distributed as distributed_module
from batgrad.ml.metrics import LossMetrics


def test_distributed_loss_is_weighted_by_global_valid_count(monkeypatch) -> None:
    remote_values = iter((torch.tensor([18.0, 18.0]), torch.tensor([6.0, 6.0])))

    def add_remote(value: torch.Tensor) -> torch.Tensor:
        value += next(remote_values)
        return value

    monkeypatch.setattr(distributed_module, "all_reduce_sum", add_remote)
    local = LossMetrics(
        loss=torch.tensor(1.0),
        feature_loss_sum=torch.tensor([2.0, 2.0]),
        feature_loss_count=torch.tensor([2.0, 2.0]),
    )

    reduced = distributed_module.all_reduce_loss_metrics(local)

    assert reduced.loss.item() == pytest.approx(2.5)
    assert reduced.feature_loss_sum is not None
    assert reduced.feature_loss_sum.tolist() == [20.0, 20.0]
    assert reduced.feature_loss_count is not None
    assert reduced.feature_loss_count.tolist() == [8.0, 8.0]


def test_backward_scale_compensates_for_ddp_gradient_averaging(monkeypatch) -> None:
    monkeypatch.setattr(
        distributed_module,
        "all_reduce_sum",
        lambda value: value.add(torch.tensor(12.0)),
    )
    monkeypatch.setattr(distributed_module, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_world_size", lambda: 2)

    scale = distributed_module.globally_normalized_backward_scale(torch.tensor(4.0))

    assert scale.item() == pytest.approx(2.0 / 16.0)
