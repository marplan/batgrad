from __future__ import annotations

from pathlib import Path

import pytest

from notebooks._support.inference_helpers import (
    checkpoint_discovery_status,
    checkpoint_frame,
    make_inference_submission,
    selected_checkpoints_from_table,
)


def test_empty_checkpoint_root_returns_visible_status(tmp_path: Path) -> None:
    checkpoints, error = checkpoint_discovery_status(tmp_path)

    assert checkpoints == ()
    assert error is not None
    assert "**/checkpoints/**/*.pt" in error
    frame = checkpoint_frame(checkpoints)
    assert frame.columns == ["alias", "checkpoint", "checkpoint_path"]
    assert selected_checkpoints_from_table(None, frame) == ()


def test_checkpoint_discovery_finds_nested_training_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run-a" / "checkpoints" / "epoch-1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    checkpoints, error = checkpoint_discovery_status(tmp_path)

    assert error is None
    assert tuple(item.path for item in checkpoints) == (str(checkpoint),)
    assert tuple(item.label for item in checkpoints) == ("run-a/checkpoints/epoch-1.pt",)


def test_inference_submission_rejects_empty_checkpoint_selection() -> None:
    with pytest.raises(ValueError, match="Select at least one checkpoint"):
        make_inference_submission(
            submit_id=1,
            checkpoints=(),
            device="cpu",
            masked_suffix_steps="0",
            rollout_steps=1,
        )
