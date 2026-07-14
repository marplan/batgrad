from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from batgrad.ml.inference import (
    CheckpointSelection,
    InferenceResult,
    discover_checkpoints as discover_checkpoint_paths,
)
from batgrad.notebook_helpers import wrap_anywidget_blocks
from batgrad.viz.ml import build_inference_widget, inference_metrics_frame

if TYPE_CHECKING:
    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: str
    label: str


@dataclass(frozen=True, slots=True)
class InferenceSubmission:
    submit_id: int
    checkpoints: tuple[CheckpointSelection, ...]
    device: str
    masked_suffix_steps: tuple[int, ...]
    rollout_steps: int


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    submission: InferenceSubmission
    selected_index_frame: pl.DataFrame
    store: DatasetStoreReader


def checkpoint_options(root: str | Path = ".") -> tuple[CheckpointInfo, ...]:
    root_path = Path(root)
    return tuple(
        CheckpointInfo(path=str(path), label=str(path.relative_to(root_path)))
        for path in discover_checkpoint_paths(root_path)
    )


def checkpoint_discovery_status(
    root: str | Path,
) -> tuple[tuple[CheckpointInfo, ...], str | None]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return (), f"Checkpoint search root does not exist: {root_path}"
    if not root_path.is_dir():
        return (), f"Checkpoint search root is not a directory: {root_path}"
    checkpoints = checkpoint_options(root_path)
    if not checkpoints:
        return (
            (),
            f"No checkpoints found under {root_path}. "
            "Expected files matching **/checkpoints/**/*.pt.",
        )
    return checkpoints, None


def checkpoint_frame(checkpoints: tuple[CheckpointInfo, ...]) -> pl.DataFrame:
    if not checkpoints:
        return pl.DataFrame(
            schema={
                "alias": pl.String,
                "checkpoint": pl.String,
                "checkpoint_path": pl.String,
            }
        )
    return pl.DataFrame(
        {
            "alias": [f"ckpt {idx}" for idx in range(1, len(checkpoints) + 1)],
            "checkpoint": [item.label for item in checkpoints],
            "checkpoint_path": [item.path for item in checkpoints],
        }
    )


def make_checkpoint_table(frame: pl.DataFrame) -> object:
    import marimo as mo  # noqa: PLC0415

    return mo.ui.table(frame, selection="multi", hidden_columns=["checkpoint_path"])


def selected_checkpoints_from_table(
    checkpoint_table_value: object,
    frame: pl.DataFrame,
) -> tuple[CheckpointSelection, ...]:
    aliases = {str(row["alias"]) for row in _table_rows(checkpoint_table_value) if row.get("alias")}
    if not aliases:
        return ()
    return tuple(
        CheckpointSelection(alias=str(row["alias"]), path=str(row["checkpoint_path"]))
        for row in frame.filter(pl.col("alias").is_in(aliases)).iter_rows(named=True)
        if row.get("checkpoint_path")
    )


def make_inference_submission(
    *,
    submit_id: int,
    checkpoints: tuple[CheckpointSelection, ...],
    device: str,
    masked_suffix_steps: str,
    rollout_steps: int,
) -> InferenceSubmission:
    if not checkpoints:
        raise ValueError("Select at least one checkpoint before running inference")
    parsed_suffix_steps = parse_masked_suffix_steps(masked_suffix_steps)
    return InferenceSubmission(
        submit_id=submit_id,
        checkpoints=checkpoints,
        device=device,
        masked_suffix_steps=parsed_suffix_steps,
        rollout_steps=rollout_steps,
    )


def parse_masked_suffix_steps(value: str) -> tuple[int, ...]:
    steps = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise ValueError("Masked suffix steps must be >= 0")
        steps.append(step)
    if not steps:
        raise ValueError("Enter at least one masked suffix step")
    return tuple(dict.fromkeys(steps))


def make_inference_request(
    *,
    store: DatasetStoreReader,
    selected_index_frame: pl.DataFrame,
    submission: InferenceSubmission,
) -> InferenceRequest:
    if store is None:
        raise ValueError("Select a valid store root before running inference")
    return InferenceRequest(submission, selected_index_frame.clone(), store)


def render_batch_result(result: InferenceResult | None, batch_index: int) -> object | None:
    import marimo as mo  # noqa: PLC0415

    if result is None:
        return None
    items = []
    if result.warning is not None:
        items.append(mo.callout(result.warning, kind="warn"))
    idx = min(max(0, int(batch_index)), int(result.inputs.shape[0]) - 1)
    items.extend(wrap_anywidget_blocks((build_inference_widget(result, idx),)))
    items.extend((mo.md("### Metrics"), mo.ui.table(inference_metrics_frame(result))))
    return mo.vstack(items)


def inference_view(
    *,
    submission_error: str | None,
    inference_error: str | None,
    result_view: object | None,
) -> object:
    import marimo as mo  # noqa: PLC0415

    if submission_error is not None:
        return mo.callout(submission_error, kind="danger")
    if inference_error is not None:
        return mo.callout(inference_error, kind="danger")
    if result_view is None:
        return mo.md("Select checkpoint and ML index rows, then click Run inference.")
    return result_view


def _table_rows(value: object) -> tuple[Mapping[str, object], ...]:
    rows: object
    if isinstance(value, pl.DataFrame):
        rows = value.iter_rows(named=True)
    elif isinstance(value, Mapping):
        selection = value.get("selection")
        rows = (
            selection.iter_rows(named=True)
            if isinstance(selection, pl.DataFrame)
            else selection or ()
        )
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        rows = value
    else:
        rows = ()
    return tuple(cast("Mapping[str, object]", row) for row in rows if isinstance(row, Mapping))
