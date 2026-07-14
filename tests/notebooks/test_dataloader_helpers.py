from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from notebooks._support import dataloader_helpers

if TYPE_CHECKING:
    import pytest


def test_batch_change_updates_existing_preview_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    widget = object()
    preview = SimpleNamespace(
        widget=widget,
        spec=SimpleNamespace(
            batch_group_index=0,
            sample_index=0,
            consecutive_step=0,
        ),
    )
    updated = SimpleNamespace(widget=widget)
    received: list[tuple[int, int, int]] = []

    def update(_preview, group, sample, step):
        received.append((group, sample, step))
        return updated

    monkeypatch.setattr(dataloader_helpers, "update_ml_batch_preview", update)

    error, result, view = dataloader_helpers.update_batch_preview(
        preview=preview,
        batch_group_index=1,
        sample_index=0,
        consecutive_step=0,
    )

    assert error is None
    assert result is updated
    assert view is None
    assert received == [(1, 0, 0)]


def test_protocol_change_requires_full_resubmission_after_plot() -> None:
    submission = SimpleNamespace(spec=SimpleNamespace(active_protocol="cycling"))

    assert not dataloader_helpers.protocol_requires_resubmit(None, "EIS")
    assert not dataloader_helpers.protocol_requires_resubmit(submission, "cycling")
    assert dataloader_helpers.protocol_requires_resubmit(submission, "EIS")


def test_batch_update_keeps_replacement_widget_view() -> None:
    initial_preview = SimpleNamespace(widget=object())
    replacement_widget = object()
    replacement_preview = SimpleNamespace(widget=replacement_widget)
    batch_updated_preview = SimpleNamespace(widget=replacement_widget)
    replacement_view = object()
    display = dataloader_helpers.BatchPreviewDisplay(initial_preview, object())

    display = dataloader_helpers.updated_batch_preview_display(
        previous=display,
        preview=replacement_preview,
        view=replacement_view,
    )
    display = dataloader_helpers.updated_batch_preview_display(
        previous=display,
        preview=batch_updated_preview,
        view=None,
    )

    assert display is not None
    assert display.preview is batch_updated_preview
    assert display.view is replacement_view
