from __future__ import annotations

from batgrad.viz.interactive import _consume_overlay_showlegend


def test_overlay_legend_is_shown_once_per_parent_run() -> None:
    shown_labels: set[str] = set()

    visibility = tuple(
        _consume_overlay_showlegend(label, shown_labels)
        for label in ("run-a", "run-a", "run-b", "run-b")
    )

    assert visibility == (True, False, True, False)
