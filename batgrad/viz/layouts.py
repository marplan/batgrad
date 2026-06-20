from __future__ import annotations

from typing import Any


def timeseries_subplot_kwargs(rows: int) -> dict[str, Any]:
    return {
        "rows": rows,
        "cols": 1,
        "shared_xaxes": True,
        "vertical_spacing": 0.08,
    }


def eis_subplot_kwargs() -> dict[str, Any]:
    return {
        "rows": 2,
        "cols": 2,
        "specs": [[{"rowspan": 2}, {}], [None, {}]],
        "column_widths": [0.5, 0.5],
        "row_heights": [0.5, 0.5],
        "horizontal_spacing": 0.12,
        "vertical_spacing": 0.2,
    }
