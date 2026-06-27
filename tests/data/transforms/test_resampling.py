from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.resampling import (
    LinearResamplingSpec,
    MinMaxLTTBResamplingSpec,
    downsample_tiny_budget_frame,
    resample_linear_frame,
    resolve_min_max_lttb_budget,
)


def test_min_max_lttb_budget_and_tiny_budget() -> None:
    spec = MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=3)
    assert resolve_min_max_lttb_budget(spec, 10) == 3
    ratio_spec = MinMaxLTTBResamplingSpec(
        BaseColumns.time,
        BaseColumns.volt,
        points_ratio=0.5,
    )
    assert resolve_min_max_lttb_budget(ratio_spec, 10) == 5
    frame = pl.DataFrame({"x": [1, 2, 3]})
    assert downsample_tiny_budget_frame(frame, 1)["x"].to_list() == [1]
    assert downsample_tiny_budget_frame(frame, 2)["x"].to_list() == [1, 3]
    with pytest.raises(ValueError):
        resolve_min_max_lttb_budget(
            MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt),
            10,
        )


def test_min_max_lttb_full_and_bounded_are_deterministic() -> None:
    frame = pl.DataFrame(
        {
            str(BaseColumns.time): list(range(10)),
            str(BaseColumns.volt): [0.0, 1.0, 0.5, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0],
        }
    )
    spec = MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=4)
    full = spec.apply_full(frame)
    assert full[str(BaseColumns.time)].to_list() == [0, 4, 5, 9]

    def chunks():
        return iter(
            (
                frame.slice(0, 5).with_columns(pl.Series("rid", np.arange(0, 5))),
                frame.slice(5, 5).with_columns(pl.Series("rid", np.arange(5, 10))),
            )
        )
    bounded = pl.concat(
        list(spec.apply_bounded(chunks, row_count=10, max_batch_rows=5, row_id_col="rid"))
    )
    assert bounded[str(BaseColumns.time)].to_list() == full[str(BaseColumns.time)].to_list()


def test_linear_resampling_interpolates_numeric_and_nearest_string() -> None:
    frame = pl.DataFrame({str(BaseColumns.time): [0.0, 2.0], "y": [0.0, 4.0], "label": ["a", "b"]})
    result = resample_linear_frame(frame, LinearResamplingSpec(BaseColumns.time, points=3))
    assert result[str(BaseColumns.time)].to_list() == [0.0, 1.0, 2.0]
    assert result["y"].to_list() == [0.0, 2.0, 4.0]
    assert result["label"].to_list() == ["a", "a", "b"]
    with pytest.raises(ValueError, match="positive range"):
        resample_linear_frame(frame, LinearResamplingSpec(BaseColumns.time, points=3, scale="log"))
