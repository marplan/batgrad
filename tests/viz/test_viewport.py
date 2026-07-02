from __future__ import annotations

import polars as pl
import pytest

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec
from batgrad.storage.segments import SegmentSource
from batgrad.viz.viewport import (
    AnnotationSource,
    TraceSource,
    sample_annotation_viewport,
    sample_trace_viewport,
)


def _trace_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            str(BaseColumns.time): list(range(10)),
            str(BaseColumns.volt): [float(value) for value in range(10)],
            str(BaseColumns.curr): [float(value * 2) for value in range(10)],
        }
    )


def test_sample_trace_viewport_filters_budget_and_customdata() -> None:
    frame = _trace_frame()
    source = TraceSource(
        trace_idx=1,
        lf=frame.lazy(),
        x_col=BaseColumns.time,
        y_col=BaseColumns.volt,
        resampling=MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=10),
        customdata_cols=(BaseColumns.curr,),
    )
    sample = sample_trace_viewport(
        source,
        x_range=(2.0, 8.0),
        y_range=(3.0, 7.0),
        budget=3,
        max_batch_rows=None,
    )
    assert sample.trace_idx == 1
    assert sample.row_count == 5
    assert sample.shown_points == 3
    assert sample.downsampled
    assert sample.customdata is not None
    assert all(2.0 <= value <= 8.0 for value in sample.x)
    with pytest.raises(ValueError, match="budget"):
        sample_trace_viewport(source, x_range=None, y_range=None, budget=0, max_batch_rows=None)


def test_sample_trace_viewport_uses_bounded_segment_source(local_store) -> None:
    frame = _trace_frame()
    local_store.write_table(frame, "trace.parquet", row_group_size=2)
    source = TraceSource(
        trace_idx=0,
        lf=frame.lazy(),
        x_col=BaseColumns.time,
        y_col=BaseColumns.volt,
        resampling=MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=4),
        segment_source=SegmentSource.from_values(
            local_store,
            ({"file path": "trace.parquet", "row start": 0, "row count": 10},),
        ),
        row_count=10,
    )
    sample = sample_trace_viewport(source, x_range=None, y_range=None, budget=4, max_batch_rows=3)
    assert sample.row_count == 10
    assert sample.shown_points == 4

    no_source = TraceSource(
        trace_idx=0,
        lf=frame.lazy(),
        x_col=BaseColumns.time,
        y_col=BaseColumns.volt,
        resampling=MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=4),
        row_count=10,
    )
    with pytest.raises(ValueError, match="store-backed"):
        sample_trace_viewport(no_source, x_range=None, y_range=None, budget=4, max_batch_rows=3)


def test_sample_annotation_viewport_returns_boundaries() -> None:
    frame = pl.DataFrame(
        {
            str(BaseColumns.time): [0.0, 1.0, 2.0, 3.0],
            str(BaseColumns.volt): [3.0, 3.1, 3.2, 3.3],
            str(BaseColumns.anns): [
                None,
                [{"column": str(BaseColumns.volt), "reason": "above column maximum"}],
                [{"column": str(BaseColumns.volt), "reason": "above column maximum"}],
                None,
            ],
        },
        schema={
            str(BaseColumns.time): BaseColumns.time.dtype,
            str(BaseColumns.volt): BaseColumns.volt.dtype,
            str(BaseColumns.anns): BaseColumns.anns.dtype,
        },
    )
    source = AnnotationSource(
        trace_idx=2,
        parent_trace_idx=1,
        lf=frame.lazy(),
        x_col=BaseColumns.time,
        y_col=BaseColumns.volt,
        annotation_columns=(str(BaseColumns.volt),),
        annotation_reason="above column maximum",
    )
    sample = sample_annotation_viewport(source, x_range=None, y_range=None, max_batch_rows=None)
    assert sample.x == [1.0, 2.0]
    assert sample.y == [3.1, 3.2]
