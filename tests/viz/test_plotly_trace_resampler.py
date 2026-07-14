from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import pytest

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.resampling import MinMaxLTTBResamplingSpec
from batgrad.viz.widgets.plotly_trace_resampler import PlotlyTraceResampler


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            str(BaseColumns.time): list(range(8)),
            str(BaseColumns.volt): [float(value) for value in range(8)],
        }
    )


def _widget(max_points_per_trace: int = 3, max_points_per_figure: int = 10) -> PlotlyTraceResampler:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[], y=[], name="a"))
    return PlotlyTraceResampler(
        fig,
        max_points_per_trace=max_points_per_trace,
        max_points_per_figure=max_points_per_figure,
        max_batch_rows=None,
    )


def _two_trace_widget(max_points_per_trace: int = 5, max_points_per_figure: int = 6):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[], y=[], name="a"))
    fig.add_trace(go.Scattergl(x=[], y=[], name="b"))
    fig.add_trace(go.Scattergl(x=[], y=[], name="ann"))
    return PlotlyTraceResampler(
        fig,
        max_points_per_trace=max_points_per_trace,
        max_points_per_figure=max_points_per_figure,
        max_batch_rows=None,
    )


def test_register_trace_show_and_relayout_update() -> None:
    frame = _frame()
    widget = _widget()
    widget.register_trace(
        0,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
    )
    shown = widget.show()
    assert shown._fig_json["data"][0]["x"]
    assert len(shown._fig_json["data"][0]["x"]) <= 3

    widget._on_evt(
        {"new": {"axes": {"xaxis": {"range": [2, 5]}}, "visible": {"0": True}, "_rid": 4}}
    )
    assert widget._update["_rid"] == 4
    assert len(widget._update["updates"][0]["x"]) <= 3


def test_widget_rejects_invalid_registration_and_annotation_parent() -> None:
    frame = _frame()
    widget = _widget()
    with pytest.raises(IndexError):
        widget.register_trace(
            4,
            frame.lazy(),
            BaseColumns.time,
            BaseColumns.volt,
            MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
        )
    with pytest.raises(ValueError, match="parent_trace_idx"):
        widget.register_annotation_trace(
            0,
            99,
            frame.lazy(),
            BaseColumns.time,
            BaseColumns.volt,
            annotation_columns=(str(BaseColumns.volt),),
            annotation_reason="missing",
        )


def test_selected_data_returns_rows_with_trace_metadata() -> None:
    frame = _frame()
    widget = _widget()
    widget.register_trace(
        0,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
    )
    selected, total = widget.selected_data(
        selection={"traces": [{"trace_idx": 0, "x_range": [2, 4], "name": "trace-a"}]},
        limit=10,
    )
    assert total == 3
    assert selected["trace"].to_list() == ["trace-a", "trace-a", "trace-a"]
    assert selected[str(BaseColumns.time)].to_list() == [2, 3, 4]


def test_multi_trace_budget_is_split_per_visible_trace() -> None:
    frame = _frame()
    widget = _two_trace_widget(max_points_per_trace=5, max_points_per_figure=6)
    for idx in (0, 1):
        widget.register_trace(
            idx,
            frame.lazy(),
            BaseColumns.time,
            BaseColumns.volt,
            MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
        )

    widget.show()
    assert len(widget._fig_json["data"][0]["x"]) <= 3
    assert len(widget._fig_json["data"][1]["x"]) <= 3


def test_partial_raw_trace_update_preserves_downsampled_status() -> None:
    stream = _frame()
    overlay = stream.slice(0, 2)
    widget = _two_trace_widget(max_points_per_trace=5, max_points_per_figure=6)
    for idx, frame in enumerate((stream, overlay)):
        widget.register_trace(
            idx,
            frame.lazy(),
            BaseColumns.time,
            BaseColumns.volt,
            MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
            row_count=frame.height,
        )

    widget.show()
    assert widget._status.startswith("Downsampled:")

    widget.update_registered_traces(((1, overlay.lazy(), overlay.height),))

    assert widget._status.startswith("Downsampled:")
    assert str(widget._update["status"]).startswith("Downsampled:")


def test_new_trace_registered_after_show_receives_budget_on_update() -> None:
    frame = _frame()
    widget = _two_trace_widget(max_points_per_trace=5, max_points_per_figure=6)
    widget.register_trace(
        0,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
    )
    widget.show()
    widget.register_trace(
        1,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
    )

    widget._on_evt({"new": {"visible": {"0": True, "1": True}, "_rid": 10}})

    updates = {update["trace_idx"]: update for update in widget._update["updates"]}
    assert set(updates) == {0, 1}
    assert len(updates[0]["x"]) <= 3
    assert len(updates[1]["x"]) <= 3


def test_hidden_parent_trace_suppresses_trace_and_annotation_updates() -> None:
    frame = _frame().with_columns(
        pl.lit([{"column": str(BaseColumns.volt), "reason": "missing"}]).alias(BaseColumns.anns)
    )
    widget = _two_trace_widget()
    widget.register_trace(
        0,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
    )
    widget.register_trace(
        1,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        MinMaxLTTBResamplingSpec(BaseColumns.time, BaseColumns.volt, points=8),
    )
    widget.register_annotation_trace(
        2,
        0,
        frame.lazy(),
        BaseColumns.time,
        BaseColumns.volt,
        annotation_columns=(str(BaseColumns.volt),),
        annotation_reason="missing",
    )

    widget.show()
    widget._on_evt({"new": {"visible": {"0": False, "1": True}, "_rid": 9}})

    updated_indices = {update["trace_idx"] for update in widget._update["updates"]}
    assert updated_indices == {1}
