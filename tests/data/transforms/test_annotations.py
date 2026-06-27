from __future__ import annotations

import polars as pl

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.annotations import (
    add_annotations,
    ensure_annotation_columns,
    finalize_annotations,
)


def test_annotations_are_added_and_finalized_to_structs() -> None:
    frame = ensure_annotation_columns(pl.DataFrame({"x": [1.0, None], "y": [1.0, 2.0]}))
    annotated = add_annotations(
        frame,
        [(pl.col("x").is_null(), BaseColumns.ann_reasons.values.missing, "x")],
    )
    assert annotated[str(BaseColumns.ann_cols)].to_list() == [None, "x"]
    finalized = finalize_annotations(annotated, include_annotations=True)
    assert str(BaseColumns.anns) in finalized.columns
    assert str(BaseColumns.ann_cols) not in finalized.columns
    assert finalized[str(BaseColumns.anns)].to_list()[1] == [
        {"column": "x", "reason": "missing"}
    ]


def test_finalize_annotations_can_be_disabled() -> None:
    frame = ensure_annotation_columns(pl.DataFrame({"x": [1.0]}))
    assert finalize_annotations(frame, include_annotations=False).columns == frame.columns
