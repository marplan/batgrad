from __future__ import annotations

import polars as pl

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.transforms import CRateTransformSpec


def test_crate_transform_adds_and_preserves_existing_values() -> None:
    transform = CRateTransformSpec(BaseColumns.curr, BaseColumns.crate, 2.0)
    frame = pl.DataFrame({str(BaseColumns.curr): [1.0, 2.0]})
    assert transform.apply(frame)[str(BaseColumns.crate)].to_list() == [0.5, 1.0]

    existing = pl.DataFrame(
        {str(BaseColumns.curr): [1.0, 2.0], str(BaseColumns.crate): [None, 9.0]}
    )
    assert transform.apply(existing)[str(BaseColumns.crate)].to_list() == [0.5, 9.0]
    assert transform.apply(pl.DataFrame({"x": [1]})).columns == ["x"]
