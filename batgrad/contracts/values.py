from __future__ import annotations

from typing import Self

import polars as pl


class ValueSpec(str):
    __slots__ = ("description", "dtype")

    dtype: pl.DataType | None
    description: str | None

    def __new__(
        cls,
        value: str,
        dtype: type[pl.DataType] | pl.DataType | None = None,
        description: str | None = None,
    ) -> Self:
        instance = super().__new__(cls, value)
        instance.dtype = dtype
        instance.description = description
        return instance

    def with_value(self, value: str) -> Self:
        return type(self)(
            value,
            dtype=self.dtype,
            description=self.description,
        )


class BaseValues:
    missing = ValueSpec("missing", dtype=pl.String, description="Missing data identifier")
    col_min = ValueSpec("below column minimum", dtype=pl.String, description="Below column minimum")
    col_max = ValueSpec("above column maximum", dtype=pl.String, description="Above column maximum")
    dup_time = ValueSpec(
        "duplicate time steps",
        dtype=pl.String,
        description="Duplicate time steps",
    )
    big_dt = ValueSpec("big time diff", dtype=pl.String, description="Big time difference")
    domain_x_axis = ValueSpec("invalid domain x-axis", dtype=pl.String)
    train = ValueSpec("train", dtype=pl.String, description="Training split identifier")
    val = ValueSpec("val", dtype=pl.String, description="Validation split identifier")

    time_domain = ValueSpec("timeseries domain", dtype=pl.String, description="Time domain")
    freq_domain = ValueSpec("frequency domain", dtype=pl.String, description="Frequency domain")

    cycling_protocol = ValueSpec("cycling", dtype=pl.String, description="Cycling protocol")
    hppc_protocol = ValueSpec("HPPC", dtype=pl.String, description="HPPC protocol")
    rpt_protocol = ValueSpec("RPT", dtype=pl.String, description="RPT protocol")
    eis_protocol = ValueSpec("EIS", dtype=pl.String, description="EIS protocol")
