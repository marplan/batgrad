from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import polars as pl

if TYPE_CHECKING:
    from batgrad.contracts.mapping import MappingSpec


@dataclass(frozen=True)
class CRateTransformSpec:
    source_col: MappingSpec
    target_col: MappingSpec
    nominal_capacity_ah: float

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        return (self.source_col,)

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        return (self.target_col,)

    @overload
    def apply(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def apply(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        columns = data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.columns
        if self.source_col not in columns:
            return data
        derived = pl.col(self.source_col).cast(pl.Float64) / pl.lit(float(self.nominal_capacity_ah))
        if self.target_col in columns:
            expr = pl.coalesce([pl.col(self.target_col), derived]).alias(self.target_col)
        else:
            expr = derived.alias(self.target_col)
        return data.with_columns(expr)
