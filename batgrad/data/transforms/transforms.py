from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import polars as pl

if TYPE_CHECKING:
    from batgrad.contracts.mapping import MappingSpec


@dataclass(frozen=True)
class CRateTransformSpec:
    """Derive C-rate from current and nominal capacity.

    If `target_col` already exists, null target values are filled from the
    derived C-rate and existing non-null values are preserved. If `source_col` is
    absent, the input frame is returned unchanged.

    Attributes:
        source_col: Current column in amps.
        target_col: Output C-rate column.
        nominal_capacity_ah: Nominal cell capacity used as the divisor.

    Examples:
        >>> CRateTransformSpec(
        ...     source_col=BaseColumns.curr,
        ...     target_col=BaseColumns.crate,
        ...     nominal_capacity_ah=5.0,
        ... )
        CRateTransformSpec(...)
    """

    source_col: MappingSpec
    target_col: MappingSpec
    nominal_capacity_ah: float

    @property
    def input_columns(self) -> tuple[MappingSpec, ...]:
        """Columns required before this transform runs.

        Returns:
            Source current column.
        """
        return (self.source_col,)

    @property
    def produced_columns(self) -> tuple[MappingSpec, ...]:
        """Columns produced or filled by this transform.

        Returns:
            Target C-rate column.
        """
        return (self.target_col,)

    @overload
    def apply(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def apply(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Apply the C-rate derivation to a frame when the source column exists.

        Args:
            data: Input dataframe or lazy frame.

        Returns:
            Frame with `target_col` added or filled when `source_col` exists;
            otherwise the original frame.
        """
        columns = data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.columns
        if self.source_col not in columns:
            return data
        derived = pl.col(self.source_col).cast(pl.Float64) / pl.lit(float(self.nominal_capacity_ah))
        if self.target_col in columns:
            expr = pl.coalesce([pl.col(self.target_col), derived]).alias(self.target_col)
        else:
            expr = derived.alias(self.target_col)
        return data.with_columns(expr)
