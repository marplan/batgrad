from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl

if TYPE_CHECKING:
    from batgrad.contracts.columns import ColumnSpec


class TransformSpecBase:
    name: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define class variable 'name'")


type FullTaskTransformHandler = Callable[[pl.LazyFrame, TransformSpecBase], pl.LazyFrame]
type BoundedTransformHandler = Callable[[pl.DataFrame, TransformSpecBase], pl.DataFrame]


@dataclass(frozen=True, slots=True)
class TransformHandler:
    full_task: FullTaskTransformHandler
    bounded: BoundedTransformHandler
    source_columns: Callable[[TransformSpecBase], tuple[ColumnSpec, ...]]


TRANSFORM_REGISTRY: dict[type[TransformSpecBase], TransformHandler] = {}


def register_transform(spec_type: type[TransformSpecBase], handler: TransformHandler) -> None:
    if spec_type in TRANSFORM_REGISTRY:
        raise ValueError(f"Transform {spec_type.__name__} is already registered")
    if any(registered.name == spec_type.name for registered in TRANSFORM_REGISTRY):
        raise ValueError(f"Transform name {spec_type.name!r} is already registered")
    TRANSFORM_REGISTRY[spec_type] = handler


@dataclass(frozen=True, slots=True)
class CRateTransformSpec(TransformSpecBase):
    source_col: ColumnSpec
    target_col: ColumnSpec
    nominal_capacity_ah: float

    name: ClassVar[str] = "c_rate"


def apply_transforms_full_task(
    data: pl.LazyFrame,
    transforms: tuple[TransformSpecBase, ...],
) -> pl.LazyFrame:
    for transform in transforms:
        data = _handler(transform).full_task(data, transform)
    return data


def apply_transforms_bounded_chunk(
    data: pl.DataFrame,
    transforms: tuple[TransformSpecBase, ...],
) -> pl.DataFrame:
    for transform in transforms:
        data = _handler(transform).bounded(data, transform)
    return data


def transform_source_columns(
    transforms: tuple[TransformSpecBase, ...],
) -> tuple[ColumnSpec, ...]:
    columns: list[ColumnSpec] = []
    for transform in transforms:
        columns.extend(_handler(transform).source_columns(transform))
    return tuple(dict.fromkeys(columns))


def _handler(transform: TransformSpecBase) -> TransformHandler:
    handler = TRANSFORM_REGISTRY.get(type(transform))
    if handler is None:
        raise ValueError(f"No handler registered for transform {type(transform).__name__}")
    return handler


def _c_rate_expr(transform: CRateTransformSpec) -> pl.Expr:
    return (
        pl.col(transform.source_col).cast(pl.Float64, strict=False)
        / pl.lit(float(transform.nominal_capacity_ah))
    ).alias(transform.target_col)


def _c_rate_full_task(data: pl.LazyFrame, transform: TransformSpecBase) -> pl.LazyFrame:
    if not isinstance(transform, CRateTransformSpec):
        raise TypeError(f"Expected CRateTransformSpec, got {type(transform).__name__}")
    return data.with_columns(_c_rate_expr(transform))


def _c_rate_bounded(data: pl.DataFrame, transform: TransformSpecBase) -> pl.DataFrame:
    if not isinstance(transform, CRateTransformSpec):
        raise TypeError(f"Expected CRateTransformSpec, got {type(transform).__name__}")
    return data.with_columns(_c_rate_expr(transform))


def _c_rate_sources(transform: TransformSpecBase) -> tuple[ColumnSpec, ...]:
    if not isinstance(transform, CRateTransformSpec):
        raise TypeError(f"Expected CRateTransformSpec, got {type(transform).__name__}")
    return (transform.source_col,)


register_transform(
    CRateTransformSpec,
    TransformHandler(
        full_task=_c_rate_full_task,
        bounded=_c_rate_bounded,
        source_columns=_c_rate_sources,
    ),
)
