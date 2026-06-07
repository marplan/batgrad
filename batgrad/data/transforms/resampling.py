from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, cast

if TYPE_CHECKING:
    import polars as pl

    from batgrad.contracts.columns import ColumnSpec


class ResamplingSpecBase:
    method: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "method" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define class variable 'method'")


type ResamplingHandler = Callable[[pl.DataFrame, ResamplingSpecBase], pl.DataFrame]
RESAMPLING_REGISTRY: dict[type[ResamplingSpecBase], ResamplingHandler] = {}


def register_resampling[SpecT: ResamplingSpecBase](
    spec_type: type[SpecT],
) -> Callable[
    [Callable[[pl.DataFrame, SpecT], pl.DataFrame]],
    Callable[[pl.DataFrame, SpecT], pl.DataFrame],
]:
    def decorator(
        fn: Callable[[pl.DataFrame, SpecT], pl.DataFrame],
    ) -> Callable[[pl.DataFrame, SpecT], pl.DataFrame]:
        if spec_type in RESAMPLING_REGISTRY:
            raise ValueError(f"Resampling {spec_type.__name__} is already registered")
        if any(registered.method == spec_type.method for registered in RESAMPLING_REGISTRY):
            raise ValueError(f"Resampling method {spec_type.method!r} is already registered")

        RESAMPLING_REGISTRY[spec_type] = cast("ResamplingHandler", fn)
        return fn

    return decorator


@dataclass(frozen=True, slots=True)
class MinMaxLTTBResamplingSpec(ResamplingSpecBase):
    x_col: ColumnSpec
    y_col: ColumnSpec
    points: int | None = None
    points_ratio: float | None = None
    min_points: int = 3

    method: ClassVar[str] = "min_max_lttb"


@dataclass(frozen=True, slots=True)
class LinearResamplingSpec(ResamplingSpecBase):
    x_col: ColumnSpec
    points: int
    scale: Literal["linear", "log"] = "linear"

    method: ClassVar[str] = "linear"


@register_resampling(MinMaxLTTBResamplingSpec)
def min_max_lttb_resampling(
    data: pl.DataFrame,
    _resampling_spec: MinMaxLTTBResamplingSpec,
) -> pl.DataFrame:
    return data


@register_resampling(LinearResamplingSpec)
def linear_resampling(
    data: pl.DataFrame,
    _resampling_spec: LinearResamplingSpec,
) -> pl.DataFrame:
    return data


def run_resampling(data: pl.DataFrame, resampling_spec: ResamplingSpecBase) -> pl.DataFrame:
    handler = RESAMPLING_REGISTRY.get(type(resampling_spec))
    if handler is None:
        raise ValueError(f"No handler registered for resampling {type(resampling_spec).__name__}")
    return handler(data, resampling_spec)
