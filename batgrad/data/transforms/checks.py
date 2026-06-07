from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from batgrad.contracts.columns import ColumnSpec
    from batgrad.data.datasets.specs import DatasetSpec


class CheckSpecBase:
    name: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define class variable 'name'")


type CheckHandler = Callable[[DatasetSpec, CheckSpecBase], bool]
CHECK_REGISTRY: dict[type[CheckSpecBase], CheckHandler] = {}


def register_check[SpecT: CheckSpecBase](
    spec_type: type[SpecT],
) -> Callable[[Callable[[DatasetSpec, SpecT], bool]], Callable[[DatasetSpec, SpecT], bool]]:
    def decorator(
        fn: Callable[[DatasetSpec, SpecT], bool],
    ) -> Callable[[DatasetSpec, SpecT], bool]:
        if spec_type in CHECK_REGISTRY:
            raise ValueError(f"Check {spec_type.__name__} is already registered")
        if any(registered.name == spec_type.name for registered in CHECK_REGISTRY):
            raise ValueError(f"Check name {spec_type.name!r} is already registered")

        CHECK_REGISTRY[spec_type] = cast("CheckHandler", fn)
        return fn

    return decorator


@dataclass(frozen=True, slots=True)
class MissingCheckSpec(CheckSpecBase):
    name: ClassVar[str] = "missing"


@dataclass(frozen=True, slots=True)
class TimeCheckSpec(CheckSpecBase):
    time_col: ColumnSpec
    dt_col: ColumnSpec
    max_big_dt_count: int = 5
    big_dt_floor_s: float = 5.0
    max_diff_factor: float | None = 100.0

    name: ClassVar[str] = "time"


@register_check(MissingCheckSpec)
def missing_check(_dataset_spec: DatasetSpec, _check_spec: MissingCheckSpec) -> bool:
    return True


@register_check(TimeCheckSpec)
def time_check(_dataset_spec: DatasetSpec, _check_spec: TimeCheckSpec) -> bool:
    return True


def run_check(dataset_spec: DatasetSpec, check_spec: CheckSpecBase) -> bool:
    handler = CHECK_REGISTRY.get(type(check_spec))
    if handler is None:
        raise ValueError(f"No handler registered for check {type(check_spec).__name__}")
    return handler(dataset_spec, check_spec)
