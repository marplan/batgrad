from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from batgrad.contracts.domains import Domains
from batgrad.data.transforms.checks import CHECK_REGISTRY, CheckSpecBase, TimeCheckSpec
from batgrad.data.transforms.resampling import RESAMPLING_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from batgrad.contracts.columns import ColumnSpec
    from batgrad.contracts.domains import DomainSpec
    from batgrad.contracts.values import ValueSpec
    from batgrad.data.transforms.resampling import ResamplingSpecBase
    from batgrad.data.transforms.transforms import TransformSpecBase

DOMAIN_REQUIRED_CHECKS: dict[ValueSpec, frozenset[type[CheckSpecBase]]] = {
    Domains.time.domain_id: frozenset({TimeCheckSpec}),
}


@dataclass(frozen=True, slots=True)
class ProtocolNormalizeSpec:
    domain: DomainSpec
    columns: tuple[ColumnSpec, ...]
    group_by: tuple[ColumnSpec, ...]
    order_by: tuple[ColumnSpec, ...] = field(default_factory=tuple)
    transforms: tuple[TransformSpecBase, ...] = field(default_factory=tuple)
    checks: tuple[CheckSpecBase, ...] = field(default_factory=tuple)
    resampling: ResamplingSpecBase | None = None


@dataclass(frozen=True, slots=True)
class NormalizeSpec:
    spec_id: str
    time_convention: str = "start_of_interval"
    protocol_specs: Mapping[str, ProtocolNormalizeSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_specs", dict(self.protocol_specs))
        self._validate_registered_required_checks()
        self._validate_required_domain_checks()
        self._validate_registered_resampling()

    def _validate_registered_required_checks(self) -> None:
        required_checks: set[type[CheckSpecBase]] = set()
        for checks in DOMAIN_REQUIRED_CHECKS.values():
            required_checks.update(checks)

        missing_handlers = required_checks - set(CHECK_REGISTRY)
        if missing_handlers:
            names = sorted(check.name for check in missing_handlers)
            raise ValueError(f"Required checks are missing handlers: {names}")

    def _validate_required_domain_checks(self) -> None:
        for protocol, protocol_spec in self.protocol_specs.items():
            required_checks = DOMAIN_REQUIRED_CHECKS.get(
                protocol_spec.domain.domain_id,
                frozenset(),
            )
            selected_checks = {type(check) for check in protocol_spec.checks}
            missing_checks = required_checks - selected_checks

            if missing_checks:
                names = sorted(check.name for check in missing_checks)
                raise ValueError(
                    f"Protocol {protocol!r} with domain "
                    f"{protocol_spec.domain.domain_id!r} is missing required checks: {names}",
                )

    def _validate_registered_resampling(self) -> None:
        for protocol, protocol_spec in self.protocol_specs.items():
            resampling = protocol_spec.resampling
            if resampling is None:
                continue

            if type(resampling) not in RESAMPLING_REGISTRY:
                raise ValueError(
                    f"Protocol {protocol!r} uses unregistered resampling "
                    f"{type(resampling).__name__}",
                )
