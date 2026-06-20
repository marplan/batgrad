from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from batgrad.contracts.mapping import BaseColumns, MappingSpec

type SelectorValues = tuple[object, ...] | None
type GroupSelector = dict[MappingSpec, SelectorValues]


@dataclass(frozen=True, slots=True)
class StageSelection:
    protocols: SelectorValues
    groups: tuple[GroupSelector, ...]

    @classmethod
    def from_values(
        cls,
        *,
        protocols: object,
        group_values: object,
        group_columns: Sequence[MappingSpec],
    ) -> StageSelection:
        return cls(
            protocols=normalize_selector_values(protocols),
            groups=normalize_group_selectors(group_values, group_columns),
        )

    def expr(self) -> pl.Expr | None:
        exprs = [self.protocol_expr(), self.group_expr()]
        expr = None
        for next_expr in exprs:
            if next_expr is None:
                continue
            expr = next_expr if expr is None else expr & next_expr
        return expr

    def protocol_expr(self) -> pl.Expr | None:
        if self.protocols is None:
            return None
        return (
            pl.col(BaseColumns.proto)
            .cast(pl.String)
            .is_in([str(value) for value in self.protocols])
        )

    def group_expr(self) -> pl.Expr | None:
        selector_exprs = []
        for selector in self.groups:
            expr = None
            for column, values in selector.items():
                if values is None:
                    continue
                col_expr = pl.col(column).is_in(list(values))
                expr = col_expr if expr is None else expr & col_expr
            if expr is not None:
                selector_exprs.append(expr)
        if not selector_exprs:
            return None
        expr = selector_exprs[0]
        for selector_expr in selector_exprs[1:]:
            expr |= selector_expr
        return expr

    def matches_protocol(self, value: object) -> bool:
        return self.protocols is None or any(
            str(value) == str(protocol) for protocol in self.protocols
        )

    def matches_row(self, row: Mapping[str, object]) -> bool:
        return self.matches_protocol(row.get(str(BaseColumns.proto))) and any(
            _row_matches_group(row, selector) for selector in self.groups
        )

    def matches_group_values(self, group_values: Mapping[MappingSpec, object]) -> bool:
        return any(_group_values_match(group_values, selector) for selector in self.groups)


def normalize_selector_values(value: object) -> SelectorValues:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set | frozenset):
        return tuple(value)
    return (value,)


def normalize_group_selectors(
    group_values: object,
    columns: Sequence[MappingSpec],
) -> tuple[GroupSelector, ...]:
    by_name = {str(column): column for column in columns}
    if group_values is None:
        return ({},)
    if isinstance(group_values, Mapping):
        return (_normalize_group_selector(group_values, by_name),)
    if isinstance(group_values, list | tuple):
        selectors = tuple(
            _normalize_group_selector(_expect_selector_mapping(selector), by_name)
            for selector in group_values
        )
        if not selectors:
            raise ValueError("group_values list must contain at least one selector dict")
        return selectors
    raise TypeError("group_values must be a dict, list of dicts, or None")


def _expect_selector_mapping(value: object) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("group_values list entries must be dicts")
    return value


def _row_matches_group(row: Mapping[str, object], selector: GroupSelector) -> bool:
    for column, values in selector.items():
        if values is None:
            continue
        if not any(row.get(str(column)) == value for value in values):
            return False
    return True


def _group_values_match(
    group_values: Mapping[MappingSpec, object],
    selector: GroupSelector,
) -> bool:
    for column, values in selector.items():
        if values is None:
            continue
        if not any(group_values.get(column) == value for value in values):
            return False
    return True


def all_group_columns(
    columns_by_protocol: Sequence[Sequence[MappingSpec]],
) -> tuple[MappingSpec, ...]:
    return tuple(dict.fromkeys(column for columns in columns_by_protocol for column in columns))


def _normalize_group_selector(
    selector: Mapping[Any, Any],
    columns_by_name: Mapping[str, MappingSpec],
) -> GroupSelector:
    normalized: GroupSelector = {}
    for key, value in selector.items():
        column = columns_by_name.get(str(key))
        if column is None:
            expected = ", ".join(sorted(columns_by_name))
            raise ValueError(f"Unknown group_values column {key!r}; expected one of: {expected}")
        normalized[column] = normalize_selector_values(value)
    return normalized
