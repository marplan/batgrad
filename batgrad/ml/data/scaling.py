from __future__ import annotations

import math
from typing import TYPE_CHECKING, overload

import polars as pl
import torch

from batgrad.contracts.mapping import BaseColumns, MappingSpec
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN
from batgrad.ml.data.config import ScalingRule

if TYPE_CHECKING:
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.storage.store import DatasetStoreReader

type BoundsByColumn = dict[str | MappingSpec, tuple[float, float]]


@overload
def scale_data(data: pl.DataFrame, rules: tuple[ScalingRule, ...]) -> pl.DataFrame: ...


@overload
def scale_data(data: pl.LazyFrame, rules: tuple[ScalingRule, ...]) -> pl.LazyFrame: ...


@overload
def scale_data(data: torch.Tensor, rules: tuple[ScalingRule, ...]) -> torch.Tensor: ...


def scale_data(
    data: pl.DataFrame | pl.LazyFrame | torch.Tensor,
    rules: tuple[ScalingRule, ...],
) -> pl.DataFrame | pl.LazyFrame | torch.Tensor:
    """Scale selected frame columns or all tensor channels into model space.

    Args:
        data: Polars frame or tensor to scale. Tensor channels occupy the last
            dimension and must match `rules` one-to-one in tuple order.
        rules: Column scaling rules.

    Returns:
        A new object of the same concrete data type. Frame columns without a rule
        are unchanged.

    Raises:
        ValueError: If a tensor's channel count differs from the rule count.
    """
    if isinstance(data, pl.DataFrame | pl.LazyFrame):
        columns = data.columns if isinstance(data, pl.DataFrame) else data.collect_schema().names()
        return data.with_columns(_scale_expr(rule) for rule in rules if rule.name in columns)
    _validate_tensor_channels(data, rules)
    result = data.clone()
    for idx, rule in enumerate(rules):
        output = _scale_channel(data[..., idx], rule)
        if rule.clip:
            output = torch.clamp(output, min=rule.output_min, max=rule.output_max)
        result[..., idx] = output
    return result


def inverse_scale_data(
    data: pl.DataFrame | torch.Tensor,
    rules: tuple[ScalingRule, ...],
) -> pl.DataFrame | torch.Tensor:
    """Reverse scaling for selected frame columns or ordered tensor channels.

    Clipping is intentionally not applied during inverse scaling.

    Args:
        data: Frame matched by rule name, or tensor matched by rule order.
        rules: Scaling rules to reverse.

    Returns:
        A new frame or tensor in physical units.

    Raises:
        ValueError: If a tensor's channel count differs from the rule count.
    """
    if isinstance(data, pl.DataFrame):
        return data.with_columns(
            _inverse_scale_expr(rule) for rule in rules if rule.name in data.columns
        )
    _validate_tensor_channels(data, rules)
    result = data.clone()
    for idx, rule in enumerate(rules):
        result[..., idx] = _inverse_scale_channel(data[..., idx], rule)
    return result


def inverse_scale_tensor(
    data: torch.Tensor,
    columns: tuple[str | MappingSpec, ...],
    scaling: tuple[ScalingRule, ...],
) -> torch.Tensor:
    """Inverse-scale named tensor channels while leaving other channels unchanged.

    Args:
        data: Tensor whose last dimension corresponds to `columns`.
        columns: Ordered tensor channel names.
        scaling: Available rules, matched to channels by name.

    Returns:
        A cloned tensor in which channels with matching rules are in physical
        units.
    """
    rules_by_name = {rule.name: rule for rule in scaling}
    selected = tuple(rules_by_name.get(str(column)) for column in columns)
    result = data.clone()
    for idx, rule in enumerate(selected):
        if rule is None:
            continue
        result[..., idx] = _inverse_scale_channel(data[..., idx], rule)
    return result


def minmax_scaling(
    bounds: BoundsByColumn,
    *,
    output_min: float = -1.0,
    output_max: float = 1.0,
    clip: bool = False,
) -> tuple[ScalingRule, ...]:
    """Construct linear scaling rules from ordered column bounds.

    Args:
        bounds: Mapping from columns to `(input_min, input_max)` physical bounds.
        output_min: Shared lower model-space bound.
        output_max: Shared upper model-space bound.
        clip: Whether forward scaling clips to output bounds.

    Returns:
        Rules preserving the mapping's iteration order.

    Examples:
        Scale voltage and current to `[-1, 1]`:

        ```python
        rules = minmax_scaling({"voltage": (2.5, 4.2), "current": (-5.0, 5.0)})
        ```
    """
    return tuple(
        ScalingRule(
            column,
            float(input_min),
            float(input_max),
            output_min=float(output_min),
            output_max=float(output_max),
            clip=clip,
        )
        for column, (input_min, input_max) in bounds.items()
    )


def validate_scaling_bounds(
    index: MlDatasetIndex,
    store: DatasetStoreReader,
    rules: tuple[ScalingRule, ...],
) -> None:
    del store
    if not rules or index.frame.height == 0:
        return
    if BaseColumns.norm_stats not in index.frame.columns:
        raise ValueError(
            f"ML scaling validation requires normalized manifests with {BaseColumns.norm_stats!r}. "
            "Regenerate normalized data with the current normalization pipeline."
        )
    for row in index.frame.iter_rows(named=True):
        for rule in rules:
            _validate_row_rule_bounds(row, rule)


def _validate_row_rule_bounds(
    row: dict[str, object],
    rule: ScalingRule,
) -> None:
    stats = _row_stats(row)
    observed = stats.get(rule.name)
    if observed is None:
        raise ValueError(
            f"Scaling bounds for column {rule.name!r} require manifest stats. "
            f"Regenerate normalized data or remove this ScalingRule. "
            f"dataset id={row.get(BaseColumns.set_id)!r}, "
            f"protocol={row.get(BaseColumns.proto)!r}, "
            f"manifest path={row.get(BaseColumns.manifest)!r}, "
            f"manifest row id={row.get(MANIFEST_ROW_ID_COLUMN)!r}"
        )
    observed_min, observed_max = observed
    if observed_min < rule.input_min or observed_max > rule.input_max:
        raise ValueError(
            f"Scaling bounds violated for column {rule.name!r}. "
            f"Configured: [{rule.input_min}, {rule.input_max}]. "
            f"Observed: [{observed_min}, {observed_max}]. "
            f"dataset id={row.get(BaseColumns.set_id)!r}, "
            f"protocol={row.get(BaseColumns.proto)!r}, "
            f"cell id={row.get(BaseColumns.cell_id)!r}, "
            f"cycle index={row.get(BaseColumns.cidx)!r}, "
            f"manifest path={row.get(BaseColumns.manifest)!r}, "
            f"manifest row id={row.get(MANIFEST_ROW_ID_COLUMN)!r}"
        )


def _row_stats(row: dict[str, object]) -> dict[str, tuple[float, float]]:
    value = row.get(BaseColumns.norm_stats)
    if not isinstance(value, list | tuple):
        raise TypeError(
            f"Manifest row is missing {BaseColumns.norm_stats!r}. Regenerate normalized data. "
            f"manifest path={row.get(BaseColumns.manifest)!r}, "
            f"manifest row id={row.get(MANIFEST_ROW_ID_COLUMN)!r}"
        )
    stats: dict[str, tuple[float, float]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        column = item.get("column")
        min_value = _float_stat_value(item.get("min"))
        max_value = _float_stat_value(item.get("max"))
        if column is None or min_value is None or max_value is None:
            continue
        stats[str(column)] = (min_value, max_value)
    return stats


def _float_stat_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str | int | float):
        return float(value)
    return None


def _scale_expr(rule: ScalingRule) -> pl.Expr:
    lower, upper = _transformed_bounds(rule)
    source = _transform_expr(pl.col(rule.name), rule)
    expr = ((source - lower) / (upper - lower)) * (
        rule.output_max - rule.output_min
    ) + rule.output_min
    if rule.clip:
        expr = expr.clip(rule.output_min, rule.output_max)
    return expr.alias(rule.name)


def _inverse_scale_expr(rule: ScalingRule) -> pl.Expr:
    lower, upper = _transformed_bounds(rule)
    transformed = ((pl.col(rule.name) - rule.output_min) / (rule.output_max - rule.output_min)) * (
        upper - lower
    ) + lower
    return _inverse_transform_expr(transformed, rule).alias(rule.name)


def _validate_tensor_channels(data: torch.Tensor, rules: tuple[ScalingRule, ...]) -> None:
    if data.shape[-1] != len(rules):
        raise ValueError(
            f"Shape mismatch: tensor has {data.shape[-1]} channels, rules={len(rules)}"
        )


def _inverse_scale_channel(data: torch.Tensor, rule: ScalingRule) -> torch.Tensor:
    lower, upper = _transformed_bounds(rule)
    transformed = ((data - rule.output_min) / (rule.output_max - rule.output_min)) * (
        upper - lower
    ) + lower
    return _inverse_transform_tensor(transformed, rule)


def _scale_channel(data: torch.Tensor, rule: ScalingRule) -> torch.Tensor:
    lower, upper = _transformed_bounds(rule)
    transformed = _transform_tensor(data, rule)
    return ((transformed - lower) / (upper - lower)) * (
        rule.output_max - rule.output_min
    ) + rule.output_min


def _transformed_bounds(rule: ScalingRule) -> tuple[float, float]:
    if rule.transform == "log1p":
        return math.log1p(rule.input_min), math.log1p(rule.input_max)
    return rule.input_min, rule.input_max


def _transform_expr(expr: pl.Expr, rule: ScalingRule) -> pl.Expr:
    if rule.transform == "log1p":
        return expr.log1p()
    return expr


def _inverse_transform_expr(expr: pl.Expr, rule: ScalingRule) -> pl.Expr:
    if rule.transform == "log1p":
        return expr.exp() - 1.0
    return expr


def _transform_tensor(data: torch.Tensor, rule: ScalingRule) -> torch.Tensor:
    if rule.transform == "log1p":
        return torch.log1p(data)
    return data


def _inverse_transform_tensor(data: torch.Tensor, rule: ScalingRule) -> torch.Tensor:
    if rule.transform == "log1p":
        return torch.expm1(data)
    return data
