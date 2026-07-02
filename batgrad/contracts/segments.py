from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from batgrad.contracts.mapping import BaseColumns


@dataclass(frozen=True, slots=True)
class ParquetSegment:
    """Contiguous row slice inside a generated parquet file."""

    path: str
    row_start: int
    row_count: int

    @classmethod
    def from_value(cls, value: ParquetSegment | Mapping[Any, Any]) -> ParquetSegment:
        if isinstance(value, ParquetSegment):
            return value
        return cls(
            path=str(_mapping_value(value, BaseColumns.path, "path")),
            row_start=_as_int(_mapping_value(value, BaseColumns.row0, "row_start")),
            row_count=_as_int(_mapping_value(value, BaseColumns.row_n, "row_count")),
        )

    def as_manifest_dict(self) -> dict[str, object]:
        return {
            str(BaseColumns.path): self.path,
            str(BaseColumns.row0): self.row_start,
            str(BaseColumns.row_n): self.row_count,
        }


type SegmentLike = ParquetSegment | Mapping[Any, Any]


def normalize_segments(segments: tuple[SegmentLike, ...]) -> tuple[ParquetSegment, ...]:
    return tuple(ParquetSegment.from_value(segment) for segment in segments)


def segment_values(value: object) -> tuple[SegmentLike, ...]:
    if not isinstance(value, list | tuple):
        return ()
    segments: list[SegmentLike] = []
    for segment in value:
        if isinstance(segment, ParquetSegment):
            segments.append(segment)
            continue
        if isinstance(segment, Mapping):
            segments.append(ParquetSegment.from_value(segment))
            continue
        raise TypeError(f"Expected parquet segment mapping, got {type(segment).__name__}")
    return tuple(segments)


def segment_row_count(segments: tuple[ParquetSegment, ...]) -> int:
    return sum(segment.row_count for segment in segments)


def segment_manifest_dicts(segments: tuple[ParquetSegment, ...]) -> tuple[dict[str, object], ...]:
    return tuple(segment.as_manifest_dict() for segment in segments)


def _mapping_value(value: Mapping[Any, Any], *keys: object) -> object:
    for key in keys:
        string_key = str(key)
        if string_key in value:
            return value[string_key]
        if key in value:
            return value[key]
    raise KeyError(str(keys[0]))


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Expected int-like value, got {type(value).__name__}")
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}")
