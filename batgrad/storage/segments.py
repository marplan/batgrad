from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.segments import SegmentLike, normalize_segments, segment_row_count
from batgrad.storage.chunks import iter_data_chunks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.contracts.segments import ParquetSegment
    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class SegmentWindowRef:
    """Concrete overlap between a logical segment stream and a requested row window."""

    segment: ParquetSegment
    window_row_start: int
    window_row_count: int


@dataclass(frozen=True, slots=True)
class SegmentSource:
    """Store plus segment list used to scan or chunk selected data."""

    store: DatasetStoreReader
    segments: tuple[ParquetSegment, ...]
    row_count: int | None = None

    @classmethod
    def from_values(
        cls,
        store: DatasetStoreReader,
        segments: tuple[SegmentLike, ...],
        *,
        row_count: int | None = None,
    ) -> SegmentSource:
        """Build a segment source from manifest segment values."""
        normalized = normalize_segments(segments)
        return cls(
            store=store,
            segments=normalized,
            row_count=segment_row_count(normalized) if row_count is None else row_count,
        )

    def scan(self) -> pl.LazyFrame:
        """Scan all referenced segments as one lazy frame."""
        return scan_segment_frames(self.store, self.segments)

    def iter_chunks(
        self,
        chunk_rows: int,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Iterate referenced segments as dataframe chunks."""
        yield from iter_segment_frames(self.store, self.segments, chunk_rows, columns=columns)


def collect_frame(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def collect_segment_frames(
    store: DatasetStoreReader,
    segments: tuple[SegmentLike, ...],
    *,
    columns: tuple[str, ...] | None = None,
    chunk_rows: int = 500_000,
) -> pl.DataFrame:
    frames = list(iter_segment_frames(store, segments, chunk_rows, columns=columns))
    if frames:
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")
    return pl.DataFrame(schema=dict.fromkeys(columns or (), pl.Float64))


def collect_segment_window_frames(
    store: DatasetStoreReader,
    segments: tuple[SegmentLike, ...],
    *,
    offset: int,
    rows: int,
    columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    frames = list(iter_segment_window_frames(store, segments, offset, rows, columns=columns))
    if frames:
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")
    return pl.DataFrame(schema=dict.fromkeys(columns or (), pl.Float64))


def iter_segment_frames(
    store: DatasetStoreReader,
    segments: tuple[SegmentLike, ...],
    chunk_rows: int,
    columns: tuple[str, ...] | None = None,
) -> Iterator[pl.DataFrame]:
    """Iterate parquet segment slices as dataframe chunks."""
    for segment in normalize_segments(segments):
        if segment.row_count <= 0:
            continue
        yield from store.iter_table_slices(
            segment.path,
            ((segment.row_start, segment.row_count),),
            chunk_rows,
            columns=columns,
        )


def iter_segment_window_frames(
    store: DatasetStoreReader,
    segments: tuple[SegmentLike, ...],
    offset: int,
    rows: int,
    columns: tuple[str, ...] | None = None,
) -> Iterator[pl.DataFrame]:
    """Iterate a row window across concatenated parquet segments."""
    for ref in iter_segment_window_refs(segments, offset, rows):
        yield from store.iter_table_slices(
            ref.segment.path,
            ((ref.window_row_start, ref.window_row_count),),
            max(1, rows),
            columns=columns,
        )


def iter_segment_window_refs(
    segments: tuple[SegmentLike, ...],
    offset: int,
    rows: int,
) -> Iterator[SegmentWindowRef]:
    """Iterate concrete segment overlaps for a row window across concatenated segments."""
    if rows <= 0:
        return
    window_start = max(0, offset)
    window_end = window_start + rows
    cursor = 0
    for segment in normalize_segments(segments):
        segment_start = cursor
        segment_end = cursor + segment.row_count
        cursor = segment_end
        overlap_start = max(window_start, segment_start)
        overlap_end = min(window_end, segment_end)
        if overlap_start >= overlap_end:
            continue
        local_start = segment.row_start + overlap_start - segment_start
        local_rows = overlap_end - overlap_start
        yield SegmentWindowRef(
            segment=segment,
            window_row_start=local_start,
            window_row_count=local_rows,
        )


def scan_segment_frames(
    store: DatasetStoreReader,
    segments: tuple[SegmentLike, ...],
) -> pl.LazyFrame:
    """Scan parquet segment slices and concatenate them lazily."""
    frames = [
        store.scan_table(segment.path).slice(
            segment.row_start,
            segment.row_count,
        )
        for segment in normalize_segments(segments)
    ]
    if not frames:
        raise ValueError("No table segments to scan")
    return frames[0] if len(frames) == 1 else pl.concat(frames)


def coalesce_frames(
    frames: Iterator[pl.DataFrame],
    chunk_rows: int,
) -> Iterator[pl.DataFrame]:
    """Coalesce small frames into chunks with up to `chunk_rows` rows."""
    pending: list[pl.DataFrame] = []
    pending_rows = 0
    for frame in frames:
        if frame.height == 0:
            continue
        pending.append(frame)
        pending_rows += frame.height
        if pending_rows < chunk_rows:
            continue
        combined = pl.concat(pending, how="vertical")
        while combined.height >= chunk_rows:
            yield combined.slice(0, chunk_rows)
            combined = combined.slice(chunk_rows)
        pending = [combined] if combined.height > 0 else []
        pending_rows = combined.height
    if pending:
        yield from iter_data_chunks(pl.concat(pending, how="vertical"), chunk_rows)
