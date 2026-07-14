from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, MappingSpec
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN
from batgrad.contracts.segments import ParquetSegment, normalize_segments, segment_values
from batgrad.ml.data.config import LoaderConfig, coerce_protocol
from batgrad.ml.data.index import MlDatasetIndex

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WindowRef:
    stream: StreamPlan
    offset: int

    @property
    def protocol(self) -> DatasetProtocolId:
        return self.stream.protocol

    @property
    def split(self) -> str:
        return self.stream.split

    @property
    def manifest_path(self) -> str:
        return self.stream.manifest_path

    @property
    def manifest_row_id(self) -> int:
        return self.stream.manifest_row_id

    @property
    def stream_identity(self) -> tuple[object, ...]:
        return self.stream.stream_identity

    @property
    def group_key(self) -> tuple[object, ...]:
        return self.stream.group_key

    @property
    def alignment_key(self) -> tuple[object, ...]:
        return self.stream.alignment_key

    @property
    def segments(self) -> tuple[ParquetSegment, ...]:
        return self.stream.segments

    @property
    def row_count(self) -> int:
        return self.stream.row_count


@dataclass(frozen=True, slots=True)
class StreamPlan:
    protocol: DatasetProtocolId
    split: str
    manifest_path: str
    manifest_row_id: int
    stream_identity: tuple[object, ...]
    group_key: tuple[object, ...]
    alignment_key: tuple[object, ...]
    segments: tuple[ParquetSegment, ...]
    row_count: int
    phase_start: int
    phase_stride: int


@dataclass(frozen=True, slots=True)
class BatchPlan:
    refs: tuple[WindowRef, ...]
    stateful_group_idx: int | None = None
    stateful_step_idx: int | None = None
    stateful_steps: int | None = None


def build_batch_plans(
    index: MlDatasetIndex,
    config: LoaderConfig,
    epoch_idx: int = 0,
    stream_plans: tuple[StreamPlan, ...] | None = None,
) -> tuple[BatchPlan, ...]:
    return tuple(
        iter_batch_plans(
            index,
            config,
            epoch_idx=epoch_idx,
            stream_plans=stream_plans,
        )
    )


def count_batch_plans(
    index: MlDatasetIndex,
    config: LoaderConfig,
    epoch_idx: int = 0,
    stream_plans: tuple[StreamPlan, ...] | None = None,
) -> int:
    protocol_order = _protocol_order(index, config)
    if config.strategy == "sequential":
        return sum(
            _count_sequential_batch_plans(index, protocol, config) for protocol in protocol_order
        )
    if config.strategy == "shuffled_protocol_groups":
        return _count_shuffled_protocol_group_batch_plans(
            index, protocol_order, config, epoch_idx, stream_plans=stream_plans
        )
    raise ValueError(f"Unknown batch strategy: {config.strategy!r}")


def iter_batch_plans(
    index: MlDatasetIndex,
    config: LoaderConfig,
    epoch_idx: int = 0,
    stream_plans: tuple[StreamPlan, ...] | None = None,
) -> Iterator[BatchPlan]:
    protocol_order = _protocol_order(index, config)
    if config.strategy == "sequential":
        for protocol in protocol_order:
            for ref in iter_window_refs(index, protocol, config):
                yield BatchPlan(refs=(ref,))
        return
    if config.strategy == "shuffled_protocol_groups":
        yield from _iter_shuffled_protocol_group_batch_plans(
            index, protocol_order, config, epoch_idx, stream_plans=stream_plans
        )
        return
    raise ValueError(f"Unknown batch strategy: {config.strategy!r}")


def build_stream_plans(
    index: MlDatasetIndex,
    protocol: DatasetProtocolId | object,
    config: LoaderConfig,
) -> tuple[StreamPlan, ...]:
    protocol = coerce_protocol(protocol)
    if protocol == DatasetProtocolId.eis and config.strategy == "shuffled_protocol_groups":
        return ()
    window = config.window_for(protocol)
    frame = index.frame.filter(pl.col(BaseColumns.proto).cast(pl.String) == str(protocol))
    plans: list[StreamPlan] = []
    seen: set[tuple[object, ...]] = set()
    for stream_idx, row in enumerate(frame.iter_rows(named=True)):
        row_count = int(row[BaseColumns.row_n])
        if row_count <= 1:
            continue
        group_key = key_values(row, config.group_key)
        stream_identity = _stream_identity(row, protocol, group_key)
        if stream_identity in seen:
            stream_identity = (*stream_identity, stream_idx)
        seen.add(stream_identity)
        phase_start, phase_stride = _resolve_stream_phase_components(
            seed=config.seed,
            seq_len=window.seq_len,
            stream_key=stream_identity,
        )
        plans.append(
            StreamPlan(
                protocol=protocol,
                split=row[BaseColumns.split],
                manifest_path=row[BaseColumns.manifest],
                manifest_row_id=int(row[MANIFEST_ROW_ID_COLUMN]),
                stream_identity=stream_identity,
                group_key=group_key,
                alignment_key=key_values(row, config.alignment_key),
                segments=row_segments(row),
                row_count=row_count,
                phase_start=phase_start,
                phase_stride=phase_stride,
            )
        )
    return tuple(plans)


def iter_window_refs(
    index: MlDatasetIndex,
    protocol: DatasetProtocolId | object,
    config: LoaderConfig,
) -> Iterator[WindowRef]:
    protocol = coerce_protocol(protocol)
    window = config.window_for(protocol)
    plans = build_stream_plans(index, protocol, config)
    for stream in plans:
        row_count = stream.row_count
        if row_count <= 1:
            continue
        max_offset = (
            row_count - 1 if not window.drop_incomplete else row_count - window.window_rows + 1
        )
        if max_offset <= 0:
            continue
        for offset in range(0, max_offset, window.step):
            yield _window_ref(stream, offset)


def row_segments(row: dict[str, object]) -> tuple[ParquetSegment, ...]:
    return normalize_segments(segment_values(row.get(BaseColumns.norm_segs)))


def key_values(
    row: dict[str, object], columns: tuple[str | MappingSpec, ...]
) -> tuple[object, ...]:
    return tuple(row[column] for column in columns)


def validate_key_columns(index: MlDatasetIndex, config: LoaderConfig) -> None:
    columns = set(index.frame.columns)
    missing = sorted(
        str(column)
        for column in (*config.group_key, *config.alignment_key)
        if column not in columns
    )
    if missing:
        raise ValueError(f"ML index is missing configured group/alignment columns: {missing}")


def first_protocol(index: MlDatasetIndex) -> DatasetProtocolId:
    if index.frame.height == 0:
        raise ValueError("Cannot infer active protocol from an empty ML index")
    return coerce_protocol(index.frame[0, BaseColumns.proto])


def protocol_index(index: MlDatasetIndex, protocol: DatasetProtocolId) -> MlDatasetIndex:
    return MlDatasetIndex(
        index.frame.filter(pl.col(BaseColumns.proto).cast(pl.String) == str(protocol))
    )


def _protocol_order(
    index: MlDatasetIndex,
    config: LoaderConfig,
) -> tuple[DatasetProtocolId, ...]:
    if config.protocol_order:
        return config.protocol_order
    return (first_protocol(index),)


def _iter_shuffled_protocol_group_batch_plans(
    index: MlDatasetIndex,
    protocol_order: tuple[DatasetProtocolId, ...],
    config: LoaderConfig,
    epoch_idx: int,
    *,
    stream_plans: tuple[StreamPlan, ...] | None = None,
) -> Iterator[BatchPlan]:
    """Yield shuffled stateful protocol-window batches.

    For whole-stream mode (`stateful_n_windows=-1`), all streams in a batch are
    shortened to the shortest stream length. With multi-protocol chains this can
    drop long tails and, in the worst case, prevent later protocols from being
    seen for longer streams in that batch. A later strategy should bucket by
    length or preserve state per active lane instead of truncating the batch.
    """
    if any(protocol == DatasetProtocolId.eis for protocol in protocol_order):
        raise NotImplementedError(
            "shuffled_protocol_groups does not support EIS yet; "
            "use sequential debug or select cycling/HPPC/RPT"
        )
    refs_by_stream = _shuffled_protocol_refs_by_stream(
        index, protocol_order, config, epoch_idx, stream_plans=stream_plans
    )
    stateful_segments = _stateful_segments(refs_by_stream, config.stateful_n_windows)
    _stable_shuffle(stateful_segments, seed=config.seed, epoch_idx=epoch_idx, salt="segments")

    window = config.window_for(protocol_order[0])
    for group_idx, batch_start in enumerate(range(0, len(stateful_segments), window.batch_size)):
        batch_segments = stateful_segments[batch_start : batch_start + window.batch_size]
        if len(batch_segments) != window.batch_size and config.drop_incomplete_batches:
            continue
        stateful_steps = min(len(segment) for segment in batch_segments)
        if config.stateful_n_windows == -1:
            lane_lengths = tuple(len(segment) for segment in batch_segments)
            dropped_windows = sum(length - stateful_steps for length in lane_lengths)
            if dropped_windows:
                logger.warning(
                    "Whole-stream stateful batch truncates longer lanes: group=%d "
                    "lane_lengths=%s emitted_steps=%d dropped_windows=%d",
                    group_idx,
                    lane_lengths,
                    stateful_steps,
                    dropped_windows,
                )
        for step_idx in range(stateful_steps):
            yield BatchPlan(
                refs=tuple(segment[step_idx] for segment in batch_segments),
                stateful_group_idx=group_idx,
                stateful_step_idx=step_idx,
                stateful_steps=stateful_steps,
            )


def _stateful_segments(
    refs_by_stream: dict[tuple[object, ...], tuple[WindowRef, ...]],
    stateful_n_windows: int,
) -> list[tuple[WindowRef, ...]]:
    if stateful_n_windows == -1:
        return [refs for refs in refs_by_stream.values() if refs]
    stateful_segments: list[tuple[WindowRef, ...]] = []
    for refs in refs_by_stream.values():
        for start in range(0, len(refs), stateful_n_windows):
            segment = refs[start : start + stateful_n_windows]
            if len(segment) == stateful_n_windows:
                stateful_segments.append(segment)
    return stateful_segments


def _count_sequential_batch_plans(
    index: MlDatasetIndex,
    protocol: DatasetProtocolId,
    config: LoaderConfig,
) -> int:
    window = config.window_for(protocol)
    total = 0
    for stream in build_stream_plans(index, protocol, config):
        if stream.row_count <= 1:
            continue
        max_offset = (
            stream.row_count - 1
            if not window.drop_incomplete
            else stream.row_count - window.window_rows + 1
        )
        total += _range_count(max_offset, window.step)
    return total


def _count_shuffled_protocol_group_batch_plans(
    index: MlDatasetIndex,
    protocol_order: tuple[DatasetProtocolId, ...],
    config: LoaderConfig,
    epoch_idx: int,
    *,
    stream_plans: tuple[StreamPlan, ...] | None = None,
) -> int:
    if any(protocol == DatasetProtocolId.eis for protocol in protocol_order):
        return 0
    window = config.window_for(protocol_order[0])
    refs_by_stream = _shuffled_protocol_refs_by_stream(
        index, protocol_order, config, epoch_idx, stream_plans=stream_plans
    )
    if config.stateful_n_windows == -1:
        stateful_segments = _stateful_segments(refs_by_stream, config.stateful_n_windows)
        _stable_shuffle(stateful_segments, seed=config.seed, epoch_idx=epoch_idx, salt="segments")
        return _count_whole_stream_stateful_batches(stateful_segments, window.batch_size, config)
    ref_counts = tuple(len(refs) for refs in refs_by_stream.values())
    ref_counts = tuple(count for count in ref_counts if count > 0)
    segment_count = sum(count // config.stateful_n_windows for count in ref_counts)
    batch_count = _batch_count(
        segment_count, window.batch_size, drop_incomplete=config.drop_incomplete_batches
    )
    return batch_count * config.stateful_n_windows


def _count_shuffled_stream_refs(
    stream: StreamPlan,
    seq_len: int,
    config: LoaderConfig,
    epoch_idx: int,
) -> int:
    if stream.row_count - seq_len <= 0:
        return 0
    if config.stateful_n_windows == -1:
        return _range_count(stream.row_count - seq_len, seq_len)
    phase = _resolve_stream_phase(
        epoch_idx=epoch_idx,
        seq_len=seq_len,
        phase_start=stream.phase_start,
        phase_stride=stream.phase_stride,
    )
    max_offset_exclusive = max(0, stream.row_count - seq_len)
    if phase >= max_offset_exclusive:
        return 0
    return _range_count(max_offset_exclusive - phase, seq_len)


def _count_whole_stream_stateful_batches(
    stateful_segments: list[tuple[WindowRef, ...]],
    batch_size: int,
    config: LoaderConfig,
) -> int:
    usable = len(stateful_segments)
    if config.drop_incomplete_batches:
        usable = (usable // batch_size) * batch_size
    total = 0
    for start in range(0, usable, batch_size):
        batch = stateful_segments[start : start + batch_size]
        if len(batch) == batch_size or not config.drop_incomplete_batches:
            total += min(len(segment) for segment in batch)
    return total


def _batch_count(item_count: int, batch_size: int, *, drop_incomplete: bool) -> int:
    if item_count <= 0:
        return 0
    if drop_incomplete:
        return item_count // batch_size
    return math.ceil(item_count / batch_size)


def _range_count(stop_exclusive: int, step: int) -> int:
    if stop_exclusive <= 0:
        return 0
    return math.ceil(stop_exclusive / step)


def _shuffled_protocol_refs_by_stream(
    index: MlDatasetIndex,
    protocol_order: tuple[DatasetProtocolId, ...],
    config: LoaderConfig,
    epoch_idx: int,
    *,
    stream_plans: tuple[StreamPlan, ...] | None = None,
) -> dict[tuple[object, ...], tuple[WindowRef, ...]]:
    refs_by_stream: dict[tuple[object, ...], list[WindowRef]] = {}
    for protocol in protocol_order:
        window = config.window_for(protocol)
        plans = (
            tuple(stream for stream in stream_plans if stream.protocol == protocol)
            if stream_plans is not None
            else build_stream_plans(index, protocol, config)
        )
        for stream in plans:
            row_count = stream.row_count
            max_offset = row_count - window.seq_len
            if max_offset <= 0:
                continue
            offsets = _whole_stream_offsets(row_count, window.seq_len)
            if config.stateful_n_windows != -1:
                phase = _resolve_stream_phase(
                    epoch_idx=epoch_idx,
                    seq_len=window.seq_len,
                    phase_start=stream.phase_start,
                    phase_stride=stream.phase_stride,
                )
                offsets = _build_phase_offsets(
                    stream_len=row_count,
                    step=window.seq_len,
                    phase=phase,
                )
            refs = [_window_ref(stream, offset) for offset in offsets]
            if not refs:
                continue
            stream_key = (
                stream.alignment_key
                if config.cross_protocol_state_carry == "chain"
                else (*stream.alignment_key, protocol)
            )
            refs_by_stream.setdefault(stream_key, []).extend(refs)
    return {key: tuple(refs) for key, refs in refs_by_stream.items() if refs}


def _window_ref(stream: StreamPlan, offset: int) -> WindowRef:
    return WindowRef(stream=stream, offset=offset)


def _stable_int(*parts: object) -> int:
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _stable_shuffle(
    items: list[tuple[WindowRef, ...]], seed: int, epoch_idx: int, salt: str
) -> None:
    items.sort(
        key=lambda refs: _stable_int(
            seed,
            epoch_idx,
            salt,
            tuple((ref.manifest_path, ref.manifest_row_id, ref.offset) for ref in refs),
        )
    )


def _resolve_stream_phase_components(
    seed: int, seq_len: int, stream_key: object
) -> tuple[int, int]:
    if seq_len <= 1:
        return 0, 1
    start = _stable_int(seed, stream_key, "phase", "start") % seq_len
    stride = (_stable_int(seed, stream_key, "phase", "stride") % (seq_len - 1)) + 1
    while math.gcd(stride, seq_len) != 1:
        stride += 1
        if stride >= seq_len:
            stride = 1
    return start, stride


def _resolve_stream_phase(epoch_idx: int, seq_len: int, phase_start: int, phase_stride: int) -> int:
    if seq_len <= 1:
        return 0
    return (phase_start + (int(epoch_idx) * phase_stride)) % seq_len


def _build_phase_offsets(stream_len: int, step: int, phase: int) -> tuple[int, ...]:
    if stream_len <= 1:
        return ()
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    max_offset_exclusive = max(0, stream_len - step)
    if phase >= max_offset_exclusive:
        return ()
    return tuple(range(phase, max_offset_exclusive, step))


def _whole_stream_offsets(stream_len: int, step: int) -> tuple[int, ...]:
    if stream_len <= 1:
        return ()
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    max_offset_exclusive = max(0, stream_len - step)
    return tuple(range(0, max_offset_exclusive, step))


def _stream_identity(
    row: dict[str, object], protocol: DatasetProtocolId, group_key: tuple[object, ...]
) -> tuple[object, ...]:
    return (
        group_key,
        protocol,
        row.get(BaseColumns.manifest),
        row.get(MANIFEST_ROW_ID_COLUMN),
    )
