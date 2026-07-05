from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.segments import ParquetSegment
from batgrad.ml.data.batch import (
    Batch,
    BatchSegmentRef,
    BatchState,
)
from batgrad.ml.data.config import PADDING_VALUE, LoaderConfig, ScalingRule, WindowConfig
from batgrad.ml.data.planning import BatchPlan, StreamPlan, WindowRef, row_segments
from batgrad.ml.data.scaling import scale_data
from batgrad.storage.segments import (
    iter_segment_frames,
    iter_segment_window_frames,
    iter_segment_window_refs,
)

if TYPE_CHECKING:
    from batgrad.ml.data.index import MlDatasetIndex
    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class StreamTensorCache:
    tensors: dict[tuple[object, ...], torch.Tensor]
    source_columns: tuple[str, ...]
    column_indices: dict[str, int]
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    input_indices: torch.Tensor
    target_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class RefTensors:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    segments: tuple[BatchSegmentRef, ...]


def materialize_window_ref(
    store: DatasetStoreReader,
    ref: WindowRef,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    config: LoaderConfig,
    batch_idx: int,
) -> Batch:
    return _materialize_window_ref_from_store_or_cache(
        store,
        ref,
        input_columns,
        target_columns,
        scaling,
        config,
        batch_idx,
    )


def materialize_batch_plan(
    store: DatasetStoreReader,
    plan: BatchPlan,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    config: LoaderConfig,
    batch_idx: int,
    *,
    schema_by_path: dict[str, set[str]] | None = None,
    stream_tensor_cache: StreamTensorCache | None = None,
) -> Batch:
    if not plan.refs:
        raise ValueError("BatchPlan must contain at least one WindowRef")
    if config.strategy == "sequential" and len(plan.refs) == 1:
        return _materialize_window_ref_from_store_or_cache(
            store,
            plan.refs[0],
            input_columns,
            target_columns,
            scaling,
            config,
            batch_idx,
            schema_by_path=schema_by_path,
            stream_tensor_cache=stream_tensor_cache,
        )

    if stream_tensor_cache is not None:
        return _materialize_batch_plan_from_cache(
            plan,
            input_columns,
            target_columns,
            config,
            batch_idx,
            stream_tensor_cache,
        )

    samples = tuple(
        _materialize_ref_tensors_from_store(
            store,
            sample_ref,
            input_columns,
            target_columns,
            config.window_for(sample_ref.protocol).seq_len,
            source_columns(input_columns, target_columns),
            selected_scaling_rules(scaling, input_columns, target_columns),
            schema_by_path=schema_by_path,
        )
        for sample_ref in plan.refs
    )
    return _batch_from_ref_tensors(
        plan.refs[0],
        plan.refs,
        samples,
        batch_idx,
        stateful_group_idx=plan.stateful_group_idx,
        stateful_step_idx=plan.stateful_step_idx,
        stateful_steps=plan.stateful_steps,
    )


def resolve_index_schema_by_protocol(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
) -> dict[DatasetProtocolId, tuple[str, ...]]:
    schemas: dict[DatasetProtocolId, set[str] | None] = {}
    for row in index.frame.iter_rows(named=True):
        protocol = DatasetProtocolId(str(row[BaseColumns.proto]))
        current = schemas.get(protocol)
        for segment in row_segments(row):
            names = set(store.scan_table(segment.path, limit=0).collect_schema().names())
            current = names if current is None else current & names
        schemas[protocol] = current
    return {protocol: tuple(sorted(columns or ())) for protocol, columns in schemas.items()}


def build_stream_tensor_cache(
    store: DatasetStoreReader,
    stream_plans: tuple[StreamPlan, ...],
    source_columns_: tuple[str, ...],
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    schema_by_path_: dict[str, set[str]],
) -> StreamTensorCache:
    column_indices = {column: idx for idx, column in enumerate(source_columns_)}
    tensors = {
        stream.stream_identity: _frame_to_tensor(
            collect_stream_data(
                store,
                stream.segments,
                source_columns_,
                scaling,
                schema_by_path_=schema_by_path_,
            )
        )
        for stream in stream_plans
    }
    return StreamTensorCache(
        tensors=tensors,
        source_columns=source_columns_,
        column_indices=column_indices,
        input_columns=input_columns,
        target_columns=target_columns,
        input_indices=torch.tensor(
            [column_indices[column] for column in input_columns], dtype=torch.long
        ),
        target_indices=torch.tensor(
            [column_indices[column] for column in target_columns], dtype=torch.long
        ),
    )


def schema_by_path(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    requested: tuple[str, ...],
) -> dict[str, set[str]]:
    schemas: dict[str, set[str]] = {}
    if not requested:
        return schemas
    for row in index.frame.iter_rows(named=True):
        for segment in row_segments(row):
            path = segment.path
            if path not in schemas:
                schemas[path] = set(store.scan_table(path, limit=0).collect_schema().names())
            missing = sorted(set(requested) - schemas[path])
            if missing:
                raise ValueError(f"Shard {path!r} is missing requested ML columns: {missing}")
    return schemas


def source_columns(
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*input_columns, *target_columns)))


def selected_scaling_rules(
    scaling: tuple[ScalingRule, ...],
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> tuple[ScalingRule, ...]:
    selected = set(source_columns(input_columns, target_columns))
    return tuple(rule for rule in scaling if rule.name in selected)


def collect_stream_data(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    *,
    schema_by_path_: dict[str, set[str]] | None = None,
) -> pl.DataFrame:
    for raw_segment in segments:
        segment = ParquetSegment.from_value(raw_segment)
        _validate_segment_columns(store, segment.path, columns, schema_by_path=schema_by_path_)
    chunks = tuple(iter_segment_frames(store, segments, 500_000, columns=columns))
    if any(ParquetSegment.from_value(segment).row_count > 0 for segment in segments) and not chunks:
        raise ValueError("Manifest segments produced no rows during ML batch materialization")
    if not chunks:
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.Float64))
    frame = chunks[0] if len(chunks) == 1 else pl.concat(chunks, how="vertical")
    return scale_data(frame, scaling) if scaling else frame


def _materialize_batch_plan_from_cache(
    plan: BatchPlan,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    config: LoaderConfig,
    batch_idx: int,
    cache: StreamTensorCache,
) -> Batch:
    # PERF: full_in_mem shuffled batches are still roughly 3x slower than
    # sequential batches in many-stream benchmarks because they assemble many
    # independent stream windows instead of reshaping one contiguous slice. A
    # grouped gather variant was benchmarked here but was slower for that common
    # one-ref-per-stream case, so keep this predictable per-ref path unless the
    # sampling/layout strategy changes.
    samples = tuple(
        _materialize_ref_tensors_from_cache(
            sample_ref,
            input_columns,
            target_columns,
            config.window_for(sample_ref.protocol).seq_len,
            config.window_for(sample_ref.protocol).seq_len + 1,
            cache,
        )
        for sample_ref in plan.refs
    )
    return _batch_from_ref_tensors(
        plan.refs[0],
        plan.refs,
        samples,
        batch_idx,
        stateful_group_idx=plan.stateful_group_idx,
        stateful_step_idx=plan.stateful_step_idx,
        stateful_steps=plan.stateful_steps,
    )


def _materialize_window_ref_from_cache(
    ref: WindowRef,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    window_config: WindowConfig,
    batch_idx: int,
    cache: StreamTensorCache,
) -> Batch:
    sample = _materialize_ref_tensors_from_cache(
        ref,
        input_columns,
        target_columns,
        window_config.batch_size * window_config.seq_len,
        window_config.window_rows,
        cache,
    )
    return _batch_from_protocol_tensors(
        ref,
        (ref,),
        sample.inputs.reshape(window_config.batch_size, window_config.seq_len, len(input_columns)),
        sample.targets.reshape(
            window_config.batch_size, window_config.seq_len, len(target_columns)
        ),
        sample.mask.reshape(window_config.batch_size, window_config.seq_len),
        sample.segments,
        batch_idx,
    )


def _batch_from_ref_tensors(
    ref: WindowRef,
    refs: tuple[WindowRef, ...],
    samples: tuple[RefTensors, ...],
    batch_idx: int,
    *,
    stateful_group_idx: int | None = None,
    stateful_step_idx: int | None = None,
    stateful_steps: int | None = None,
) -> Batch:
    segments = tuple(segment for sample in samples for segment in sample.segments)
    return _batch_from_protocol_tensors(
        ref,
        refs,
        torch.stack([sample.inputs for sample in samples], dim=0),
        torch.stack([sample.targets for sample in samples], dim=0),
        torch.stack([sample.mask for sample in samples], dim=0),
        segments,
        batch_idx,
        stateful_group_idx=stateful_group_idx,
        stateful_step_idx=stateful_step_idx,
        stateful_steps=stateful_steps,
    )


def _batch_from_protocol_tensors(
    ref: WindowRef,
    refs: tuple[WindowRef, ...],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    segments: tuple[BatchSegmentRef, ...],
    batch_idx: int,
    *,
    stateful_group_idx: int | None = None,
    stateful_step_idx: int | None = None,
    stateful_steps: int | None = None,
) -> Batch:
    return Batch(
        inputs=inputs,
        targets=targets,
        mask=mask,
        all_valid=bool(mask.all().item()),
        state=BatchState(
            split=ref.split,
            batch_idx=batch_idx,
            protocols=tuple(sample_ref.protocol for sample_ref in refs),
            manifest_paths=tuple(sample_ref.manifest_path for sample_ref in refs),
            manifest_row_ids=tuple(sample_ref.manifest_row_id for sample_ref in refs),
            group_keys=tuple(sample_ref.group_key for sample_ref in refs),
            alignment_keys=tuple(sample_ref.alignment_key for sample_ref in refs),
            segments=segments,
            window_offsets=tuple(sample_ref.offset for sample_ref in refs),
            stateful_group_idx=stateful_group_idx,
            stateful_step_idx=stateful_step_idx,
            stateful_steps=stateful_steps,
        ),
    )


def _cache_window(cache: StreamTensorCache, ref: WindowRef, rows: int) -> tuple[torch.Tensor, int]:
    stream = cache.tensors[ref.stream_identity]
    window = stream[ref.offset : ref.offset + rows]
    real_rows = int(window.shape[0])
    if real_rows >= rows:
        return window, real_rows
    padding = torch.full(
        (rows - real_rows, stream.shape[1]),
        PADDING_VALUE,
        dtype=stream.dtype,
        device=stream.device,
    )
    return torch.cat((window, padding), dim=0), real_rows


def _cache_window_tensors(
    window: torch.Tensor,
    real_rows: int,
    cache: StreamTensorCache,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = _select_cached_columns(window[:seq_len], cache, input_columns)
    targets = _select_cached_columns(window[1 : seq_len + 1], cache, target_columns)
    return inputs, targets, _window_mask(real_rows, seq_len)


def _materialize_ref_tensors_from_cache(
    ref: WindowRef,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    seq_len: int,
    sample_rows: int,
    cache: StreamTensorCache,
) -> RefTensors:
    window, real_rows = _cache_window(cache, ref, sample_rows)
    inputs, targets, mask = _cache_window_tensors(
        window,
        real_rows,
        cache,
        input_columns,
        target_columns,
        seq_len,
    )
    return _ref_tensors(ref, inputs, targets, mask, real_rows, sample_rows)


def _select_cached_columns(
    tensor: torch.Tensor,
    cache: StreamTensorCache,
    columns: tuple[str, ...],
) -> torch.Tensor:
    if columns == cache.input_columns:
        return tensor.index_select(1, cache.input_indices)
    if columns == cache.target_columns:
        return tensor.index_select(1, cache.target_indices)
    indices = torch.tensor([cache.column_indices[column] for column in columns], dtype=torch.long)
    return tensor.index_select(1, indices)


def _materialize_window_ref_from_store_or_cache(
    store: DatasetStoreReader,
    ref: WindowRef,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    config: LoaderConfig,
    batch_idx: int,
    *,
    schema_by_path: dict[str, set[str]] | None = None,
    stream_tensor_cache: StreamTensorCache | None = None,
) -> Batch:
    window_config = config.window_for(ref.protocol)
    if stream_tensor_cache is not None:
        return _materialize_window_ref_from_cache(
            ref,
            input_columns,
            target_columns,
            window_config,
            batch_idx,
            stream_tensor_cache,
        )
    source_columns_ = source_columns(input_columns, target_columns)
    sample = _materialize_ref_tensors_from_store(
        store,
        ref,
        input_columns,
        target_columns,
        window_config.batch_size * window_config.seq_len,
        source_columns_,
        selected_scaling_rules(scaling, input_columns, target_columns),
        schema_by_path=schema_by_path,
        sample_rows=window_config.window_rows,
    )
    return _batch_from_protocol_tensors(
        ref,
        (ref,),
        sample.inputs.reshape(window_config.batch_size, window_config.seq_len, len(input_columns)),
        sample.targets.reshape(
            window_config.batch_size, window_config.seq_len, len(target_columns)
        ),
        sample.mask.reshape(window_config.batch_size, window_config.seq_len),
        sample.segments,
        batch_idx,
    )


def _materialize_ref_tensors_from_store(
    store: DatasetStoreReader,
    ref: WindowRef,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    seq_len: int,
    source_columns_: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    *,
    schema_by_path: dict[str, set[str]] | None = None,
    sample_rows: int | None = None,
) -> RefTensors:
    rows = sample_rows or seq_len + 1
    window = _collect_window_data(
        store,
        ref.segments,
        ref.offset,
        rows,
        source_columns_,
        scaling,
        schema_by_path=schema_by_path,
    )
    real_rows = window.height
    if real_rows < rows:
        window = _pad_window(window, rows, source_columns_)
    inputs, targets, mask = _frame_window_tensors(
        window,
        real_rows,
        input_columns,
        target_columns,
        seq_len,
    )
    return _ref_tensors(ref, inputs, targets, mask, real_rows, rows)


def _frame_window_tensors(
    window: pl.DataFrame,
    real_rows: int,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = _frame_to_tensor(window.slice(0, seq_len).select(input_columns))
    targets = _frame_to_tensor(window.slice(1, seq_len).select(target_columns))
    return inputs, targets, _window_mask(real_rows, seq_len)


def _window_mask(real_rows: int, seq_len: int) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[: max(0, min(real_rows - 1, seq_len))] = True
    return mask


def _ref_tensors(
    ref: WindowRef,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    real_rows: int,
    requested_rows: int,
) -> RefTensors:
    return RefTensors(
        inputs=inputs,
        targets=targets,
        mask=mask,
        segments=_window_segment_refs(
            ref.segments, ref.offset, max(0, min(real_rows, requested_rows))
        ),
    )


def _collect_window_data(
    store: DatasetStoreReader,
    segments: tuple[ParquetSegment, ...],
    window_offset: int,
    window_rows: int,
    columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    *,
    schema_by_path: dict[str, set[str]] | None = None,
) -> pl.DataFrame:
    refs = tuple(iter_segment_window_refs(segments, window_offset, window_rows))
    for ref in refs:
        _validate_segment_columns(store, ref.segment.path, columns, schema_by_path=schema_by_path)
    chunks = tuple(
        iter_segment_window_frames(
            store,
            segments,
            window_offset,
            window_rows,
            columns=columns,
        )
    )
    if any(ref.window_row_count > 0 for ref in refs) and not chunks:
        raise ValueError("Manifest segments produced no rows during ML window materialization")
    if not chunks:
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.Float64))
    frame = chunks[0] if len(chunks) == 1 else pl.concat(chunks, how="vertical")
    return scale_data(frame, scaling) if scaling else frame


def _validate_segment_columns(
    store: DatasetStoreReader,
    segment_path: str,
    columns: tuple[str, ...],
    *,
    schema_by_path: dict[str, set[str]] | None = None,
) -> None:
    if schema_by_path is None:
        schema_names = set(store.scan_table(segment_path, limit=0).collect_schema().names())
    else:
        schema_names = schema_by_path.get(segment_path)
        if schema_names is None:
            schema_names = set(store.scan_table(segment_path, limit=0).collect_schema().names())
            schema_by_path[segment_path] = schema_names
    missing = [column for column in columns if column not in schema_names]
    if missing:
        raise ValueError(
            f"Normalized segment {segment_path!r} is missing requested ML columns: {missing}"
        )


def _frame_to_tensor(frame: pl.DataFrame) -> torch.Tensor:
    filled = frame.fill_null(PADDING_VALUE)
    to_torch = getattr(filled, "to_torch", None)
    if callable(to_torch):
        try:
            tensor = to_torch(return_type="tensor")
            if isinstance(tensor, torch.Tensor):
                return tensor.to(dtype=torch.float32)
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.from_numpy(filled.to_numpy()).to(dtype=torch.float32)


def _pad_window(frame: pl.DataFrame, rows: int, columns: tuple[str, ...]) -> pl.DataFrame:
    missing = rows - frame.height
    if missing <= 0:
        return frame
    padding = pl.DataFrame({column: [PADDING_VALUE] * missing for column in columns})
    return pl.concat((frame, padding), how="vertical")


def _window_segment_refs(
    segments: tuple[ParquetSegment, ...],
    window_offset: int,
    window_rows: int,
) -> tuple[BatchSegmentRef, ...]:
    return tuple(
        BatchSegmentRef(
            path=ref.segment.path,
            row_start=ref.segment.row_start,
            row_count=ref.segment.row_count,
            window_row_start=ref.window_row_start,
            window_row_count=ref.window_row_count,
        )
        for ref in iter_segment_window_refs(segments, window_offset, window_rows)
    )
