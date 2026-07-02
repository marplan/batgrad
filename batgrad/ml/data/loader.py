from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from batgrad.logging import get_logger
from batgrad.ml.data import materialization, planning
from batgrad.ml.data.batch import Batch
from batgrad.ml.data.config import (
    LoaderConfig,
    ScalingRule,
    ValidationConfig,
    coerce_protocol,
)
from batgrad.ml.data.index import (
    ManifestPaths,
    MlDatasetIndex,
    ProtocolMode,
    build_index,
)
from batgrad.ml.data.scaling import validate_scaling_bounds

logger = get_logger(__name__)
StreamTensorCache = materialization.StreamTensorCache

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.data._utils.worker import WorkerInfo

    from batgrad.contracts.mapping import DatasetProtocolId, MappingSpec
    from batgrad.ml.data.planning import BatchPlan, StreamPlan
    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class DistributedInfo:
    rank: int = 0
    world_size: int = 1


class MlDataIterable(IterableDataset[Batch]):
    def __init__(
        self,
        store: DatasetStoreReader,
        index: MlDatasetIndex,
        input_columns: tuple[str | MappingSpec, ...],
        target_columns: tuple[str | MappingSpec, ...],
        scaling: tuple[ScalingRule, ...],
        config: LoaderConfig,
        active_protocol: DatasetProtocolId | None = None,
    ) -> None:
        self.store = store
        self.full_index = index
        self.index = index.filter_split(config.split)
        self.input_columns = tuple(input_columns)
        self.target_columns = tuple(target_columns)
        self.scaling = scaling
        self.config = config
        self.active_protocol = (
            config.active_protocol or active_protocol or planning.first_protocol(self.index)
        )
        self.active_index = planning.protocol_index(self.index, self.active_protocol)
        planning.validate_key_columns(self.active_index, self.config)
        self.stream_plans = planning.build_stream_plans(
            self.active_index, self.active_protocol, self.config
        )
        self.active_scaling = materialization.selected_scaling_rules(
            self.scaling,
            self.input_columns,
            self.target_columns,
        )
        source_columns_ = materialization.source_columns(self.input_columns, self.target_columns)
        self.schema_by_path = (
            materialization.schema_by_path(
                self.store,
                self.active_index,
                source_columns_,
            )
            if self.config.data_access == "full_in_mem"
            else {}
        )
        validate_scaling_bounds(self.active_index, self.store, self.active_scaling)
        _log_data_access_plan(
            self.stream_plans,
            self.config,
            self.active_protocol,
            source_columns_,
        )
        stream_tensor_cache = (
            materialization.build_stream_tensor_cache(
                self.store,
                self.stream_plans,
                source_columns_,
                self.input_columns,
                self.target_columns,
                self.active_scaling,
                self.schema_by_path,
            )
            if self.config.data_access == "full_in_mem"
            else None
        )
        if self.config.data_access == "full_in_mem" and not stream_tensor_cache:
            raise ValueError("data_access='full_in_mem' could not load any active protocol streams")
        if stream_tensor_cache is not None and not stream_tensor_cache.tensors:
            raise ValueError("data_access='full_in_mem' could not load any active protocol streams")
        self.stream_tensor_cache = stream_tensor_cache
        self._epoch_idx = 0
        if not self.input_columns:
            raise ValueError("input_columns must not be empty")
        if not self.target_columns:
            raise ValueError("target_columns must not be empty")

    def set_epoch(self, epoch_idx: int) -> None:
        if epoch_idx < 0:
            raise ValueError(f"epoch_idx must be >= 0, got {epoch_idx}")
        self._epoch_idx = int(epoch_idx)

    def steps_per_epoch(self, epoch_idx: int = 0) -> int:
        return planning.count_batch_plans(
            self.index,
            self.active_protocol,
            self.config,
            epoch_idx=epoch_idx,
            stream_plans=self.stream_plans,
        )

    def __iter__(self) -> Iterator[Batch]:
        epoch_idx = self._epoch_idx
        self._epoch_idx += 1
        yield from _iter_batches(
            self.store,
            self.index,
            self.input_columns,
            self.target_columns,
            self.active_scaling,
            self.config,
            self.active_protocol,
            self.stream_plans,
            self.schema_by_path,
            self.stream_tensor_cache,
            epoch_idx,
            get_worker_info(),
        )


class CudaBatchPrefetchIterator:
    def __init__(
        self,
        base_iter: Iterator[Batch],
        *,
        device: torch.device,
        non_blocking: bool,
    ) -> None:
        self._base_iter = base_iter
        self._device = device
        self._non_blocking = non_blocking
        self._stream = torch.cuda.Stream(device=self._device)
        self._next_batch: Batch | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._base_iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = batch.to(self._device, non_blocking=self._non_blocking)

    def __iter__(self) -> CudaBatchPrefetchIterator:
        return self

    def __next__(self) -> Batch:
        if self._next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(device=self._device).wait_stream(self._stream)
        batch = self._next_batch
        self._preload()
        return batch


class DevicePrefetchDataLoader:
    def __init__(
        self,
        base_loader: DataLoader[Batch],
        *,
        device: torch.device,
        non_blocking: bool,
    ) -> None:
        self._base_loader = base_loader
        self._device = device
        self._non_blocking = non_blocking

    def __iter__(self) -> CudaBatchPrefetchIterator:
        return CudaBatchPrefetchIterator(
            cast("Iterator[Batch]", iter(self._base_loader)),
            device=self._device,
            non_blocking=self._non_blocking,
        )

    @property
    def dataset(self) -> object:
        return self._base_loader.dataset


def create_dataloader(
    store: DatasetStoreReader,
    manifest_paths: ManifestPaths,
    input_columns: tuple[str | MappingSpec, ...],
    target_columns: tuple[str | MappingSpec, ...],
    protocols: tuple[object, ...] | None = None,
    protocol_mode: ProtocolMode = "strict",
    active_protocol: DatasetProtocolId | object | None = None,
    validation: ValidationConfig | None = None,
    scaling: tuple[ScalingRule, ...] = (),
    config: LoaderConfig | None = None,
) -> DataLoader[Batch] | DevicePrefetchDataLoader:
    resolved_config = LoaderConfig() if config is None else config
    if active_protocol is not None:
        resolved_config = replace(
            resolved_config,
            active_protocol=coerce_protocol(active_protocol),
        )
    index = build_index(
        store,
        manifest_paths,
        protocols=protocols,
        protocol_mode=protocol_mode,
        validation=validation,
    )
    dataset = MlDataIterable(
        store=store,
        index=index,
        input_columns=input_columns,
        target_columns=target_columns,
        scaling=scaling,
        config=resolved_config,
        active_protocol=resolved_config.active_protocol,
    )
    device = torch.device(resolved_config.device)
    use_cuda_prefetch = resolved_config.prefetch_to_device and device.type == "cuda"
    if resolved_config.prefetch_to_device and device.type != "cuda":
        raise ValueError("prefetch_to_device=True requires a CUDA device")
    if use_cuda_prefetch and not torch.cuda.is_available():
        raise ValueError("CUDA prefetch requested but CUDA is not available")
    loader = _torch_dataloader(
        dataset,
        resolved_config,
        use_cuda_prefetch=use_cuda_prefetch,
    )
    if use_cuda_prefetch:
        return DevicePrefetchDataLoader(
            loader,
            device=device,
            non_blocking=resolved_config.non_blocking,
        )
    return loader


def create_index(
    store: DatasetStoreReader,
    manifest_paths: ManifestPaths,
    protocols: tuple[object, ...] | None = None,
    protocol_mode: ProtocolMode = "strict",
    validation: ValidationConfig | None = None,
) -> MlDatasetIndex:
    return build_index(
        store,
        manifest_paths,
        protocols=protocols,
        protocol_mode=protocol_mode,
        validation=validation,
    )


def _torch_dataloader(
    dataset: MlDataIterable,
    config: LoaderConfig,
    *,
    use_cuda_prefetch: bool = False,
) -> DataLoader[Batch]:
    pin_memory = config.pin_memory or use_cuda_prefetch
    if config.num_workers <= 0:
        return DataLoader[Batch](
            dataset,
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
    if config.multiprocessing_context is None:
        return DataLoader[Batch](
            dataset,
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
        )
    return DataLoader[Batch](
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        multiprocessing_context=config.multiprocessing_context,
    )


def _iter_batches(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    scaling: tuple[ScalingRule, ...],
    config: LoaderConfig,
    active_protocol: DatasetProtocolId,
    stream_plans: tuple[StreamPlan, ...] | None,
    schema_by_path: dict[str, set[str]] | None,
    stream_tensor_cache: StreamTensorCache | None,
    epoch_idx: int,
    worker_info: WorkerInfo | None,
) -> Iterator[Batch]:
    plans = planning.iter_batch_plans(
        index,
        active_protocol,
        config,
        epoch_idx=epoch_idx,
        stream_plans=stream_plans,
    )
    plans = _iter_distributed_plans(plans, _distributed_info(), config)
    plans = _iter_worker_plans(plans, worker_info)
    for batch_idx, plan in enumerate(plans):
        yield materialization.materialize_batch_plan(
            store,
            plan,
            input_columns,
            target_columns,
            scaling,
            config,
            batch_idx,
            schema_by_path=schema_by_path,
            stream_tensor_cache=stream_tensor_cache,
        )


def _log_data_access_plan(
    stream_plans: tuple[StreamPlan, ...],
    config: LoaderConfig,
    active_protocol: DatasetProtocolId,
    source_columns: tuple[str, ...],
) -> None:
    window = config.window_for(active_protocol)
    total_rows = sum(stream.row_count for stream in stream_plans)
    if config.data_access == "full_in_mem":
        payload_gb = total_rows * len(source_columns) * 4 / 1e9
        logger.info(
            "ML loader data_access=full_in_mem split=%s protocol=%s streams=%d "
            "rows=%d columns=%d dtype=float32 estimated_tensor_payload_gb=%.2f. "
            "Full selected split/protocol/columns will be cached in CPU RAM.",
            config.split,
            active_protocol,
            len(stream_plans),
            total_rows,
            len(source_columns),
            payload_gb,
        )
        return

    batch_payload_gb = window.batch_size * window.seq_len * len(source_columns) * 4 / 1e9
    logger.info(
        "ML loader data_access=windowed split=%s protocol=%s streams=%d columns=%d "
        "batch_size=%d seq_len=%d dtype=float32 estimated_batch_tensor_payload_gb=%.4f. "
        "Reads only needed windows from parquet; actual IO depends on row-group size.",
        config.split,
        active_protocol,
        len(stream_plans),
        len(source_columns),
        window.batch_size,
        window.seq_len,
        batch_payload_gb,
    )


def _distributed_plans(
    plans: tuple[BatchPlan, ...],
    distributed: DistributedInfo,
    config: LoaderConfig,
) -> tuple[BatchPlan, ...]:
    if distributed.world_size <= 1:
        return plans
    usable_count = len(plans)
    if config.drop_incomplete_distributed:
        usable_count = (usable_count // distributed.world_size) * distributed.world_size
    return tuple(
        plan
        for idx, plan in enumerate(plans[:usable_count])
        if (idx % distributed.world_size) == distributed.rank
    )


def _iter_distributed_plans(
    plans: Iterator[BatchPlan],
    distributed: DistributedInfo,
    config: LoaderConfig,
) -> Iterator[BatchPlan]:
    if distributed.world_size <= 1:
        yield from plans
        return
    if config.drop_incomplete_distributed:
        # Preserve old drop-incomplete semantics without requiring all callers to
        # materialize plans. This branch is only used for distributed training.
        yield from _distributed_plans(tuple(plans), distributed, config)
        return
    for idx, plan in enumerate(plans):
        if (idx % distributed.world_size) == distributed.rank:
            yield plan


def _iter_worker_plans(
    plans: Iterator[BatchPlan], worker_info: WorkerInfo | None
) -> Iterator[BatchPlan]:
    if worker_info is None:
        yield from plans
        return
    for idx, plan in enumerate(plans):
        if (idx % worker_info.num_workers) == worker_info.id:
            yield plan


def _distributed_info() -> DistributedInfo:
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return DistributedInfo()
    try:
        if not distributed.is_available() or not distributed.is_initialized():
            return DistributedInfo()
        return DistributedInfo(
            rank=int(distributed.get_rank()), world_size=int(distributed.get_world_size())
        )
    except RuntimeError:
        return DistributedInfo()


def dataloader_for_split(
    loader: DataLoader[Batch] | DevicePrefetchDataLoader,
    split: str,
) -> DataLoader[Batch] | DevicePrefetchDataLoader:
    dataset = loader.dataset
    if not isinstance(dataset, MlDataIterable):
        raise TypeError("dataloader_for_split expects a loader created by create_dataloader")
    config = replace(dataset.config, split=split)
    split_loader = _torch_dataloader(
        MlDataIterable(
            dataset.store,
            dataset.full_index,
            dataset.input_columns,
            dataset.target_columns,
            dataset.scaling,
            config,
            dataset.active_protocol,
        ),
        config,
        use_cuda_prefetch=config.prefetch_to_device and torch.device(config.device).type == "cuda",
    )
    device = torch.device(config.device)
    if config.prefetch_to_device and device.type == "cuda":
        return DevicePrefetchDataLoader(
            split_loader,
            device=device,
            non_blocking=config.non_blocking,
        )
    return split_loader
