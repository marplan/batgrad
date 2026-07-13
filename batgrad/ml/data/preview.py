from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.paths import dataset_id_from_manifest_path
from batgrad.ml.data.config import LoaderConfig, ScalingRule, WindowConfig
from batgrad.ml.data.index import MlDatasetIndex, sort_index_frame
from batgrad.ml.data.materialization import materialize_batch_plan
from batgrad.ml.data.planning import iter_batch_plans

if TYPE_CHECKING:
    from batgrad.ml.data.batch import Batch
    from batgrad.ml.data.planning import WindowRef
    from batgrad.storage.store import DatasetStoreReader


@dataclass(frozen=True, slots=True)
class MlBatchPreviewSpec:
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    batch_size: int
    seq_len: int
    batch_group_index: int
    sample_index: int = 0
    consecutive_step: int = 0
    strategy: str = "shuffled_protocol_groups"
    stateful_n_windows: int = 1
    active_protocol: str = str(DatasetProtocolId.cycling)
    scaling: tuple[ScalingRule, ...] = ()

    @property
    def preview_rows(self) -> int:
        if self.strategy != "sequential":
            return self.seq_len
        return self.batch_size * self.seq_len

    @property
    def raw_plan_index(self) -> int:
        consecutive_step = min(
            max(0, int(self.consecutive_step)),
            max(0, int(self.stateful_n_windows) - 1),
        )
        return self.batch_group_index * self.stateful_n_windows + consecutive_step


@dataclass(frozen=True, slots=True)
class MlBatchPreviewData:
    store: DatasetStoreReader
    index: MlDatasetIndex
    spec: MlBatchPreviewSpec
    ref: WindowRef
    batch: Batch
    sample_index: int
    batch_index: int
    total_batches: int


def prepare_ml_batch_preview(
    store: DatasetStoreReader,
    index: MlDatasetIndex,
    spec: MlBatchPreviewSpec,
) -> MlBatchPreviewData:
    config = _preview_loader_config(spec)
    sorted_index = type(index)(sort_index_frame(index.frame))
    selected_plan = None
    selected_index = 0
    total_batches = 0
    for plan_idx, plan in enumerate(iter_batch_plans(sorted_index, config)):
        total_batches = plan_idx + 1
        if plan_idx <= spec.raw_plan_index:
            selected_plan = plan
            selected_index = plan_idx
    if selected_plan is None:
        raise ValueError("No batch windows are available for this preview selection")

    sample_index = min(max(0, int(spec.sample_index)), len(selected_plan.refs) - 1)
    return MlBatchPreviewData(
        store=store,
        index=sorted_index,
        spec=spec,
        ref=selected_plan.refs[sample_index],
        batch=materialize_batch_plan(
            store,
            selected_plan,
            spec.input_columns,
            spec.target_columns,
            spec.scaling,
            config,
            selected_index,
        ),
        sample_index=sample_index,
        batch_index=selected_index,
        total_batches=total_batches,
    )


def count_ml_batch_preview_groups(
    index: MlDatasetIndex,
    *,
    strategy: str,
    active_protocol: str,
    batch_size: int,
    seq_len: int,
    stateful_n_windows: int,
) -> int:
    if index.frame.is_empty() or BaseColumns.proto not in index.frame.columns:
        return 0
    if ml_batch_preview_unavailable_message(
        strategy=strategy,
        active_protocol=active_protocol,
    ):
        return 0
    spec = MlBatchPreviewSpec(
        input_columns=("__unused__",),
        target_columns=("__unused__",),
        batch_size=batch_size,
        seq_len=seq_len,
        batch_group_index=0,
        strategy=strategy,
        stateful_n_windows=stateful_n_windows,
        active_protocol=active_protocol,
    )
    config = _preview_loader_config(spec)
    sorted_index = type(index)(sort_index_frame(index.frame))
    batch_count = sum(1 for _plan in iter_batch_plans(sorted_index, config))
    if batch_count == 0:
        return 0
    return (batch_count + stateful_n_windows - 1) // stateful_n_windows


def ml_batch_preview_unavailable_message(*, strategy: str, active_protocol: str) -> str | None:
    try:
        protocol = DatasetProtocolId(active_protocol)
    except ValueError:
        return None
    if protocol == DatasetProtocolId.eis and strategy != "sequential":
        return (
            "EIS batch preview is not supported with shuffled protocol groups yet. "
            "Use Sequential debug or select cycling/HPPC/RPT."
        )
    return None


def _preview_loader_config(spec: MlBatchPreviewSpec) -> LoaderConfig:
    return LoaderConfig(
        strategy="sequential" if spec.strategy == "sequential" else "shuffled_protocol_groups",
        protocol_order=(DatasetProtocolId(spec.active_protocol),),
        stateful_n_windows=int(spec.stateful_n_windows),
        default_window=WindowConfig(
            batch_size=spec.batch_size,
            seq_len=spec.seq_len,
            drop_incomplete=False,
        ),
    )


def load_manifest_preview(
    store: DatasetStoreReader | None,
    manifest_commits: dict[str, str],
) -> pl.DataFrame:
    if store is None or not manifest_commits:
        return pl.DataFrame()
    frames = []
    for manifest_path in manifest_commits:
        try:
            frame = (
                store.scan_table(manifest_path)
                .collect()
                .with_columns(pl.lit(manifest_path).alias(BaseColumns.manifest))
            )
        except FileNotFoundError:
            continue
        if BaseColumns.set_id not in frame.columns:
            frame = frame.with_columns(
                pl.lit(dataset_id_from_manifest_path(manifest_path)).alias(BaseColumns.set_id)
            )
        frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def available_protocols(raw_manifest: pl.DataFrame) -> tuple[str, ...]:
    if raw_manifest.height and BaseColumns.proto in raw_manifest.columns:
        return tuple(sorted(str(value) for value in raw_manifest[BaseColumns.proto].unique()))
    return ()


def validation_group_options(raw_manifest: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in raw_manifest.columns
        if column not in {BaseColumns.norm_segs, BaseColumns.raw_paths}
    )


def default_validation_group_by(group_options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in (BaseColumns.set_id, BaseColumns.cell_id, BaseColumns.cidx)
        if column in group_options
    )


def shard_columns_for_protocols(
    schema_by_protocol: dict[object, tuple[str, ...]],
) -> tuple[str, ...]:
    if not schema_by_protocol:
        return ()
    column_sets = [set(columns) for columns in schema_by_protocol.values()]
    return tuple(sorted(set.intersection(*column_sets))) if column_sets else ()


def default_input_columns(shard_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in (BaseColumns.time, BaseColumns.curr, BaseColumns.volt)
        if column in shard_columns
    )


def default_target_columns(shard_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str(column) for column in (BaseColumns.curr, BaseColumns.volt) if column in shard_columns
    )


def active_protocol_options(schema_by_protocol: dict[object, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(str(protocol) for protocol in schema_by_protocol) or ("cycling",)
