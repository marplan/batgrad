from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

import polars as pl
import pyarrow.parquet as pq

from batgrad.contracts.mapping import BaseColumns
from batgrad.contracts.paths import dataset_id_from_manifest_path
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN, ML_INDEX_ROW_ID_COLUMN
from batgrad.ml.data.config import ValidationConfig, coerce_protocol, column_name

if TYPE_CHECKING:
    from batgrad.storage.store import DatasetStoreReader

GIT_COMMIT_PREFIX_LEN = 7
CANONICAL_MANIFEST_PARTS = 4
MIN_CANONICAL_SEGMENT_PARTS = 4
logger = getLogger(__name__)
type ManifestPaths = dict[str, str]
type ProtocolMode = Literal["strict", "available"]


@dataclass(frozen=True)
class MlDatasetIndex:
    frame: pl.DataFrame

    def validate(self) -> None:
        required = {
            BaseColumns.set_id,
            BaseColumns.proto,
            BaseColumns.row_n,
            BaseColumns.norm_segs,
            BaseColumns.split,
            BaseColumns.manifest,
            MANIFEST_ROW_ID_COLUMN,
        }
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"ML dataset index is missing required columns: {missing}")

    def filter_split(self, split: str) -> MlDatasetIndex:
        return MlDatasetIndex(self.frame.filter(pl.col(BaseColumns.split) == split))


def build_index(
    store: DatasetStoreReader,
    manifest_paths: ManifestPaths,
    protocols: tuple[object, ...] | None = None,
    protocol_mode: ProtocolMode = "strict",
    validation: ValidationConfig | None = None,
) -> MlDatasetIndex:
    if not manifest_paths:
        raise ValueError("manifest_paths must not be empty")
    for manifest_path in manifest_paths:
        _validate_canonical_normalized_manifest_path(manifest_path)
    frames = tuple(
        _load_manifest(store, manifest_path, git_commit)
        for manifest_path, git_commit in manifest_paths.items()
    )
    frame = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
    _validate_required_manifest_columns(frame)
    frame = _filter_protocols(frame, protocols, protocol_mode)
    frame = _assign_splits(
        frame,
        validation or ValidationConfig.sample(fraction=0.0),
    )
    index = MlDatasetIndex(_with_ml_index_row_id(sort_index_frame(frame)))
    index.validate()
    return index


def available_manifest_paths(store: DatasetStoreReader) -> tuple[str, ...]:
    return tuple(
        path
        for path in store.list_files(pattern="type=*/dataset=*/source=normalized/manifest.parquet")
        if _is_canonical_normalized_manifest_path(path)
    )


def sort_index_frame(frame: pl.DataFrame) -> pl.DataFrame:
    preferred = (
        BaseColumns.set_id,
        BaseColumns.cell_id,
        BaseColumns.cidx,
        BaseColumns.proto,
        BaseColumns.soc_pct,
        BaseColumns.manifest,
        MANIFEST_ROW_ID_COLUMN,
    )
    sort_columns = [column for column in preferred if column in frame.columns]
    return frame.sort(*sort_columns) if sort_columns else frame


def _with_ml_index_row_id(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop(ML_INDEX_ROW_ID_COLUMN, strict=False).with_row_index(ML_INDEX_ROW_ID_COLUMN)


def _load_manifest(store: DatasetStoreReader, manifest_path: str, git_commit: str) -> pl.DataFrame:
    _validate_manifest_git(store, manifest_path, git_commit)
    try:
        manifest = store.scan_table(manifest_path).collect()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}") from exc
    if manifest.height == 0:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    _validate_normalized_segment_paths(manifest, manifest_path)
    dataset_id = _dataset_id_for_manifest(manifest, manifest_path)
    if BaseColumns.set_id not in manifest.columns:
        manifest = manifest.with_columns(pl.lit(dataset_id).alias(BaseColumns.set_id))
    return manifest.with_row_index(MANIFEST_ROW_ID_COLUMN).with_columns(
        pl.lit(manifest_path).alias(BaseColumns.manifest)
    )


def _validate_manifest_git(
    store: DatasetStoreReader,
    manifest_path: str,
    expected_commit: str,
) -> None:
    expected = expected_commit.strip()
    if len(expected) < GIT_COMMIT_PREFIX_LEN:
        raise ValueError(
            f"Manifest git commit for {manifest_path!r} must contain at least 7 characters"
        )
    with store.local_file(manifest_path) as path:
        metadata = pq.ParquetFile(path).metadata.metadata
    if metadata is None:
        raise ValueError(f"Manifest has no parquet footer metadata: {manifest_path}")

    commit = _footer_value(metadata, BaseColumns.git_commit)
    status = _footer_value(metadata, BaseColumns.git_status)
    if not commit or commit == "na":
        raise ValueError(
            f"Manifest {manifest_path!r} has no git commit metadata. "
            "Regenerate the normalized data with git available."
        )
    if not status or status == "na":
        raise ValueError(
            f"Manifest {manifest_path!r} has no git status metadata. "
            "Regenerate the normalized data with git available."
        )
    if not commit.startswith(expected):
        raise ValueError(
            f"Manifest git commit mismatch for {manifest_path!r}: "
            f"expected prefix {expected!r}, found {commit!r}"
        )
    if status != BaseColumns.git_status.values.clean:
        logger.warning(
            "Manifest %s was generated from a dirty git worktree: git_status=%s git_commit=%s",
            manifest_path,
            status,
            commit,
        )


def _footer_value(metadata: dict[bytes, bytes], key: str) -> str | None:
    value = metadata.get(key.encode())
    return None if value is None else value.decode()


def _dataset_id_for_manifest(manifest: pl.DataFrame, manifest_path: str) -> str:
    if BaseColumns.set_id in manifest.columns:
        values = manifest[BaseColumns.set_id].drop_nulls().unique().to_list()
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            raise ValueError(
                f"Manifest has multiple dataset ids: path={manifest_path!r} values={values}"
            )
    if value := dataset_id_from_manifest_path(manifest_path):
        return value
    raise ValueError(
        "Could not infer dataset id. Add a 'dataset id' manifest column or use a "
        f"path containing 'dataset=...': {manifest_path}"
    )


def _validate_required_manifest_columns(frame: pl.DataFrame) -> None:
    required = {BaseColumns.proto, BaseColumns.row_n, BaseColumns.norm_segs}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing normalized ML columns: {missing}")


def _is_canonical_normalized_manifest_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) == CANONICAL_MANIFEST_PARTS
        and parts[0].startswith("type=")
        and parts[1].startswith("dataset=")
        and parts[2] == "source=normalized"
        and parts[3] == "manifest.parquet"
    )


def _validate_canonical_normalized_manifest_path(path: str) -> None:
    if _is_canonical_normalized_manifest_path(path):
        return
    raise ValueError(
        "Normalized manifest paths must be store-relative and canonical: "
        "type=*/dataset=*/source=normalized/manifest.parquet. "
        f"Got {path!r}. Set the store root to the directory containing type=published/ "
        "or type=synthetic/, not its parent."
    )


def _validate_normalized_segment_paths(manifest: pl.DataFrame, manifest_path: str) -> None:
    if BaseColumns.norm_segs not in manifest.columns:
        return
    for row_idx, row_segments in enumerate(manifest[BaseColumns.norm_segs].to_list()):
        if not isinstance(row_segments, list | tuple):
            continue
        for segment in row_segments:
            if not isinstance(segment, dict):
                continue
            segment_path = str(
                segment.get(BaseColumns.path) or segment.get(str(BaseColumns.path)) or ""
            )
            parts = PurePosixPath(segment_path).parts
            if len(parts) < MIN_CANONICAL_SEGMENT_PARTS or not parts[0].startswith("type="):
                raise ValueError(
                    "Normalized segment paths must be store-relative and start with type=*. "
                    f"manifest={manifest_path!r} row={row_idx} segment_path={segment_path!r}. "
                    "Regenerate normalized data with the store root set to the directory "
                    "containing type=published/ or type=synthetic/."
                )
            if parts[1].startswith("dataset=") and parts[2] == "source=normalized":
                continue
            raise ValueError(
                "Normalized segment paths must use canonical layout "
                "type=*/dataset=*/source=normalized/.... "
                f"manifest={manifest_path!r} row={row_idx} segment_path={segment_path!r}"
            )


def _filter_protocols(
    frame: pl.DataFrame,
    protocols: tuple[object, ...] | None,
    protocol_mode: ProtocolMode,
) -> pl.DataFrame:
    if protocols is None:
        return frame
    if protocol_mode not in {"strict", "available"}:
        raise ValueError(f"protocol_mode must be 'strict' or 'available', got {protocol_mode!r}")
    requested = tuple(str(coerce_protocol(protocol)) for protocol in protocols)
    if not requested:
        raise ValueError("protocols must not be empty when provided")
    available_by_dataset = frame.group_by(BaseColumns.set_id).agg(
        pl.col(BaseColumns.proto).unique().alias("__protocols")
    )
    _validate_protocol_availability(available_by_dataset, requested, protocol_mode)
    filtered = frame.filter(pl.col(BaseColumns.proto).cast(pl.String).is_in(list(requested)))
    if filtered.height == 0:
        raise ValueError(f"No rows remain after protocol selection: requested={list(requested)}")
    return filtered


def _validate_protocol_availability(
    available_by_dataset: pl.DataFrame,
    requested: tuple[str, ...],
    protocol_mode: ProtocolMode,
) -> None:
    missing_messages = []
    empty_messages = []
    for row in available_by_dataset.iter_rows(named=True):
        available = {str(value) for value in row["__protocols"]}
        missing = [protocol for protocol in requested if protocol not in available]
        present = [protocol for protocol in requested if protocol in available]
        dataset_id = row[BaseColumns.set_id]
        if not present:
            empty_messages.append(
                f"dataset={dataset_id!r} requested={list(requested)} available={sorted(available)}"
            )
            continue
        if missing:
            message = f"dataset={dataset_id!r} missing={missing} available={sorted(available)}"
            if protocol_mode == "strict":
                missing_messages.append(message)
            else:
                logger.warning(
                    "Dataset %r is missing requested protocols %s; using available requested "
                    "protocols %s.",
                    dataset_id,
                    missing,
                    present,
                )
    if empty_messages:
        raise ValueError(
            "Requested protocols are absent for one or more datasets: " + "; ".join(empty_messages)
        )
    if protocol_mode == "strict" and missing_messages:
        raise ValueError("Requested protocols not found: " + "; ".join(missing_messages))


def _assign_splits(
    frame: pl.DataFrame,
    validation: ValidationConfig,
) -> pl.DataFrame:
    group_cols = tuple(column_name(column) for column in validation.group_by)
    missing = sorted(set(group_cols) - set(frame.columns))
    if missing:
        raise ValueError(f"Validation group_by columns missing from manifest index: {missing}")

    groups = frame.select(group_cols).unique(maintain_order=True)
    val_groups = _validation_groups(groups, validation, group_cols)
    if val_groups.height == 0:
        return frame.with_columns(pl.lit(BaseColumns.split.values.train).alias(BaseColumns.split))
    val_groups = val_groups.with_columns(pl.lit(value=True).alias("__is_val_group"))
    return (
        frame.join(val_groups, on=list(group_cols), how="left")
        .with_columns(
            pl.when(pl.col("__is_val_group").fill_null(value=False))
            .then(pl.lit(BaseColumns.split.values.val))
            .otherwise(pl.lit(BaseColumns.split.values.train))
            .alias(BaseColumns.split)
        )
        .drop("__is_val_group")
    )


def _validation_groups(
    groups: pl.DataFrame,
    validation: ValidationConfig,
    group_cols: tuple[str, ...],
) -> pl.DataFrame:
    provided = _provided_groups(groups, validation, group_cols)
    if validation.strategy == "provide":
        return provided
    count = groups.height
    sample_count = int(count * validation.fraction)
    if validation.strategy == "merge":
        sample_count = max(0, sample_count - provided.height)
        candidates = (
            groups.join(provided, on=list(group_cols), how="anti") if provided.height else groups
        )
        sampled = _sample_groups(candidates, sample_count, validation.seed, group_cols)
        return pl.concat((provided, sampled), how="vertical") if provided.height else sampled
    return _sample_groups(groups, sample_count, validation.seed, group_cols)


def _sample_groups(
    groups: pl.DataFrame,
    count: int,
    seed: int,
    group_cols: tuple[str, ...],
) -> pl.DataFrame:
    if count <= 0 or groups.height == 0:
        return groups.limit(0)
    return (
        groups.with_columns(pl.struct(list(group_cols)).hash(seed=seed).alias("__hash"))
        .sort("__hash", *group_cols)
        .head(count)
        .drop("__hash")
    )


def _provided_groups(
    groups: pl.DataFrame,
    validation: ValidationConfig,
    group_cols: tuple[str, ...],
) -> pl.DataFrame:
    if not validation.provided:
        return groups.limit(0)
    selected = groups.limit(0)
    group_col_set = set(group_cols)
    for selector in validation.provided:
        normalized = {column_name(key): value for key, value in selector.items()}
        unknown = sorted(set(normalized) - group_col_set)
        if unknown:
            raise ValueError(f"Validation selector contains unknown group columns: {unknown}")
        if not normalized:
            raise ValueError("Validation selector must not be empty")
        expr = None
        for column, value in normalized.items():
            values = value if isinstance(value, tuple | list | set | frozenset) else (value,)
            next_expr = pl.col(column).is_in(list(values))
            expr = next_expr if expr is None else expr & next_expr
        matches = groups.filter(expr) if expr is not None else groups.limit(0)
        if matches.height == 0:
            raise ValueError(f"Validation selector did not match any group: {normalized}")
        selected = pl.concat((selected, matches), how="vertical").unique(maintain_order=True)
    return selected.select(group_cols)
