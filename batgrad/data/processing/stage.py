from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol

from batgrad.data.processing.io import coalesce_frames
from batgrad.data.processing.runtime import (
    ProcessTaskSpec,
    iter_process_task_results,
    log_task_progress_if_due,
)
from batgrad.data.processing.sharding import ShardWriter

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Iterator, Sequence

    import polars as pl

    from batgrad.contracts.mapping import DatasetStageId, MappingSpec
    from batgrad.contracts.metadata import MetadataLayout
    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.data.processing.runtime import ProcessTaskResult
    from batgrad.data.processing.sharding import ShardWriteConfig
    from batgrad.storage.store import DataProcessingStore


class StageRunConfig(Protocol):
    n_jobs: int
    worker_polars_max_threads: int | None
    chunk_rows: int
    compression: str
    use_content_defined_chunking: bool
    row_group_size: int
    max_shard_size_bytes: int


@dataclass(frozen=True)
class StageOutputSpec:
    """Resolved output-writing configuration for one processing stage."""

    dataset_spec: DatasetSpec
    stage_id: DatasetStageId
    output_root: str
    manifest_path: str
    manifest_metadata: MetadataLayout
    footer_metadata: MetadataLayout
    shard_key_col: MappingSpec
    segment_col: MappingSpec
    source_paths_col: MappingSpec


@dataclass(frozen=True)
class PreparedTable:
    """Temporary parquet table produced by one stage task."""

    temp_path: str
    metadata: dict[MappingSpec, object]
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class TaskOutput[TaskT]:
    """Result payload returned by one stage task before final sharding."""

    task: TaskT
    tables: tuple[PreparedTable, ...]
    warnings: tuple[str, ...] = ()
    chunks: int = 0
    rows: int = 0


@dataclass(frozen=True)
class ConsumeStats:
    """Rows and chunks consumed from prepared task tables."""

    chunks: int = 0
    rows: int = 0


@dataclass(frozen=True)
class StageRunStats:
    """Aggregate task and row counts for a stage run."""

    succeeded: int = 0
    failed: int = 0
    warnings: int = 0
    chunks: int = 0
    rows: int = 0


def stage_scratch_root(output_spec: StageOutputSpec) -> str:
    """Return a unique scratch root under a stage output root."""
    value = f"{time.time_ns()}-{os.getpid()}-{output_spec.output_root}".encode()
    digest = hashlib.blake2s(value, digest_size=5).hexdigest()
    return f"{output_spec.output_root}/_tmp_{output_spec.stage_id}_{os.getpid():x}-{digest}"


def stage_temp_path(
    scratch_root: str,
    task_index: int,
    batch_index: int | None = None,
) -> str:
    """Return the temporary parquet path for one task or task batch."""
    suffix = "" if batch_index is None else f"_batch-{batch_index:04d}"
    return f"{scratch_root}/task-{task_index:06d}{suffix}.parquet"


def interactive_run_root(dataset_spec: DatasetSpec, stage_id: DatasetStageId) -> str:
    """Return a unique scratch root for an interactive stage run."""
    value = f"{time.time_ns()}-{os.getpid()}".encode()
    digest = hashlib.blake2s(value, digest_size=5).hexdigest()
    run_id = f"{os.getpid():x}-{digest}"
    return f"{dataset_spec.source_root(stage_id)}/scratch/{run_id}"


def validate_scratch_run_root(run_root: str) -> None:
    """Validate that a path looks like a single scratch run before deletion."""
    parts = PurePosixPath(run_root).parts
    if "scratch" not in parts:
        raise ValueError(f"Refusing to clean non-scratch run: {run_root}")
    scratch_index = parts.index("scratch")
    if len(parts) != scratch_index + 2:
        raise ValueError(f"Refusing to clean non-run scratch path: {run_root}")


def create_stage_writer(
    output_store: DataProcessingStore,
    output_spec: StageOutputSpec,
    config: ShardWriteConfig,
) -> ShardWriter:
    """Create the protocol-sharding writer used by ingest and normalize."""
    return ShardWriter(
        output_store=output_store,
        dataset_spec=output_spec.dataset_spec,
        stage_id=output_spec.stage_id,
        output_root=output_spec.output_root,
        manifest_path=output_spec.manifest_path,
        manifest_metadata=output_spec.manifest_metadata,
        footer_metadata=output_spec.footer_metadata,
        shard_key_col=output_spec.shard_key_col,
        segment_col=output_spec.segment_col,
        source_paths_col=output_spec.source_paths_col,
        config=config,
    )


def protocol_spec_by_id[SpecT](
    protocol_specs: Sequence[SpecT],
    protocol: object,
) -> SpecT:
    """Return a protocol spec whose `protocol_id` matches `protocol` as a string."""
    for spec in protocol_specs:
        protocol_id = getattr(spec, "protocol_id", None)
        if str(protocol_id) == str(protocol):
            return spec
    raise ValueError(f"Protocol {protocol!r} is not declared in stage protocol specs")


def stage_output_spec(
    *,
    dataset_spec: DatasetSpec,
    stage_id: DatasetStageId,
    output_root: str,
    manifest_path: str,
    manifest_metadata: MetadataLayout,
    footer_metadata: MetadataLayout,
    shard_key_col: MappingSpec,
    segment_col: MappingSpec,
    source_paths_col: MappingSpec,
) -> StageOutputSpec:
    """Construct a stage output spec from resolved dataset and metadata paths."""
    return StageOutputSpec(
        dataset_spec=dataset_spec,
        stage_id=stage_id,
        output_root=output_root,
        manifest_path=manifest_path,
        manifest_metadata=manifest_metadata,
        footer_metadata=footer_metadata,
        shard_key_col=shard_key_col,
        segment_col=segment_col,
        source_paths_col=source_paths_col,
    )


def close_stage_writer(
    writer: ShardWriter,
    *,
    succeeded: int,
    failed: int = 0,
    aborted: bool,
) -> None:
    """Close the writer and choose whether to write, skip, or error the manifest."""
    manifest: Literal["write", "error", "skip"] = "error" if aborted or failed else "write"
    if succeeded == 0 and failed == 0:
        manifest = "skip"
    writer.close(manifest=manifest)


def run_stage_task_outputs[TaskT](
    *,
    output_store: DataProcessingStore,
    scratch_store: DataProcessingStore,
    output_spec: StageOutputSpec,
    config: StageRunConfig,
    tasks: Sequence[TaskT],
    iter_results: Callable[[str], Iterator[ProcessTaskResult[TaskOutput[TaskT]]]],
    logger: logging.Logger,
    delete_output_root: bool = True,
    dry_run: bool = False,
) -> StageRunStats:
    """Run stage tasks and consume their temporary tables into final shards.

    This shared runner creates a scratch root, starts the shard writer, iterates
    task results, appends successful task tables to final output, writes an error
    manifest when tasks fail, and always cleans scratch files.
    """
    scratch_root = stage_scratch_root(output_spec)
    if delete_output_root and not dry_run:
        output_store.delete_dir(output_spec.output_root, missing_ok=True)
    prepare_stage_scratch(scratch_store, scratch_root, dry_run=dry_run)

    writer = None if dry_run else create_stage_writer(output_store, output_spec, config)
    succeeded = failed = warnings = chunks = rows = 0
    aborted = False
    started_at = time.perf_counter()
    last_progress_at = started_at
    task_count = len(tasks)
    stage_name = str(output_spec.stage_id)
    if tasks:
        last_progress_at = log_task_progress_if_due(
            stage_name,
            output_spec.dataset_spec.dataset_id,
            1,
            task_count,
            succeeded,
            failed,
            warnings,
            last_progress_at,
            force=True,
        )
    try:
        for result in iter_results(scratch_root):
            if result.value is None:
                failed += 1
                logger.error(
                    "%s task failed task=%d/%d task_id=%s error=%s",
                    stage_name,
                    result.task_index,
                    task_count,
                    result.task_id,
                    result.error,
                )
                last_progress_at = log_task_progress_if_due(
                    stage_name,
                    output_spec.dataset_spec.dataset_id,
                    result.task_index,
                    task_count,
                    succeeded,
                    failed,
                    warnings,
                    last_progress_at,
                )
                continue

            warnings += len(result.value.warnings)
            for warning in result.value.warnings:
                logger.warning(
                    "%s warning task_id=%s %s",
                    stage_name,
                    result.task_id,
                    warning,
                )
            if writer is None:
                stats = ConsumeStats(chunks=result.value.chunks, rows=result.value.rows)
            else:
                table_stats = consume_prepared_tables(
                    scratch_store,
                    writer,
                    result.value.tables,
                    config.chunk_rows,
                    getattr(config, "max_batch_rows", None) or config.chunk_rows,
                )
                stats = ConsumeStats(
                    chunks=table_stats.chunks + result.value.chunks,
                    rows=table_stats.rows + result.value.rows,
                )
            succeeded += 1
            chunks += stats.chunks
            rows += stats.rows
            if stats.chunks == 0 and not dry_run:
                logger.warning(
                    "%s task produced no rows task_id=%s",
                    stage_name,
                    result.task_id,
                )
            last_progress_at = log_task_progress_if_due(
                stage_name,
                output_spec.dataset_spec.dataset_id,
                result.task_index,
                task_count,
                succeeded,
                failed,
                warnings,
                last_progress_at,
            )
    except BaseException:
        aborted = True
        raise
    finally:
        if writer is not None:
            close_stage_writer(writer, succeeded=succeeded, failed=failed, aborted=aborted)
        clean_stage_scratch(scratch_store, scratch_root, dry_run=dry_run)

    logger.info(
        "%s finished dataset=%s tasks=%d succeeded=%d failed=%d warnings=%d rows=%d "
        "dry_run=%s elapsed_s=%.1f",
        stage_name,
        output_spec.dataset_spec.dataset_id,
        task_count,
        succeeded,
        failed,
        warnings,
        rows,
        dry_run,
        time.perf_counter() - started_at,
    )
    return StageRunStats(succeeded, failed, warnings, chunks, rows)


def prepare_stage_scratch(
    scratch_store: DataProcessingStore,
    scratch_root: str,
    *,
    dry_run: bool,
) -> None:
    """Create scratch storage unless running in dry-run mode."""
    if not dry_run:
        scratch_store.create_dir(scratch_root)


def clean_stage_scratch(
    scratch_store: DataProcessingStore,
    scratch_root: str,
    *,
    dry_run: bool,
) -> None:
    """Delete scratch storage unless running in dry-run mode."""
    if not dry_run:
        scratch_store.delete_dir(scratch_root, missing_ok=True)


def iter_stage_task_results[TaskT, PayloadT, OutputT](
    *,
    tasks: Sequence[TaskT],
    task_id: Callable[[TaskT], str],
    make_payload: Callable[[int, TaskT], PayloadT],
    worker: Callable[[PayloadT], OutputT],
    config: StageRunConfig,
    ordered: bool = True,
) -> Iterator[ProcessTaskResult[OutputT]]:
    """Build process task specs and yield worker results."""
    specs = tuple(
        ProcessTaskSpec(
            task_index=idx,
            task_id=task_id(task),
            arg=make_payload(idx, task),
        )
        for idx, task in enumerate(tasks, start=1)
    )
    yield from iter_process_task_results(worker, specs, config, ordered=ordered)


def consume_prepared_tables(
    store: DataProcessingStore,
    writer: ShardWriter,
    tables: tuple[PreparedTable, ...],
    chunk_rows: int,
    coalesce_rows: int | None = None,
) -> ConsumeStats:
    """Append temporary task tables into final shards and delete them.

    Args:
        store: Store containing temporary parquet tables.
        writer: Final shard writer.
        tables: Prepared task tables.
        chunk_rows: Rows read from each temporary table at a time.
        coalesce_rows: Optional larger output batch size before appending.

    Returns:
        Number of chunks and rows appended to final output.
    """
    chunks = rows = 0
    for table in tables:
        table_chunks = table_rows = 0

        def append_chunk(chunk: pl.DataFrame, prepared: PreparedTable = table) -> None:
            nonlocal table_chunks, table_rows
            writer.append(chunk, prepared.metadata, prepared.source_paths)
            table_chunks += 1
            table_rows += chunk.height

        try:
            table_frames = store.iter_table_chunks(table.temp_path, chunk_rows)
            if coalesce_rows is not None and coalesce_rows > chunk_rows:
                table_frames = coalesce_frames(table_frames, coalesce_rows)
            for chunk in table_frames:
                append_chunk(chunk)
        finally:
            store.delete_file(table.temp_path, missing_ok=True)
        chunks += table_chunks
        rows += table_rows
    return ConsumeStats(chunks=chunks, rows=rows)
