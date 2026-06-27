from __future__ import annotations

from dataclasses import dataclass

import pytest

from batgrad.contracts.mapping import BaseColumns, DatasetStageId
from batgrad.contracts.metadata import INGEST_STAGE_METADATA
from batgrad.data.processing.runtime import ProcessTaskResult
from batgrad.data.processing.stage import TaskOutput, run_stage_task_outputs, stage_output_spec
from tests.fixtures import dataset_spec


@dataclass(frozen=True)
class StageConfig:
    n_jobs: int = 1
    worker_polars_max_threads: int | None = -1
    chunk_rows: int = 2
    compression: str = "zstd"
    use_content_defined_chunking: bool = False
    row_group_size: int = 2
    max_shard_size_bytes: int = 0


class Logger:
    def error(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass


def test_run_stage_task_outputs_writes_error_manifest_and_cleans_scratch(local_store) -> None:
    spec = dataset_spec()
    output_spec = stage_output_spec(
        dataset_spec=spec,
        stage_id=DatasetStageId.ingested,
        output_root="stage-out",
        manifest_path="stage-out/manifest.parquet",
        manifest_metadata=INGEST_STAGE_METADATA.manifest,
        footer_metadata=INGEST_STAGE_METADATA.footer,
        shard_key_col=BaseColumns.proto,
        segment_col=BaseColumns.parq_segs,
        source_paths_col=BaseColumns.raw_paths,
    )

    def iter_results(_scratch_root: str):
        yield ProcessTaskResult[TaskOutput[str]](1, "bad-task", error="boom")

    stats = run_stage_task_outputs(
        output_store=local_store,
        scratch_store=local_store,
        output_spec=output_spec,
        config=StageConfig(),
        tasks=("bad-task",),
        iter_results=iter_results,
        logger=Logger(),
    )

    assert stats.failed == 1
    files = local_store.list_files()
    assert "stage-out/err_manifest.parquet" in files
    assert not any("_tmp" in path for path in files)


def test_run_stage_task_outputs_cleans_scratch_on_interrupt(local_store) -> None:
    spec = dataset_spec()
    output_spec = stage_output_spec(
        dataset_spec=spec,
        stage_id=DatasetStageId.ingested,
        output_root="stage-interrupt",
        manifest_path="stage-interrupt/manifest.parquet",
        manifest_metadata=INGEST_STAGE_METADATA.manifest,
        footer_metadata=INGEST_STAGE_METADATA.footer,
        shard_key_col=BaseColumns.proto,
        segment_col=BaseColumns.parq_segs,
        source_paths_col=BaseColumns.raw_paths,
    )

    def iter_results(_scratch_root: str):
        raise KeyboardInterrupt
        yield

    with pytest.raises(KeyboardInterrupt):
        run_stage_task_outputs(
            output_store=local_store,
            scratch_store=local_store,
            output_spec=output_spec,
            config=StageConfig(),
            tasks=("task",),
            iter_results=iter_results,
            logger=Logger(),
        )

    assert not any("_tmp" in path for path in local_store.list_files())
