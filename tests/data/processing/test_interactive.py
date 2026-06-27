from __future__ import annotations

import pytest

from batgrad.contracts.mapping import BaseColumns, DatasetStageId
from batgrad.data.processing.interactive import run_load_interactive, run_load_interactive_manifest
from batgrad.data.processing.normalize import (
    NormalizeStageConfig,
    run_normalize,
    run_normalize_interactive,
)
from batgrad.data.processing.raw import IngestStageConfig, run_ingest
from batgrad.data.processing.stage import validate_scratch_run_root
from tests.fixtures import SyntheticAdapter, dataset_spec


def _normalized_store(local_store):
    spec = dataset_spec()
    run_ingest(
        SyntheticAdapter(spec),
        local_store,
        local_store,
        IngestStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )
    return spec


def test_run_load_interactive_filters_manifest_and_scans(local_store) -> None:
    spec = _normalized_store(local_store)
    run = run_load_interactive(
        spec,
        local_store,
        source=DatasetStageId.normalized,
        group_values={BaseColumns.cell_id: "cell-a"},
    )
    manifest = run.manifest()
    assert manifest.height == 1
    assert manifest.row(0, named=True)[str(BaseColumns.cell_id)] == "cell-a"
    assert run.scan().collect().height == 4
    sources = list(run.iter_sources())
    assert len(sources) == 1
    assert sources[0][1].row_count == 4


def test_run_load_interactive_manifest_uses_supplied_frame(local_store) -> None:
    spec = _normalized_store(local_store)
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect().limit(1)
    run = run_load_interactive_manifest(
        spec,
        local_store,
        source=DatasetStageId.normalized,
        manifest=manifest,
    )
    assert run.manifest().height == 1


def test_interactive_clean_only_deletes_valid_scratch_run(local_store) -> None:
    spec = _normalized_store(local_store)
    run = run_load_interactive(spec, local_store, source=DatasetStageId.normalized)
    run.clean()

    local_store.create_dir("type=synthetic/dataset=synthetic-test/source=normalized/scratch/run-1")
    scratch_run = run_load_interactive(spec, local_store, source=DatasetStageId.normalized)
    object.__setattr__(
        scratch_run,
        "run_root",
        "type=synthetic/dataset=synthetic-test/source=normalized/scratch/run-1",
    )
    scratch_run.clean()
    assert (
        "type=synthetic/dataset=synthetic-test/source=normalized/scratch/run-1"
        not in local_store.list_files()
    )
    with pytest.raises(ValueError, match="scratch"):
        validate_scratch_run_root("not/a/scratch/root/with/extra")


def test_run_normalize_interactive_writes_selected_scratch_run(local_store) -> None:
    spec = dataset_spec()
    run_ingest(
        SyntheticAdapter(spec),
        local_store,
        local_store,
        IngestStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )

    run = run_normalize_interactive(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
        group_values={BaseColumns.cell_id: "cell-b"},
    )

    manifest = run.manifest()
    assert manifest.height == 1
    assert manifest.row(0, named=True)[str(BaseColumns.cell_id)] == "cell-b"
    assert run.scan().collect()[str(BaseColumns.curr)].to_list() == [11.0, 11.1, 11.2, 11.3]
    assert run.run_root is not None
    assert any(path.startswith(run.run_root) for path in local_store.list_files())

    run.clean()
    assert not any(path.startswith(run.run_root) for path in local_store.list_files())
