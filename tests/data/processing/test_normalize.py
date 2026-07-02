from __future__ import annotations

from dataclasses import replace

import polars as pl
import pyarrow.parquet as pq

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId
from batgrad.data.processing.normalize import (
    NormalizeStageConfig,
    _task_metadata,
    plan_normalize_tasks,
    run_normalize,
)
from batgrad.data.processing.raw import IngestBatch, IngestStageConfig, IngestTask, run_ingest
from batgrad.data.transforms.checks import ColumnBoundsCheckSpec, MissingCheckSpec, TimeCheckSpec
from batgrad.data.transforms.transforms import CRateTransformSpec
from batgrad.storage.segments import SegmentSource
from tests.fixtures import SyntheticAdapter, dataset_spec


class BadRawAdapter(SyntheticAdapter):
    def plan_raw_tasks(self, input_store, raw_spec):
        del input_store, raw_spec
        return (IngestTask("bad-task", ("raw/bad.csv",)),)

    def load_raw_task(self, task, input_store, raw_spec):
        del task, input_store, raw_spec
        yield IngestBatch(
            data=pl.DataFrame(
                {
                    "time_s": [0.0, 1.0, 1.0, 10.0],
                    "current_a": [1.0, None, 1.2, 1.3],
                    "voltage_v": [3.0, 3.1, 6.0, 3.3],
                    str(BaseColumns.cell_id): ["cell-bad"] * 4,
                    str(BaseColumns.cidx): [1] * 4,
                }
            ),
            protocol_id=DatasetProtocolId.cycling,
            source_paths=("raw/bad.csv",),
            metadata={
                BaseColumns.proto: DatasetProtocolId.cycling,
                BaseColumns.cell_id: "cell-bad",
                BaseColumns.cidx: 1,
            },
        )


def _ingest_fixture(store, *, resample_points: int | None = None):
    spec = dataset_spec(resample_points=resample_points)
    run_ingest(
        SyntheticAdapter(spec),
        store,
        store,
        IngestStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )
    return spec


def test_plan_normalize_tasks_groups_manifest_rows(local_store) -> None:
    spec = _ingest_fixture(local_store)
    normalize_spec = spec.processing_stages[DatasetStageId.normalized]
    tasks = plan_normalize_tasks(spec, local_store, normalize_spec)

    assert [task.task_id for task in tasks] == [
        "cycling_cell_id-cell_a_cycle_index-1",
        "cycling_cell_id-cell_b_cycle_index-1",
    ]
    assert [task.row_count for task in tasks] == [5, 5]
    assert tasks[0].raw_paths == ("raw/a.csv",)
    assert [segment.row_count for segment in tasks[0].parquet_segments] == [2, 2, 1]


def test_plan_normalize_tasks_filters_protocol_and_group_values(local_store) -> None:
    spec = _ingest_fixture(local_store)
    normalize_spec = spec.processing_stages[DatasetStageId.normalized]
    tasks = plan_normalize_tasks(
        spec,
        local_store,
        normalize_spec,
        protocols="cycling",
        group_values={BaseColumns.cell_id: "cell-b"},
    )
    assert len(tasks) == 1
    assert tasks[0].group_values[BaseColumns.cell_id] == "cell-b"


def test_task_metadata_contains_manifest_and_resampling_values(local_store) -> None:
    spec = _ingest_fixture(local_store, resample_points=3)
    normalize_spec = spec.processing_stages[DatasetStageId.normalized]
    task = plan_normalize_tasks(spec, local_store, normalize_spec)[0]
    metadata = _task_metadata(
        spec.dataset_id,
        normalize_spec,
        task,
        NormalizeStageConfig(apply_resampling=True),
    )
    assert metadata[BaseColumns.set_id] == spec.dataset_id
    assert metadata[BaseColumns.proto] == "cycling"
    assert metadata[BaseColumns.raw_paths] == ["raw/a.csv"]
    assert metadata[BaseColumns.resamp] == "min_max_lttb"
    assert metadata[BaseColumns.time_conv] == "start_of_interval"


def test_run_normalize_writes_outputs_manifest_and_cleans_scratch(local_store) -> None:
    spec = _ingest_fixture(local_store)
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect()
    assert manifest.height == 2
    assert manifest[str(BaseColumns.row_n)].sum() == 8
    assert set(manifest[str(BaseColumns.resamp)].to_list()) == {"none"}
    assert not any("_tmp" in path for path in local_store.list_files())

    segments = manifest.row(0, named=True)[str(BaseColumns.norm_segs)]
    source = SegmentSource.from_values(local_store, tuple(segments))
    frame = source.scan().collect()
    assert str(BaseColumns.crate) in frame.columns
    assert frame[str(BaseColumns.time)].to_list() == [0.0, 1.0, 2.0, 3.0]


def test_run_normalize_writes_footer_metadata(local_store) -> None:
    spec = _ingest_fixture(local_store, resample_points=3)
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect()
    segment = manifest.row(0, named=True)[str(BaseColumns.norm_segs)][0]
    footer = pq.ParquetFile(local_store.resolve(segment[str(BaseColumns.path)])).metadata.metadata

    assert footer is not None
    assert footer[str(BaseColumns.set_id).encode()] == spec.dataset_id.encode()
    assert footer[str(BaseColumns.stage).encode()] == b"normalized"
    assert (
        footer[str(BaseColumns.manifest).encode()]
        == spec.manifest(DatasetStageId.normalized).encode()
    )
    assert footer[str(BaseColumns.time_conv).encode()] == b"start_of_interval"


def test_run_normalize_reconstructs_segments_across_shard_rollover(local_store) -> None:
    spec = _ingest_fixture(local_store)
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(
            chunk_rows=2,
            row_group_size=2,
            max_batch_rows=2,
            max_shard_size_bytes=1,
        ),
    )
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect()
    cell_a = manifest.filter(pl.col(BaseColumns.cell_id) == "cell-a").row(0, named=True)
    segments = tuple(cell_a[str(BaseColumns.norm_segs)])

    assert len({segment[str(BaseColumns.path)] for segment in segments}) > 1
    reconstructed = SegmentSource.from_values(local_store, segments).scan().collect()
    assert reconstructed[str(BaseColumns.time)].to_list() == [0.0, 1.0, 2.0, 3.0]
    assert reconstructed[str(BaseColumns.curr)].to_list() == [1.0, 1.1, 1.2, 1.3]


def test_run_normalize_failed_checks_write_error_manifest_and_clean_scratch(local_store) -> None:
    spec = dataset_spec()
    run_ingest(
        BadRawAdapter(spec),
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

    files = local_store.list_files()
    assert spec.manifest(DatasetStageId.normalized) not in files
    assert (
        spec.manifest(DatasetStageId.normalized).replace(
            "manifest.parquet",
            "err_manifest.parquet",
        )
        in files
    )
    assert not any("_tmp" in path for path in files)


def test_run_normalize_dry_run_leaves_no_normalized_outputs(local_store) -> None:
    spec = _ingest_fixture(local_store)

    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
        dry_run=True,
    )

    files = local_store.list_files()
    assert spec.manifest(DatasetStageId.ingested) in files
    assert spec.manifest(DatasetStageId.normalized) not in files
    assert not any(path.startswith(spec.source_root(DatasetStageId.normalized)) for path in files)
    assert not any("_tmp" in path for path in files)


def test_normalize_resolves_alias_columns_and_constants(local_store) -> None:
    spec = dataset_spec()
    curr_alias = BaseColumns.curr.with_alias("current")
    normalize_spec = spec.processing_stages[DatasetStageId.normalized]
    protocol_spec = normalize_spec.protocol_specs[0]
    aliased_protocol_spec = replace(
        protocol_spec,
        columns=(BaseColumns.time, curr_alias, BaseColumns.volt, BaseColumns.crate),
        transforms=(CRateTransformSpec(curr_alias, BaseColumns.crate, 2.0),),
        checks=(
            MissingCheckSpec((curr_alias, BaseColumns.volt)),
            TimeCheckSpec(BaseColumns.time, BaseColumns.dt, max_dt_s=5.0),
            ColumnBoundsCheckSpec({BaseColumns.volt: (2.0, 5.0)}),
        ),
    )
    spec = replace(
        spec,
        processing_stages={
            **spec.processing_stages,
            DatasetStageId.normalized: replace(
                normalize_spec,
                protocol_specs=(aliased_protocol_spec,),
            ),
        },
    )
    local_store.write_table(
        pl.DataFrame(
            {
                str(BaseColumns.time): [0.0, 1.0, 2.0],
                "current": [1.0, 2.0, 3.0],
                str(BaseColumns.volt): [3.0, 3.1, 3.2],
                str(BaseColumns.cell_id): ["cell-a"] * 3,
                str(BaseColumns.cidx): [1] * 3,
            }
        ),
        "ingested/cycling/cycling_part-000000.parquet",
    )
    local_store.write_table(
        pl.DataFrame(
            {
                str(BaseColumns.raw_paths): [["raw/a.csv"]],
                str(BaseColumns.ingest_segs): [
                    [
                        {
                            str(BaseColumns.path): "ingested/cycling/cycling_part-000000.parquet",
                            str(BaseColumns.row0): 0,
                            str(BaseColumns.row_n): 3,
                        }
                    ]
                ],
                str(BaseColumns.row_n): [3],
                str(BaseColumns.proto): ["cycling"],
                str(BaseColumns.cell_id): ["cell-a"],
                str(BaseColumns.cidx): [1],
            }
        ),
        spec.manifest(DatasetStageId.ingested),
        row_group_size=1,
    )

    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=3, row_group_size=3, max_shard_size_bytes=0),
    )
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect()
    frame = (
        SegmentSource.from_values(
            local_store,
            tuple(manifest.row(0, named=True)[str(BaseColumns.norm_segs)]),
        )
        .scan()
        .collect()
    )

    assert frame[str(BaseColumns.curr)].to_list() == [1.0, 2.0]
    assert frame[str(BaseColumns.cap_nom)].to_list() == [2.0, 2.0]
    assert frame[str(BaseColumns.crate)].to_list() == [0.5, 1.0]


def test_run_normalize_missing_required_input_writes_error_manifest(local_store) -> None:
    spec = dataset_spec()
    local_store.write_table(
        pl.DataFrame(
            {
                str(BaseColumns.time): [0.0, 1.0, 2.0],
                str(BaseColumns.curr): [1.0, 2.0, 3.0],
                str(BaseColumns.cell_id): ["cell-a"] * 3,
                str(BaseColumns.cidx): [1] * 3,
            }
        ),
        "ingested/cycling/cycling_part-000000.parquet",
    )
    local_store.write_table(
        pl.DataFrame(
            {
                str(BaseColumns.raw_paths): [["raw/a.csv"]],
                str(BaseColumns.ingest_segs): [
                    [
                        {
                            str(BaseColumns.path): "ingested/cycling/cycling_part-000000.parquet",
                            str(BaseColumns.row0): 0,
                            str(BaseColumns.row_n): 3,
                        }
                    ]
                ],
                str(BaseColumns.row_n): [3],
                str(BaseColumns.proto): ["cycling"],
                str(BaseColumns.cell_id): ["cell-a"],
                str(BaseColumns.cidx): [1],
            }
        ),
        spec.manifest(DatasetStageId.ingested),
        row_group_size=1,
    )

    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )

    files = local_store.list_files()
    assert spec.manifest(DatasetStageId.normalized) not in files
    assert (
        spec.manifest(DatasetStageId.normalized).replace("manifest.parquet", "err_manifest.parquet")
        in files
    )
    assert not any("_tmp" in path for path in files)


def test_normalize_manifest_contains_contract_columns_and_types(local_store) -> None:
    spec = _ingest_fixture(local_store, resample_points=3)
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )

    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect()

    for column in spec.processing_stages[DatasetStageId.normalized].manifest_metadata.columns:
        assert str(column) in manifest.columns
        assert manifest.schema[str(column)] == column.dtype
    assert set(manifest[str(BaseColumns.resamp)].to_list()) == {"min_max_lttb"}
    assert set(manifest[str(BaseColumns.time_conv)].to_list()) == {"start_of_interval"}


def test_plan_normalize_tasks_accepts_multiple_group_selectors(local_store) -> None:
    spec = _ingest_fixture(local_store)
    tasks = plan_normalize_tasks(
        spec,
        local_store,
        spec.processing_stages[DatasetStageId.normalized],
        group_values=[
            {BaseColumns.cell_id: "cell-a"},
            {str(BaseColumns.cell_id): "cell-b"},
        ],
    )

    assert [task.group_values[BaseColumns.cell_id] for task in tasks] == ["cell-a", "cell-b"]


def test_run_normalize_bounded_resampling_rolls_manifest_segments(local_store) -> None:
    spec = _ingest_fixture(local_store, resample_points=3)
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(
            chunk_rows=2,
            row_group_size=2,
            max_batch_rows=2,
            max_shard_size_bytes=1,
            apply_resampling=True,
        ),
    )
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.normalized)).collect()
    assert manifest[str(BaseColumns.row_n)].sum() <= 6
    assert all(value == "min_max_lttb" for value in manifest[str(BaseColumns.resamp)].to_list())
    assert any("part-000001" in path for path in local_store.list_files())


def test_plan_normalize_tasks_returns_no_tasks_for_unknown_protocol(local_store) -> None:
    spec = _ingest_fixture(local_store)
    run_normalize(
        spec,
        local_store,
        local_store,
        NormalizeStageConfig(chunk_rows=2, row_group_size=2, n_jobs=2, max_shard_size_bytes=0),
    )
    assert spec.manifest(DatasetStageId.normalized) in local_store.list_files()

    bad = (
        local_store.scan_table(spec.manifest(DatasetStageId.ingested))
        .collect()
        .with_columns(pl.lit("bad").alias(str(BaseColumns.proto)))
    )
    local_store.delete_file(spec.manifest(DatasetStageId.ingested))
    local_store.write_table(bad, spec.manifest(DatasetStageId.ingested))
    tasks = plan_normalize_tasks(
        spec,
        local_store,
        spec.processing_stages[DatasetStageId.normalized],
    )
    assert tasks == ()
