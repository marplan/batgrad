from __future__ import annotations

from dataclasses import replace

import polars as pl
import pyarrow.parquet as pq
import pytest

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, MappingSpec
from batgrad.contracts.protocols import BatteryProtocols
from batgrad.data.processing.raw import (
    IngestBatch,
    IngestProtocolSpec,
    IngestStageConfig,
    align_to_protocol_spec,
    prepare_raw_batch,
    run_ingest,
    validate_raw_batch_metadata,
)
from batgrad.storage.segments import SegmentSource
from tests.fixtures import SyntheticAdapter, dataset_spec, raw_stage_spec, synthetic_raw_frame


def test_raw_stage_file_patterns() -> None:
    spec = raw_stage_spec()
    assert spec.is_included_file("data.csv")
    assert not spec.is_included_file("data.txt")
    assert not spec.is_included_file("data_skip.csv")


def test_prepare_raw_batch_aligns_aliases_and_casts() -> None:
    spec = raw_stage_spec()
    batch = IngestBatch(
        data=synthetic_raw_frame(),
        protocol_id=DatasetProtocolId.cycling,
        source_paths=("raw/a.csv",),
        metadata={
            BaseColumns.proto: DatasetProtocolId.cycling,
            BaseColumns.cell_id: "cell-a",
            BaseColumns.cidx: 1,
        },
    )
    frame, warnings = prepare_raw_batch(batch, spec)
    assert warnings == ()
    assert frame.columns == [
        str(BaseColumns.time),
        str(BaseColumns.curr),
        str(BaseColumns.volt),
        str(BaseColumns.cell_id),
        str(BaseColumns.cidx),
    ]
    assert frame.schema[str(BaseColumns.time)] == pl.Float64


def test_prepare_raw_batch_honors_dropped_columns_and_current_sign_flip() -> None:
    dropped = MappingSpec("ignored", dtype=pl.Int64)
    spec = replace(
        raw_stage_spec(),
        protocol_specs=(
            IngestProtocolSpec(
                protocol=BatteryProtocols.cyc,
                columns=(
                    BaseColumns.time.with_alias("time_s"),
                    BaseColumns.curr.with_alias("current_a"),
                    BaseColumns.volt.with_alias("voltage_v"),
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                ),
                dropped_columns=(dropped,),
                flip_current_sign=True,
            ),
        ),
    )
    batch = IngestBatch(
        data=synthetic_raw_frame().with_columns(pl.lit(1).alias("ignored")),
        protocol_id=DatasetProtocolId.cycling,
        source_paths=("raw/a.csv",),
        metadata={
            BaseColumns.proto: DatasetProtocolId.cycling,
            BaseColumns.cell_id: "cell-a",
            BaseColumns.cidx: 1,
        },
    )

    frame, warnings = prepare_raw_batch(batch, spec)

    assert "dropped declared columns" in warnings[0]
    assert frame[str(BaseColumns.curr)].to_list()[:2] == [-1.0, -1.1]
    assert "ignored" not in frame.columns


def test_raw_batch_validation_and_unknown_columns_fail() -> None:
    spec = raw_stage_spec()
    batch = IngestBatch(
        data=synthetic_raw_frame().with_columns(pl.lit(1).alias("unknown")),
        protocol_id=DatasetProtocolId.cycling,
        source_paths=("raw/a.csv",),
        metadata={BaseColumns.proto: DatasetProtocolId.cycling},
    )
    with pytest.raises(ValueError, match="missing required"):
        validate_raw_batch_metadata(batch, spec)
    protocol_spec = spec.protocol_spec(DatasetProtocolId.cycling)
    with pytest.raises(ValueError, match="unknown columns"):
        align_to_protocol_spec(batch.data, protocol_spec, batch.source_paths)


def test_run_ingest_writes_manifest_and_shards(local_store) -> None:
    spec = dataset_spec()
    run_ingest(
        SyntheticAdapter(spec),
        local_store,
        local_store,
        IngestStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )
    manifest = local_store.scan_table(spec.manifest(DatasetStageId.ingested)).collect()
    assert manifest.height == 2
    assert manifest[str(BaseColumns.row_n)].sum() == 10
    assert set(manifest[str(BaseColumns.proto)].to_list()) == {"cycling"}
    assert not any("_tmp" in path for path in local_store.list_files())


def test_run_ingest_writes_expected_shard_data_and_footer_metadata(local_store) -> None:
    spec = dataset_spec()
    run_ingest(
        SyntheticAdapter(spec),
        local_store,
        local_store,
        IngestStageConfig(chunk_rows=2, row_group_size=2, max_shard_size_bytes=0),
    )

    manifest = local_store.scan_table(spec.manifest(DatasetStageId.ingested)).collect()
    segments = tuple(
        segment
        for row_segments in manifest[str(BaseColumns.ingest_segs)].to_list()
        for segment in row_segments
    )
    frame = SegmentSource.from_values(local_store, segments).scan().collect()

    assert frame.columns == [
        str(BaseColumns.time),
        str(BaseColumns.curr),
        str(BaseColumns.volt),
        str(BaseColumns.cell_id),
        str(BaseColumns.cidx),
    ]
    assert frame.height == 10
    assert frame[str(BaseColumns.cell_id)].to_list() == ["cell-a"] * 5 + ["cell-b"] * 5
    assert frame[str(BaseColumns.curr)].to_list()[:3] == [1.0, 1.1, 1.2]
    assert frame[str(BaseColumns.curr)].to_list()[5:8] == [11.0, 11.1, 11.2]

    footer = pq.ParquetFile(
        local_store.resolve(segments[0][str(BaseColumns.path)])
    ).metadata.metadata
    assert footer is not None
    assert footer[str(BaseColumns.set_id).encode()] == spec.dataset_id.encode()
    assert footer[str(BaseColumns.stage).encode()] == b"ingested"
    assert (
        footer[str(BaseColumns.manifest).encode()]
        == spec.manifest(DatasetStageId.ingested).encode()
    )
    assert footer[str(BaseColumns.git_status).encode()] in {b"clean", b"dirty", b"na"}
    assert str(BaseColumns.git_commit).encode() in footer
