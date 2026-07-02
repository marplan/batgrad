from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetStageId
from batgrad.contracts.metadata import INGEST_STAGE_METADATA
from batgrad.data.processing.sharding import ShardWriter, manifest_row
from tests.fixtures import dataset_spec


@dataclass(frozen=True)
class ShardConfig:
    compression: str = "zstd"
    use_content_defined_chunking: bool = False
    row_group_size: int = 2
    max_shard_size_bytes: int = 1


def test_manifest_row_records_segment_and_sources() -> None:
    row = manifest_row(
        BaseColumns.ingest_segs,
        BaseColumns.raw_paths,
        "out/a.parquet",
        2,
        3,
        ("a.csv",),
        {BaseColumns.proto: "cycling"},
    )
    assert row[BaseColumns.raw_paths] == ["a.csv"]
    assert row[BaseColumns.ingest_segs] == [
        {"file path": "out/a.parquet", "row start": 2, "row count": 3}
    ]
    assert row[BaseColumns.row_n] == 3


def test_shard_writer_rolls_over_and_writes_manifest(local_store) -> None:
    spec = dataset_spec()
    writer = ShardWriter(
        output_store=local_store,
        dataset_spec=spec,
        stage_id=DatasetStageId.ingested,
        output_root="out",
        manifest_path="manifest.parquet",
        manifest_metadata=INGEST_STAGE_METADATA.manifest,
        footer_metadata=INGEST_STAGE_METADATA.footer,
        shard_key_col=BaseColumns.proto,
        segment_col=BaseColumns.ingest_segs,
        source_paths_col=BaseColumns.raw_paths,
        config=ShardConfig(),
    )
    metadata = {BaseColumns.proto: "cycling"}
    writer.append(pl.DataFrame({"x": [1, 2]}), metadata, ("a.csv",))
    writer.append(pl.DataFrame({"x": [3, 4]}), metadata, ("a.csv",))
    writer.append(pl.DataFrame({"x": [5]}), {BaseColumns.proto: "HPPC"}, ("b.csv",))
    writer.close()

    files = local_store.list_files()
    assert "manifest.parquet" in files
    assert any(path.endswith("cycling_part-000000.parquet") for path in files)
    assert any(path.endswith("cycling_part-000001.parquet") for path in files)
    assert any(path.endswith("hppc_part-000000.parquet") for path in files)

    manifest = local_store.scan_table("manifest.parquet").collect()
    assert manifest[str(BaseColumns.row_n)].sum() == 5
    cycling = manifest.filter(pl.col(BaseColumns.proto) == "cycling").row(0, named=True)
    assert cycling[str(BaseColumns.raw_paths)] == ["a.csv"]
    assert len(cycling[str(BaseColumns.ingest_segs)]) == 2


def test_shard_writer_error_and_skip_manifest(local_store) -> None:
    spec = dataset_spec()
    for mode in ("error", "skip"):
        writer = ShardWriter(
            output_store=local_store,
            dataset_spec=spec,
            stage_id=DatasetStageId.ingested,
            output_root=f"out-{mode}",
            manifest_path=f"{mode}/manifest.parquet",
            manifest_metadata=INGEST_STAGE_METADATA.manifest,
            footer_metadata=INGEST_STAGE_METADATA.footer,
            shard_key_col=BaseColumns.proto,
            segment_col=BaseColumns.ingest_segs,
            source_paths_col=BaseColumns.raw_paths,
            config=ShardConfig(max_shard_size_bytes=0),
        )
        writer.close(manifest=mode)
    assert "error/err_manifest.parquet" in local_store.list_files()
    assert "skip/manifest.parquet" not in local_store.list_files()
