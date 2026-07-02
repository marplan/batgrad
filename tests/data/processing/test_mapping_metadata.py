from __future__ import annotations

import polars as pl
import pytest

from batgrad.contracts.mapping import BaseColumns, MappingSpec
from batgrad.contracts.metadata import MetadataLayout
from batgrad.data.processing.metadata import encode_footer_values
from batgrad.data.processing.sharding import build_manifest, resolve_footer_values


def test_mapping_spec_aliases_and_values() -> None:
    spec = MappingSpec("canonical", dtype=pl.Float64, alias=("Raw", "raw2"))
    assert spec.matching_name(["RAW"]) == "RAW"
    assert spec.has_match("raw2")
    assert spec.with_alias("x").alias == ("canonical", "x")
    valued = spec.with_values(("a", "b"))
    assert valued.values == ("a", "b")


def test_footer_encoding_and_resolution_priority() -> None:
    layout = MetadataLayout(
        required=(BaseColumns.set_id,),
        optional={BaseColumns.stage: "layout", BaseColumns.git_status: None},
    )
    values = resolve_footer_values(
        layout,
        {BaseColumns.stage: "task"},
        {BaseColumns.set_id: "dataset", BaseColumns.stage: "runtime"},
    )
    assert values[BaseColumns.set_id] == "dataset"
    assert values[BaseColumns.stage] == "task"
    assert values[BaseColumns.git_status] is None
    assert encode_footer_values(values)[str(BaseColumns.git_status)] == "null"

    with pytest.raises(ValueError, match="missing required"):
        resolve_footer_values(layout, {}, {})


def test_build_manifest_groups_rows_and_merges_segments() -> None:
    layout = MetadataLayout(
        required=(
            BaseColumns.raw_paths,
            BaseColumns.ingest_segs,
            BaseColumns.row_n,
            BaseColumns.proto,
        ),
    )
    rows = [
        {
            BaseColumns.raw_paths: ["a.csv"],
            BaseColumns.ingest_segs: [{"file path": "a.parquet", "row start": 0, "row count": 2}],
            BaseColumns.row_n: 2,
            BaseColumns.proto: "cycling",
        },
        {
            BaseColumns.raw_paths: ["a.csv"],
            BaseColumns.ingest_segs: [{"file path": "b.parquet", "row start": 0, "row count": 3}],
            BaseColumns.row_n: 3,
            BaseColumns.proto: "cycling",
        },
    ]
    manifest = build_manifest(layout, BaseColumns.ingest_segs, rows)
    assert manifest.height == 1
    row = manifest.row(0, named=True)
    assert row[str(BaseColumns.row_n)] == 5
    assert row[str(BaseColumns.raw_paths)] == ["a.csv"]
    assert len(row[str(BaseColumns.ingest_segs)]) == 2
