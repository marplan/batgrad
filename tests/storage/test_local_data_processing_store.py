from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batgrad.storage.local import LocalDataProcessingStore
from tests.fixtures import store_at


def test_constructor_requires_absolute_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        LocalDataProcessingStore("relative/path")
    with pytest.raises(FileNotFoundError):
        LocalDataProcessingStore(tmp_path / "missing")
    file_path = tmp_path / "file"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError):
        LocalDataProcessingStore(file_path)


def test_create_resolve_delete_and_list_files(tmp_path: Path) -> None:
    store = store_at(tmp_path / "store")
    assert Path(store.root).exists()
    assert store.resolve() == store.root
    assert store.resolve("a/b.txt") == str(Path(store.root) / "a/b.txt")
    assert store.resolve(tmp_path / "abs.txt") == str(tmp_path / "abs.txt")

    store.create_dir("a/.hidden")
    Path(store.resolve("a/b.txt")).write_text("b")
    Path(store.resolve("a/.hidden/c.txt")).write_text("c")
    Path(store.resolve("z.txt")).write_text("z")

    assert store.list_files() == ("a/b.txt", "z.txt")
    assert store.list_files("a") == ("a/b.txt",)
    with store.local_file("a/b.txt") as path:
        assert path == Path(store.root) / "a/b.txt"

    store.delete_file("a/b.txt")
    assert not Path(store.resolve("a/b.txt")).exists()
    store.delete_file("a/b.txt")
    with pytest.raises(FileNotFoundError):
        store.delete_file("a/b.txt", missing_ok=False)
    store.delete_dir("a")
    assert not Path(store.resolve("a")).exists()
    with pytest.raises(FileNotFoundError):
        store.delete_dir("a", missing_ok=False)


def test_write_scan_chunks_slices_and_size(local_store: LocalDataProcessingStore) -> None:
    frame = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    local_store.write_table(
        frame,
        "data/table.parquet",
        metadata={"stage": "test"},
        row_group_size=2,
    )

    assert local_store.table_size_bytes("data/table.parquet") is not None
    assert local_store.table_size_bytes("missing.parquet") is None
    with pytest.raises(FileExistsError):
        local_store.write_table(frame, "data/table.parquet")

    scanned = local_store.scan_table(
        "data/table.parquet",
        columns=("x",),
        filters=pl.col("x") > 2,
        limit=2,
    ).collect()
    assert scanned.to_dict(as_series=False) == {"x": [3, 4]}

    chunks = list(local_store.iter_table_chunks("data/table.parquet", 2, columns=("x",)))
    assert [chunk["x"].to_list() for chunk in chunks] == [[1, 2], [3, 4], [5]]
    filtered = list(
        local_store.iter_table_chunks("data/table.parquet", 3, filters=pl.col("y") > 30)
    )
    assert pl.concat(filtered)["x"].to_list() == [4, 5]
    with pytest.raises(ValueError, match="chunk_rows"):
        list(local_store.iter_table_chunks("data/table.parquet", 0))

    slices = list(
        local_store.iter_table_slices(
            "data/table.parquet",
            ((1, 3), (4, 1)),
            chunk_rows=2,
            columns=("x",),
        )
    )
    assert [value for chunk in slices for value in chunk["x"].to_list()] == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="chunk_rows"):
        list(local_store.iter_table_slices("data/table.parquet", ((0, 1),), 0))

    metadata = pq.ParquetFile(local_store.resolve("data/table.parquet")).metadata.metadata
    assert metadata is not None
    assert metadata[b"stage"] == b"test"


def test_open_table_writer_writes_metadata(local_store: LocalDataProcessingStore) -> None:
    schema = pa.schema([pa.field("x", pa.int64())])
    writer = local_store.open_table_writer("writer/out.parquet", schema, "zstd")
    writer.write_table(pl.DataFrame({"x": [1, 2]}), row_group_size=1)
    writer.close({"kind": "writer"})

    assert local_store.scan_table("writer/out.parquet").collect()["x"].to_list() == [1, 2]
    metadata = pq.ParquetFile(local_store.resolve("writer/out.parquet")).metadata.metadata
    assert metadata is not None
    assert metadata[b"kind"] == b"writer"
