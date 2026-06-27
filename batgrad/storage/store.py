from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    import polars as pl
    import pyarrow as pa


class TableWriter(Protocol):
    """Interface for writing one table incrementally."""

    def write_table(self, data: pl.DataFrame, row_group_size: int | None = None) -> None:
        """Append one dataframe chunk to the table."""
        ...

    def close(self, metadata: dict[str, str] | None = None) -> None:
        """Finalize the table and attach optional metadata."""
        ...


class DataProcessingStore(Protocol):
    """Storage interface used by data processing stages.

    A data processing store resolves logical dataset locations to a concrete
    storage backend and provides the file, parquet scanning, chunk iteration, and
    table writing operations required by ingest, normalization, and sharding.
    """

    root: str

    def create_dir(self, location: str | Path) -> None:
        """Create a directory location, including missing parents."""
        ...

    def delete_dir(self, location: str | Path, *, missing_ok: bool = True) -> None:
        """Delete a directory location."""
        ...

    def delete_file(self, location: str | Path, *, missing_ok: bool = True) -> None:
        """Delete a file location."""
        ...

    def list_files(
        self,
        location: str | Path | None = None,
        pattern: str = "*",
    ) -> tuple[str, ...]:
        """List file locations matching a backend-specific pattern."""
        ...

    def local_file(self, location: str | Path) -> AbstractContextManager:
        """Open a context manager that yields a local path for a file location."""
        ...

    def scan_table(
        self,
        location: str | Path | tuple[str | Path, ...],
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        """Create a lazy scan for one or more parquet table locations."""
        ...

    def iter_table_chunks(
        self,
        location: str | Path | tuple[str | Path, ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Yield sequential chunks from complete parquet table locations."""
        ...

    def iter_table_slices(
        self,
        location: str | Path,
        slices: tuple[tuple[int, int], ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Yield chunks from selected row ranges in one parquet table."""
        ...

    def write_table(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        location: str | Path,
        metadata: dict[str, str] | None = None,
        row_group_size: int | None = None,
    ) -> None:
        """Write a complete parquet table to one logical location."""
        ...

    def open_table_writer(
        self,
        location: str | Path,
        schema: pa.Schema,
        compression: str,
        *,
        use_content_defined_chunking: bool = False,
    ) -> TableWriter:
        """Open a writer for chunked output to one table location."""
        ...

    def table_size_bytes(self, location: str | Path) -> int | None:
        """Return a table size in bytes, or `None` when it is unavailable."""
        ...
