from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    import polars as pl
    import pyarrow as pa


class TableWriter(Protocol):
    def write_table(self, data: pl.DataFrame, row_group_size: int | None = None) -> None: ...

    def close(self, metadata: dict[str, str] | None = None) -> None: ...


class DataStore(Protocol):
    root: str

    def resolve(self, location: str | Path | None = None) -> str: ...

    def exists(self, location: str | Path | None = None) -> bool: ...

    def delete_dir(self, location: str | Path, *, missing_ok: bool = True) -> None: ...

    def list_dirs(
        self,
        location: str | Path | None = None,
        pattern: str = "*",
    ) -> tuple[str, ...]: ...

    def list_files(
        self,
        location: str | Path | None = None,
        pattern: str = "*",
    ) -> tuple[str, ...]: ...

    def local_file(self, location: str | Path) -> AbstractContextManager: ...

    def scan_table(
        self,
        location: str | Path | tuple[str | Path, ...],
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame: ...

    def iter_table_chunks(
        self,
        location: str | Path | tuple[str | Path, ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
    ) -> Iterator[pl.DataFrame]: ...

    def write_table(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        location: str | Path,
        metadata: dict[str, str] | None = None,
        row_group_size: int | None = None,
    ) -> None: ...

    def open_table_writer(
        self,
        location: str | Path,
        schema: pa.Schema,
        compression: str,
        *,
        use_content_defined_chunking: bool = False,
    ) -> TableWriter: ...

    def table_size_bytes(self, location: str | Path) -> int | None: ...
