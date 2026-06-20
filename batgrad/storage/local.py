from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from batgrad.data.processing.io import iter_data_chunks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.storage.store import TableWriter


class LocalTableWriter:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        compression: str,
        *,
        use_content_defined_chunking: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"File exists: {path}")
        self._writer = pq.ParquetWriter(
            path,
            schema,
            compression=compression,
            use_content_defined_chunking=use_content_defined_chunking,
        )

    def write_table(self, data: pl.DataFrame, row_group_size: int | None = None) -> None:
        self._writer.write_table(data.to_arrow(), row_group_size=row_group_size)

    def close(self, metadata: dict[str, str] | None = None) -> None:
        if metadata is not None:
            self._writer.add_key_value_metadata(metadata)
        self._writer.close()


class LocalDataProcessingStore:
    def __init__(self, root: str | Path, *, create: bool = False) -> None:
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError(f"Local data root must be an absolute path: {root_path}")
        if create:
            root_path.mkdir(parents=True, exist_ok=True)
        if not root_path.exists():
            raise FileNotFoundError(f"Local data root does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Local data root is not a directory: {root_path}")
        self.root = str(root_path.resolve())

    def resolve(self, location: str | Path | None = None) -> str:
        if location is None:
            return self.root
        path = Path(location)
        if path.is_absolute():
            return str(path)
        return str(Path(self.root) / path)

    def create_dir(self, location: str | Path) -> None:
        Path(self.resolve(location)).mkdir(parents=True, exist_ok=True)

    def delete_dir(self, location: str | Path, *, missing_ok: bool = True) -> None:
        path = Path(self.resolve(location))
        if not path.exists():
            if missing_ok:
                return
            raise FileNotFoundError(f"Directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        shutil.rmtree(path)

    def delete_file(self, location: str | Path, *, missing_ok: bool = True) -> None:
        path = Path(self.resolve(location))
        if not path.exists():
            if missing_ok:
                return
            raise FileNotFoundError(f"File does not exist: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")
        path.unlink()

    def list_files(
        self,
        location: str | Path | None = None,
        pattern: str = "*",
    ) -> tuple[str, ...]:
        root = Path(self.resolve(location))
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")
        paths = [
            path.relative_to(self.root).as_posix()
            for path in root.rglob(pattern)
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        ]
        return tuple(sorted(paths))

    @contextmanager
    def local_file(self, location: str | Path) -> Iterator[Path]:
        yield Path(self.resolve(location))

    def scan_table(
        self,
        location: str | Path | tuple[str | Path, ...],
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        resolved = (
            [self.resolve(path) for path in location]
            if isinstance(location, tuple)
            else self.resolve(location)
        )
        lf = pl.scan_parquet(resolved)
        if columns is not None:
            lf = lf.select(list(columns))
        if filters is not None:
            lf = lf.filter(filters)
        if limit is not None:
            lf = lf.limit(limit)
        return lf

    def iter_table_chunks(
        self,
        location: str | Path | tuple[str | Path, ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
    ) -> Iterator[pl.DataFrame]:
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        resolved = (
            [self.resolve(path) for path in location]
            if isinstance(location, tuple)
            else [self.resolve(location)]
        )
        selected_columns = list(columns) if columns is not None else None
        for path in resolved:
            parquet_file = pq.ParquetFile(path)
            for arrow_batch in parquet_file.iter_batches(
                batch_size=chunk_rows,
                columns=selected_columns,
            ):
                frame = pl.from_arrow(arrow_batch)
                if not isinstance(frame, pl.DataFrame):
                    raise TypeError(
                        f"Expected batch conversion to return DataFrame, got {type(frame).__name__}"
                    )
                if filters is not None:
                    frame = frame.filter(filters)
                if frame.height > 0:
                    yield frame

    def iter_table_slices(
        self,
        location: str | Path,
        slices: tuple[tuple[int, int], ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[pl.DataFrame]:
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        if not slices:
            return
        selected_columns = list(columns) if columns is not None else None
        parquet_file = pq.ParquetFile(self.resolve(location))
        row_group_starts: list[int] = []
        cursor = 0
        for idx in range(parquet_file.metadata.num_row_groups):
            row_group_starts.append(cursor)
            cursor += parquet_file.metadata.row_group(idx).num_rows
        sorted_slices = tuple(sorted(slices))
        for row_group_idx, group_start in enumerate(row_group_starts):
            group_rows = parquet_file.metadata.row_group(row_group_idx).num_rows
            group_end = group_start + group_rows
            overlaps = [
                (max(0, start - group_start), min(group_rows, start + count - group_start))
                for start, count in sorted_slices
                if count > 0 and start < group_end and start + count > group_start
            ]
            if not overlaps:
                continue
            table = parquet_file.read_row_group(row_group_idx, columns=selected_columns)
            frame = pl.from_arrow(table)
            if not isinstance(frame, pl.DataFrame):
                raise TypeError(
                    f"Expected row-group conversion to return DataFrame, got {type(frame).__name__}"
                )
            for local_start, local_end in overlaps:
                yield from iter_data_chunks(
                    frame.slice(local_start, local_end - local_start),
                    chunk_rows,
                )

    def write_table(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        location: str | Path,
        metadata: dict[str, str] | None = None,
        row_group_size: int | None = None,
    ) -> None:
        out_path = Path(self.resolve(location))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            raise FileExistsError(f"File exists: {out_path}")
        if isinstance(data, pl.LazyFrame):
            data.sink_parquet(out_path, metadata=metadata, row_group_size=row_group_size)
            return
        data.write_parquet(out_path, metadata=metadata, row_group_size=row_group_size)

    def open_table_writer(
        self,
        location: str | Path,
        schema: pa.Schema,
        compression: str,
        *,
        use_content_defined_chunking: bool = False,
    ) -> TableWriter:
        return LocalTableWriter(
            Path(self.resolve(location)),
            schema,
            compression,
            use_content_defined_chunking=use_content_defined_chunking,
        )

    def table_size_bytes(self, location: str | Path) -> int | None:
        path = Path(self.resolve(location))
        if not path.exists():
            return None
        return path.stat().st_size
