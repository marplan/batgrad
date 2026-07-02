from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from batgrad.storage.chunks import iter_data_chunks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.storage.store import TableWriter


class LocalTableWriter:
    """Write one parquet table from multiple in-memory chunks.

    `LocalTableWriter` owns an open `pyarrow.parquet.ParquetWriter` for a single
    output file. Each call to `write_table` appends one `polars.DataFrame` as
    another parquet table chunk. The file is created
    eagerly, parent directories are created as needed, and existing output files
    are never overwritten.

    Use this writer when a processing stage produces a table incrementally and
    should not keep the complete result in memory. Call `close` when all chunks
    have been written so parquet metadata can be attached and the file can be
    flushed.

    Examples:
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> from pathlib import Path
        >>> from batgrad.storage.local import LocalDataProcessingStore
        >>> store = LocalDataProcessingStore(Path("/data/batgrad"), create=True)
        >>> schema = pa.schema([pa.field("voltage", pa.float64())])
        >>> writer = store.open_table_writer("normalized/cell.parquet", schema, "zstd")
        >>> writer.write_table(pl.DataFrame({"voltage": [3.7, 3.8]}))
        >>> writer.write_table(pl.DataFrame({"voltage": [3.9]}))
        >>> writer.close({"stage": "normalized"})
    """

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        compression: str,
        *,
        use_content_defined_chunking: bool,
    ) -> None:
        """Create a parquet writer for a new local file.

        Args:
            path: Absolute output path for the parquet file.
            schema: Arrow schema used for all chunks written by this writer.
            compression: Parquet compression codec passed to PyArrow.
            use_content_defined_chunking: Whether PyArrow should use content-defined
                chunking when writing parquet data.

        Raises:
            FileExistsError: If `path` already exists.
        """
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
        """Append one dataframe chunk to the open parquet file.

        Args:
            data: Dataframe chunk to append. Its schema must match the writer schema.
            row_group_size: Optional parquet row-group size for this chunk.
        """
        self._writer.write_table(data.to_arrow(), row_group_size=row_group_size)

    def close(self, metadata: dict[str, str] | None = None) -> None:
        """Attach optional footer metadata and close the writer.

        Args:
            metadata: Optional parquet key-value footer metadata to write before closing.
        """
        if metadata is not None:
            self._writer.add_key_value_metadata(metadata)
        self._writer.close()


class LocalDataProcessingStore:
    """Local filesystem implementation of the data processing store.

    The store exposes a local directory as a dataset storage root. Relative
    locations are resolved below that root, while absolute paths are passed
    through unchanged. It provides the file and parquet operations used by the
    processing pipeline: directory management, source-file discovery, local file
    access, lazy parquet scans, bounded-memory parquet iteration, one-shot table
    writes, and chunked table writers.

    The root must be absolute. Passing `create=True` creates the root directory
    before validation, which is useful for scratch stores and test fixtures.

    Examples:
        Discover raw Excel files while using the ingest spec to ignore
        excluded files such as `README.xlsx`:

        >>> from pathlib import Path
        >>> import polars as pl
        >>> from batgrad.contracts.mapping import DatasetStageId
        >>> from batgrad.data.datasets.pozzato_2022.config import DATASET_SPEC
        >>> from batgrad.storage.local import LocalDataProcessingStore
        >>> store = LocalDataProcessingStore(Path("/data/loc_datasets"))
        >>> raw_spec = DATASET_SPEC.processing_stages[DatasetStageId.ingested]
        >>> raw_root = DATASET_SPEC.source_root(DatasetStageId.raw)
        >>> paths = []
        >>> for pattern in raw_spec.included_file_patterns:
        ...     paths.extend(
        ...         path
        ...         for path in store.list_files(raw_root, pattern=pattern)
        ...         if raw_spec.is_included_file(path)
        ...     )

        Read a parquet table lazily when the whole result does not need to be loaded
        immediately:

        >>> frame = store.scan_table(
        ...     "normalized/cell.parquet",
        ...     columns=("voltage",),
        ...     filters=pl.col("voltage") > 3.6,
        ... ).collect()

        Iterate over a full table in bounded-size chunks:

        >>> def process(chunk):
        ...     pass
        >>> for chunk in store.iter_table_chunks("normalized/cell.parquet", 100_000):
        ...     process(chunk)

        Extract explicit row windows when a manifest points to table slices:

        >>> slices = ((0, 1024), (10_000, 2048))
        >>> for chunk in store.iter_table_slices("normalized/cell.parquet", slices, 512):
        ...     process(chunk)
    """

    def __init__(self, root: str | Path, *, create: bool = False) -> None:
        """Create a local store rooted at an absolute directory.

        Args:
            root: Absolute filesystem directory used as the store root.
            create: Create `root` before validation if it does not exist.

        Raises:
            ValueError: If `root` is not absolute.
            FileNotFoundError: If `root` does not exist and `create` is false.
            NotADirectoryError: If `root` exists but is not a directory.
        """
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
        """Resolve a logical location to a filesystem path.

        Relative locations are resolved below `root`. Absolute locations are
        returned unchanged, which lets callers pass explicit local paths when they
        already have them.

        Args:
            location: Relative or absolute path-like location. If omitted, the
                store root is returned.

        Returns:
            Resolved filesystem path as a string.
        """
        if location is None:
            return self.root
        path = Path(location)
        if path.is_absolute():
            return str(path)
        return str(Path(self.root) / path)

    def create_dir(self, location: str | Path) -> None:
        """Create a directory, including missing parents.

        Args:
            location: Relative or absolute directory location to create.
        """
        Path(self.resolve(location)).mkdir(parents=True, exist_ok=True)

    def delete_dir(self, location: str | Path, *, missing_ok: bool = True) -> None:
        """Delete a directory tree.

        Args:
            location: Relative or absolute directory location to delete.
            missing_ok: Return silently when the directory does not exist.

        Raises:
            FileNotFoundError: If the directory is missing and `missing_ok` is false.
            NotADirectoryError: If the resolved location exists but is not a directory.
        """
        path = Path(self.resolve(location))
        if not path.exists():
            if missing_ok:
                return
            raise FileNotFoundError(f"Directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        shutil.rmtree(path)

    def delete_file(self, location: str | Path, *, missing_ok: bool = True) -> None:
        """Delete a file.

        Args:
            location: Relative or absolute file location to delete.
            missing_ok: Return silently when the file does not exist.

        Raises:
            FileNotFoundError: If the file is missing and `missing_ok` is false.
            IsADirectoryError: If the resolved location exists but is not a file.
        """
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
        """List non-hidden files below a store location.

        Files are matched with `pathlib.Path.rglob`, returned as sorted POSIX
        paths relative to `root`, and skipped when any path component below
        `location` starts with `.`.

        Args:
            location: Directory to search. Defaults to the store root.
            pattern: Recursive glob pattern, such as `"*.xlsx"` or
                `"**/*.parquet"`.

        Returns:
            Sorted tuple of matching file paths relative to the store root.

        Raises:
            FileNotFoundError: If `location` is missing or is not a directory.

        Examples:
            Use the dataset ingest spec to find Pozzato raw Excel files while
            excluding paths such as `README.xlsx`:

            >>> paths = []
            >>> for pattern in raw_spec.included_file_patterns:
            ...     paths.extend(
            ...         path
            ...         for path in store.list_files(raw_root, pattern=pattern)
            ...         if raw_spec.is_included_file(path)
            ...     )
        """
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
        """Yield a concrete local path for a logical store location.

        Args:
            location: Relative or absolute file location.

        Yields:
            Resolved local `pathlib.Path`.

        Examples:
            The Pozzato raw adapter uses this to pass a local Excel path to
            `fastexcel`:

            >>> with store.local_file(source_path) as path:
            ...     excel = fastexcel.read_excel(path)
        """
        yield Path(self.resolve(location))

    def scan_table(
        self,
        location: str | Path | tuple[str | Path, ...],
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        """Create a lazy parquet scan for one or more table locations.

        Args:
            location: One parquet location, or a tuple of locations to scan as one
                lazy frame.
            columns: Optional columns to select from the scan.
            filters: Optional Polars expression applied to the lazy frame.
            limit: Optional maximum number of rows to read.

        Returns:
            Polars lazy frame for the selected parquet data.

        Examples:
            Load selected manifest rows without materializing unrelated columns:

            >>> manifest = store.scan_table(
            ...     "manifest.parquet",
            ...     columns=("protocol", "segments"),
            ...     filters=pl.col("protocol") == "cycling",
            ... ).collect()
        """
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
        """Yield sequential parquet batches from one or more complete tables.

        This method streams every row from each table in order, with batches of
        at most `chunk_rows` before any optional filter is applied. Use it when
        a stage should process a complete table without loading it all into
        memory.

        Args:
            location: One parquet location, or a tuple of locations to process in
                order.
            chunk_rows: Maximum number of rows requested per parquet batch before
                filtering.
            columns: Optional columns to read from the parquet file.
            filters: Optional Polars expression applied to each batch after it is
                read.

        Yields:
            Non-empty dataframe chunks.

        Raises:
            ValueError: If `chunk_rows` is less than one.

        Examples:
            Consume a scratch table in bounded-size chunks before deleting it:

            >>> for chunk in store.iter_table_chunks(temp_path, chunk_rows):
            ...     writer.append(chunk, metadata, source_paths)
            >>> store.delete_file(temp_path, missing_ok=True)
        """
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
        """Yield selected row windows from a single parquet table.

        Unlike `iter_table_chunks`, this method does not scan the whole table. It
        reads only row groups that overlap the requested `slices` and emits the
        selected rows in chunks of at most `chunk_rows`. Use it for
        manifest segments or sharded selections that reference specific row
        ranges.

        Args:
            location: Parquet table location to read from.
            slices: Row windows as `(row_start, row_count)` tuples.
            chunk_rows: Maximum number of rows yielded per chunk.
            columns: Optional columns to read from the parquet file.

        Yields:
            Non-empty dataframe chunks for the requested row windows.

        Raises:
            ValueError: If `chunk_rows` is less than one.

        Examples:
            Read a manifest segment that points to a row range in a shard:

            >>> slices = ((row_start, row_count),)
            >>> for chunk in store.iter_table_slices(path, slices, 100_000):
            ...     process(chunk)
        """
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
        """Write one parquet table and fail if the output already exists.

        Args:
            data: Polars dataframe or lazy frame to write.
            location: Relative or absolute output table location.
            metadata: Optional parquet key-value footer metadata.
            row_group_size: Optional parquet row-group size.

        Raises:
            FileExistsError: If `location` already exists.

        Examples:
            Write a prepared scratch table during ingestion:

            >>> store.write_table(data, temp_path, row_group_size=config.row_group_size)

            Write a manifest with parquet footer metadata:

            >>> store.write_table(manifest, "manifest.parquet", metadata={"stage": "ingested"})
        """
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
        """Open a parquet writer for incrementally writing one table.

        Args:
            location: Relative or absolute output table location.
            schema: Arrow schema for all chunks written through the returned writer.
            compression: Parquet compression codec passed to PyArrow.
            use_content_defined_chunking: Whether PyArrow should use content-defined
                chunking when writing parquet data.

        Returns:
            Table writer that keeps the parquet file open until closed.

        Raises:
            FileExistsError: If `location` already exists.

        Examples:
            Write bounded normalization output as chunks become available:

            >>> writer = store.open_table_writer(temp_path, chunk.to_arrow().schema, "zstd")
            >>> writer.write_table(chunk, row_group_size=config.row_group_size)
            >>> writer.close()
        """
        return LocalTableWriter(
            Path(self.resolve(location)),
            schema,
            compression,
            use_content_defined_chunking=use_content_defined_chunking,
        )

    def table_size_bytes(self, location: str | Path) -> int | None:
        """Return the table file size in bytes.

        This is used by sharding to decide when an open shard should roll over to
        a new file.

        Args:
            location: Relative or absolute table location.

        Returns:
            File size in bytes, or `None` if the file does not exist.
        """
        path = Path(self.resolve(location))
        if not path.exists():
            return None
        return path.stat().st_size
