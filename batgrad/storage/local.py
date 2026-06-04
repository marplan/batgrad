from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator


class LocalDataStore:
    def __init__(self, root: str | Path) -> None:
        self.root = str(Path(root).resolve())

    def resolve(self, location: str | Path | None = None) -> str:
        if location is None:
            return self.root
        path = Path(location)
        if path.is_absolute():
            return str(path)
        return str(Path(self.root) / path)

    def exists(self, location: str | Path | None = None) -> bool:
        return Path(self.resolve(location)).exists()

    def _relative_location(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @staticmethod
    def _is_visible_path(path: Path, root: Path) -> bool:
        return not any(part.startswith(".") for part in path.relative_to(root).parts)

    def _list_paths(
        self,
        location: str | Path | None,
        pattern: str,
        *,
        dirs: bool,
    ) -> tuple[str, ...]:
        root = Path(self.resolve(location))
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")
        paths: list[str] = []
        for path in root.rglob(pattern):
            if not self._is_visible_path(path, root):
                continue
            if dirs and path.is_dir():
                paths.append(self._relative_location(path))
            if not dirs and path.is_file():
                paths.append(self._relative_location(path))
        return tuple(sorted(paths))

    def list_dirs(
        self,
        location: str | Path | None = None,
        pattern: str = "*",
    ) -> tuple[str, ...]:
        return self._list_paths(location, pattern, dirs=True)

    def list_files(
        self,
        location: str | Path | None = None,
        pattern: str = "*",
    ) -> tuple[str, ...]:
        return self._list_paths(location, pattern, dirs=False)

    def _resolve_table_locations(
        self,
        location: str | Path | tuple[str | Path, ...],
    ) -> str | list[str]:
        if isinstance(location, tuple):
            return [self.resolve(path) for path in location]
        return self.resolve(location)

    def scan_table(
        self,
        location: str | Path | tuple[str | Path, ...],
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        lf = pl.scan_parquet(self._resolve_table_locations(location))

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

        resolved = self._resolve_table_locations(location)
        paths = resolved if isinstance(resolved, list) else [resolved]
        selected_columns = list(columns) if columns is not None else None

        # NOTE: Using pyarrow here because polars collect_batches is marked unstable
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            for arrow_batch in parquet_file.iter_batches(
                batch_size=chunk_rows,
                columns=selected_columns,
            ):
                df = pl.from_arrow(arrow_batch)
                if not isinstance(df, pl.DataFrame):
                    raise TypeError(
                        f"Expected batch conversion to return DataFrame, got {type(df).__name__}",
                    )

                if filters is not None:
                    df = df.filter(filters)

                if df.height > 0:
                    yield df

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
