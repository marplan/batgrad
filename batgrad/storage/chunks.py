from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl


def iter_data_chunks(data: pl.DataFrame, chunk_rows: int) -> Iterator[pl.DataFrame]:
    """Yield non-empty chunks from an in-memory dataframe."""
    if data.height <= chunk_rows:
        if data.height > 0:
            yield data
        return
    for offset in range(0, data.height, chunk_rows):
        chunk = data.slice(offset, chunk_rows)
        if chunk.height > 0:
            yield chunk
