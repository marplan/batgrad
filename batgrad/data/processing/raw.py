from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import DatasetSpec
    from batgrad.data.processing.config import RawStageConfig
    from batgrad.storage.store import DataStore


def raw_to_parquet(
    spec: DatasetSpec,
    input_store: DataStore,
    output_store: DataStore,
    config: RawStageConfig,
) -> None:
    """Convert raw data to parquet format while keeping all of the original data.

    Args:
        spec: Dataset specification.
        input_store: Input data store.
        output_store: Output data store.
        config: Raw stage configuration.

    """
