from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import DatasetSpec
    from batgrad.data.processing.config import NormalizeStageConfig
    from batgrad.storage.store import DataStore


def normalize_dataset(
    spec: DatasetSpec,
    input_store: DataStore,
    output_store: DataStore,
    config: NormalizeStageConfig,
) -> None:
    """Normalize a dataset to a standard parquet format following PyBaMM's naming conventions.

    Args:
        spec: Dataset specification.
        input_store: Input data store.
        output_store: Output data store.
        config: Normalization stage configuration.

    """
