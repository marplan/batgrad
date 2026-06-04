from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from batgrad.data.datasets.pozzato_2022.config import DATASET_SPEC as POZZATO_2022

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import DatasetSpec

DatasetIds = Literal["pozzato-2022"]

_DATASETS: dict[str, DatasetSpec] = {
    POZZATO_2022.dataset_id: POZZATO_2022,
}


def get_dataset(dataset_id: DatasetIds) -> DatasetSpec:
    """Get a dataset specification by its ID.

    Central registry of all datasets. All dataset access should go through this function.

    Args:
        dataset_id: The unique ID of the dataset.

    Returns:
        The dataset specification.

    Raises:
        ValueError: If the dataset ID is unknown.

    Examples:
        >>> from batgrad.data.datasets.registry import get_dataset
        >>> pozzato_2022 = get_dataset("pozzato-2022")
        >>> data_store = get_storage()  # env var holds root path
        >>> data_paths = data_store.list_files(pozzato_2022.location.root(), pattern="*.parquet")
        >>> lf = data_store.scan_table(data_paths[0])

    """
    dataset = _DATASETS.get(dataset_id)
    if dataset is None:
        known = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset_id={dataset_id!r}. Known datasets: {known}")
    return dataset
