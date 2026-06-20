from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from batgrad.data.datasets.pozzato_2022.config import DATASET as POZZATO_2022

if TYPE_CHECKING:
    from batgrad.data.datasets.config import Dataset

DatasetId = Literal["pozzato-2022"]

_DATASETS: dict[str, Dataset] = {
    POZZATO_2022.spec.dataset_id: POZZATO_2022,
}


def get_dataset(dataset_id: DatasetId) -> Dataset:
    """Get a registered dataset bundle by ID."""
    dataset = _DATASETS.get(dataset_id)
    if dataset is None:
        known = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset_id={dataset_id!r}. Known datasets: {known}")
    return dataset
