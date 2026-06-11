from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from batgrad.data.datasets.pozzato_2022.pipeline import Pozzato2022Dataset
from batgrad.data.processing.config import FailureMode, RawStageConfig

if TYPE_CHECKING:
    from batgrad.data.datasets.specs import Dataset

DatasetIds = Literal["pozzato-2022"]

_DATASETS: dict[str, type[Dataset]] = {
    Pozzato2022Dataset.spec.dataset_id: Pozzato2022Dataset,
}


@overload
def get_dataset(dataset_id: Literal["pozzato-2022"]) -> Pozzato2022Dataset: ...


# FIX: 2 identical overloads only for debugging purposes
@overload
def get_dataset(dataset_id: Literal["pozzato-2022"]) -> Pozzato2022Dataset: ...


def get_dataset(dataset_id: DatasetIds) -> Dataset:
    """Get a dataset by its ID.

    Central registry of all datasets. Dataset access should go through this function.

    Args:
        dataset_id: The unique ID of the dataset.

    Returns:
        The dataset object.

    Raises:
        ValueError: If the dataset ID is unknown.

    Examples:
        >>> from batgrad.data.datasets.registry import get_dataset
        >>> pozzato_2022 = get_dataset("pozzato-2022")
        >>> data_store = get_storage()  # env var holds root path
        >>> data_paths = data_store.list_files(
        ...     pozzato_2022.spec.location.root(), pattern="*.parquet"
        ... )
        >>> lf = data_store.scan_table(data_paths[0])

    """
    dataset_cls = _DATASETS.get(dataset_id)
    if dataset_cls is None:
        known = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset_id={dataset_id!r}. Known datasets: {known}")
    return dataset_cls()


if __name__ == "__main__":
    import sys

    from batgrad._loggers import configure
    from batgrad.data.processing.config import NormalizeStageConfig
    from batgrad.storage.factory import get_storage

    configure("INFO")

    new_data_store = get_storage(root="/data/loc_datasets/", backend="local")
    pozzato_2022 = get_dataset("pozzato-2022")
    pozzato_2022.raw_to_parquet(
        new_data_store,
        new_data_store,
        RawStageConfig(
            n_jobs=3,
            worker_polars_max_threads=2,
            failure_mode=FailureMode.CONTINUE,
        ),
    )
    sys.exit(0)
    pozzato_2022.normalize(
        new_data_store,
        new_data_store,
        NormalizeStageConfig(
            max_batch_rows=500_000,
            n_jobs=3,
            worker_polars_max_threads=2,
            failure_mode=FailureMode.CONTINUE,
            apply_resampling=True,
        ),
    )

    t = 4

    data_files = [
        path
        for path in new_data_store.list_files(
            pozzato_2022.spec.location.root(),
            pattern="*.parquet",
        )
        if "source=parquet" in path
    ][0:10]
